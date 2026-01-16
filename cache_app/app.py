import tempfile
import redis
import json
import pandas as pd
import geopandas as gpd
import folium
import numpy as np
from functools import wraps
import pymongo
from typing import Callable, List, Dict, Any
from PyQt5.QtCore import QUrl
from shapely.geometry import Point, shape
from shapely import bounds
from matplotlib import colormaps, colors
from time import perf_counter

from GUII import *

HOST = 'localhost'

# Redis client setup
redis_client = redis.Redis(
    host=HOST,
    port=6379,
    db=0,
    decode_responses=True
)

# MongoDB client setup
mongo_connection = pymongo.MongoClient('mongodb://' + HOST)
mongo_client = mongo_connection.PAG2

mongo_powiaty = list(mongo_client['powiaty_wojewodztwa'].find())
POWIATY = [
    (
        shape(powiat['geometry']),
        powiat['nazwa_powiat'].lower(),
        powiat['nazwa_woj'].lower()
    )
    for powiat in mongo_powiaty
]


# Cache setup
def generate_cache_key(args: List, kwargs: Dict[str, Any]) -> str:
    key_string = ':'.join(str(arg) for arg in args)
    if kwargs:
        key_string += ':' + ':'.join(f"{k}={v}" for k, v in sorted(kwargs.items()))
    return key_string


def cache_result() -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> (Dict, bool, str):
            cache_key = generate_cache_key(args, kwargs)
            try:
                cached_data = redis_client.get(cache_key)
                if cached_data:
                    print("Cache hit @ " + cache_key)
                    return json.loads(cached_data), True, cache_key
            except redis.RedisError as e:
                print(f"Redis error: {e}")

            # Cache miss - compute the result
            print("Cache miss, computing result...")
            result = func(*args, **kwargs)
            expiry = 1800  # 30 minut cache
            try:
                redis_client.setex(cache_key, expiry, json.dumps(result))
                print("Cached result @ " + cache_key)
            except redis.RedisError as e:
                print(f"Redis error: {e}")

            return result, False, cache_key

        return wrapper

    return decorator


# Load collection from CSV to MongoDB and support functions
def load_collection_to_mongo(csv_path: str):
    with open('../dane/effacility.geojson', 'r', encoding='utf-8') as f:
        effacility = json.load(f)['features']
    effacility_map = {str(feature['properties']['ifcid']): feature for feature in effacility}

    df = pd.read_csv(csv_path, sep=';', decimal=',', dtype={'ID': str}, header=None, usecols=[0, 1, 2, 3],
                     names=['ID', 'Kod', 'DataCzas', 'value'])
    df[['date', 'time']] = df['DataCzas'].str.split(' ', expand=True)
    df.drop(columns=['DataCzas'])
    code = df['Kod'].iloc[0]

    def process_group(group, effacility_map):
        group_name = group.name
        effacility_data = effacility_map.get(group_name, {})
        geometry = effacility_data.get('geometry', {})
        coordinates = geometry.get('coordinates', [None, None])
        print('znajdywanie')
        powiat, wojewodztwo = find_powiat_and_woj(coordinates)
        print('znaleziono')
        records = group.groupby('date', group_keys=False).apply(
            lambda day_group: day_group[['time', 'value']].to_dict(orient='records'),
            include_groups=False
        ).to_dict()

        return {
            'properties': effacility_data.get('properties', {}),
            'powiat': powiat,
            'wojewodztwo': wojewodztwo,
            'geometry': geometry,
            'records': records
        }

    result = (
        df.groupby('ID', group_keys=False)
        .apply(lambda group: process_group(group, effacility_map), include_groups=False).to_dict()
    )

    collection = mongo_client[code]
    collection.delete_many({})
    for id_, data in result.items():
        document = {
            'ID': id_,
            'properties': data['properties'],
            'powiat': data['powiat'],
            'wojewodztwo': data['wojewodztwo'],
            'geometry': data['geometry'],
            'records': data['records']
        }
        collection.insert_one(document)
    return collection


def powiaty_wojewodztwa_to_mongo():
    with open('../dane/powiaty.geojson', 'r', encoding='utf-8') as f:
        powiaty = json.load(f)['features']
    with open('../dane/woj.geojson', 'r', encoding='utf-8') as f:
        wojewodztwa = json.load(f)['features']
    # Konwersja na GeoDataFrame
    gdf_powiaty = gpd.GeoDataFrame.from_features(powiaty).set_crs("EPSG:2180")
    gdf_wojewodztwa = gpd.GeoDataFrame.from_features(wojewodztwa).set_crs("EPSG:2180")

    gdf_wojewodztwa = gdf_wojewodztwa[['name', 'geometry']].rename(columns={'name': 'nazwa_woj'})
    gdf_powiaty = gdf_powiaty[['name', 'geometry']].rename(columns={'name': 'nazwa_powiat'})

    gdf_powiaty['geom'] = gdf_powiaty.geometry.copy()
    gdf_powiaty['centroid'] = gdf_powiaty.geometry.centroid
    gdf_powiaty = gdf_powiaty.set_geometry('centroid')

    powiaty_z_wojewodztwem = gpd.sjoin(
        left_df=gdf_powiaty,
        right_df=gdf_wojewodztwa,
        predicate='within',
        how='left')

    powiaty_z_wojewodztwem['geometry'] = powiaty_z_wojewodztwem['geom']
    powiaty_z_wojewodztwem = powiaty_z_wojewodztwem.set_geometry('geometry')

    powiaty_z_wojewodztwem = powiaty_z_wojewodztwem[['nazwa_powiat', 'nazwa_woj', 'geometry']]

    collection = mongo_client['powiaty_wojewodztwa']
    collection.delete_many({})
    for _, row in powiaty_z_wojewodztwem.iterrows():
        document = {
            'nazwa_powiat': row['nazwa_powiat'],
            'nazwa_woj': row['nazwa_woj'],
            'geometry': row['geometry'].__geo_interface__,
            'bbox': bounds(row['geometry']).tolist()
        }
        collection.insert_one(document)
    return powiaty_z_wojewodztwem


def find_powiat_and_woj(point):
    try:
        point = Point(point)
    except TypeError as e:
        return None, None

    for geom, powiat, woj in POWIATY:
        if geom.contains(point):
            return powiat, woj
    return None, None


# Main function to calculate stats - now returns complete map data
@cache_result()
def calculate_statistics(code: str, woj_name: str, /, type: str = 'average') -> Dict:
    woj_name = woj_name.lower()

    # Get data from Mongo
    collection = mongo_client[code]
    documents = list(collection.find({'wojewodztwo': woj_name}))

    powiaty = mongo_client['powiaty_wojewodztwa']
    powiaty = list(powiaty.find({'nazwa_woj': woj_name}))

    powiaty_with_station = np.unique([doc['powiat'] for doc in documents])

    # Calculate hourly stats for each powiat
    hourly_stats = {}
    for powiat in powiaty:
        nazwa = powiat['nazwa_powiat']
        if nazwa not in powiaty_with_station:
            print('No station in powiat', nazwa)
            hourly_stats[nazwa] = [None] * 24
            continue

        stacje_powiatu = [doc for doc in documents if doc['powiat'] == nazwa]
        stats = [0] * 24
        div = [0] * 24

        for stacja in stacje_powiatu:
            for days in stacja['records'].values():
                for record in days:
                    idx = int(record['time'][:2])
                    stats[idx] += record['value']
                    div[idx] += 1

        for i, (opad_miesieczny_w_godzine, n) in enumerate(zip(stats, div)):
            stats[i] = opad_miesieczny_w_godzine / n if n > 0 else None
        hourly_stats[nazwa] = stats

    # Get geometries and prepare features for GeoJSON
    collection = mongo_client['powiaty_wojewodztwa']
    powiaty_features = list(collection.find({'nazwa_woj': woj_name}))

    features = []
    for feature in powiaty_features:
        nazwa = feature['nazwa_powiat']

        # Create GeoJSON-compatible feature
        geojson_feature = {
            'type': 'Feature',
            'properties': {
                'nazwa_powiat': nazwa,
                'nazwa_woj': feature['nazwa_woj'],
                'hourly_stats': hourly_stats[nazwa]  # Store all 24 hours
            },
            'geometry': feature['geometry']
        }
        features.append(geojson_feature)

    # Calculate centroid using shapely (can't serialize GeoDataFrame to JSON)
    temp_gdf = gpd.GeoDataFrame.from_features(features, crs='EPSG:2180')
    temp_gdf = temp_gdf.to_crs('EPSG:4326')
    combined_centroid = temp_gdf.union_all().centroid

    return {
        'features': features,
        'centroid': [combined_centroid.y, combined_centroid.x]  # lat, lon
    }


class MyApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.start_slider.sliderReleased.connect(self.update_map)
        self.ui.end_slider.sliderReleased.connect(self.update_map)
        self.ui.comboBox.currentIndexChanged.connect(self.update_map)
        self.update_map()

    def update_map(self):

        code = 'B00606S'
        wojewodztwo_name = self.ui.comboBox.currentText()
        time_range = self.ui.get_time_range()

        # Get cached map data
        t0 = perf_counter()
        map_data, from_cache, redis_key = calculate_statistics(code, wojewodztwo_name)
        t1 = perf_counter()
        self.ui.cache_info_label.setText(
            f'Ostatnie wczytanie danych z: {"Redis:"+redis_key if from_cache else "MongoDB"}'
        )
        self.ui.cache_hit_label.setText(f'Czas pobierania: {round((t1-t0)*1000)} ms')

        # Calculate summed stats for the selected time range
        summed_stats = {}
        for feature in map_data['features']:
            nazwa = feature['properties']['nazwa_powiat']
            hourly = feature['properties']['hourly_stats']

            stat_sum = None
            for i, stat in enumerate(hourly):
                if stat is None or i not in time_range:
                    continue
                stat_sum = stat_sum + stat if stat_sum is not None else stat

            summed_stats[nazwa] = stat_sum

        # Calculate min/max for color scaling
        valid_values = [v for v in summed_stats.values() if v is not None]
        max_val = max(valid_values) if valid_values else 1
        min_val = min(valid_values) if valid_values else 0

        # Add summed stats to features
        for feature in map_data['features']:
            nazwa = feature['properties']['nazwa_powiat']
            feature['properties']['stat'] = summed_stats[nazwa]

        # Create GeoDataFrame for rendering
        gdf = gpd.GeoDataFrame.from_features(map_data['features'], crs='EPSG:2180')
        gdf = gdf.to_crs('EPSG:4326')

        # Style function
        def style_powiaty(feature):
            color = '#FFA500'
            weight = 1
            fill_opacity = 0.1

            if feature['properties']['stat'] is not None:
                cmap = colormaps['coolwarm']
                normalized = (feature['properties']['stat'] - min_val) / (
                            max_val - min_val) if max_val != min_val else 0.5
                color = colors.to_hex(cmap(normalized))
                fill_opacity = 0.6
                weight = 3

            return {
                'fillColor': color,
                'color': 'black',
                'weight': weight,
                'fillOpacity': fill_opacity
            }

        # Create map
        c_lat, c_lon = map_data['centroid']
        m = folium.Map(location=(c_lat, c_lon), zoom_start=8)

        folium.GeoJson(
            gdf,
            name='Powiaty',
            style_function=style_powiaty,
            tooltip=folium.GeoJsonTooltip(
                fields=['nazwa_powiat', 'stat'],
                aliases=['Powiat:', 'Średnia:']
            )
        ).add_to(m)

        tmpfile = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
        m.save(tmpfile.name)
        tmpfile.close()

        self.ui.webEngineView.load(QUrl.fromLocalFile(tmpfile.name))
        t2 = perf_counter()
        print('Czas renderowania mapy:', round((t2 - t1) * 1000), 'ms')
        self.ui.load_time_label.setText(f'Łączny czas ładowania: {round((t2 - t0) * 1000)} ms')


if __name__ == "__main__":
    redis_client.flushall()
    # MONGO SETUP:
    # powiaty_wojewodztwa_to_mongo()
    load_collection_to_mongo('../dane/B00606S_2023_04.csv')

    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
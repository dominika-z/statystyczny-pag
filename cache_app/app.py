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
from collections import defaultdict
from PyQt5.QtCore import QUrl
from shapely.geometry import Point, shape, Polygon, MultiPolygon
from shapely import bounds
from matplotlib import colormaps, colors

from GUII import *

HOST = 'localhost'

# Redis client setup
redis_client = redis.Redis(
    host = HOST,
    port = 6379,
    db = 0,
    decode_responses=True
)

# MongoDB client setup
mongo_connection = pymongo.MongoClient('mongodb://' + HOST)
mongo_client = mongo_connection.PAG2

# Cache setup
def generate_cache_key(args: List, kwargs: Dict[str, Any]) -> str:
    key_string = ':'.join(args)
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
            expiry = 1800                       # 30 minut cache
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

    df = pd.read_csv(csv_path, sep=';', decimal=',',  dtype={'ID': str},header=None, usecols=[0,1,2,3], names=['ID', 'Kod', 'DataCzas', 'value'])
    df[['date', 'time']] = df['DataCzas'].str.split(' ', expand=True)
    df.drop(columns=['DataCzas'])
    code = df['Kod'].iloc[0]

    # To jest bardzo skomplikowana operacja z tym grupowaniem i szczerze sam to mało rozumiem
    # Chodzi o to, że grupuje po ID (kod stacji), a potem po dacie, ale pandas.groupby jest jakieś zagmatwane
    def process_group(group, effacility_map):
        group_name = group.name
        effacility_data = effacility_map.get(group_name, {})
        geometry = effacility_data.get('geometry', {})
        coordinates = geometry.get('coordinates', [None, None])
        powiat, wojewodztwo = find_powiat_and_woj(coordinates)

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
        predicate='within',  # Operacja: powiat (left) wewnątrz województwa (right)
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
            'bbox': bounds(row['geometry']).tolist()        # minx, miny, maxx, maxy
        }
        collection.insert_one(document)
    return powiaty_z_wojewodztwem
def find_powiat_and_woj(point):
    try:
        point = Point(point)
    except TypeError as e:
        return None, None

    # Query the MongoDB collection
    powiaty_wojewodztwa = mongo_client['powiaty_wojewodztwa']
    documents = list(powiaty_wojewodztwa.find({}))

    for doc in documents:
        geometry = shape(doc['geometry'])
        if geometry.contains(point):
            return doc['nazwa_powiat'], doc['nazwa_woj']
    return None, None

# Main funtion to calculate stats
@cache_result()
def calculate_statistics(code: str, woj_name: str, /, type: str = 'average') -> Dict:
    woj_name = woj_name.lower()

    # Get data from Mongo
    collection = mongo_client[code]
    documents = list(collection.find({'wojewodztwo': woj_name}))

    powiaty = mongo_client['powiaty_wojewodztwa']
    powiaty = list(powiaty.find({'nazwa_woj': woj_name}))

    powiaty_with_station = np.unique([doc['powiat'] for doc in documents])
    d = defaultdict(list)
    for powiat in powiaty:
        nazwa = powiat['nazwa_powiat']
        if nazwa not in powiaty_with_station:
            print('No station in powiat', nazwa)
            d[nazwa] = [None] * 24
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
            stats[i] = opad_miesieczny_w_godzine / n
        d[nazwa] = stats
    d = dict(d)
    return d

def expand_bbox(bbox, bbox2):
    try:
        if bbox2[0] < bbox[0]:
            bbox[0] = bbox2[0]
        if bbox2[1] < bbox[1]:
            bbox[1] = bbox2[1]
        if bbox2[2] > bbox[2]:
            bbox[2] = bbox2[2]
        if bbox2[3] > bbox[3]:
            bbox[3] = bbox2[3]
    except TypeError:
        bbox = bbox2
    return bbox

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

        stats, from_chache, redis_key  = calculate_statistics(code, wojewodztwo_name)

        # Filter by time range
        summed = dict()
        for nazwa, value in stats.items():
            stat_sum = None
            for i, stat in enumerate(value):
                if stat is None:
                    continue
                if i not in time_range:
                    continue
                try:
                    stat_sum += stat
                except TypeError:
                    stat_sum = stat
            summed[nazwa] = stat_sum
        max_val = max([value for value in summed.values() if value is not None])
        min_val = min([value for value in summed.values() if value is not None])

        collection = mongo_client['powiaty_wojewodztwa']
        powiaty_features = list(collection.find({'nazwa_woj': wojewodztwo_name}))
        bbox = [None, None, None, None]
        for feature in powiaty_features:
            feature.pop('_id') # Usunięcie _id
            feature['geometry'] = shape(feature['geometry'])
            feature_bbox = feature['geometry'].bounds
            bbox = expand_bbox(bbox, feature_bbox)

            # dodanie statystyk do powiatu
            nazwa = feature['nazwa_powiat']
            feature['stat'] = summed[nazwa]

        powiaty_gdf = gpd.GeoDataFrame(powiaty_features, geometry='geometry', crs='EPSG:2180')
        powiaty_gdf = powiaty_gdf.to_crs('EPSG:4326')
        combined_centroid = powiaty_gdf.union_all().centroid
        c_lon, c_lat = combined_centroid.x, combined_centroid.y

        def style_powiaty(feature):
            # Domyślny styl
            color = '#FFA500'  # Pomarańczowy
            weight = 1
            fill_opacity = 0.1

            if feature['properties']['stat'] is not None:
                cmap = colormaps['coolwarm']
                normalized = (feature['properties']['stat'] - min_val) / max_val
                color = colors.to_hex(cmap(normalized))
                fill_opacity = 0.6
                weight = 3

            return {
                'fillColor': color,
                'color': 'black',
                'weight': weight,
                'fillOpacity': fill_opacity
            }
        m = folium.Map(location=(c_lat, c_lon), zoom_start=8)

        folium.GeoJson(
            powiaty_gdf,
            name='Powiaty',
            style_function=style_powiaty,
            tooltip=folium.GeoJsonTooltip(fields=['nazwa_powiat', 'stat'], aliases=['Powiat:', 'Średia:'])
        ).add_to(m)

        tmpfile = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
        m.save(tmpfile.name)
        tmpfile.close()

        # Wczytanie mapy w QWebEngineView
        self.ui.webEngineView.load(QUrl.fromLocalFile(tmpfile.name))














if __name__ == "__main__":
    # MONGO SETUP:
    # powiaty_wojewodztwa_to_mongo()
    # load_collection_to_mongo('../dane/B00606S_2023_04.csv')

    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
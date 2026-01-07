import sys
import io
import folium # pip install folium
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout
from PyQt5.QtWebEngineWidgets import QWebEngineView # pip install PyQtWebEngin
from GUII import *
import redis 
import json
import folium
import geopandas as gpd
import pandas as pd
from PyQt5.QtCore import QUrl
import tempfile
import pymongo


def setdb():
    with open("powiaty.geojson", "r", encoding="utf-8") as f:
        powiaty_raw = json.load(f)["features"] 

    with open("woj.geojson", "r", encoding="utf-8") as f:
        wojewodztwa_raw = json.load(f)["features"] 

    #Connect to the Redis database 
    pool= redis.ConnectionPool(host='127.0.0.1', port=6379, db=0) 
    db = redis.Redis(connection_pool=pool) 

    combine_data(powiaty_raw, wojewodztwa_raw)

    for i, feature in enumerate(powiaty_raw):
        objectkey = f"powiat:{i}"
        objectval = json.dumps(feature) #konwersja string na JSON
        db.set(objectkey, objectval)

    for i, feature in enumerate(wojewodztwa_raw):
        objectkey = f"wojewodztwo:{i}"
        objectval = json.dumps(feature) #konwersja string na JSON
        db.set(objectkey, objectval)


def get_geojson_data(db, key_prefix):
    #pobiera wszystkie obiekty z Redis pasujące do danego prefiksu klucza

    keys = db.keys(f"{key_prefix}:*") 
    
    features = []
    if not keys:
        print(f"Nie znaleziono kluczy z prefiksem: {key_prefix}")
        return None
        
    for key in keys:
        feature_data = db.get(key)
        if feature_data:
            features.append(json.loads(feature_data))
            
    return features

def combine_data(powiaty, wojewodztwa):
    # Konwersja na GeoDataFrame
    gdf_powiaty = gpd.GeoDataFrame.from_features(powiaty).set_crs("EPSG:2180")
    gdf_wojewodztwa = gpd.GeoDataFrame.from_features(wojewodztwa).set_crs("EPSG:2180")

    gdf_wojewodztwa = gdf_wojewodztwa[['name', 'geometry']].rename(columns={'name': 'nazwa_woj'})
    gdf_powiaty = gdf_powiaty[['name', 'geometry']].rename(columns={'name': 'nazwa_powiat'})

    powiaty_z_wojewodztwem = gpd.sjoin(
    left_df=gdf_powiaty,
    right_df=gdf_wojewodztwa,
    predicate='within',  # Operacja: powiat (left) wewnątrz województwa (right)
    how='left')

    powiaty_z_wojewodztwem = powiaty_z_wojewodztwem[['nazwa_powiat', 'nazwa_woj', 'geometry']]

    for i, feature in enumerate(powiaty_z_wojewodztwem.iterfeatures()):
        objectkey = f"powzwoj:{i}"
        objectval = json.dumps(feature) 
        db.set(objectkey, objectval)
   
   

def get_powiaty_for_woj(db, woj_name):
    keys = db.keys("powzwoj:*")
    features = []

    for key in keys:
        raw = db.get(key)
        #print(type(raw), raw[:100])
        if raw is None:
            continue
        data = json.loads(raw.decode("utf-8")) #dekodyje bytes na str

        if data.get("properties", {}).get("nazwa_woj") == woj_name:
            features.append(data)
    if not features:
        print(f"Brak danych dla województwa: {woj_name}")
        return gpd.GeoDataFrame(columns=['geometry'], geometry='geometry').set_crs("EPSG:2180")
    return gpd.GeoDataFrame.from_features(features).set_crs("EPSG:2180")


def get_woj(db, woj_name):
    keys = db.keys("wojewodztwo:*")

    for key in keys:
        data = json.loads(db.get(key))
        if data["properties"]["name"] == woj_name:
            gdf = gpd.GeoDataFrame.from_features([data]).set_crs("EPSG:2180")
            gdf = gdf.rename(columns={"name": "nazwa_woj"})
            return gdf

class MyApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        self.ui.comboBox.currentIndexChanged.connect(self.update_map)
        self.update_map()

    def update_map(self):
        pool= redis.ConnectionPool(host='127.0.0.1', port=6379, db=0) 
        redisdb = redis.Redis(connection_pool=pool) 
        woj_name = self.ui.comboBox.currentText()
    
        gdf_pow = get_powiaty_for_woj(redisdb, woj_name)
        gdf_woj = get_woj(redisdb, woj_name)

        centroid = gdf_woj.geometry.centroid.iloc[0]
                
        gdf_woj = gdf_woj.to_crs("EPSG:4326")
        gdf_pow = gdf_pow.to_crs("EPSG:4326")

        c_gdf = gpd.GeoDataFrame(geometry=[centroid], crs="EPSG:2180").to_crs("EPSG:4326")
        c_lon, c_lat = c_gdf.geometry.iloc[0].x, c_gdf.geometry.iloc[0].y

        m = folium.Map(location=[c_lat, c_lon], zoom_start=7)

        folium.GeoJson(
            gdf_woj,
            name="Województwo",
            style_function=lambda x: {
                'fillColor': '#FFA500', 
                'color': 'white', 
                'weight': 3, 
                'fillOpacity': 0.1
            },
            #tooltip=folium.GeoJsonTooltip(fields=['nazwa_woj'], aliases=['Województwo:']),
            #highlight_function=lambda x: {'weight': 5, 'color': 'red'}
        ).add_to(m)
        folium.GeoJson(
            gdf_pow,
            name="Powiaty",
            style_function=lambda x: {
                'fillColor': '#FFA500', 
                'color': 'white', 
                'weight': 3, 
                'fillOpacity': 0.1
            },
            tooltip=folium.GeoJsonTooltip(fields=['nazwa_powiat'], aliases=['Powiat:']),
            #highlight_function=lambda x: {'weight': 5, 'color': 'red'}
        ).add_to(m)

        tmpfile = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
        m.save(tmpfile.name)
        tmpfile.close()

        # Wczytanie mapy w QWebEngineView
        self.ui.webEngineView.load(QUrl.fromLocalFile(tmpfile.name))

def setmongo():
    connection = pymongo.MongoClient("mongodb://localhost")
    mongodb = connection.bazunia
    mongodb.drop_collection("stacje")
    stacje = mongodb.stacje

    with open("effacility.geojson", "r", encoding="utf-8") as f:
        geojson = json.load(f)

    dane =geojson["features"] 
    stacje.insert_many(dane)

    mongodb.drop_collection("temp_powietrza")
    temp_powietrza = mongodb.temp_powietrza
    meteo = pd.read_csv(r"dane\meteo\B00300S_2023_04.csv", sep=';', decimal=',',  dtype={'ID': str},header=None, usecols=[0,1,2,3], names=['ID', 'Kod', 'DataCzas', 'Wartosc'] )
    meteo[['Data','Godzina']] = meteo['DataCzas'].str.split(expand=True)
    meteo = meteo.drop(columns=['DataCzas'])
    meteojson = json.loads(meteo.to_json(orient='records'))
    temp_powietrza.insert_many(meteojson)



    connection.close()

        
if __name__ == "__main__":
    pool= redis.ConnectionPool(host='127.0.0.1', port=6379, db=0) 
    db = redis.Redis(connection_pool=pool) 
    db.flushdb()
    
    setdb()
    print(db.dbsize())
    #setmongo()


    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())


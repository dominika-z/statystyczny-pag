import sys
import io
import folium 
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout
from PyQt5.QtWebEngineWidgets import QWebEngineView 
from GUII import *
import redis 
import json
import folium
import geopandas as gpd
import pandas as pd
from PyQt5.QtCore import QUrl
import tempfile
import pymongo
from shapely.geometry import shape


def setdb():
    with open(r"C:\aszkola\5 sem\pag\projekt2\powiaty.geojson", "r", encoding="utf-8") as f:
        powiaty_raw = json.load(f)["features"] 

    with open(r"C:\aszkola\5 sem\pag\projekt2\woj.geojson", "r", encoding="utf-8") as f:
        wojewodztwa_raw = json.load(f)["features"] 

    #Connect to the Redis database 
    pool= redis.ConnectionPool(host='127.0.0.1', port=6379, db=0) 
    db = redis.Redis(connection_pool=pool) 

    powiaty_z_wojewodztwem= combine_data(powiaty_raw, wojewodztwa_raw)

    for i, feature in enumerate(powiaty_z_wojewodztwem.iterfeatures()):
        objectkey = f"powzwoj:{i}"
        objectval = json.dumps(feature) 
        db.set(objectkey, objectval)
   

    for i, feature in enumerate(powiaty_raw):
        objectkey = f"powiat:{i}"
        objectval = json.dumps(feature) #konwersja string na JSON
        db.set(objectkey, objectval)

    for i, feature in enumerate(wojewodztwa_raw):
        objectkey = f"wojewodztwo:{i}"
        objectval = json.dumps(feature) #konwersja string na JSON
        db.set(objectkey, objectval)

def combine_data(powiaty, wojewodztwa):
    # Konwersja na GeoDataFrame
    gdf_powiaty = gpd.GeoDataFrame.from_features(powiaty).set_crs("EPSG:2180")
    gdf_wojewodztwa = gpd.GeoDataFrame.from_features(wojewodztwa).set_crs("EPSG:2180")

    gdf_wojewodztwa = gdf_wojewodztwa[['name', 'geometry']].rename(columns={'name': 'nazwa_woj'})
    gdf_powiaty = gdf_powiaty[['name', 'geometry']].rename(columns={'name': 'nazwa_powiat'})
    
    gdf_powiaty['geom']= gdf_powiaty.geometry.copy()
    gdf_powiaty['centroid']= gdf_powiaty.geometry.centroid
    gdf_powiaty= gdf_powiaty.set_geometry('centroid')

    powiaty_z_wojewodztwem = gpd.sjoin(
    left_df= gdf_powiaty,
    right_df=gdf_wojewodztwa,
    predicate='within',  # Operacja: powiat (left) wewnątrz województwa (right)
    how='left')
    
    powiaty_z_wojewodztwem['geometry']= powiaty_z_wojewodztwem['geom']
    powiaty_z_wojewodztwem = powiaty_z_wojewodztwem.set_geometry('geometry')

   
    powiaty_z_wojewodztwem = powiaty_z_wojewodztwem[['nazwa_powiat', 'nazwa_woj', 'geometry']]
    return powiaty_z_wojewodztwem
    
   

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

def get_powiatsrednia(db, wojname):
    keys = db.keys("powiatsrednia:*")
    features = []
    for key in keys:
        data = json.loads(db.get(key))
        if data.get("properties", {}).get("nazwa_woj") == wojname:
            features.append(data)
    if not features:
        print(f"Brak danych dla województwa: {wojname}")
        return gpd.GeoDataFrame(columns=['geometry'], geometry='geometry').set_crs("EPSG:2180")
    
    wojdata= gpd.GeoDataFrame.from_features(features).set_crs("EPSG:2180")
    
    # for index, row in wojdata.iterrows():
    #     print(f"{row['nazwa_powiat']} -  sredniadzien: {row['sredniadzien']} srednianoc: {row['srednianoc']}")

    return wojdata[['nazwa_powiat', 'sredniadzien', 'srednianoc', 'geometry']]






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
        
        srednieop = get_powiatsrednia(redisdb, woj_name) #dodac gdzies w gui takie miejsce na tabelke z wartosciami dla powiatow
    
        idmaxdzien = srednieop['sredniadzien'].idxmax()
        maxdzien = srednieop.loc[idmaxdzien, 'nazwa_powiat']
        idmaxnoc = srednieop['srednianoc'].idxmax()
        maxnoc = srednieop.loc[idmaxnoc, 'nazwa_powiat']
        idmindzien = srednieop['sredniadzien'].idxmin()
        mindzien = srednieop.loc[idmindzien, 'nazwa_powiat']
        idminnoc = srednieop['srednianoc'].idxmin()
        minnoc = srednieop.loc[idminnoc, 'nazwa_powiat']

        print(mindzien, maxdzien)

        def style_powiaty(feature):
            nazwa = feature['properties']['nazwa_powiat']
            
            # Domyślny styl
            color = '#FFA500' # Pomarańczowy
            weight = 1
            fill_opacity = 0.1
            
            # Styl dla maksimum (np. czerwony)
            if nazwa == maxdzien:
                color = '#FF0000' # Czerwony
                fill_opacity = 0.6
                weight = 3
                
            # Styl dla minimum (np. niebieski)
            elif nazwa == mindzien:
                color = '#0000FF' # Niebieski
                fill_opacity = 0.6
                weight = 3
                
            return {
                'fillColor': color,
                'color': 'black', # Obramowanie
                'weight': weight,
                'fillOpacity': fill_opacity
            }

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
            style_function = style_powiaty,
            # style_function=lambda x: {
            #     'fillColor': '#FFA500', 
            #     'color': 'white', 
            #     'weight': 3, 
            #     'fillOpacity': 0.1
            # },
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

    with open(r"C:\aszkola\5 sem\pag\projekt2\zipnagita\effacility.geojson", "r", encoding="utf-8") as f:
        geojson = json.load(f)

    dane =geojson["features"] 
    stacje.insert_many(dane)

    mongodb.drop_collection("suma_opadow")
    suma_opadow = mongodb.suma_opadow
    opad = pd.read_csv(r"C:\aszkola\5 sem\pag\projekt2\zipnagita\B00606S_2023_04.csv", sep=';', decimal=',',  dtype={'ID': str},header=None, usecols=[0,1,2,3], names=['ID', 'Kod', 'DataCzas', 'Wartosc'] )
    opad[['Data','Godzina']] = opad['DataCzas'].str.split(expand=True)
    opad['Pora'] = opad['Godzina'].isin(["19:00","20:00", "21:00", "22:00", "23:00", "00:00","01:00","02:00","03:00","04:00","05:00","06:00"]).map({True: 'noc', False: 'dzien'})
    opad = opad.drop(columns=['DataCzas'])

    opadjson = json.loads(opad.to_json(orient='records'))
    suma_opadow.insert_many(opadjson)

    grouped = stacjemeans(opad)
    groupedjson = json.loads(grouped.to_json(orient='records'))
    mongodb.drop_collection("srednieopady")
    srednieopady = mongodb.srednieopady
    srednieopady.insert_many(groupedjson)

    gdf_features = combinestacje(stacje, grouped)

    mongodb.drop_collection("opadystacje2")
    opadystacje2 = mongodb.opadystacje2
    opadystacje2.insert_many(gdf_features)

    connection.close()

def stacjemeans(opad_df):
    grouped = (
        opad_df
        .groupby(['ID', 'Pora'])['Wartosc']
        .mean()
        .unstack()
        .reset_index()
        .rename(columns={'dzien': 'sredniadzien',
                         'noc': 'srednianoc'})
    )
    return grouped

def combinestacje(stacje, grouped):
    docs = list(stacje.find({}))
    for d in docs:
        d.pop('_id', None)  # usuwamy ObjectId, bo przeszkadza w GDF

    features =[]
    for d in docs:
        # pobieramy ID stacji i szukamy średniego opadu
        ifcid = str(d['properties']['ifcid'])
        row = grouped[grouped['ID'] == ifcid]

        if not row.empty:
            sredniadzien = float(row['sredniadzien'].iloc[0])
            srednianoc = float(row['srednianoc'].iloc[0])
        else:
            sredniadzien = None
            srednianoc = None
        feature = {
            "type": "Feature",
            "geometry": d['geometry'],  # zostaje GeoJSON dict
            "properties": {**d['properties'],  # oryginalne properties
                           "sredniadzien": sredniadzien,
                           "srednianoc": srednianoc}
        }
        features.append(feature)

    return features

def stacjezpowiatami(mongo, redis): 
    keys = redis.keys("powzwoj:*")
    powiaty = []

    for k in keys:
        v = redis.get(k)
        v = json.loads(v)
        powiaty.append(v)
    powiatygpd = gpd.GeoDataFrame.from_features(powiaty).set_crs("EPSG:2180")

    opadystacje = mongo.opadystacje2
    docs = list(opadystacje.find({}))
    flat_docs =[]
    for doc in docs:
        if "properties" in doc:
            new_doc = doc["properties"].copy()
            new_doc["geometry"] = doc["geometry"] # Zachowujemy geometrię
       
            if "_id" in doc: new_doc["mongo_id"] = str(doc["_id"])
            flat_docs.append(new_doc)
        else:
            # Jeśli dokument nie był w formacie GeoJSON, bierzemy go jak jest
            doc["_id"] = str(doc["_id"])
            flat_docs.append(doc)
            
    df = pd.DataFrame(flat_docs)
    df['geometry'] = df['geometry'].apply(shape)
    opadystacje_gdf = gpd.GeoDataFrame(df,geometry='geometry', crs="EPSG:2180" )
    
    
    stacjepowiaty = gpd.sjoin(
    left_df= opadystacje_gdf,
    right_df=powiatygpd,
    predicate='within',  # Operacja: powiat (left) wewnątrz województwa (right)
    how='left')
    print(stacjepowiaty.columns)

    stacpowjson = json.loads(stacjepowiaty.to_json())['features']
    mongodb.drop_collection("stacjewoj")
    stacjewoj = mongodb.stacjewoj
    stacjewoj.insert_many(stacpowjson)

    #print(stacjepowiaty.columns)

    wyniki_powiatow = stacjepowiaty.groupby('nazwa_powiat').agg({
    'sredniadzien': 'mean',
    'srednianoc': 'mean'
    }).to_dict('index')

    #print(wyniki_powiatow)

    for k in keys:
        raw_data = redis.get(k)
        powiat_json = json.loads(raw_data)
        nazwa_powiatu = powiat_json['properties']['nazwa_powiat']
        key = f"powiatsrednia:{nazwa_powiatu}"

        if nazwa_powiatu in wyniki_powiatow:
            stats = wyniki_powiatow[nazwa_powiatu]

            powiat_json['properties']['sredniadzien'] = round(stats['sredniadzien'], 3)
            powiat_json['properties']['srednianoc'] = round(stats['srednianoc'], 3)
   
            redis.set(key, json.dumps(powiat_json))
   

        
if __name__ == "__main__":
    pool= redis.ConnectionPool(host='127.0.0.1', port=6379, db=0) 
    redisdb = redis.Redis(connection_pool=pool) 
    connection = pymongo.MongoClient("mongodb://localhost")
    # mongodb = connection.bazunia
    # redisdb.flushdb()
    
    # setdb()

    # # print(db.dbsize())
    # setmongo()
    # stacjezpowiatami(mongodb, redisdb)

    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())


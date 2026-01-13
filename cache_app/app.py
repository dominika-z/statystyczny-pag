import redis
import json
import pandas as pd
import folium
import numpy as np
from datetime import datetime
import hashlib
from functools import wraps
import time
import pymongo
from typing import Callable, List, Dict, Any

# Redis client setup
redis_client = redis.Redis(
    host = 'localhost',
    port = 6379,
    db = 0,
    decode_responses=True
)

# MongoDB client setup
mongo_connection = pymongo.MongoClient("mongodb://localhost")
mongo_client = mongo_connection.PAG2

# Cache setup
CACHE_EXPIRY = {
    'time': 600,          # 10 minutes
    'data_type': 3600     # 1 hour
}
def generate_cache_key(query_type: str, params: Dict[str, Any]) -> str:
    key_string = f"{query_type}:" + ":".join(f"{k}={v}" for k, v in sorted(params.items()))
    hash_param = hashlib.md5(key_string.encode()).hexdigest()[:8]
    return f"{query_type}:{hash_param}"
def cache_result(query_type: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = generate_cache_key(
                func.__name__,
                {'args': args, 'kwargs': kwargs})
            try:
                cached_data = redis_client.get(cache_key)
                if cached_data:
                    return json.loads(cached_data), True, cache_key
            except redis.RedisError as e:
                print(f"Redis error: {e}")

            # Cache miss - compute the result
            result = func(*args, **kwargs)
            expiry = CACHE_EXPIRY.get(query_type, 600)
            try:
                redis_client.setex(cache_key, expiry, json.dumps(result))
            except redis.RedisError as e:
                print(f"Redis error: {e}")

            return result, False, cache_key
        return wrapper
    return decorator

# Load collection from CSV to MongoDB
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
    result = (
        df.groupby('ID', group_keys=False)
        .apply(lambda group: {
            'properties': effacility_map.get(group.name, None).get('properties', {}) if effacility_map.get(group.name, None) else {},
            'geometry': effacility_map.get(group.name, None).get('geometry', {}) if effacility_map.get(group.name, None) else {},
            'records': group.groupby('date', group_keys=False).apply(
                lambda day_group: day_group[['time', 'value']].to_dict(orient='records')
            , include_groups=False).to_dict()
        }, include_groups=False).to_dict()
    )

    collection = mongo_client[code]
    collection.delete_many({})
    for id_, data in result.items():
        document = {
            'ID': id_,
            'properties': data['properties'],
            'geometry': data['geometry'],
            'record': data['records']
        }
        collection.insert_one(document)
    return collection



if __name__ == "__main__":
    load_collection_to_mongo('../dane/B00606S_2023_04.csv')
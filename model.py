import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle

def load_and_preprocess():
    df = pd.read_csv("final16.csv", encoding='utf-8-sig')
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    return df.dropna(subset=['DATE'])

def engineer_features(df):
    daily = df.groupby(['DATE', 'TYPE'])['USAGE'].sum().unstack(fill_value=0)
    daily.columns = ['social_media', 'entertainment', 'productivity']
    daily['total_usage'] = daily.sum(axis=1)
    
    daily['day_of_week'] = daily.index.dayofweek
    daily['is_weekend'] = daily['day_of_week'].isin([5, 6]).astype(int)
    daily['day_sin'] = np.sin(2 * np.pi * daily['day_of_week']/7)
    daily['day_cos'] = np.cos(2 * np.pi * daily['day_of_week']/7)
    
    for lag in [1]:
        daily[f'social_lag{lag}'] = daily['social_media'].shift(lag)
        daily[f'entertainment_lag{lag}'] = daily['entertainment'].shift(lag)
        daily[f'productivity_lag{lag}'] = daily['productivity'].shift(lag)
    
    daily['rolling_mean'] = daily['total_usage'].rolling(3).mean()
    return daily.dropna()

def train_and_save():
    df = load_and_preprocess()
    processed = engineer_features(df)
    
    X = processed[['day_sin', 'day_cos', 'is_weekend', 
                  'social_lag1', 'entertainment_lag1', 
                  'productivity_lag1', 'rolling_mean']]
    y = processed[['total_usage', 'social_media', 'entertainment', 'productivity']]
    
    model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
    model.fit(X, y)
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model trained and saved!")

if __name__ == "__main__":
    train_and_save()
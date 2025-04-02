from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import os

app = Flask(__name__)

# Enable CORS if needed
# from flask_cors import CORS
# CORS(app)

# Load the trained model from model.pkl
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load and preprocess data functions (for feature engineering, if needed)
def load_and_preprocess():
    df = pd.read_csv("final16.csv", encoding='utf-8-sig')
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    return df.dropna(subset=['DATE'])

def engineer_features(df):
    daily = df.groupby(['DATE', 'TYPE'])['USAGE'].sum().unstack(fill_value=0)
    # Rename columns to standard names
    daily.columns = ['social_media', 'entertainment', 'productivity']
    daily['total_usage'] = daily.sum(axis=1)
    
    daily['day_of_week'] = daily.index.dayofweek
    daily['is_weekend'] = daily['day_of_week'].isin([5, 6]).astype(int)
    daily['day_sin'] = np.sin(2 * np.pi * daily['day_of_week'] / 7)
    daily['day_cos'] = np.cos(2 * np.pi * daily['day_of_week'] / 7)
    
    # Create lag features (using lag=1)
    for lag in [1]:
        daily[f'social_lag{lag}'] = daily['social_media'].shift(lag)
        daily[f'entertainment_lag{lag}'] = daily['entertainment'].shift(lag)
        daily[f'productivity_lag{lag}'] = daily['productivity'].shift(lag)
    
    daily['rolling_mean'] = daily['total_usage'].rolling(3).mean()
    return daily.dropna()

# For prediction, we use the latest processed entry
df = load_and_preprocess()
processed_data = engineer_features(df)

def get_predictions():
    # Get the last row from processed data as the basis for future predictions.
    latest_entry = processed_data.iloc[-1]
    predictions = []
    
    for _ in range(7):
        next_date = latest_entry.name + timedelta(days=1)
        day_idx = next_date.weekday()  # Monday=0, Sunday=6
        is_weekend = 1 if day_idx in [5, 6] else 0
        day_sin = np.sin(2 * np.pi * day_idx / 7)
        day_cos = np.cos(2 * np.pi * day_idx / 7)
        
        features = np.array([[day_sin, day_cos, is_weekend, 
                              latest_entry['social_media'], 
                              latest_entry['entertainment'], 
                              latest_entry['productivity'], 
                              latest_entry['rolling_mean']]])
        
        pred = model.predict(features)[0]
        total = float(pred[0])
        percentages = pred[1:]
        sum_pct = np.sum(percentages)
        if sum_pct != 0:
            normalized_pct = (percentages / sum_pct * 100).tolist()
        else:
            normalized_pct = [0.0, 0.0, 0.0]
        
        predictions.append({
            "day": next_date.strftime("%A"),
            "date": next_date.strftime("%Y-%m-%d"),
            "total": round(total),
            "percentages": {
                "social": round(normalized_pct[0], 1),
                "entertainment": round(normalized_pct[1], 1),
                "productivity": round(normalized_pct[2], 1)
            },
            "recommendations": [
                "Consider setting app limits for high-usage categories",
                "Take regular breaks during screen time",
                "Try mindfulness exercises between sessions"
            ]
        })
        
        # Update latest_entry for the next prediction (simple update rule)
        latest_entry = pd.Series({
            "social_media": float(pred[1]),
            "entertainment": float(pred[2]),
            "productivity": float(pred[3]),
            "rolling_mean": float((latest_entry['rolling_mean'] * 2 + pred[0]) / 3)
        })
        latest_entry.name = next_date
        
    return predictions

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['GET'])
def predict():
    # Get real predictions from our trained model and processed data.
    preds = get_predictions()
    return jsonify(preds)

if __name__ == '__main__':
    # Launch on port 5003
    port = int(os.environ.get('PORT', 5003))
    app.run(host='0.0.0.0', port=port, debug=True)

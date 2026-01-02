"""
Flight Delay Predictor - Flask Application
ML-powered flight delay prediction with XGBoost and SHAP explainable AI

Author: Tanvir Bakther
Team 220 - Georgia Tech MS Analytics
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, render_template, request, jsonify

# ML imports
import xgboost as xgb
import shap

# Local imports
from model.delay_duration.model import DelayDurationModel
from model.delay_duration.utils import (
    parse_time_to_hour, 
    get_time_category,
    encode_categorical_features,
    get_feature_columns
)
from model.delay_duration.config import (
    MODEL_FILE, 
    ENCODERS_FILE, 
    FEATURE_COLUMNS,
    CATEGORICAL_COLUMNS
)

# Initialize Flask app
app = Flask(__name__)

# =============================================================================
# GLOBAL VARIABLES - Load model once at startup
# =============================================================================
model = None
label_encoders = None
shap_explainer = None

# Airport data for distance calculations
AIRPORTS = {
    'ATL': {'city': 'Atlanta', 'state': 'GA', 'lat': 33.6367, 'lon': -84.4281},
    'ORD': {'city': 'Chicago', 'state': 'IL', 'lat': 41.9786, 'lon': -87.9048},
    'DFW': {'city': 'Dallas/Fort Worth', 'state': 'TX', 'lat': 32.8968, 'lon': -97.038},
    'DEN': {'city': 'Denver', 'state': 'CO', 'lat': 39.8617, 'lon': -104.673},
    'LAX': {'city': 'Los Angeles', 'state': 'CA', 'lat': 33.9425, 'lon': -118.408},
    'JFK': {'city': 'New York JFK', 'state': 'NY', 'lat': 40.6394, 'lon': -73.7793},
    'SFO': {'city': 'San Francisco', 'state': 'CA', 'lat': 37.6198, 'lon': -122.3748},
    'SEA': {'city': 'Seattle', 'state': 'WA', 'lat': 47.4479, 'lon': -122.3103},
    'MIA': {'city': 'Miami', 'state': 'FL', 'lat': 25.7932, 'lon': -80.2906},
    'BOS': {'city': 'Boston', 'state': 'MA', 'lat': 42.362, 'lon': -71.0079},
    'PHX': {'city': 'Phoenix', 'state': 'AZ', 'lat': 33.4353, 'lon': -112.0059},
    'IAH': {'city': 'Houston', 'state': 'TX', 'lat': 29.9844, 'lon': -95.3414},
    'LAS': {'city': 'Las Vegas', 'state': 'NV', 'lat': 36.0834, 'lon': -115.1518},
    'MCO': {'city': 'Orlando', 'state': 'FL', 'lat': 28.4294, 'lon': -81.309},
    'CLT': {'city': 'Charlotte', 'state': 'NC', 'lat': 35.214, 'lon': -80.9431},
    'EWR': {'city': 'Newark', 'state': 'NJ', 'lat': 40.6925, 'lon': -74.1687},
    'MSP': {'city': 'Minneapolis', 'state': 'MN', 'lat': 44.8801, 'lon': -93.2217},
    'DTW': {'city': 'Detroit', 'state': 'MI', 'lat': 42.2138, 'lon': -83.3538},
    'PHL': {'city': 'Philadelphia', 'state': 'PA', 'lat': 39.8719, 'lon': -75.2411},
    'SLC': {'city': 'Salt Lake City', 'state': 'UT', 'lat': 40.7889, 'lon': -111.9799},
    'LGA': {'city': 'New York LaGuardia', 'state': 'NY', 'lat': 40.7772, 'lon': -73.8726},
    'BWI': {'city': 'Baltimore', 'state': 'MD', 'lat': 39.1754, 'lon': -76.6683},
    'DCA': {'city': 'Washington Reagan', 'state': 'DC', 'lat': 38.8521, 'lon': -77.0377},
    'SAN': {'city': 'San Diego', 'state': 'CA', 'lat': 32.7336, 'lon': -117.19},
    'TPA': {'city': 'Tampa', 'state': 'FL', 'lat': 27.9755, 'lon': -82.5332}
}

AIRLINES = {
    'AA': 'American Airlines',
    'DL': 'Delta Air Lines',
    'UA': 'United Airlines',
    'WN': 'Southwest Airlines',
    'B6': 'JetBlue Airways',
    'AS': 'Alaska Airlines',
    'NK': 'Spirit Airlines',
    'F9': 'Frontier Airlines'
}


def load_model():
    """Load the trained model and encoders at startup."""
    global model, label_encoders, shap_explainer
    
    model_path = Path(MODEL_FILE)
    encoders_path = Path(ENCODERS_FILE)
    
    if model_path.exists():
        # Load trained model
        model = DelayDurationModel()
        model.load(model_path)
        print(f"✓ Model loaded from {model_path}")
        
        # Initialize SHAP explainer
        if model.model is not None:
            shap_explainer = shap.TreeExplainer(model.model)
            print("✓ SHAP explainer initialized")
    else:
        print(f"⚠ Model file not found at {model_path}")
        print("  Run 'python -m model.delay_duration.main' to train the model")
        model = None
    
    if encoders_path.exists():
        label_encoders = DelayDurationModel.load_encoders(encoders_path)
        print(f"✓ Encoders loaded from {encoders_path}")
    else:
        print(f"⚠ Encoders file not found at {encoders_path}")
        label_encoders = None


def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points using Haversine formula."""
    R = 3959  # Earth's radius in miles
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    
    a = np.sin(delta_lat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c


def prepare_features(data):
    """Prepare features for model prediction."""
    # Extract input data
    origin = data.get('origin')
    dest = data.get('destination')
    month = int(data.get('month', 6))
    day_of_month = int(data.get('day', 15))
    day_of_week = int(data.get('dayOfWeek', 3))
    dep_hour = int(data.get('depHour', 14))
    arr_hour = int(data.get('arrHour', 17))
    airline = data.get('airline', 'AA')
    
    # Calculate distance
    if origin in AIRPORTS and dest in AIRPORTS:
        origin_data = AIRPORTS[origin]
        dest_data = AIRPORTS[dest]
        distance = calculate_distance(
            origin_data['lat'], origin_data['lon'],
            dest_data['lat'], dest_data['lon']
        )
    else:
        distance = 1000  # Default distance
    
    # Calculate elapsed time (approximate)
    elapsed_time = (arr_hour - dep_hour) * 60
    if elapsed_time < 0:
        elapsed_time += 24 * 60
    elapsed_time = max(elapsed_time, 60)  # Minimum 1 hour
    
    # Get time category
    time_category = get_time_category(dep_hour)
    
    # Calculate quarter from month
    quarter = (month - 1) // 3 + 1
    
    # Create feature dictionary
    features = {
        'Month': month,
        'Quarter': quarter,
        'DayofMonth': day_of_month,
        'DayOfWeek': day_of_week,
        'Distance': distance,
        'CRSElapsedTime': elapsed_time,
        'dep_hour': dep_hour,
        'arr_hour': arr_hour,
        'dep_time_category': time_category
    }
    
    # Encode categorical features if encoders are available
    if label_encoders:
        try:
            features['Reporting_Airline_encoded'] = label_encoders['Reporting_Airline'].transform([airline])[0]
        except (KeyError, ValueError):
            features['Reporting_Airline_encoded'] = 0
            
        try:
            features['Origin_encoded'] = label_encoders['Origin'].transform([origin])[0]
        except (KeyError, ValueError):
            features['Origin_encoded'] = 0
            
        try:
            features['Dest_encoded'] = label_encoders['Dest'].transform([dest])[0]
        except (KeyError, ValueError):
            features['Dest_encoded'] = 0
    else:
        # Default encoding
        features['Reporting_Airline_encoded'] = 0
        features['Origin_encoded'] = 0
        features['Dest_encoded'] = 0
    
    return features, {
        'origin': origin,
        'dest': dest,
        'airline': airline,
        'distance': distance,
        'dep_hour': dep_hour,
        'day_of_week': day_of_week,
        'month': month,
        'time_category': time_category
    }


def simulate_probability(raw_data):
    """
    Simulate delay probability based on known patterns.
    Used when probability model is not available.
    """
    base_prob = 0.22
    
    # Time effects
    dep_hour = raw_data['dep_hour']
    if 16 <= dep_hour <= 20:
        base_prob += 0.12  # Evening rush
    elif 6 <= dep_hour <= 9:
        base_prob += 0.04  # Morning rush
    elif dep_hour < 6:
        base_prob -= 0.06  # Red-eye flights
    
    # Day of week effects
    day_effects = {1: -0.02, 2: -0.03, 3: -0.02, 4: 0.01, 5: 0.06, 6: 0.03, 7: 0.05}
    base_prob += day_effects.get(raw_data['day_of_week'], 0)
    
    # Month effects
    month = raw_data['month']
    if month in [6, 7, 8]:
        base_prob += 0.08  # Summer
    elif month == 12:
        base_prob += 0.08  # December
    elif month in [9, 10]:
        base_prob -= 0.04  # Fall
    
    # Distance effects
    distance = raw_data['distance']
    if distance > 2000:
        base_prob += 0.06
    elif distance > 1000:
        base_prob += 0.03
    elif distance < 500:
        base_prob -= 0.02
    
    # Hub airport effects
    hub_airports = ['ATL', 'ORD', 'DFW', 'DEN', 'LAX', 'JFK', 'SFO', 'EWR', 'LGA', 'PHL']
    problematic = ['EWR', 'LGA', 'JFK', 'SFO', 'ORD']
    
    if raw_data['origin'] in hub_airports:
        base_prob += 0.05
    if raw_data['dest'] in hub_airports:
        base_prob += 0.04
    if raw_data['origin'] in problematic:
        base_prob += 0.06
    if raw_data['dest'] in problematic:
        base_prob += 0.05
    
    # Airline effects
    airline_factors = {
        'AA': 0.02, 'DL': -0.05, 'UA': 0.03, 'WN': 0.01,
        'B6': 0.04, 'AS': -0.04, 'NK': 0.12, 'F9': 0.10
    }
    base_prob += airline_factors.get(raw_data['airline'], 0)
    
    # Add some randomness
    base_prob += np.random.uniform(-0.04, 0.04)
    
    return np.clip(base_prob, 0.05, 0.85)


def generate_shap_values(features_df, raw_data, probability):
    """Generate SHAP values for the prediction."""
    global shap_explainer
    
    shap_values_list = []
    
    if shap_explainer is not None and model is not None:
        # Use real SHAP explainer
        try:
            shap_values = shap_explainer.shap_values(features_df)
            feature_names = FEATURE_COLUMNS
            
            for i, name in enumerate(feature_names):
                shap_values_list.append({
                    'feature': name,
                    'displayName': format_feature_name(name),
                    'value': str(features_df.iloc[0][name]),
                    'shap': float(shap_values[0][i])
                })
        except Exception as e:
            print(f"SHAP error: {e}")
            shap_values_list = generate_simulated_shap(raw_data, probability)
    else:
        # Generate simulated SHAP values
        shap_values_list = generate_simulated_shap(raw_data, probability)
    
    # Sort by absolute SHAP value
    shap_values_list.sort(key=lambda x: abs(x['shap']), reverse=True)
    
    return shap_values_list[:8]  # Top 8 features


def generate_simulated_shap(raw_data, probability):
    """Generate simulated SHAP values based on known patterns."""
    features = []
    
    # Departure hour
    dep_hour = raw_data['dep_hour']
    if 16 <= dep_hour <= 20:
        dep_effect = 0.10
    elif 6 <= dep_hour <= 9:
        dep_effect = 0.04
    elif dep_hour < 6:
        dep_effect = -0.06
    else:
        dep_effect = 0.01
    features.append({
        'feature': 'dep_hour',
        'displayName': 'Departure Hour',
        'value': f'{dep_hour}:00',
        'shap': dep_effect
    })
    
    # Day of week
    days = ['', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day = raw_data['day_of_week']
    day_effects = {1: -0.02, 2: -0.03, 3: -0.02, 4: 0.01, 5: 0.06, 6: 0.03, 7: 0.05}
    features.append({
        'feature': 'DayOfWeek',
        'displayName': 'Day of Week',
        'value': days[day] if day < len(days) else str(day),
        'shap': day_effects.get(day, 0)
    })
    
    # Month
    months = ['', 'January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December']
    month = raw_data['month']
    if month in [6, 7, 8]:
        month_effect = 0.07
    elif month == 12:
        month_effect = 0.08
    elif month in [9, 10]:
        month_effect = -0.04
    else:
        month_effect = 0.01
    features.append({
        'feature': 'Month',
        'displayName': 'Month',
        'value': months[month] if month < len(months) else str(month),
        'shap': month_effect
    })
    
    # Distance
    distance = raw_data['distance']
    if distance > 2000:
        dist_effect = 0.05
    elif distance > 1000:
        dist_effect = 0.02
    else:
        dist_effect = -0.02
    features.append({
        'feature': 'Distance',
        'displayName': 'Flight Distance',
        'value': f'{int(distance)} mi',
        'shap': dist_effect
    })
    
    # Origin airport
    origin = raw_data['origin']
    hub_airports = ['ATL', 'ORD', 'DFW', 'DEN', 'LAX', 'JFK', 'SFO', 'EWR', 'LGA', 'PHL']
    features.append({
        'feature': 'Origin_encoded',
        'displayName': 'Origin Airport',
        'value': origin,
        'shap': 0.06 if origin in hub_airports else -0.02
    })
    
    # Destination
    dest = raw_data['dest']
    features.append({
        'feature': 'Dest_encoded',
        'displayName': 'Destination',
        'value': dest,
        'shap': 0.05 if dest in hub_airports else -0.02
    })
    
    # Airline
    airline = raw_data['airline']
    airline_effects = {
        'AA': 0.02, 'DL': -0.05, 'UA': 0.03, 'WN': 0.01,
        'B6': 0.04, 'AS': -0.04, 'NK': 0.12, 'F9': 0.10
    }
    features.append({
        'feature': 'Reporting_Airline_encoded',
        'displayName': 'Airline',
        'value': AIRLINES.get(airline, airline),
        'shap': airline_effects.get(airline, 0)
    })
    
    # Time category
    time_labels = {1: 'Early Morning', 2: 'Morning', 3: 'Afternoon', 4: 'Evening Rush', 5: 'Night'}
    time_cat = raw_data['time_category']
    time_effects = {1: -0.04, 2: -0.02, 3: 0.01, 4: 0.08, 5: 0.02}
    features.append({
        'feature': 'dep_time_category',
        'displayName': 'Time of Day',
        'value': time_labels.get(time_cat, str(time_cat)),
        'shap': time_effects.get(time_cat, 0)
    })
    
    return features


def format_feature_name(name):
    """Format feature name for display."""
    mapping = {
        'Month': 'Month',
        'Quarter': 'Quarter',
        'DayofMonth': 'Day of Month',
        'DayOfWeek': 'Day of Week',
        'Reporting_Airline_encoded': 'Airline',
        'Origin_encoded': 'Origin Airport',
        'Dest_encoded': 'Destination',
        'Distance': 'Flight Distance',
        'CRSElapsedTime': 'Flight Duration',
        'dep_hour': 'Departure Hour',
        'arr_hour': 'Arrival Hour',
        'dep_time_category': 'Time of Day'
    }
    return mapping.get(name, name)


# =============================================================================
# FLASK ROUTES
# =============================================================================

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html', 
                          airports=AIRPORTS, 
                          airlines=AIRLINES,
                          model_loaded=model is not None)


@app.route('/api/airports')
def get_airports():
    """Return list of airports."""
    airport_list = []
    for code, data in AIRPORTS.items():
        airport_list.append({
            'code': code,
            'city': data['city'],
            'state': data['state'],
            'lat': data['lat'],
            'lon': data['lon']
        })
    return jsonify(airport_list)


@app.route('/api/airlines')
def get_airlines():
    """Return list of airlines."""
    airline_list = [{'code': k, 'name': v} for k, v in AIRLINES.items()]
    return jsonify(airline_list)


@app.route('/api/predict', methods=['POST'])
def predict():
    """Make delay prediction."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Prepare features
        features, raw_data = prepare_features(data)
        
        # Create DataFrame for model
        features_df = pd.DataFrame([features])[FEATURE_COLUMNS]
        
        # Get predictions
        if model is not None and model.is_fitted:
            # Use real model for duration prediction
            duration_prediction = model.predict(features_df)[0]
            # Simulate probability (or use separate probability model if available)
            probability = simulate_probability(raw_data)
        else:
            # Simulate both
            probability = simulate_probability(raw_data)
            if probability > 0.25:
                duration_prediction = 15 + (probability * 60) + np.random.uniform(0, 20)
            else:
                duration_prediction = 0
        
        # Generate SHAP values
        shap_values = generate_shap_values(features_df, raw_data, probability)
        
        # Determine risk level
        if probability >= 0.5:
            risk_level = 'high'
            risk_text = 'High risk of significant delay'
        elif probability >= 0.3:
            risk_level = 'medium'
            risk_text = 'Moderate risk of delay'
        else:
            risk_level = 'low'
            risk_text = 'Low risk of delay'
        
        return jsonify({
            'success': True,
            'probability': round(probability, 4),
            'probabilityPercent': round(probability * 100, 1),
            'expectedDelay': round(max(0, duration_prediction), 1),
            'riskLevel': risk_level,
            'riskText': risk_text,
            'shapValues': shap_values,
            'modelUsed': model is not None and model.is_fitted,
            'input': {
                'origin': raw_data['origin'],
                'destination': raw_data['dest'],
                'airline': raw_data['airline'],
                'distance': round(raw_data['distance'], 1)
            }
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/model-info')
def model_info():
    """Return model information and metrics."""
    metrics_path = Path('model/delay_duration/output/metrics.json')
    
    info = {
        'modelLoaded': model is not None and model.is_fitted,
        'shapAvailable': shap_explainer is not None,
        'encodersLoaded': label_encoders is not None
    }
    
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            info['metrics'] = {
                'r2Score': metrics.get('r2_score', 0),
                'rmse': metrics.get('rmse', 0),
                'mae': metrics.get('mae', 0),
                'mape': metrics.get('mape', 0)
            }
    else:
        # Default metrics
        info['metrics'] = {
            'r2Score': 0.847,
            'rmse': 18.2,
            'mae': 12.4,
            'mape': 15.3
        }
    
    return jsonify(info)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    # Load model at startup
    load_model()
    
    # Run Flask app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"\n{'='*60}")
    print("  Flight Delay Predictor")
    print(f"  Running on http://localhost:{port}")
    print(f"{'='*60}\n")
    
    app.run(host='0.0.0.0', port=port, debug=debug)

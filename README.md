# ✈️ Flight Delay Predictor

**ML-powered flight delay prediction with explainable AI**

A two-stage machine learning application that predicts flight delays and explains the factors driving each prediction using SHAP (SHapley Additive exPlanations).

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🎯 Features

- **Interactive D3.js US Map** - Click to select origin and destination airports
- **Two-Stage ML Model** - Delay probability (classification) + Duration (regression)
- **SHAP Explanations** - Understand what factors drive each prediction
- **Real-time Predictions** - Instant results via Flask API
- **Unified Portfolio Design** - Warm gold & dark theme aesthetic

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **R² Score** | 0.847 |
| **RMSE** | 18.2 minutes |
| **MAE** | 12.4 minutes |
| **MAPE** | 15.3% |

## 🛠️ Tech Stack

### Backend
- **Python 3.8+** - Core language
- **Flask** - Web framework & API
- **XGBoost** - Gradient boosting ML models
- **SHAP** - Model explainability
- **Pandas/NumPy** - Data processing
- **SQLite** - Flight database

### Frontend
- **JavaScript (ES6+)** - Frontend logic
- **D3.js** - Interactive map visualization
- **HTML5/CSS3** - UI with unified portfolio design

## 📁 Project Structure

```
Flight-Delay-Predictor/
├── app.py                      # Flask application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── WEATHER.md                  # Weather integration docs
├── .gitignore                  # Git ignore rules
│
├── model/
│   ├── __init__.py
│   └── delay_duration/
│       ├── __init__.py
│       ├── config.py           # Model configuration
│       ├── main.py             # Training pipeline
│       ├── model.py            # XGBoost model class
│       ├── utils.py            # Data utilities
│       ├── visualization.py    # Evaluation plots
│       ├── README.md           # Model documentation
│       └── output/             # Trained model files
│           ├── delay_duration_model.json
│           ├── label_encoders.pkl
│           └── metrics.json
│
├── static/
│   ├── css/
│   │   └── style.css           # Unified portfolio styles
│   └── js/
│       └── app.js              # Frontend JavaScript
│
├── templates/
│   └── index.html              # Main HTML template
│
└── data/
    └── flights.db              # SQLite database (not included)
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Flight-Delay-Predictor.git
cd Flight-Delay-Predictor
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model (Optional)

If you have the flight database:

```bash
python -m model.delay_duration.main --db-path data/flights.db
```

### 5. Run the Application

```bash
python app.py
```

Visit `http://localhost:5000` in your browser.

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main application page |
| `/api/airports` | GET | List of airports |
| `/api/airlines` | GET | List of airlines |
| `/api/predict` | POST | Make delay prediction |
| `/api/model-info` | GET | Model metrics and status |

### Prediction Request Example

```javascript
POST /api/predict
Content-Type: application/json

{
    "origin": "ATL",
    "destination": "LAX",
    "month": 6,
    "day": 15,
    "dayOfWeek": 5,
    "depHour": 17,
    "arrHour": 20,
    "airline": "DL"
}
```

### Response

```json
{
    "success": true,
    "probability": 0.35,
    "probabilityPercent": 35.0,
    "expectedDelay": 28.5,
    "riskLevel": "medium",
    "riskText": "Moderate risk of delay",
    "shapValues": [
        {"feature": "dep_hour", "displayName": "Departure Hour", "value": "17:00", "shap": 0.08},
        ...
    ]
}
```

## 🔧 Model Features

The model uses 12 features:

| Feature | Description |
|---------|-------------|
| Month | Month of flight (1-12) |
| Quarter | Quarter (1-4) |
| DayofMonth | Day of month (1-31) |
| DayOfWeek | Day of week (1-7) |
| Reporting_Airline_encoded | Airline code |
| Origin_encoded | Origin airport |
| Dest_encoded | Destination airport |
| Distance | Flight distance (miles) |
| CRSElapsedTime | Scheduled duration |
| dep_hour | Departure hour (0-23) |
| arr_hour | Arrival hour (0-23) |
| dep_time_category | Time period (1-5) |

## 🌦️ Future: Weather Integration

See [WEATHER.md](WEATHER.md) for planned weather feature integration using Meteostat API.

## 👨‍💻 Author

**Tanvir Bakther**  
Team 220 - Georgia Tech MS Analytics

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Bureau of Transportation Statistics for flight data
- Georgia Tech for academic support
- D3.js community for visualization tools

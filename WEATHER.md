# Weather Features Strategy for Flight Delay ML Models

## Overview
This guide outlines the weather features for the delay probability and delay duration models using **Meteostat API** historical weather data.

## Data Source: Meteostat API

### Available Weather Fields

Meteostat provides hourly weather observations with the following fields:

| Field | Database Column | Unit | Description | Typical Coverage |
|-------|----------------|------|-------------|------------------|
| `temp` | `temperature` | °C | Air temperature | ~95% |
| `dwpt` | `dew_point` | °C | Dew point temperature | ~95% |
| `rhum` | `relative_humidity` | % | Relative humidity (0-100) | ~90% |
| `prcp` | `precipitation` | mm | Precipitation amount | ~85% |
| `snow` | `snow_depth` | mm | Snow depth | ~60% (seasonal) |
| `wdir` | `wind_direction` | degrees | Wind direction (0-360) | ~90% |
| `wspd` | `wind_speed` | km/h | Wind speed | ~90% |
| `wpgt` | `wind_gust` | km/h | Peak wind gust | ~70% |
| `pres` | `pressure` | hPa | Atmospheric pressure | ~85% |
| `tsun` | `total_sunshine` | minutes | Total sunshine duration | ~50% |
| `coco` | `condition_code` | - | Weather condition code | ~80% |

### Fields NOT Available from Meteostat
⚠️ The following fields are **NOT** provided by Meteostat:
- Visibility (use humidity/precipitation as proxy)
- Cloud ceiling height (use condition code as proxy)
- Runway conditions
- Convective weather (thunderstorms) - must derive from condition codes
- Crosswind components (requires runway data)

## Database Schema

### `weather_observations` Table
Stores raw weather observations from Meteostat stations.

```sql
CREATE TABLE weather_observations (
    obs_id INTEGER PRIMARY KEY,
    station_id TEXT NOT NULL,
    observation_time DATETIME NOT NULL,

    -- Temperature and humidity
    temperature REAL,           -- temp (°C)
    dew_point REAL,             -- dwpt (°C)
    relative_humidity REAL,     -- rhum (%)

    -- Atmospheric
    pressure REAL,              -- pres (hPa)

    -- Precipitation and snow
    precipitation REAL,         -- prcp (mm)
    snow_depth REAL,            -- snow (mm)

    -- Wind
    wind_speed REAL,            -- wspd (km/h)
    wind_gust REAL,             -- wpgt (km/h)
    wind_direction REAL,        -- wdir (degrees)

    -- Condition and sunshine
    condition_code REAL,        -- coco
    total_sunshine REAL,        -- tsun (minutes)

    -- Derived flags for ML
    is_precipitation INTEGER,   -- prcp > 0
    is_high_wind INTEGER,       -- wspd > 46 km/h (~25 knots)
    is_extreme_temp INTEGER,    -- temp < 0°C or > 35°C
    is_high_humidity INTEGER    -- rhum > 85%
);
```

### `flight_weather` Table
Links flights to weather conditions at origin and destination.

```sql
CREATE TABLE flight_weather (
    flight_id INTEGER PRIMARY KEY,

    -- Origin weather (at departure time)
    origin_temp REAL,
    origin_dew_point REAL,
    origin_humidity REAL,
    origin_precip REAL,
    origin_precip_flag INTEGER,
    origin_wind_speed REAL,
    origin_wind_direction REAL,
    origin_pressure REAL,
    origin_condition_code REAL,

    -- Destination weather (at scheduled arrival time)
    dest_temp REAL,
    dest_dew_point REAL,
    dest_humidity REAL,
    dest_precip REAL,
    dest_precip_flag INTEGER,
    dest_wind_speed REAL,
    dest_wind_direction REAL,
    dest_pressure REAL,
    dest_condition_code REAL,

    -- Composite risk scores (calculated)
    origin_weather_risk REAL,
    dest_weather_risk REAL,
    combined_weather_risk REAL
);
```

## Feature Categories

### 1. **Primary Weather Features** (Available from Meteostat)

#### At Origin (Departure Time)
- `origin_precip_flag` - Binary: any precipitation
- `origin_precip` - Precipitation amount (mm)
- `origin_wind_speed` - Sustained winds (km/h)
- `origin_wind_gust` - Wind gusts (km/h)
- `origin_temp` - Temperature (°C)
- `origin_humidity` - Relative humidity (%)
- `origin_pressure` - Atmospheric pressure (hPa)

#### At Destination (Scheduled Arrival Time)
- `dest_precip_flag` - Binary: any precipitation
- `dest_precip` - Precipitation amount (mm)
- `dest_wind_speed` - Sustained winds (km/h)
- `dest_wind_gust` - Wind gusts (km/h)
- `dest_temp` - Temperature (°C)
- `dest_humidity` - Relative humidity (%)
- `dest_pressure` - Atmospheric pressure (hPa)

### 2. **Derived Weather Risk Scores**

#### Wind Risk (0-10 scale)
```python
def calculate_wind_risk(wind_speed_kmh):
    """
    Convert wind speed to risk score.
    Based on aviation weather minimums.
    """
    if wind_speed_kmh < 28:  # < 15 knots (normal operations)
        return 0
    elif wind_speed_kmh < 46:  # 15-25 knots (caution)
        return 4
    elif wind_speed_kmh < 65:  # 25-35 knots (high wind)
        return 7
    else:  # > 35 knots (severe)
        return 10
```

#### Precipitation Risk (0-10 scale)
```python
def calculate_precip_risk(precip_mm, temp_c, snow_depth_mm=None):
    """
    Convert precipitation to risk score.
    Accounts for frozen precipitation (higher risk).
    """
    if precip_mm == 0:
        return 0

    # Snow/ice is worse than rain
    is_frozen = temp_c < 2 or (snow_depth_mm and snow_depth_mm > 0)

    if precip_mm < 2.5:  # Light
        return 5 if is_frozen else 2
    elif precip_mm < 7.6:  # Moderate
        return 8 if is_frozen else 5
    else:  # Heavy (> 7.6mm)
        return 10 if is_frozen else 7
```

#### Temperature Extremes Risk (0-10 scale)
```python
def calculate_temp_risk(temp_c):
    """
    Extreme temperatures affect aircraft performance.
    """
    if -5 <= temp_c <= 30:  # Normal operating range
        return 0
    elif temp_c < -18:  # < 0°F - extreme cold
        return 8
    elif temp_c > 40:  # > 104°F - extreme heat
        return 8
    elif temp_c < 0:  # Freezing
        return 5
    elif temp_c > 35:  # > 95°F
        return 5
    else:
        return 2
```

#### Humidity Risk (0-10 scale)
```python
def calculate_humidity_risk(humidity_pct, temp_c):
    """
    High humidity + low temp = fog/ice risk.
    High humidity + high temp = reduced visibility.
    """
    if humidity_pct < 70:
        return 0
    elif humidity_pct >= 90:
        # Very high humidity
        if temp_c < 5:  # Fog/freezing fog risk
            return 8
        else:  # Reduced visibility likely
            return 5
    elif humidity_pct >= 85:
        if temp_c < 5:
            return 5
        else:
            return 3
    else:
        return 1
```

#### Composite Weather Risk Score
```python
def calculate_weather_risk(wind_risk, precip_risk, temp_risk, humidity_risk):
    """
    Combine individual risks into composite score (0-10).
    Weighted by aviation impact.
    """
    composite = (
        0.30 * wind_risk +
        0.35 * precip_risk +
        0.20 * temp_risk +
        0.15 * humidity_risk
    )
    return min(composite, 10)
```

### 3. **Advanced Derived Features**

#### Winter Weather Detection
```python
def detect_winter_weather(temp_c, precip_mm, snow_depth_mm, humidity_pct):
    """Detect winter weather conditions."""
    freezing_precip = (temp_c < 2) and (precip_mm > 0)
    snow_present = (snow_depth_mm and snow_depth_mm > 0)
    ice_risk = (temp_c < 2) and (temp_c > -5) and (humidity_pct > 85)

    return {
        'freezing_precip_flag': 1 if freezing_precip else 0,
        'snow_flag': 1 if snow_present else 0,
        'ice_risk_flag': 1 if ice_risk else 0
    }
```

#### Visibility Proxy (from humidity + precipitation)
```python
def estimate_visibility_category(humidity_pct, precip_mm, temp_c):
    """
    Estimate visibility category (not actual visibility).
    0 = Good, 1 = Marginal, 2 = Poor
    """
    if humidity_pct >= 95 and temp_c < 15:
        return 2  # Likely fog - poor visibility
    elif precip_mm > 5:
        return 2  # Heavy precip - poor visibility
    elif humidity_pct >= 90 or precip_mm > 2:
        return 1  # Marginal visibility
    else:
        return 0  # Good visibility
```

### 4. **Temporal Weather Features**

#### Weather Differentials (Origin vs Destination)
```python
# Weather difference between origin and destination
temp_differential = abs(dest_temp - origin_temp)
humidity_differential = abs(dest_humidity - origin_humidity)
pressure_differential = abs(dest_pressure - origin_pressure)
wind_speed_differential = abs(dest_wind_speed - origin_wind_speed)

# Large differentials can indicate frontal systems
frontal_activity = (pressure_differential > 10) or (temp_differential > 15)
```

#### Dew Point Spread (Fog Risk)
```python
def calculate_dew_point_spread(temp_c, dew_point_c):
    """
    Small spread indicates fog risk.
    Returns: spread in °C and fog risk flag.
    """
    spread = temp_c - dew_point_c
    fog_risk = 1 if spread < 2.5 else 0
    return spread, fog_risk
```

## Feature Engineering for ML Models

### For **Delay Probability Model** (Binary Classification)

**Top 10 Most Important Features (Meteostat-based):**
1. `dest_precip_flag` - Any precipitation at destination
2. `origin_precip_flag` - Any precipitation at origin
3. `dest_wind_risk` - Destination wind score
4. `combined_weather_risk` - Composite score
5. `origin_wind_risk` - Origin wind score
6. `dest_humidity_risk` - Destination humidity/fog risk
7. `freezing_precip_flag` - Ice/snow conditions
8. `dest_precip` - Precipitation amount at destination
9. `weather_differential` - Weather change origin→destination
10. `extreme_temp_flag` - Temperature extremes at either end

### For **Delay Duration Model** (Regression)

**Top Features:**
1. `dest_precip` - Precipitation amount (mm)
2. `origin_precip` - Precipitation amount (mm)
3. `combined_weather_risk` - Composite risk score
4. `dest_wind_speed` - Raw wind speed
5. `origin_wind_speed` - Raw wind speed
6. `dest_humidity` - Raw humidity value
7. `freezing_conditions_flag` - Ice/snow present
8. `wind_gust_differential` - Gustiness at dest vs origin
9. `temp_extreme_severity` - How extreme the temperature is
10. `pressure_differential` - Barometric pressure change

## Interaction Features

These capture complex relationships:

```python
# High wind + precipitation is worse than either alone
high_wind_and_precip = (wind_speed > 46) & (precip_flag == 1)

# High humidity + freezing = ice/fog risk
freezing_fog_risk = (humidity > 90) & (temp_c < 5) & (temp_c > -5)

# Temperature near freezing + precipitation = ice risk
freezing_precip_risk = (temp_c < 2) & (temp_c > -10) & (precip_flag == 1)

# Extreme weather at either end
worst_case_weather = max(origin_weather_risk, dest_weather_risk)

# Deteriorating conditions (dest worse than origin)
deteriorating_weather = dest_weather_risk - origin_weather_risk
```
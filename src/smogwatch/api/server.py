"""
AirAware FastAPI REST API Backend
---------------------------------
Production asynchronous REST service providing:
1. Real-time CPCB NAQI air quality & meteorological readings.
2. 24 to 72-hour recursive XGBoost multi-horizon forecast endpoints with 90% confidence bands.
3. NASA FIRMS satellite active fire telemetry & Stubble Fire Index (SFI).
4. Group-specific clinical & operational health advisories.
5. Personalized Air Vulnerability Index (AVI) scoring.
6. What-If Policy Intervention Simulation engine.
7. XGBoost vs Prophet model comparative benchmarks.
8. Multi-channel automated alert dispatch & subscription management.
"""

import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from data_pipeline import (
    NORTH_INDIAN_CITIES,
    fetch_nasa_firms_data,
    calculate_stubble_fire_index,
    fetch_live_city_data,
    calculate_cpcb_aqi,
    get_aqi_color
)
from ml_engine import get_model
from advisory_engine import (
    get_grap_stage,
    get_group_specific_advice,
    calculate_personal_vulnerability_index
)
from model_comparison import get_comparison_lab
from alert_system import get_alert_system

# Initialize FastAPI App
app = FastAPI(
    title="AirAware Environmental Intelligence REST API",
    description="High-performance REST API for North India Smog Forecasting, NASA FIRMS Telemetry, and Health Protocols.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for cross-origin web/mobile integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# PYDANTIC SCHEMAS
# -----------------------------------------------------------------------------
class VulnerabilityRequest(BaseModel):
    age: int = Field(..., ge=1, le=105, description="Age of the individual in years")
    conditions: List[str] = Field(default=[], description="List of pre-existing conditions (Asthma, COPD, Heart Disease, etc.)")
    daily_outdoor_hours: float = Field(..., ge=0.0, le=24.0, description="Hours spent outdoors per day")
    activity_level: str = Field(default="Moderate (Walking / Light Chores)", description="Sedentary, Moderate, or Vigorous")
    current_aqi: int = Field(default=320, description="Ambient CPCB AQI level")

class SimulationRequest(BaseModel):
    base_pm25: float = Field(default=220.0, description="Baseline PM2.5 concentration (ug/m3)")
    fire_reduction_pct: float = Field(default=60.0, ge=0.0, le=100.0, description="Farm stubble fire reduction percentage (0-100%)")
    emission_cut_pct: float = Field(default=30.0, ge=0.0, le=80.0, description="Urban vehicular and industrial curbs percentage (0-80%)")
    wind_speed_delta: float = Field(default=1.5, description="Wind speed anomaly in m/s")
    temp_delta: float = Field(default=0.0, description="Temperature anomaly in degrees Celsius")
    is_stubble_season: bool = Field(default=True, description="Whether simulation is in October-November season")

class SubscriberRequest(BaseModel):
    name: str = Field(..., description="Subscriber or organization name")
    role: str = Field(default="School Principal", description="User role / persona")
    city: str = Field(default="Delhi", description="City to monitor")
    contact: str = Field(..., description="Phone number or email address")
    channel: str = Field(default="WhatsApp", description="WhatsApp, SMS, or Email")
    aqi_threshold: int = Field(default=300, description="AQI threshold triggering dispatch")

class AlertDispatchRequest(BaseModel):
    city: str = Field(default="Delhi", description="Target city")
    target_channel: str = Field(default="WhatsApp", description="WhatsApp, SMS, School_Circular, or Occupational_Directive")
    recipient: str = Field(default="Public Broadcast", description="Recipient identifier")


# -----------------------------------------------------------------------------
# ROOT & SYSTEM ENDPOINTS
# -----------------------------------------------------------------------------
@app.get("/", tags=["System"])
def root():
    return {
        "service": "AirAware Environmental Intelligence REST API",
        "version": "1.0.0",
        "status": "online",
        "documentation": "/docs",
        "sdgs": ["SDG 3 (Good Health)", "SDG 11 (Sustainable Cities)", "SDG 13 (Climate Action)"],
        "endpoints": [
            "/api/v1/cities",
            "/api/v1/current/{city}",
            "/api/v1/forecast/{city}",
            "/api/v1/fires",
            "/api/v1/fires/stubble-index/{city}",
            "/api/v1/advisory/{city}",
            "/api/v1/vulnerability",
            "/api/v1/simulate",
            "/api/v1/benchmarks",
            "/api/v1/alerts/subscribers",
            "/api/v1/alerts/dispatch-test"
        ]
    }

@app.get("/health", tags=["System"])
def health_check():
    model = get_model()
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "xgboost_trained": model.is_trained,
        "available_cities": len(NORTH_INDIAN_CITIES)
    }


# -----------------------------------------------------------------------------
# AIR QUALITY & FORECAST ENDPOINTS
# -----------------------------------------------------------------------------
@app.get("/api/v1/cities", tags=["Air Quality"])
def get_cities():
    """Returns list of monitored North Indian cities with spatial coordinates."""
    return {
        "count": len(NORTH_INDIAN_CITIES),
        "cities": [
            {
                "city": name,
                "state": meta["state"],
                "latitude": meta["lat"],
                "longitude": meta["lon"],
                "baseline_pm25": meta["baseline_pm25"],
                "description": meta["description"]
            }
            for name, meta in NORTH_INDIAN_CITIES.items()
        ]
    }

# In-memory TTL Caches for sub-millisecond API response
_CURRENT_CACHE: Dict[str, Tuple[float, Any]] = {}
_FORECAST_CACHE: Dict[Tuple[str, int], Tuple[float, Any]] = {}
_FIRES_CACHE: Optional[Tuple[float, Any]] = None

@app.get("/api/v1/current/{city}", tags=["Air Quality"])
def get_current_air_quality(city: str):
    """Fetches real-time CPCB NAQI, individual pollutant sub-indices, and weather covariates."""
    city_name = city.capitalize()
    if city_name not in NORTH_INDIAN_CITIES:
        city_name = "Delhi"

    now_t = time.time()
    if city_name in _CURRENT_CACHE:
        t_cached, data_cached = _CURRENT_CACHE[city_name]
        if now_t - t_cached < 900:  # 15 minutes
            return data_cached

    city_data = fetch_live_city_data(city_name)
    current = city_data["current"]

    aqi, category, dominant, sub_indices = calculate_cpcb_aqi({
        "PM2.5": current.get("pm2_5"),
        "PM10": current.get("pm10"),
        "NO2": current.get("no2"),
        "SO2": current.get("so2"),
        "CO": current.get("co"),
        "O3": current.get("o3")
    })

    grap = get_grap_stage(aqi)

    result = {
        "city": city_name,
        "timestamp": str(current.get("timestamp", datetime.now())),
        "cpcb_aqi": aqi,
        "category": category,
        "hex_color": get_aqi_color(category),
        "dominant_pollutant": dominant,
        "pollutant_sub_indices": sub_indices,
        "raw_concentrations": {
            "pm2_5": current.get("pm2_5"),
            "pm10": current.get("pm10"),
            "no2": current.get("no2"),
            "so2": current.get("so2"),
            "co": current.get("co"),
            "o3": current.get("o3")
        },
        "meteorology": {
            "temperature_c": current.get("temperature"),
            "relative_humidity_pct": current.get("humidity"),
            "wind_speed_ms": current.get("wind_speed"),
            "wind_direction_deg": current.get("wind_direction")
        },
        "grap_status": grap
    }
    _CURRENT_CACHE[city_name] = (now_t, result)
    return result

@app.get("/api/v1/forecast/{city}", tags=["Air Quality"])
def get_aqi_forecast(city: str, horizon: int = Query(72, ge=12, le=72, description="Forecast horizon in hours (12-72)")):
    """Generates multi-horizon recursive XGBoost AQI predictions with 90% confidence bounds."""
    city_name = city.capitalize()
    if city_name not in NORTH_INDIAN_CITIES:
        city_name = "Delhi"

    now_t = time.time()
    cache_key = (city_name, horizon)
    if cache_key in _FORECAST_CACHE:
        t_cached, data_cached = _FORECAST_CACHE[cache_key]
        if now_t - t_cached < 900:  # 15 minutes
            return data_cached

    city_data = fetch_live_city_data(city_name)
    fires_df = fetch_nasa_firms_data()
    model = get_model()

    forecast_df = model.forecast_72h(city_name, city_data["hourly_df"], fires_df)
    forecast_df = forecast_df.iloc[:horizon]

    records = []
    for _, row in forecast_df.iterrows():
        records.append({
            "timestamp": str(row["timestamp"]),
            "hour_ahead": int(row["hour_ahead"]),
            "predicted_aqi": int(row["predicted_aqi"]),
            "aqi_lower_ci90": int(row["aqi_lower"]),
            "aqi_upper_ci90": int(row["aqi_upper"]),
            "category": row["aqi_category"],
            "predicted_pm25": float(row["predicted_pm25"]),
            "temperature": float(row["temperature"]),
            "wind_speed": float(row["wind_speed"]),
            "inversion_risk": float(row["inversion_risk"])
        })

    result = {
        "city": city_name,
        "horizon_hours": horizon,
        "model": "XGBoost Regressor (Multi-Step Lag Propagator)",
        "forecast_series": records
    }
    _FORECAST_CACHE[cache_key] = (now_t, result)
    return result


# -----------------------------------------------------------------------------
# SATELLITE TELEMETRY (NASA FIRMS)
# -----------------------------------------------------------------------------
@app.get("/api/v1/fires", tags=["Satellite Telemetry"])
def get_active_fires(limit: int = Query(250, ge=10, le=1000)):
    """Fetches Near Real-Time active fire hotspots from NASA FIRMS MODIS & VIIRS satellites."""
    global _FIRES_CACHE
    now_t = time.time()
    if _FIRES_CACHE is not None:
        t_cached, df_cached = _FIRES_CACHE
        if now_t - t_cached < 1800:
            fires_df = df_cached
        else:
            fires_df = fetch_nasa_firms_data()
            _FIRES_CACHE = (now_t, fires_df)
    else:
        fires_df = fetch_nasa_firms_data()
        _FIRES_CACHE = (now_t, fires_df)

    total_detected = len(fires_df)
    subset = fires_df.head(limit).to_dict(orient="records")
    return {
        "total_active_clusters": total_detected,
        "returned_clusters": len(subset),
        "source": "NASA FIRMS (MODIS C6.1 & SUOMI VIIRS 375m)",
        "bounding_box": "North India Agriculture Belt (Punjab, Haryana, West UP)",
        "clusters": subset
    }

@app.get("/api/v1/fires/stubble-index/{city}", tags=["Satellite Telemetry"])
def get_city_stubble_index(city: str):
    """Calculates distance-weighted Stubble Fire Index (SFI) and smoke plume advection risk."""
    city_name = city.capitalize()
    meta = NORTH_INDIAN_CITIES.get(city_name, NORTH_INDIAN_CITIES["Delhi"])
    fires_df = fetch_nasa_firms_data()
    metrics = calculate_stubble_fire_index(meta["lat"], meta["lon"], fires_df)
    return {
        "city": city_name,
        "stubble_fire_metrics": metrics
    }


# -----------------------------------------------------------------------------
# HEALTH ADVISORY & PERSONAL VULNERABILITY
# -----------------------------------------------------------------------------
@app.get("/api/v1/advisory/{city}", tags=["Health Advisory"])
def get_city_health_advisory(city: str):
    """Returns group-specific clinical and operational guidelines for the city's current AQI."""
    city_name = city.capitalize()
    city_data = fetch_live_city_data(city_name)
    aqi = int(city_data["current"].get("cpcb_aqi", 150))
    grap = get_grap_stage(aqi)
    advice = get_group_specific_advice(aqi)

    return {
        "city": city_name,
        "current_aqi": aqi,
        "grap_stage": grap["stage"],
        "advisories": advice
    }

@app.post("/api/v1/vulnerability", tags=["Health Advisory"])
def compute_personal_vulnerability(req: VulnerabilityRequest):
    """Computes personalized 1-100 Air Vulnerability Index (AVI) score and exposure budget."""
    result = calculate_personal_vulnerability_index(
        age=req.age,
        conditions=req.conditions,
        daily_outdoor_hours=req.daily_outdoor_hours,
        activity_level=req.activity_level,
        current_aqi=req.current_aqi
    )
    return result


# -----------------------------------------------------------------------------
# POLICY WHAT-IF SIMULATION LAB
# -----------------------------------------------------------------------------
@app.post("/api/v1/simulate", tags=["Policy Simulation"])
def run_what_if_simulation(req: SimulationRequest):
    """Evaluates the atmospheric delta of agricultural fire cuts and urban emission curbs."""
    model = get_model()
    sim_result = model.simulate_what_if_scenario(
        base_pm25=req.base_pm25,
        fire_reduction_pct=req.fire_reduction_pct,
        emission_cut_pct=req.emission_cut_pct,
        wind_speed_delta=req.wind_speed_delta,
        temp_delta=req.temp_delta,
        is_stubble_season=req.is_stubble_season
    )
    return sim_result


# -----------------------------------------------------------------------------
# BENCHMARK COMPARISONS (XGBOOST VS PROPHET)
# -----------------------------------------------------------------------------
@app.get("/api/v1/benchmarks", tags=["Model Comparison"])
def get_model_benchmarks(city: str = "Delhi"):
    """Returns empirical validation metrics comparing XGBoost vs Additive Prophet."""
    lab = get_comparison_lab()
    if not lab.benchmarks:
        bench = lab.run_benchmark(city)
    else:
        bench = lab.benchmarks
    return bench


# -----------------------------------------------------------------------------
# AUTOMATED ALERT DISPATCH & SUBSCRIBERS
# -----------------------------------------------------------------------------
@app.get("/api/v1/alerts/subscribers", tags=["Alerts"])
def list_subscribers():
    """Lists registered alert subscribers."""
    alert_sys = get_alert_system()
    return {
        "count": len(alert_sys.subscribers),
        "subscribers": alert_sys.subscribers
    }

@app.post("/api/v1/alerts/subscribe", tags=["Alerts"])
def subscribe_to_alerts(req: SubscriberRequest):
    """Registers a new user/organization for automated AQI emergency broadcasts."""
    alert_sys = get_alert_system()
    sub = alert_sys.register_subscriber(
        name=req.name,
        role=req.role,
        city=req.city,
        contact=req.contact,
        channel=req.channel,
        threshold=req.aqi_threshold
    )
    return {
        "message": "Successfully registered for AirAware emergency broadcasts.",
        "subscriber": sub
    }

@app.post("/api/v1/alerts/dispatch-test", tags=["Alerts"])
def test_alert_dispatch(req: AlertDispatchRequest):
    """Triggers simulated multi-channel emergency broadcast."""
    city_data = fetch_live_city_data(req.city)
    aqi = int(city_data["current"].get("cpcb_aqi", 350))
    _, cat, _, _ = calculate_cpcb_aqi({"PM2.5": aqi * 0.75})
    grap = get_grap_stage(aqi)

    alert_sys = get_alert_system()
    result = alert_sys.dispatch_test_alert(
        city_name=req.city,
        current_aqi=aqi,
        aqi_category=cat,
        grap_stage=grap["roman"],
        target_channel=req.target_channel,
        recipient=req.recipient
    )
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

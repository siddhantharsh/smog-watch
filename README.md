# smog-watch 🌫️

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange?logo=data:image/svg+xml;base64,)](https://xgboost.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://docs.docker.com/compose/)

> **smog-watch** is a real-time air quality intelligence system built for North India's annual smog crisis.  
> It forecasts PM₂.₅/AQI up to 72 hours ahead, tracks active crop fires via NASA satellite feeds, and  
> delivers group-specific health advisories — all through a live dashboard and REST API.

---

## Why this exists

Every October–December, the Indo-Gangetic Plain suffocates under a blanket of agricultural smoke, vehicle exhaust, and stagnant winter air. Existing AQI portals tell you *what is happening* — a single backward-looking number. They don't tell you what's coming, or what to *do* about it.

**smog-watch** is built to fix that. It combines machine learning forecasting, satellite fire telemetry, and clinical health protocols into a single deployable platform.

---

## What it does

| Feature | Details |
|---|---|
| **72-hour AQI Forecast** | XGBoost gradient boosting with lag features, rolling stats, and atmospheric physics proxies. Outputs hourly trajectories with 90% confidence intervals. |
| **Active Fire Tracking** | Pulls NASA FIRMS MODIS C6.1 + VIIRS 375m data. Computes a distance-weighted Stubble Fire Index for any city. Renders hotspots on interactive 3D maps. |
| **Health Advisories** | Tailored protocols for children, asthma/COPD patients, outdoor workers, and the general public. Triggers GRAP Stage alerts automatically. |
| **Vulnerability Score** | A 1–100 Air Vulnerability Index (AVI) based on age, pre-existing conditions, and outdoor exposure. Includes a mask prescription and daily outdoor time budget. |
| **What-If Simulator** | Lets policymakers model the AQI impact of interventions: stubble-burning reductions, vehicle bans, wind changes. |
| **Alert Dispatcher** | Sends SMS blasts, WhatsApp messages, school circulars, and occupational safety directives when AQI crosses critical thresholds. |
| **REST API** | Async FastAPI backend with Swagger UI and ReDoc. Ready for mobile apps, IoT devices, and municipal dashboards. |
| **Model Benchmarks** | Head-to-head comparison: XGBoost (R²=0.986) vs Facebook Prophet (R²=0.862) on held-out CPCB test data. |

---

## Repository layout

```
smog-watch/
│
├── src/smogwatch/              # Main Python package
│   ├── core/
│   │   ├── pipeline.py         # CPCB NAQI calculator, NASA FIRMS downloader, Open-Meteo client
│   │   └── advisory.py         # Group-specific health protocols, GRAP engine, AVI scoring
│   ├── ml/
│   │   ├── engine.py           # Feature engineering, XGBoost training, 72h forecast + simulator
│   │   └── benchmarks.py       # XGBoost vs Prophet comparison lab
│   ├── api/
│   │   └── server.py           # FastAPI REST backend (Port 8000)
│   ├── dashboard/
│   │   └── ui.py               # Streamlit dashboard with Pydeck 3D maps + Plotly charts
│   └── alerts/
│       └── dispatcher.py       # SMS / WhatsApp / circular notification system
│
├── scripts/
│   ├── train.py                # Standalone model training + validation script
│   └── launch.py               # One-command launcher for API + Dashboard
│
├── config/
│   ├── Dockerfile              # Production container image
│   ├── docker-compose.yml      # Multi-service orchestration
│   ├── .dockerignore
│   └── services.yaml           # Port/path configuration
│
├── artifacts/
│   ├── models/                 # Serialized XGBoost model + benchmark metrics
│   │   ├── airaware_xgboost.json
│   │   ├── model_metadata.json
│   │   └── comparison_metrics.json
│   └── data/                   # Cached CPCB CSVs + NASA fire JSON
│       ├── cpcb_historical_air_quality.csv
│       ├── firms_active_fires.json
│       ├── alert_subscribers.json
│       └── alert_dispatch_logs.json
│
├── tests/
│   └── test_airaware.py        # 11 unit + integration tests (CPCB math, XGBoost, API routes)
│
├── notebooks/                  # Exploratory analysis (Jupyter)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Getting started

### Prerequisites

- Python 3.11+
- `pip` or `uv`
- Docker + Docker Compose (optional, for containerised deployment)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the test suite

```bash
python -m unittest tests/test_airaware.py -v
```

Runs 11 tests covering CPCB index formulas, XGBoost inference, Prophet benchmarks, advisory logic, and API route responses.

### 3. Launch

#### One command (recommended)

```bash
python scripts/launch.py
```

| Service | URL |
|---|---|
| Streamlit Dashboard | http://localhost:8501 |
| FastAPI Swagger Docs | http://localhost:8000/docs |
| ReDoc API Reference | http://localhost:8000/redoc |

#### Manual

```bash
# API server
uvicorn src.smogwatch.api.server:app --host 0.0.0.0 --port 8000 --reload

# Dashboard
streamlit run src/smogwatch/dashboard/ui.py
```

#### Docker Compose

```bash
docker-compose -f config/docker-compose.yml up --build
```

---

## API reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health + model load status |
| `GET` | `/api/v1/cities` | List of monitored cities |
| `GET` | `/api/v1/current/{city}` | Live AQI, pollutant breakdown, GRAP stage |
| `GET` | `/api/v1/forecast/{city}?horizon=72` | Hourly XGBoost forecast with 90% C.I. |
| `GET` | `/api/v1/fires` | NASA FIRMS active fire hotspots |
| `GET` | `/api/v1/fires/stubble-index/{city}` | Distance-weighted Stubble Fire Index |
| `GET` | `/api/v1/advisory/{city}` | Group-specific health protocols |
| `POST` | `/api/v1/vulnerability` | Compute personalised AVI score |
| `POST` | `/api/v1/simulate` | Run a policy What-If simulation |
| `GET` | `/api/v1/benchmarks` | XGBoost vs Prophet metrics |
| `POST` | `/api/v1/alerts/subscribe` | Register for emergency alerts |
| `POST` | `/api/v1/alerts/dispatch-test` | Trigger a test broadcast |

---

## Model performance

Benchmarked on held-out 2024–25 CPCB Delhi AQI data (Oct–Jan).

| Model | R² | RMSE (µg/m³) | Inference latency |
|---|---|---|---|
| **XGBoost** (primary) | **0.9861** | **20.88** | 0.38 ms |
| Facebook Prophet | 0.8623 | 65.82 | 0.01 ms |

XGBoost is used for all production forecasts. Prophet serves as an interpretable seasonal baseline.

---

## CPCB AQI scale

| Range | Category | Guidance |
|---|---|---|
| 0–50 | Good | No restrictions |
| 51–100 | Satisfactory | Sensitive groups: reduce prolonged outdoor exertion |
| 101–200 | Moderate | Asthma/heart patients: limit outdoor activity |
| 201–300 | Poor | Most people: discomfort on prolonged exposure. GRAP Stage I |
| 301–400 | Very Poor | Respiratory illness risk. GRAP Stage II |
| 401–500 | Severe | GRAP Stage III — school restrictions, construction bans |
| >500 | Severe+ | GRAP Stage IV — public health emergency |

---

## Coverage

Monitored cities: **Delhi, Gurugram, Noida, Ghaziabad, Faridabad, Agra, Kanpur, Lucknow, Patna, Chandigarh, Amritsar, Ludhiana, Varanasi.**

Fire hotspot tracking region: Punjab · Haryana · Western Uttar Pradesh (paddy stubble burning zone).

---

## Tech stack

- **ML:** XGBoost ≥ 3.2, Prophet, NumPy, Pandas, SciPy  
- **API:** FastAPI, Uvicorn, Pydantic v2  
- **Dashboard:** Streamlit, Plotly, Pydeck (Deck.gl)  
- **Satellite data:** NASA FIRMS (MODIS C6.1 / VIIRS 375m)  
- **Met data:** Open-Meteo API  
- **AQI standard:** CPCB National AQI (NAQI) methodology  
- **Container:** Docker, Docker Compose  
- **Testing:** Python `unittest`

---

## Alignment with UN SDGs

- **SDG 3 – Good Health and Well-being** (Target 3.9): reduce deaths from air pollution through early warning and actionable guidance  
- **SDG 11 – Sustainable Cities** (Target 11.6): reduce the per capita environmental impact of cities  
- **SDG 13 – Climate Action** (Target 13.1): strengthen resilience and adaptive capacity to climate-related hazards

---

## License

MIT © 2025 Siddhant Harsh

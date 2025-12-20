"""
AirAware Comprehensive Automated Test Suite
-------------------------------------------
Tests:
1. CPCB National AQI calculation & breakpoint compliance.
2. Spatial Haversine Stubble Fire Index calculations.
3. XGBoost recursive multi-horizon forecast bounds.
4. Additive Prophet forecaster fit & prediction.
5. Advisory Engine GRAP mapping & Personal Vulnerability (AVI) scoring.
6. Alert Notification multi-channel payload generation.
7. FastAPI REST API endpoints via TestClient.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_pipeline import (
    calculate_sub_index,
    calculate_cpcb_aqi,
    calculate_stubble_fire_index,
    NORTH_INDIAN_CITIES
)
from ml_engine import get_model, compute_metrics, compute_atmospheric_indices
from model_comparison import AdditiveProphetForecaster
from advisory_engine import (
    get_grap_stage,
    get_group_specific_advice,
    calculate_personal_vulnerability_index
)
from alert_system import AlertNotificationSystem
from fastapi.testclient import TestClient
from api import app


class TestCPCBIndexCalculation(unittest.TestCase):
    """Verifies that air quality sub-indices match official CPCB breakpoints."""

    def test_pm25_breakpoints(self):
        # Good (0-30 -> 0-50)
        self.assertEqual(calculate_sub_index("PM2.5", 15.0), 25.0)
        self.assertEqual(calculate_sub_index("PM2.5", 30.0), 50.0)
        # Satisfactory (31-60 -> 51-100)
        self.assertTrue(51.0 <= calculate_sub_index("PM2.5", 45.0) <= 100.0)
        # Severe (250-500 -> 401-500)
        self.assertTrue(401.0 <= calculate_sub_index("PM2.5", 350.0) <= 500.0)
        # Above 500 (Hazardous emergency)
        self.assertGreater(calculate_sub_index("PM2.5", 600.0), 500.0)

    def test_overall_aqi_prominence(self):
        # PM2.5 dominant
        pollutants = {"PM2.5": 280.0, "PM10": 150.0, "NO2": 40.0}
        aqi, cat, dom, _ = calculate_cpcb_aqi(pollutants)
        self.assertEqual(cat, "Severe")
        self.assertEqual(dom, "PM2.5")
        self.assertGreaterEqual(aqi, 400)


class TestStubbleFireIndex(unittest.TestCase):
    """Tests spatial distance calculation and smoke plume risk weighting."""

    def test_stubble_fire_scoring(self):
        fires_df = pd.DataFrame([
            {"latitude": 30.9, "longitude": 75.8, "frp": 45.0, "confidence": 95},
            {"latitude": 30.5, "longitude": 76.0, "frp": 60.0, "confidence": 90},
        ])
        # Test distance to Delhi
        metrics = calculate_stubble_fire_index(28.6139, 77.2090, fires_df)
        self.assertIn("stubble_fire_index", metrics)
        self.assertIn("plume_risk", metrics)
        self.assertGreater(metrics["fire_count_350km"], 0)


class TestMachineLearningModels(unittest.TestCase):
    """Tests XGBoost and Additive Prophet models."""

    def test_xgboost_forecast(self):
        model = get_model()
        self.assertTrue(model.is_trained)
        meta = model.metadata
        self.assertIn("metrics", meta)
        self.assertGreater(meta["metrics"]["r2"], 0.85)

    def test_additive_prophet_forecaster(self):
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        y = 50.0 + 20.0 * np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 2, 100)
        forecaster = AdditiveProphetForecaster(n_changepoints=5)
        forecaster.fit(pd.Series(dates), y)
        preds = forecaster.predict(pd.Series(dates))
        self.assertEqual(len(preds), 100)
        self.assertTrue(np.all(preds > 0))


class TestAdvisoryEngine(unittest.TestCase):
    """Tests clinical and regulatory advisory mapping."""

    def test_grap_stages(self):
        stage_good = get_grap_stage(45)
        self.assertIn("Normal", stage_good["stage"])

        stage_severe = get_grap_stage(420)
        self.assertIn("Stage III", stage_severe["stage"])

        stage_emerg = get_grap_stage(475)
        self.assertIn("Stage IV", stage_emerg["stage"])

    def test_personal_vulnerability_scoring(self):
        # Elderly patient with asthma
        v_high = calculate_personal_vulnerability_index(
            age=72,
            conditions=["Asthma", "Heart Disease / Hypertension"],
            daily_outdoor_hours=4.0,
            activity_level="Moderate (Walking / Light Chores)",
            current_aqi=380
        )
        self.assertGreaterEqual(v_high["avi_score"], 70)
        self.assertIn("Vulnerability", v_high["risk_tier"])
        self.assertLessEqual(v_high["safe_outdoor_budget_minutes"], 40)


class TestAlertNotificationSystem(unittest.TestCase):
    """Tests multi-channel notification generation."""

    def test_alert_formatting(self):
        sys = AlertNotificationSystem()
        alerts = sys.evaluate_and_format_alerts(
            city_name="Delhi",
            current_aqi=410,
            aqi_category="Severe",
            grap_stage="Stage III",
            stubble_fire_index=72.0,
            inversion_risk=3.5
        )
        channels = alerts["channels"]
        self.assertIn("AIRAWARE ALERT", channels["sms"])
        self.assertIn("AIRAWARE EMERGENCY AIR ALERT", channels["whatsapp"])
        self.assertIn("OFFICIAL ADVISORY", channels["school_circular"])
        self.assertIn("OCCUPATIONAL SAFETY", channels["occupational_directive"])


class TestFastAPIEndpoints(unittest.TestCase):
    """Tests API routes via FastAPI TestClient."""

    def setUp(self):
        self.client = TestClient(app)

    def test_health_and_root(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "healthy")

    def test_cities_endpoint(self):
        r = self.client.get("/api/v1/cities")
        self.assertEqual(r.status_code, 200)
        self.assertGreaterEqual(r.json()["count"], 10)

    def test_current_aqi_endpoint(self):
        r = self.client.get("/api/v1/current/Delhi")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("cpcb_aqi", data)
        self.assertIn("category", data)
        self.assertIn("dominant_pollutant", data)


if __name__ == "__main__":
    unittest.main()

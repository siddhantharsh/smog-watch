"""
AirAware Machine Learning Engine
---------------------------------
Features:
1. Feature engineering pipeline capturing time-lags, diurnal cycles, stubble burning indicators,
   and atmospheric temperature inversion proxies.
2. Native XGBoost Booster training (independent of sklearn cython/dll dependencies).
3. Multi-horizon 24-72 hour recursive forecasting with uncertainty confidence bounds.
4. "What-If" Policy Simulation Engine (stubble fire reduction, wind speed variations, emission cuts).
5. Feature importance extraction to explain drivers of North Indian pollution spikes.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import xgboost as xgb

from data_pipeline import (
    load_or_generate_historical_cpcb_dataset,
    calculate_cpcb_aqi,
    calculate_stubble_fire_index,
    NORTH_INDIAN_CITIES,
    DATA_DIR
)

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_FILE = os.path.join(MODELS_DIR, "airaware_xgboost.json")
METADATA_FILE = os.path.join(MODELS_DIR, "model_metadata.json")

# Core features used in the ML model
FEATURE_COLUMNS = [
    "pm25_lag_1",
    "pm25_lag_2",
    "pm25_lag_3",
    "pm25_lag_24",
    "pm25_roll_mean_6h",
    "pm25_roll_mean_24h",
    "hour",
    "day_of_week",
    "month",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "is_stubble_season",
    "stubble_fire_count",
    "temperature",
    "humidity",
    "wind_speed",
    "inversion_risk_factor",
    "ventilation_index"
]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, float]:
    """Pure numpy computation of RMSE, MAE, R2, and MAPE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1.0))) * 100.0)
    return rmse, mae, r2, mape


def compute_atmospheric_indices(temp: float, humidity: float, wind_speed: float) -> Tuple[float, float]:
    """
    Computes physical atmospheric proxies:
    1. Inversion Risk Factor: High humidity + low temperature + calm wind = severe smog trapping.
    2. Ventilation Index: Wind speed * boundary layer mixing height proxy.
    """
    wind = max(0.4, float(wind_speed))
    h_ratio = min(1.0, max(0.2, float(humidity) / 100.0))
    t_diff = max(0.0, 32.0 - float(temp))  # Colder air inversions trap ground pollutants

    inversion_risk = (h_ratio * t_diff) / wind
    ventilation = wind * max(100.0, 150.0 + 35.0 * max(0.0, float(temp)))
    return round(float(inversion_risk), 2), round(float(ventilation), 1)


def create_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw timeseries DataFrame into a rich feature matrix for XGBoost.
    Handles lag features, rolling statistics, cyclical encodings, and atmospheric physics.
    """
    df = df.copy()

    # Ensure timestamp column exists
    if "Date" in df.columns and "timestamp" not in df.columns:
        df["timestamp"] = pd.to_datetime(df["Date"])
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Target PM2.5
    pm_col = "PM2.5" if "PM2.5" in df.columns else "pm2_5"
    df["target_pm25"] = df[pm_col]

    # Temporal features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["day_of_year"] = df["timestamp"].dt.dayofyear

    # Cyclical representations
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    # Stubble season flag (Oct 10 to Nov 25)
    def check_stubble(row):
        m = row["month"]
        d = row["timestamp"].day
        return 1 if ((m == 10 and d >= 10) or (m == 11 and d <= 25)) else 0

    if "Is_Stubble_Season" in df.columns:
        df["is_stubble_season"] = df["Is_Stubble_Season"]
    else:
        df["is_stubble_season"] = df.apply(check_stubble, axis=1)

    # Fire count
    if "Stubble_Fire_Count" in df.columns:
        df["stubble_fire_count"] = df["Stubble_Fire_Count"]
    elif "stubble_fire_count" not in df.columns:
        df["stubble_fire_count"] = df["is_stubble_season"] * np.random.randint(200, 1500, size=len(df))

    # Weather columns mapping
    temp_col = "Temperature" if "Temperature" in df.columns else "temperature"
    hum_col = "Humidity" if "Humidity" in df.columns else "humidity"
    wind_col = "Wind_Speed" if "Wind_Speed" in df.columns else "wind_speed"

    df["temperature"] = df[temp_col].fillna(25.0)
    df["humidity"] = df[hum_col].fillna(60.0)
    df["wind_speed"] = df[wind_col].fillna(2.5)

    # Atmospheric indices
    inv_risks, vents = [], []
    for _, r in df.iterrows():
        ir, vi = compute_atmospheric_indices(r["temperature"], r["humidity"], r["wind_speed"])
        inv_risks.append(ir)
        vents.append(vi)
    df["inversion_risk_factor"] = inv_risks
    df["ventilation_index"] = vents

    # Lags & rolling windows for PM2.5
    df["pm25_lag_1"] = df["target_pm25"].shift(1)
    df["pm25_lag_2"] = df["target_pm25"].shift(2)
    df["pm25_lag_3"] = df["target_pm25"].shift(3)
    df["pm25_lag_24"] = df["target_pm25"].shift(24).fillna(df["target_pm25"].shift(1))

    df["pm25_roll_mean_6h"] = df["target_pm25"].rolling(window=6, min_periods=1).mean()
    df["pm25_roll_mean_24h"] = df["target_pm25"].rolling(window=24, min_periods=1).mean()

    # Backfill earliest rows
    df["pm25_lag_1"] = df["pm25_lag_1"].bfill()
    df["pm25_lag_2"] = df["pm25_lag_2"].bfill()
    df["pm25_lag_3"] = df["pm25_lag_3"].bfill()
    df["pm25_lag_24"] = df["pm25_lag_24"].bfill()
    df["pm25_roll_mean_6h"] = df["pm25_roll_mean_6h"].bfill()
    df["pm25_roll_mean_24h"] = df["pm25_roll_mean_24h"].bfill()

    return df


class AirAwareXGBoostModel:
    """Manages training, persistence, and forecasting of the native AirAware XGBoost Booster."""

    def __init__(self):
        self.booster: Optional[xgb.Booster] = None
        self.metadata: Dict[str, Any] = {}
        self.feature_columns = FEATURE_COLUMNS
        self.is_trained = False
        self._load_existing_model()

    def _load_existing_model(self):
        """Loads saved model and metadata from disk if available."""
        if os.path.exists(MODEL_FILE) and os.path.exists(METADATA_FILE):
            try:
                self.booster = xgb.Booster()
                self.booster.load_model(MODEL_FILE)
                with open(METADATA_FILE, "r") as f:
                    self.metadata = json.load(f)
                self.is_trained = True
                logger.info("Successfully loaded pre-trained native XGBoost Booster.")
            except Exception as e:
                logger.warning(f"Could not load pre-trained model: {e}")
                self.is_trained = False

    def train(self, df_raw: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Trains native XGBoost Booster on historical multi-city dataset.
        Evaluates on test split and saves model artifacts.
        """
        if df_raw is None:
            df_raw = load_or_generate_historical_cpcb_dataset()

        logger.info("Building feature matrix for training...")
        target_cities = ["Delhi", "Amritsar", "Ludhiana", "Gurugram", "Lucknow", "Patna"]
        df_subset = df_raw[df_raw["City"].isin(target_cities)].copy()

        feature_dfs = []
        for city in target_cities:
            c_df = df_subset[df_subset["City"] == city].copy()
            c_feats = create_feature_matrix(c_df)
            feature_dfs.append(c_feats)

        all_feats = pd.concat(feature_dfs, ignore_index=True)

        # Train/Test Split by date (Train: < 2025, Test: >= 2025)
        all_feats["Date_dt"] = pd.to_datetime(all_feats["Date"])
        train_mask = all_feats["Date_dt"] < pd.to_datetime("2025-01-01")
        test_mask = ~train_mask

        X_train = all_feats.loc[train_mask, self.feature_columns]
        y_train = all_feats.loc[train_mask, "target_pm25"]
        X_test = all_feats.loc[test_mask, self.feature_columns]
        y_test = all_feats.loc[test_mask, "target_pm25"]

        logger.info(f"Training Native XGBoost on {len(X_train)} samples, testing on {len(X_test)} samples.")

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_columns)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=self.feature_columns)

        params = {
            "max_depth": 6,
            "learning_rate": 0.045,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "gamma": 1.0,
            "min_child_weight": 3,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "seed": 42
        }

        # Train model
        self.booster = xgb.train(
            params,
            dtrain,
            num_boost_round=250,
            evals=[(dtrain, "train"), (dtest, "test")],
            verbose_eval=False
        )

        # Evaluate predictions
        y_pred = self.booster.predict(dtest)
        rmse, mae, r2, mape = compute_metrics(y_test.values, y_pred)

        # Feature Importance by Gain
        score_gain = self.booster.get_score(importance_type="gain")
        total_gain = sum(score_gain.values()) if score_gain else 1.0
        feature_importance_dict = {
            feat: round(float(score_gain.get(feat, 0.0) / total_gain), 4)
            for feat in self.feature_columns
        }
        # Sort by importance descending
        feature_importance_dict = dict(sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True))

        residual_std = float(np.std(y_test.values - y_pred))

        self.metadata = {
            "metrics": {
                "rmse": round(rmse, 2),
                "mae": round(mae, 2),
                "r2": round(r2, 4),
                "mape": round(mape, 2),
                "residual_std": round(residual_std, 2)
            },
            "feature_importance": feature_importance_dict,
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "features_used": self.feature_columns,
            "model_architecture": "Gradient Boosted Trees (XGBoost Booster)"
        }

        # Save artifacts
        self.booster.save_model(MODEL_FILE)
        with open(METADATA_FILE, "w") as f:
            json.dump(self.metadata, f, indent=2)

        self.is_trained = True
        logger.info(f"Model training complete! Test R²: {r2:.3f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")
        return self.metadata

    def forecast_72h(
        self,
        city_name: str,
        hourly_df: pd.DataFrame,
        fires_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Produces multi-horizon 72-hour forecast of PM2.5, CPCB AQI, and confidence bands.
        Combines Open-Meteo forecast atmospheric variables with XGBoost recursive predictions.
        """
        if not self.is_trained:
            logger.info("Model not trained yet, training now...")
            self.train()

        city_meta = NORTH_INDIAN_CITIES.get(city_name, NORTH_INDIAN_CITIES["Delhi"])
        fire_metrics = calculate_stubble_fire_index(city_meta["lat"], city_meta["lon"], fires_df)

        df = hourly_df.copy().sort_values("timestamp").reset_index(drop=True)
        residual_std = self.metadata.get("metrics", {}).get("residual_std", 18.5)

        # Ensure features are computed
        df = create_feature_matrix(df)

        forecast_rows = []
        recent_pm25_buffer = list(df["target_pm25"].tail(24).values)

        last_known_idx = max(0, len(df) - 72)
        future_horizon = df.iloc[last_known_idx:last_known_idx + 72].copy()

        step_ahead = 1
        for idx, row in future_horizon.iterrows():
            pm_lag_1 = recent_pm25_buffer[-1]
            pm_lag_2 = recent_pm25_buffer[-2] if len(recent_pm25_buffer) >= 2 else pm_lag_1
            pm_lag_3 = recent_pm25_buffer[-3] if len(recent_pm25_buffer) >= 3 else pm_lag_2
            pm_lag_24 = recent_pm25_buffer[-24] if len(recent_pm25_buffer) >= 24 else pm_lag_1

            roll_6 = float(np.mean(recent_pm25_buffer[-6:]))
            roll_24 = float(np.mean(recent_pm25_buffer[-24:]))

            stubble_fire_count = fire_metrics.get("fire_count_350km", 0) * 12

            feature_dict = {
                "pm25_lag_1": pm_lag_1,
                "pm25_lag_2": pm_lag_2,
                "pm25_lag_3": pm_lag_3,
                "pm25_lag_24": pm_lag_24,
                "pm25_roll_mean_6h": roll_6,
                "pm25_roll_mean_24h": roll_24,
                "hour": row["hour"],
                "day_of_week": row["day_of_week"],
                "month": row["month"],
                "hour_sin": row["hour_sin"],
                "hour_cos": row["hour_cos"],
                "month_sin": row["month_sin"],
                "month_cos": row["month_cos"],
                "is_stubble_season": row["is_stubble_season"],
                "stubble_fire_count": stubble_fire_count,
                "temperature": row["temperature"],
                "humidity": row["humidity"],
                "wind_speed": row["wind_speed"],
                "inversion_risk_factor": row["inversion_risk_factor"],
                "ventilation_index": row["ventilation_index"]
            }

            feat_arr = np.array([[feature_dict[col] for col in self.feature_columns]], dtype=np.float32)
            dstep = xgb.DMatrix(feat_arr, feature_names=self.feature_columns)
            pred_pm25 = float(self.booster.predict(dstep)[0])
            pred_pm25 = max(15.0, pred_pm25)

            # Update rolling buffer
            recent_pm25_buffer.append(pred_pm25)
            if len(recent_pm25_buffer) > 48:
                recent_pm25_buffer.pop(0)

            # Uncertainty expands with forecast horizon
            horizon_uncertainty = residual_std * (1.0 + 0.012 * step_ahead)
            pm25_lower = max(10.0, pred_pm25 - 1.645 * horizon_uncertainty)  # 90% CI
            pm25_upper = pred_pm25 + 1.645 * horizon_uncertainty

            # Calculate CPCB AQI for central, lower, and upper
            pred_aqi, cat, dom, _ = calculate_cpcb_aqi({
                "PM2.5": pred_pm25,
                "PM10": pred_pm25 * 1.6,
                "NO2": row.get("no2", 40.0),
                "SO2": row.get("so2", 15.0),
                "CO": row.get("co", 1.2),
                "O3": row.get("o3", 35.0)
            })

            aqi_lower, _, _, _ = calculate_cpcb_aqi({"PM2.5": pm25_lower, "PM10": pm25_lower * 1.5})
            aqi_upper, _, _, _ = calculate_cpcb_aqi({"PM2.5": pm25_upper, "PM10": pm25_upper * 1.7})

            forecast_rows.append({
                "timestamp": row["timestamp"],
                "hour_ahead": step_ahead,
                "predicted_pm25": round(pred_pm25, 1),
                "pm25_lower": round(pm25_lower, 1),
                "pm25_upper": round(pm25_upper, 1),
                "predicted_aqi": pred_aqi,
                "aqi_lower": aqi_lower,
                "aqi_upper": aqi_upper,
                "aqi_category": cat,
                "dominant_pollutant": dom,
                "temperature": row["temperature"],
                "humidity": row["humidity"],
                "wind_speed": row["wind_speed"],
                "inversion_risk": row["inversion_risk_factor"]
            })
            step_ahead += 1

        return pd.DataFrame(forecast_rows)

    def simulate_what_if_scenario(
        self,
        base_pm25: float,
        fire_reduction_pct: float,
        emission_cut_pct: float,
        wind_speed_delta: float,
        temp_delta: float,
        is_stubble_season: bool = True
    ) -> Dict[str, Any]:
        """
        Interactive Policy & Weather Simulation Engine.
        Simulates the effect of interventions on predicted AQI.
        """
        if not self.is_trained:
            self.train()

        base_temp = 20.0 + temp_delta
        base_humidity = 70.0
        base_wind = max(0.5, 1.8 + wind_speed_delta)
        base_fires = int(1200 * (1.0 - (fire_reduction_pct / 100.0))) if is_stubble_season else 50

        # Adjust baseline PM2.5 with urban emission cuts
        adjusted_base_pm25 = base_pm25 * (1.0 - 0.40 * (emission_cut_pct / 100.0))

        inv_risk, vent = compute_atmospheric_indices(base_temp, base_humidity, base_wind)

        feature_dict = {
            "pm25_lag_1": adjusted_base_pm25,
            "pm25_lag_2": adjusted_base_pm25 * 1.05,
            "pm25_lag_3": adjusted_base_pm25 * 1.08,
            "pm25_lag_24": adjusted_base_pm25 * 1.1,
            "pm25_roll_mean_6h": adjusted_base_pm25,
            "pm25_roll_mean_24h": adjusted_base_pm25,
            "hour": 9,
            "day_of_week": 2,
            "month": 11,
            "hour_sin": np.sin(2 * np.pi * 9 / 24),
            "hour_cos": np.cos(2 * np.pi * 9 / 24),
            "month_sin": np.sin(2 * np.pi * 11 / 12),
            "month_cos": np.cos(2 * np.pi * 11 / 12),
            "is_stubble_season": 1 if is_stubble_season else 0,
            "stubble_fire_count": base_fires,
            "temperature": base_temp,
            "humidity": base_humidity,
            "wind_speed": base_wind,
            "inversion_risk_factor": inv_risk,
            "ventilation_index": vent
        }

        X_sim = pd.DataFrame([feature_dict])[self.feature_columns]
        dsim = xgb.DMatrix(X_sim, feature_names=self.feature_columns)
        simulated_pm25 = float(self.booster.predict(dsim)[0])
        simulated_pm25 = max(15.0, simulated_pm25)

        # Baseline conditions without intervention
        baseline_fires = 1200 if is_stubble_season else 50
        inv_base, vent_base = compute_atmospheric_indices(20.0, 70.0, 1.8)
        base_features = feature_dict.copy()
        base_features.update({
            "pm25_lag_1": base_pm25,
            "pm25_lag_2": base_pm25,
            "pm25_lag_3": base_pm25,
            "pm25_lag_24": base_pm25,
            "pm25_roll_mean_6h": base_pm25,
            "pm25_roll_mean_24h": base_pm25,
            "stubble_fire_count": baseline_fires,
            "temperature": 20.0,
            "wind_speed": 1.8,
            "inversion_risk_factor": inv_base,
            "ventilation_index": vent_base
        })
        X_base = pd.DataFrame([base_features])[self.feature_columns]
        dbase = xgb.DMatrix(X_base, feature_names=self.feature_columns)
        unmodified_pm25 = float(self.booster.predict(dbase)[0])

        sim_aqi, sim_cat, _, _ = calculate_cpcb_aqi({"PM2.5": simulated_pm25, "PM10": simulated_pm25 * 1.6})
        base_aqi, base_cat, _, _ = calculate_cpcb_aqi({"PM2.5": unmodified_pm25, "PM10": unmodified_pm25 * 1.6})

        aqi_delta = sim_aqi - base_aqi
        pm25_reduction_pct = ((unmodified_pm25 - simulated_pm25) / unmodified_pm25) * 100.0

        return {
            "baseline_pm25": round(unmodified_pm25, 1),
            "baseline_aqi": base_aqi,
            "baseline_category": base_cat,
            "simulated_pm25": round(simulated_pm25, 1),
            "simulated_aqi": sim_aqi,
            "simulated_category": sim_cat,
            "aqi_change": aqi_delta,
            "pm25_reduction_pct": round(pm25_reduction_pct, 1),
            "inversion_risk": inv_risk,
            "ventilation_index": vent
        }


# Singleton model instance
_model_instance: Optional[AirAwareXGBoostModel] = None

def get_model() -> AirAwareXGBoostModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = AirAwareXGBoostModel()
    return _model_instance


if __name__ == "__main__":
    print("Initializing AirAware Machine Learning Engine...")
    model = get_model()
    if not model.is_trained:
        print("Training model from scratch...")
        meta = model.train()
        print("Training Results:", meta["metrics"])
        print("Top 5 Features:", list(meta["feature_importance"].items())[:5])
    else:
        print("Model already trained! Metrics:", model.metadata.get("metrics"))

    # Test what-if simulation
    sim_result = model.simulate_what_if_scenario(
        base_pm25=220.0,
        fire_reduction_pct=60.0,
        emission_cut_pct=30.0,
        wind_speed_delta=1.5,
        temp_delta=0.0
    )
    print("What-If Simulation Result:")
    for k, v in sim_result.items():
        print(f"  {k}: {v}")

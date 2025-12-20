"""
AirAware Standalone Model Training & Benchmark Script
------------------------------------------------------
Trains the native XGBoost Booster on multi-city historical CPCB data,
evaluates out-of-sample metrics, logs feature importances, and updates model artifacts.
"""

import sys
import logging
from ml_engine import get_model
from data_pipeline import load_or_generate_historical_cpcb_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting AirAware Model Training & Validation Pipeline...")
    print("=" * 70)
    print("AIRAWARE: XGBoost Smog Forecasting Model Training")
    print("Target: 24-72 Hour PM2.5 & CPCB NAQI Multi-Horizon Prediction")
    print("=" * 70)

    # 1. Load / Verify Dataset
    print("\n[Step 1/3] Loading CPCB Multi-City Training Dataset...")
    df = load_or_generate_historical_cpcb_dataset()
    print(f"Loaded {len(df)} records across {df['City'].nunique()} North Indian cities.")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Features present: {list(df.columns)}")

    # 2. Train Model
    print("\n[Step 2/3] Training Native XGBoost Regressor with Agricultural Stubble & Inversion Features...")
    model = get_model()
    meta = model.train(df)

    # 3. Report Benchmarks
    print("\n[Step 3/3] Out-of-Sample Test Evaluation Metrics:")
    metrics = meta.get("metrics", {})
    print(f"  * Test R² Score        : {metrics.get('r2', 0):.4f}")
    print(f"  * Root Mean Sq Error   : {metrics.get('rmse', 0):.2f} µg/m³")
    print(f"  * Mean Absolute Error  : {metrics.get('mae', 0):.2f} µg/m³")
    print(f"  * Mean Absolute % Err  : {metrics.get('mape', 0):.2f}%")
    print(f"  * Residual Std Dev     : {metrics.get('residual_std', 0):.2f} µg/m³")

    print("\nTop 7 Model Feature Importances (Normalized Gain):")
    for feat, imp in list(meta.get("feature_importance", {}).items())[:7]:
        bar = "█" * int(imp * 40)
        print(f"  {feat:22s} | {imp:.4f} {bar}")

    print("\n" + "=" * 70)
    print("Model successfully trained, evaluated, and saved to models/ directory.")
    print("Ready for Streamlit dashboard integration!")
    print("=" * 70)

if __name__ == "__main__":
    main()

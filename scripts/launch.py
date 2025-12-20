"""
smog-watch Unified Services Launcher
-------------------------------------
Starts the FastAPI REST API (port 8000) and the Streamlit Dashboard (port 8501)
concurrently or independently.

Usage:
    python scripts/launch.py          # Starts both API and Dashboard
    python scripts/launch.py --api    # Starts only FastAPI backend
    python scripts/launch.py --ui     # Starts only Streamlit dashboard
"""

import sys
import time
import subprocess
import argparse


def start_both():
    print("=" * 70)
    print("smog-watch: Starting Unified Services")
    print("=" * 70)
    print("1. Launching FastAPI REST Backend at http://localhost:8000 ...")
    api_proc = subprocess.Popen([
        sys.executable, "-m", "uvicorn", "src.smogwatch.api.server:app",
        "--host", "0.0.0.0", "--port", "8000"
    ])

    time.sleep(2)

    print("2. Launching Streamlit Dashboard at http://localhost:8501 ...")
    ui_proc = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", "src/smogwatch/dashboard/ui.py",
        "--server.port", "8501"
    ])

    print("\n" + "=" * 70)
    print("Both services are running.")
    print("  * Streamlit Dashboard : http://localhost:8501")
    print("  * REST API + Swagger  : http://localhost:8000/docs")
    print("  * ReDoc Reference     : http://localhost:8000/redoc")
    print("=" * 70)
    print("Press Ctrl+C to stop all services.\n")

    try:
        api_proc.wait()
        ui_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down smog-watch services...")
        api_proc.terminate()
        ui_proc.terminate()
        print("Done.")


def main():
    parser = argparse.ArgumentParser(description="smog-watch Services Launcher")
    parser.add_argument("--api", action="store_true", help="Launch only FastAPI backend")
    parser.add_argument("--ui",  action="store_true", help="Launch only Streamlit dashboard")
    args = parser.parse_args()

    if args.api:
        print("Launching FastAPI on http://localhost:8000 ...")
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "src.smogwatch.api.server:app", "--host", "0.0.0.0", "--port", "8000"
        ])
    elif args.ui:
        print("Launching Streamlit on http://localhost:8501 ...")
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "src/smogwatch/dashboard/ui.py", "--server.port", "8501"
        ])
    else:
        start_both()


if __name__ == "__main__":
    main()

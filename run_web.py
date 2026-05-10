import subprocess
import sys
import os

def run_web():
    print("--- Launching IPMM LUT Web Server ---")
    
    # 1. Start Backend (FastAPI)
    # We use uvicorn to run the app in web/backend/main.py
    backend_dir = os.path.join(os.getcwd(), "web", "backend")
    sys.path.append(os.getcwd()) # Add root to path for core imports
    
    print("Starting Backend on http://localhost:8000...")
    try:
        import uvicorn
        uvicorn.run("web.backend.main:app", host="0.0.0.0", port=8000, reload=True)
    except ImportError:
        print("Error: 'uvicorn' or 'fastapi' not installed.")
        print("Please run: pip install fastapi uvicorn")
    except Exception as e:
        print(f"Failed to start server: {e}")

if __name__ == "__main__":
    run_web()

# IPMSM Motor LUT Tool

An interactive tool for exploring optimal operation trajectories (MTPA, Field Weakening) and generating control Look-Up Tables (LUT) for IPMSM (Interior Permanent Magnet Synchronous Motors).

This project provides two interfaces: a **Desktop GUI** and a **Web Dashboard**.

## 1. Setup

Python 3.9+ is required. Choose one of the following installation methods.

### Option A: Conda (Recommended)
Install all dependencies using the provided `environment.yml` file.
`ash
conda env create -f environment.yml
conda activate ipmm_lut
`

### Option B: pip & venv
Install dependencies using the `requirements.txt` file.
`ash
# Create and activate virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
`


## 2. Desktop Version (PyQt5 GUI)

The Desktop version offers a high-performance GUI for offline physics calculations and interactive visualization.

### How to Run
`ash
python run_desktop.py
`
- **Key Features**: Real-time parameter tuning, high-resolution graph rendering, and LUT data export.


## 3. Web Version (Dashboard)

The Web version consists of a FastAPI backend and a React frontend, providing a modern dashboard interface.

### 3.1 Backend Server
The backend handles the motor physics calculation API.
`ash
python run_web.py
`
- Server runs on `http://localhost:8000` by default.

### 3.2 Frontend UI
The frontend provides a responsive dashboard. (Node.js required)
`ash
cd web/frontend
npm install   # Run once
npm run dev   # Start development server
`
- Open the local URL (typically `http://localhost:5173`) in your browser.


## 4. Documentation

- **[Algorithm Guide (ALGORITHM.md)](docs/ALGORITHM.md)**: Detailed explanation of motor control formulas and optimization techniques.

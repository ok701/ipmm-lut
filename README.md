# IPMSM Motor LUT Tool

Interactive tool for exploring optimal trajectories (MTPA, Field Weakening) and generating 2D LUTs for IPMSM.

## Setup

```bash
# Via Conda
conda env create -f environment.yml && conda activate ipmm_lut

# Via pip
pip install -r requirements.txt
```

## Desktop Version
```bash
python run_desktop.py
```

## Web Version
```bash
# 1. Start Backend
python run_web.py

# 2. Start Frontend
cd web/frontend && npm install && npm run dev
```

## Documentation
- [Algorithm Guide](docs/ALGORITHM.md)
- [Notebooks](notebooks/)

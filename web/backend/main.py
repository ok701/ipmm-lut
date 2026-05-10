from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import numpy as np
import uuid
import time

# Shared Core Logic
from core.motor_model import (
    Te, part1_lambda_max_ff, build_part3_LUT, 
    solve_min_current_for_T_lam, solve_zero_torque_point_for_lam
)
from scipy.interpolate import RegularGridInterpolator

app = FastAPI(title="IPMM LUT API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tasks storage
tasks = {}

class MotorParams(BaseModel):
    pole_pairs: int
    Ld: float
    Lq: float
    psi_f: float
    Imax: float
    alpha: float
    rpm_max: float
    n_grid: int
    Vdc: float

@app.post("/v1/calculate")
async def calculate_lut(params: MotorParams, background_tasks: BackgroundTasks):
    # Simple cleanup: keep only the last 5 tasks to save memory
    if len(tasks) > 5:
        oldest_tasks = sorted(tasks.keys(), key=lambda x: time.time())[:len(tasks)-5]
        for t_id in oldest_tasks:
            tasks.pop(t_id, None)

    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "processing", "progress": 0, "result": None, "created_at": time.time()}
    background_tasks.add_task(run_full_calculation, task_id, params.dict())
    return {"task_id": task_id}

@app.get("/v1/status/{task_id}")
async def get_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]

def run_full_calculation(task_id: str, p: Dict[str, Any]):
    try:
        n_grid = p["n_grid"]
        
        Vmax = p["alpha"] * p["Vdc"]
        omega_max = p["rpm_max"] * 2 * np.pi / 60.0
        lam_min_req = Vmax / (max(omega_max, 1.0) * p["pole_pairs"])

        # Exact formulas from desktop worker.py
        lam_upper = np.hypot(p["psi_f"] + p["Ld"] * p["Imax"],
                             p["Lq"] * p["Imax"]) * 1.05
        lam_lower = max(lam_min_req * 0.98, lam_upper * 0.01)
        
        lam_grid = np.linspace(lam_lower, lam_upper, n_grid)
        
        # Build 1D Tmax LUT
        Tmax_LUT, Id_at_Tmax, Iq_at_Tmax = build_part3_LUT(lam_grid, p)
        tasks[task_id]["progress"] = 50
        
        # Build 2D Id/Iq LUT (Match Phase 2 from worker.py)
        n_ratio = n_grid 
        Tratio_grid = np.linspace(0.0, 0.999, n_ratio)
        Id_2D = np.full((len(lam_grid), len(Tratio_grid)), np.nan)
        Iq_2D = np.full((len(lam_grid), len(Tratio_grid)), np.nan)



        for i, lam_max in enumerate(lam_grid):
            Tmax_i = Tmax_LUT[i]
            id0_w, iq0_w = Id_at_Tmax[i], Iq_at_Tmax[i]
            
            for j, ratio in enumerate(Tratio_grid):
                Tref_ij = ratio * Tmax_i
                
                if ratio == 0.0:
                    id0, iq0 = solve_zero_torque_point_for_lam(float(lam_max), p)
                    Id_2D[i, j], Iq_2D[i, j] = id0, iq0
                else:
                    x0_used = [id0_w, iq0_w]
                    # Same logic as worker.py for stability
                    if ratio < 0.01:
                        z_id, z_iq = solve_zero_torque_point_for_lam(float(lam_max), p)
                        if not np.isnan(z_id): x0_used = [z_id, z_iq]

                    sol_ij, _ = solve_min_current_for_T_lam(Tref_ij, float(lam_max), p, x0=x0_used)
                    
                    if sol_ij is None and ratio < 0.01:
                         sol_ij, _ = solve_min_current_for_T_lam(Tref_ij, float(lam_max), p, x0=[id0_w, iq0_w])

                    if sol_ij is not None:
                        Id_2D[i, j], Iq_2D[i, j] = sol_ij[0], sol_ij[1]
            
            # Update progress
            tasks[task_id]["progress"] = 50 + int((i / len(lam_grid)) * 45)

        tasks[task_id] = {
            "status": "completed",
            "progress": 100,
            "result": {
                "lam_grid": lam_grid.tolist(),
                "Tratio_grid": Tratio_grid.tolist(),
                "Tmax_LUT": Tmax_LUT.tolist(),
                "Id_at_Tmax": Id_at_Tmax.tolist(),
                "Iq_at_Tmax": Iq_at_Tmax.tolist(),
                "Id_2D": Id_2D.tolist(),
                "Iq_2D": Iq_2D.tolist()
            }
        }
    except Exception as e:
        tasks[task_id] = {"status": "error", "message": str(e)}



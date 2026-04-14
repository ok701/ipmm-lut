"""
IPMM LUT Tool – FastAPI Web Server
"""
from __future__ import annotations

import asyncio
from typing import Optional
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator

app = FastAPI(title="IPMM LUT Tool")


# ─────────────────────────────────────────────
# Motor model helpers  (ported from lut.py)
# ─────────────────────────────────────────────

def Te(id_: float, iq: float, p: dict) -> float:
    return 1.5 * p["pole_pairs"] * (p["psi_f"] * iq + (p["Ld"] - p["Lq"]) * id_ * iq)


def lam_d(id_: float, p: dict) -> float:
    return p["psi_f"] + p["Ld"] * id_


def lam_q(iq: float, p: dict) -> float:
    return p["Lq"] * iq


def lam_mag(id_: float, iq: float, p: dict) -> float:
    return float(np.hypot(lam_d(id_, p), lam_q(iq, p)))


def part1_lambda_max_ff(rpm: float, Vdc: float, p: dict) -> float:
    omega_mech = rpm * 2 * np.pi / 60.0
    omega_e = p["pole_pairs"] * omega_mech
    Vmax = p["alpha"] * Vdc
    return float(Vmax / max(abs(omega_e), 1e-9))


def solve_Tmax_for_lammax(lam_max: float, p: dict):
    Imax = p["Imax"]
    bounds = [(-Imax, 0.0), (0.0, Imax)]

    def obj(x):
        return -Te(x[0], x[1], p)

    cons = [
        {"type": "ineq", "fun": lambda x: Imax**2 - (x[0]**2 + x[1]**2)},
        {"type": "ineq", "fun": lambda x: lam_max - lam_mag(x[0], x[1], p)},
    ]

    inits = [np.array([0.0, min(Imax, 5.0)])]
    for frac in [0.2, 0.5, 0.8, 1.0]:
        iq0 = frac * Imax
        id0 = -np.sqrt(max(Imax**2 - iq0**2, 0.0))
        inits.append(np.array([id0, iq0]))
    inits.append(np.array([-0.5 * Imax, 0.5 * Imax]))

    best_T = -np.inf
    best_x = (np.nan, np.nan)

    for x0 in inits:
        res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 800, "ftol": 1e-12, "disp": False})
        if not res.success:
            continue
        id_, iq = res.x
        if (id_**2 + iq**2) > Imax**2 + 1e-6:
            continue
        if lam_mag(id_, iq, p) > lam_max + 1e-6:
            continue
        val = Te(id_, iq, p)
        if val > best_T:
            best_T = float(val)
            best_x = (float(id_), float(iq))

    if not np.isfinite(best_T):
        return np.nan, (np.nan, np.nan)
    return best_T, best_x


def build_part3_LUT(lam_max_grid: np.ndarray, p: dict):
    Tmax_LUT   = np.full_like(lam_max_grid, np.nan, dtype=float)
    Id_at_Tmax = np.full_like(lam_max_grid, np.nan, dtype=float)
    Iq_at_Tmax = np.full_like(lam_max_grid, np.nan, dtype=float)
    for k, lm in enumerate(lam_max_grid):
        Tmax, (id_opt, iq_opt) = solve_Tmax_for_lammax(float(lm), p)
        Tmax_LUT[k]   = Tmax
        Id_at_Tmax[k] = id_opt
        Iq_at_Tmax[k] = iq_opt
    return Tmax_LUT, Id_at_Tmax, Iq_at_Tmax


def solve_min_current_for_T_lam(Tref: float, lam_ref: float, p: dict, x0=None):
    Imax = p["Imax"]
    bounds = [(-Imax, 0.0), (0.0, Imax)]

    def cost(x):
        return x[0]**2 + x[1]**2

    cons = [
        {"type": "eq",  "fun": lambda x: Te(x[0], x[1], p) - Tref},
        {"type": "ineq","fun": lambda x: lam_ref - lam_mag(x[0], x[1], p)},
        {"type": "ineq","fun": lambda x: Imax**2 - (x[0]**2 + x[1]**2)},
    ]

    inits = []
    if x0 is not None:
        inits.append(np.array(x0, dtype=float))
    denom = 1.5 * p["pole_pairs"] * max(p["psi_f"], 1e-9)
    iq0 = Tref / denom
    if 0.0 <= iq0 <= Imax:
        inits.append(np.array([0.0, iq0]))
    for frac in [0.2, 0.5, 0.8, 1.0]:
        iq_g = frac * Imax
        id_g = -np.sqrt(max(Imax**2 - iq_g**2, 0.0))
        inits.append(np.array([id_g, iq_g]))
    inits.append(np.array([-0.2 * Imax, 0.2 * Imax]))

    best_cost = np.inf
    best_x = None
    for x0_try in inits:
        res = minimize(cost, x0_try, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 1200, "ftol": 1e-12, "disp": False})
        if not res.success:
            continue
        id_, iq = res.x
        if abs(Te(id_, iq, p) - Tref) > 1e-3:
            continue
        if lam_mag(id_, iq, p) > lam_ref + 1e-6:
            continue
        if (id_**2 + iq**2) > Imax**2 + 1e-6:
            continue
        c = cost(res.x)
        if c < best_cost:
            best_cost = c
            best_x = res.x.copy()
    return best_x


def compute_lut(p: dict):
    """Run the full LUT computation (blocking). Returns a dict of arrays."""
    N_lam = 60
    N_tref = 30
    lam_upper = np.hypot(p["psi_f"] + p["Ld"] * p["Imax"],
                         p["Lq"] * p["Imax"]) * 1.05
    lam_lower = lam_upper * 0.03
    lam_grid = np.linspace(lam_lower, lam_upper, N_lam)
    Tratio_grid = np.linspace(0.0, 0.999, N_tref)

    Tmax_LUT, Id_at_Tmax, Iq_at_Tmax = build_part3_LUT(lam_grid, p)

    Id_LUT_2D = np.full((N_lam, N_tref), np.nan)
    Iq_LUT_2D = np.full((N_lam, N_tref), np.nan)

    for i, lam_max in enumerate(lam_grid):
        Tmax_i = float(np.interp(lam_max, lam_grid, Tmax_LUT))
        id0_w  = float(np.interp(lam_max, lam_grid, Id_at_Tmax))
        iq0_w  = float(np.interp(lam_max, lam_grid, Iq_at_Tmax))
        for j, ratio in enumerate(Tratio_grid):
            Tref_ij = ratio * Tmax_i
            if ratio < 0.01:
                continue
            sol_ij = solve_min_current_for_T_lam(
                Tref_ij, lam_max, p, x0=[id0_w, iq0_w])
            if sol_ij is not None:
                Id_LUT_2D[i, j] = sol_ij[0]
                Iq_LUT_2D[i, j] = sol_ij[1]

    return {
        "lam_grid":    lam_grid,
        "lam_upper":   float(lam_upper),
        "Tratio_grid": Tratio_grid,
        "Tmax_LUT":    Tmax_LUT,
        "Id_at_Tmax":  Id_at_Tmax,
        "Iq_at_Tmax":  Iq_at_Tmax,
        "Id_LUT_2D":   Id_LUT_2D,
        "Iq_LUT_2D":   Iq_LUT_2D,
    }


def _nan_to_none(arr: np.ndarray) -> list:
    """Convert numpy array to nested list, replacing nan with None for JSON."""
    if arr.ndim == 1:
        return [None if np.isnan(v) else float(v) for v in arr]
    return [[None if np.isnan(v) else float(v) for v in row] for row in arr]


def _build_sim_bg(p: dict):
    """Compute background meshgrid data for simulation plot."""
    Imax = p["Imax"]
    id_vec = np.linspace(-Imax * 1.1, Imax * 0.1, 60)
    iq_vec = np.linspace(0, Imax * 1.1, 60)
    ID_bg, IQ_bg = np.meshgrid(id_vec, iq_vec)
    I_bg  = np.sqrt(ID_bg**2 + IQ_bg**2)
    LAM_d = p["psi_f"] + p["Ld"] * ID_bg
    LAM_q = p["Lq"] * IQ_bg
    LAM_bg = np.sqrt(LAM_d**2 + LAM_q**2)
    TE_bg  = 1.5 * p["pole_pairs"] * (
        p["psi_f"] * IQ_bg
        + (p["Ld"] - p["Lq"]) * ID_bg * IQ_bg)

    # MTPA trajectory
    delta = p["Lq"] - p["Ld"]
    I_sweep = np.linspace(0.0, Imax, 80)
    if abs(delta) < 1e-12:
        id_mtpa = np.zeros_like(I_sweep)
        iq_mtpa = I_sweep.copy()
    else:
        k = p["psi_f"] / delta
        id_mtpa = (k - np.sqrt(k**2 + 8.0 * I_sweep**2)) / 4.0
        iq_mtpa = np.sqrt(np.maximum(I_sweep**2 - id_mtpa**2, 0.0))
    v = (np.isfinite(id_mtpa) & np.isfinite(iq_mtpa)
         & (id_mtpa <= 1e-9) & (iq_mtpa >= -1e-9))

    return {
        "id_vec":    id_vec.tolist(),
        "iq_vec":    iq_vec.tolist(),
        "I_bg":      _nan_to_none(I_bg),
        "LAM_bg":    _nan_to_none(LAM_bg),
        "TE_bg":     _nan_to_none(TE_bg),
        "id_mtpa":   _nan_to_none(id_mtpa[v]),
        "iq_mtpa":   _nan_to_none(iq_mtpa[v]),
    }


# ─────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────

class MotorParams(BaseModel):
    pole_pairs: int   = 4
    Ld:         float = 0.004
    Lq:         float = 0.008
    psi_f:      float = 0.01
    Imax:       float = 20.0
    alpha:      float = 1 / 3
    Vdc:        float = 48.0


class SimRequest(BaseModel):
    # motor params
    pole_pairs:  int
    Ld:          float
    Lq:          float
    psi_f:       float
    Imax:        float
    alpha:       float
    Vdc:         float
    # lut data (flat, row-major)
    lam_grid:    list[float]
    Tratio_grid: list[float]
    Tmax_LUT:    list[Optional[float]]
    Id_at_Tmax:  list[Optional[float]]
    Iq_at_Tmax:  list[Optional[float]]
    Id_LUT_2D:   list[list[Optional[float]]]
    Iq_LUT_2D:   list[list[Optional[float]]]
    # sim controls
    rpm:         float = 3000.0
    tref:        float = 2.0
    const_torque: bool = True


# ─────────────────────────────────────────────
# API routes
# ─────────────────────────────────────────────

@app.post("/api/compute-lut")
async def api_compute_lut(req: MotorParams):
    p = req.model_dump()
    Vdc = p.pop("Vdc")

    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(None, compute_lut, p)

    bg = _build_sim_bg(p)

    # Compute Tmax and Power as a function of RPM (for the performance chart)
    rpm_sweep = np.linspace(100.0, 12000.0, 400)
    tmax_vs_rpm: list = []
    power_vs_rpm_kw: list = []
    lam_grid_arr  = data["lam_grid"]
    lam_lower_val = float(lam_grid_arr[0])
    lam_upper_val = float(lam_grid_arr[-1])

    for rpm_i in rpm_sweep:
        lam_i = part1_lambda_max_ff(float(rpm_i), Vdc, p)
        # Clamp into the grid range:
        #  - above upper → current-limited (rated) regime: use lam_upper
        #  - below lower → deeply in flux-weakening beyond solver range: interpolate to 0
        if lam_i >= lam_upper_val:
            tmax_i = float(np.interp(lam_upper_val, lam_grid_arr, data["Tmax_LUT"]))
        elif lam_i <= lam_lower_val:
            # smoothly interpolate to 0 by ratio
            ratio = max(lam_i / lam_lower_val, 0.0)
            tmax_i = float(np.interp(lam_lower_val, lam_grid_arr, data["Tmax_LUT"])) * ratio
        else:
            tmax_i = float(np.interp(lam_i, lam_grid_arr, data["Tmax_LUT"]))
        omega_i = float(rpm_i) * 2.0 * np.pi / 60.0
        pow_kw  = (tmax_i * omega_i) / 1000.0
        tmax_vs_rpm.append(round(tmax_i, 4))
        power_vs_rpm_kw.append(round(pow_kw, 3))

    return {
        "lam_grid":       data["lam_grid"].tolist(),
        "lam_upper":      float(lam_upper_val),
        "Tratio_grid":    data["Tratio_grid"].tolist(),
        "Tmax_LUT":       _nan_to_none(data["Tmax_LUT"]),
        "Id_at_Tmax":     _nan_to_none(data["Id_at_Tmax"]),
        "Iq_at_Tmax":     _nan_to_none(data["Iq_at_Tmax"]),
        "Id_LUT_2D":      _nan_to_none(data["Id_LUT_2D"]),
        "Iq_LUT_2D":      _nan_to_none(data["Iq_LUT_2D"]),
        "sim_bg":         bg,
        "rpm_sweep":      rpm_sweep.tolist(),
        "tmax_vs_rpm":    tmax_vs_rpm,
        "power_vs_rpm_kw": power_vs_rpm_kw,
    }


@app.post("/api/simulate")
async def api_simulate(req: SimRequest):
    p = {
        "pole_pairs": req.pole_pairs,
        "Ld": req.Ld,
        "Lq": req.Lq,
        "psi_f": req.psi_f,
        "Imax": req.Imax,
        "alpha": req.alpha,
    }

    lam_grid    = np.array(req.lam_grid)
    Tmax_LUT    = np.array([v if v is not None else np.nan for v in req.Tmax_LUT])
    Id_at_Tmax  = np.array([v if v is not None else np.nan for v in req.Id_at_Tmax])
    Iq_at_Tmax  = np.array([v if v is not None else np.nan for v in req.Iq_at_Tmax])
    Tratio_grid = np.array(req.Tratio_grid)
    Id_LUT_2D   = np.array([[v if v is not None else np.nan for v in row]
                             for row in req.Id_LUT_2D])
    Iq_LUT_2D   = np.array([[v if v is not None else np.nan for v in row]
                             for row in req.Iq_LUT_2D])

    lam_ref     = part1_lambda_max_ff(req.rpm, req.Vdc, p)
    Tmax_at_lam = float(np.interp(lam_ref, lam_grid, Tmax_LUT))

    if req.const_torque:
        Tref    = float(np.clip(req.tref, 0.0, Tmax_at_lam * 0.999))
        T_ratio = Tref / max(Tmax_at_lam, 1e-6)
        interp_id = RegularGridInterpolator(
            (lam_grid, Tratio_grid), Id_LUT_2D,
            method='linear', bounds_error=False, fill_value=None)
        interp_iq = RegularGridInterpolator(
            (lam_grid, Tratio_grid), Iq_LUT_2D,
            method='linear', bounds_error=False, fill_value=None)
        pt    = np.array([[lam_ref, T_ratio]])
        id_op = float(interp_id(pt).item())
        iq_op = float(interp_iq(pt).item())
        Tref_actual = Tref
        Tref_cmd    = req.tref
    else:
        Tref_actual = Tmax_at_lam
        Tref_cmd    = 0.0
        v = np.isfinite(Id_at_Tmax) & np.isfinite(Iq_at_Tmax)
        id_op = float(np.interp(lam_ref, lam_grid[v], Id_at_Tmax[v]))
        iq_op = float(np.interp(lam_ref, lam_grid[v], Iq_at_Tmax[v]))

    te_op = Te(id_op, iq_op, p)

    return {
        "id_op":        id_op,
        "iq_op":        iq_op,
        "te_op":        te_op,
        "lam_ref":      lam_ref,
        "tmax_at_lam":  Tmax_at_lam,
        "tref_actual":  Tref_actual,
        "tref_cmd":     Tref_cmd,
    }


# ─────────────────────────────────────────────
# Static files
# ─────────────────────────────────────────────

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")

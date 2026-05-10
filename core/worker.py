import numpy as np
from scipy.interpolate import RegularGridInterpolator
from PyQt5.QtCore import QThread, pyqtSignal
from core.motor_model import (
    solve_Tmax_for_lammax, solve_min_current_for_T_lam,
    solve_zero_torque_point_for_lam
)

class LutWorker(QThread):
    """Background worker to compute the Motor LUT optimization."""
    finished = pyqtSignal(object)
    progress = pyqtSignal(int)  # 0-100

    def __init__(self, p_, parent=None):
        super().__init__(parent)
        self.p_ = dict(p_)

    def run(self):
        p = self.p_
        N_lam = int(p.get("n_grid", 50))
        N_tref = int(p.get("n_grid", 50))
        total_steps = N_lam + N_lam * N_tref  # phase1: N_lam, phase2: N_lam*N_tref
        done = 0

        Vmax = p["alpha"] * p["Vdc"]
        omega_max = p["rpm_max"] * 2 * np.pi / 60.0
        lam_min_req = Vmax / (max(omega_max, 1.0) * p["pole_pairs"])

        lam_upper = np.hypot(p["psi_f"] + p["Ld"] * p["Imax"],
                             p["Lq"] * p["Imax"]) * 1.05
        # Set lam_lower to match rpm_max, with a safety floor
        lam_lower = max(lam_min_req * 0.98, lam_upper * 0.01)
        
        lam_grid = np.linspace(lam_lower, lam_upper, N_lam)
        Tratio_grid = np.linspace(0.0, 0.999, N_tref)

        # Phase 1: Tmax sweep
        Tmax_LUT   = np.full_like(lam_grid, np.nan, dtype=float)
        Id_at_Tmax = np.full_like(lam_grid, np.nan, dtype=float)
        Iq_at_Tmax = np.full_like(lam_grid, np.nan, dtype=float)
        for k, lm in enumerate(lam_grid):
            Tmax, (id_opt, iq_opt) = solve_Tmax_for_lammax(float(lm), p)
            Tmax_LUT[k]   = Tmax
            Id_at_Tmax[k] = id_opt
            Iq_at_Tmax[k] = iq_opt
            done += 1
            self.progress.emit(round(done / total_steps * 100))

        # Phase 2: 2D LUT
        Id_LUT_2D = np.full((N_lam, N_tref), np.nan)
        Iq_LUT_2D = np.full((N_lam, N_tref), np.nan)

        for i, lam_max in enumerate(lam_grid):
            Tmax_i = float(np.interp(lam_max, lam_grid, Tmax_LUT))
            id0_w  = float(np.interp(lam_max, lam_grid, Id_at_Tmax))
            iq0_w  = float(np.interp(lam_max, lam_grid, Iq_at_Tmax))
            for j, ratio in enumerate(Tratio_grid):
                Tref_ij = ratio * Tmax_i
                
                if ratio == 0.0:
                    id0, iq0 = solve_zero_torque_point_for_lam(lam_max, p)
                    Id_LUT_2D[i, j] = id0
                    Iq_LUT_2D[i, j] = iq0
                else:
                    x0_used = [id0_w, iq0_w]
                    z_id = np.nan
                    if ratio < 0.01:
                        z_id, z_iq = solve_zero_torque_point_for_lam(lam_max, p)
                        if not np.isnan(z_id):
                            x0_used = [z_id, z_iq]

                    sol_ij, _ = solve_min_current_for_T_lam(
                        Tref_ij, lam_max, p, x0=x0_used)
                        
                    # Fallback if x0=zero_torque failed
                    if sol_ij is None and ratio < 0.01 and not np.isnan(z_id):
                        sol_ij, _ = solve_min_current_for_T_lam(
                            Tref_ij, lam_max, p, x0=[id0_w, iq0_w])

                    if sol_ij is not None:
                        Id_LUT_2D[i, j] = sol_ij[0]
                        Iq_LUT_2D[i, j] = sol_ij[1]
                        
                done += 1
                if (done % 5) == 0:
                    self.progress.emit(round(done / total_steps * 100))

        self.progress.emit(100)

        interp_id = RegularGridInterpolator(
            (lam_grid, Tratio_grid), Id_LUT_2D,
            method='linear', bounds_error=False, fill_value=None)
        interp_iq = RegularGridInterpolator(
            (lam_grid, Tratio_grid), Iq_LUT_2D,
            method='linear', bounds_error=False, fill_value=None)

        self.finished.emit({
            "lam_grid":    lam_grid,
            "Tratio_grid": Tratio_grid,
            "Tmax_LUT":    Tmax_LUT,
            "Id_at_Tmax":  Id_at_Tmax,
            "Iq_at_Tmax":  Iq_at_Tmax,
            "Id_LUT_2D":   Id_LUT_2D,
            "Iq_LUT_2D":   Iq_LUT_2D,
            "interp_id":   interp_id,
            "interp_iq":   interp_iq,
        })

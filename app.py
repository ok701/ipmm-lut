import sys
import warnings
warnings.filterwarnings('ignore', category=UserWarning,
                        message='.*not compatible with tight_layout.*')
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator

import matplotlib
matplotlib.use("Qt5Agg")
matplotlib.rcParams['figure.autolayout'] = False
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QTabWidget, QLabel, QLineEdit, QPushButton, QFormLayout,
    QSizePolicy, QGroupBox, QSlider, QFrame, QProgressBar, QCheckBox,
    QComboBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont


# =========================
# Motor model helpers
# =========================
def Te(id_, iq, p_):
    return 1.5 * p_["pole_pairs"] * (p_["psi_f"] * iq + (p_["Ld"] - p_["Lq"]) * id_ * iq)

def lam_d(id_, p_):
    return p_["psi_f"] + p_["Ld"] * id_

def lam_q(iq, p_):
    return p_["Lq"] * iq

def lam_mag(id_, iq, p_):
    return float(np.hypot(lam_d(id_, p_), lam_q(iq, p_)))

def part1_lambda_max_ff(rpm, Vdc, p_):
    omega_mech = rpm * 2 * np.pi / 60.0
    omega_e = p_["pole_pairs"] * omega_mech
    Vmax = p_["alpha"] * Vdc
    return float(Vmax / max(abs(omega_e), 1e-9))

def solve_Tmax_for_lammax(lam_max, p_):
    Imax = p_["Imax"]
    bounds = [(-Imax, 0.0), (0.0, Imax)]

    def obj(x):
        return -Te(x[0], x[1], p_)

    cons = [
        {"type": "ineq", "fun": lambda x: Imax**2 - (x[0]**2 + x[1]**2)},
        {"type": "ineq", "fun": lambda x: lam_max - lam_mag(x[0], x[1], p_)},
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
        if lam_mag(id_, iq, p_) > lam_max + 1e-6:
            continue
        val = Te(id_, iq, p_)
        if val > best_T:
            best_T = float(val)
            best_x = (float(id_), float(iq))

    if not np.isfinite(best_T):
        return np.nan, (np.nan, np.nan)
    return best_T, best_x

def build_part3_LUT(lam_max_grid, p_):
    Tmax_LUT   = np.full_like(lam_max_grid, np.nan, dtype=float)
    Id_at_Tmax = np.full_like(lam_max_grid, np.nan, dtype=float)
    Iq_at_Tmax = np.full_like(lam_max_grid, np.nan, dtype=float)
    for k, lm in enumerate(lam_max_grid):
        Tmax, (id_opt, iq_opt) = solve_Tmax_for_lammax(float(lm), p_)
        Tmax_LUT[k]   = Tmax
        Id_at_Tmax[k] = id_opt
        Iq_at_Tmax[k] = iq_opt
    return Tmax_LUT, Id_at_Tmax, Iq_at_Tmax

def solve_min_current_for_T_lam(Tref, lam_ref, p_, x0=None):
    Imax = p_["Imax"]
    bounds = [(-Imax, 0.0), (0.0, Imax)]

    def cost(x):
        return x[0]**2 + x[1]**2

    cons = [
        {"type": "eq",  "fun": lambda x: Te(x[0], x[1], p_) - Tref},
        {"type": "ineq","fun": lambda x: lam_ref - lam_mag(x[0], x[1], p_)},
        {"type": "ineq","fun": lambda x: Imax**2 - (x[0]**2 + x[1]**2)},
    ]

    inits = []
    if x0 is not None:
        inits.append(np.array(x0, dtype=float))
    denom = 1.5 * p_["pole_pairs"] * max(p_["psi_f"], 1e-9)
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
        if abs(Te(id_, iq, p_) - Tref) > 1e-3:
            continue
        if lam_mag(id_, iq, p_) > lam_ref + 1e-6:
            continue
        if (id_**2 + iq**2) > Imax**2 + 1e-6:
            continue
        c = cost(res.x)
        if c < best_cost:
            best_cost = c
            best_x = res.x.copy()
    return best_x, None


# =========================
# Background worker thread
# =========================
class LutWorker(QThread):
    finished = pyqtSignal(object)

    def __init__(self, p_, parent=None):
        super().__init__(parent)
        self.p_ = dict(p_)

    def run(self):
        p = self.p_
        N_lam = 20
        N_tref = 20
        lam_upper = np.hypot(p["psi_f"] + p["Ld"] * p["Imax"],
                             p["Lq"] * p["Imax"]) * 1.05
        # Use a low fixed floor so the trajectory covers most of the id-iq plane.
        # Physically unreachable cells will be NaN and rendered transparently.
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
                    Id_LUT_2D[i, j] = np.nan
                    Iq_LUT_2D[i, j] = np.nan
                    continue
                sol_ij, _ = solve_min_current_for_T_lam(
                    Tref_ij, lam_max, p, x0=[id0_w, iq0_w])
                if sol_ij is not None:
                    Id_LUT_2D[i, j] = sol_ij[0]
                    Iq_LUT_2D[i, j] = sol_ij[1]

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


# =========================
# Matplotlib canvas
# =========================
class MplCanvas(FigureCanvas):
    def __init__(self, fig):
        super().__init__(fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


# =========================
# Right-side tab contents
# =========================

class TmaxTab(QWidget):
    """최대토크 탭"""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # X-axis selection
        ctrl_layout = QHBoxLayout()
        ctrl_layout.setContentsMargins(10, 5, 10, 0)
        self.combo_x = QComboBox()
        self.combo_x.addItems(["Flux Linkage [Wb]", "Speed [rpm]"])
        self.combo_x.currentIndexChanged.connect(self._redraw)
        ctrl_layout.addWidget(QLabel("X-축 선택:"))
        ctrl_layout.addWidget(self.combo_x)
        ctrl_layout.addStretch()
        layout.addLayout(ctrl_layout)

        fig = Figure(constrained_layout=True)
        self.ax = fig.add_subplot(111)
        self.canvas = MplCanvas(fig)
        layout.addWidget(self.canvas)

        self.lam_grid = None
        self.Tmax_LUT = None
        self.p_ = None
        self.Vdc = None

    def update_plot(self, lam_grid, Tmax_LUT, p_=None, Vdc=None):
        self.lam_grid = lam_grid
        self.Tmax_LUT = Tmax_LUT
        self.p_ = p_
        self.Vdc = Vdc
        self._redraw()

    def _redraw(self):
        if self.lam_grid is None or self.Tmax_LUT is None:
            return

        self.ax.clear()
        
        mode = self.combo_x.currentText()
        if "Speed" in mode and self.p_ is not None and self.Vdc is not None:
            # Convert lam_max to RPM
            # lam_max = (alpha * Vdc) / (pp * omega_mech)
            # omega_mech = rpm * 2 * pi / 60
            # rpm = (alpha * Vdc * 60) / (lam_max * pp * 2 * pi)
            Vmax = self.p_["alpha"] * self.Vdc
            pp = self.p_["pole_pairs"]
            x_data = (Vmax * 60.0) / (np.maximum(self.lam_grid, 1e-9) * pp * 2.0 * np.pi)
            x_label = "Speed [rpm]"
            title = "Max Torque vs Speed"
            # Sort by speed for better plotting if necessary, but lam_grid is linear so x_data will be monotonic
            idx = np.argsort(x_data)
            self.ax.plot(x_data[idx], self.Tmax_LUT[idx], 'r-o', ms=4, lw=1.8)
        else:
            x_data = self.lam_grid
            x_label = "lam_max [Wb]"
            title = "Max Torque vs Flux Linkage Limit"
            self.ax.plot(x_data, self.Tmax_LUT, 'b-o', ms=4, lw=1.8)

        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel("Tmax [Nm]")
        self.ax.set_title(title)
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()


class LutTab(QWidget):
    """LUT 3D surface 탭"""
    def __init__(self, parent=None):
        super().__init__(parent)
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self._fig = Figure(figsize=(12, 5))
        self.canvas = MplCanvas(self._fig)
        layout.addWidget(self.canvas)
        self._init_3d_axes()

    def _init_3d_axes(self):
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        self._fig.clear()
        self.ax_id = self._fig.add_subplot(121, projection='3d')
        self.ax_iq = self._fig.add_subplot(122, projection='3d')

    def update_plot(self, lam_grid, Tratio_grid, Id_LUT_2D, Iq_LUT_2D):
        self._init_3d_axes()
        T, L = np.meshgrid(Tratio_grid, lam_grid)
        Id_masked = np.ma.masked_invalid(Id_LUT_2D)
        Iq_masked = np.ma.masked_invalid(Iq_LUT_2D)

        self.ax_id.plot_surface(T, L, Id_masked, cmap='coolwarm',
                                edgecolor='none', alpha=0.92)
        self.ax_id.set_title("Id LUT [A]", fontsize=9)
        self.ax_id.set_xlabel("T_ratio", fontsize=7, labelpad=2)
        self.ax_id.set_ylabel("lam_max [Wb]", fontsize=7, labelpad=2)
        self.ax_id.set_zlabel("Id [A]", fontsize=7, labelpad=2)
        self.ax_id.tick_params(labelsize=6)

        self.ax_iq.plot_surface(T, L, Iq_masked, cmap='viridis',
                                edgecolor='none', alpha=0.92)
        self.ax_iq.set_title("Iq LUT [A]", fontsize=9)
        self.ax_iq.set_xlabel("T_ratio", fontsize=7, labelpad=2)
        self.ax_iq.set_ylabel("lam_max [Wb]", fontsize=7, labelpad=2)
        self.ax_iq.set_zlabel("Iq [A]", fontsize=7, labelpad=2)
        self.ax_iq.tick_params(labelsize=6)
        self.canvas.draw()

    def update_trajectory(self, lam_grid, Id_at_Tmax, Iq_at_Tmax):
        """MTPA/MTPV mode: show MTPV surface extruded along T_ratio (3D)."""
        v = np.isfinite(Id_at_Tmax) & np.isfinite(Iq_at_Tmax)
        lam_v  = lam_grid[v]
        Id_v   = Id_at_Tmax[v]
        Iq_v   = Iq_at_Tmax[v]
        # Extrude 1D trajectory into a surface along a dummy T_ratio axis
        t_fake = np.linspace(0.0, 1.0, 4)
        T2d, L2d = np.meshgrid(t_fake, lam_v)
        Id_surf  = np.tile(Id_v[:, None], (1, 4))
        Iq_surf  = np.tile(Iq_v[:, None], (1, 4))

        self._init_3d_axes()
        self.ax_id.plot_surface(T2d, L2d, Id_surf, cmap='coolwarm',
                                edgecolor='none', alpha=0.92)
        self.ax_id.set_title("Id at Tmax [A] (MTPV)", fontsize=9)
        self.ax_id.set_xlabel("(T_ratio)", fontsize=7, labelpad=2)
        self.ax_id.set_ylabel("lam_max [Wb]", fontsize=7, labelpad=2)
        self.ax_id.set_zlabel("Id [A]", fontsize=7, labelpad=2)
        self.ax_id.tick_params(labelsize=6)

        self.ax_iq.plot_surface(T2d, L2d, Iq_surf, cmap='viridis',
                                edgecolor='none', alpha=0.92)
        self.ax_iq.set_title("Iq at Tmax [A] (MTPV)", fontsize=9)
        self.ax_iq.set_xlabel("(T_ratio)", fontsize=7, labelpad=2)
        self.ax_iq.set_ylabel("lam_max [Wb]", fontsize=7, labelpad=2)
        self.ax_iq.set_zlabel("Iq [A]", fontsize=7, labelpad=2)
        self.ax_iq.tick_params(labelsize=6)
        self.canvas.draw()


class SimTab(QWidget):
    """시뮬레이션 탭"""
    sim_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(6)

        # ---- controls (left of sim tab) ----
        ctrl = QWidget()
        ctrl.setFixedWidth(220)
        vl = QVBoxLayout(ctrl)
        vl.setContentsMargins(6, 6, 6, 6)
        vl.setSpacing(12)

        grp_rpm = QGroupBox("회전수 [rpm]")
        gl = QVBoxLayout(grp_rpm)
        self.lbl_rpm = QLabel("3000 rpm")
        self.lbl_rpm.setAlignment(Qt.AlignCenter)
        self.sl_rpm = QSlider(Qt.Horizontal)
        self.sl_rpm.setRange(100, 6000)
        self.sl_rpm.setValue(3000)
        self.sl_rpm.setTickInterval(500)
        self.sl_rpm.setTickPosition(QSlider.TicksBelow)
        self.sl_rpm.valueChanged.connect(self._on_change)
        gl.addWidget(self.lbl_rpm)
        gl.addWidget(self.sl_rpm)
        vl.addWidget(grp_rpm)

        grp_t = QGroupBox("토크 지령 Tref [Nm]")
        gl2 = QVBoxLayout(grp_t)
        self.lbl_tref = QLabel("2.0 Nm")
        self.lbl_tref.setAlignment(Qt.AlignCenter)
        self.sl_tref = QSlider(Qt.Horizontal)
        self.sl_tref.setRange(0, 500)   # 0 ~ 50.0 Nm (×0.1)
        self.sl_tref.setValue(20)
        self.sl_tref.setTickInterval(50)
        self.sl_tref.setTickPosition(QSlider.TicksBelow)
        self.sl_tref.valueChanged.connect(self._on_change)
        gl2.addWidget(self.lbl_tref)
        gl2.addWidget(self.sl_tref)
        vl.addWidget(grp_t)



        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        vl.addWidget(sep)

        res_title = QLabel("동작점 결과")
        res_title.setFont(QFont("Arial", 9, QFont.Bold))
        vl.addWidget(res_title)

        form = QFormLayout()
        form.setSpacing(6)
        self.lbl_id_op  = QLabel("—")
        self.lbl_iq_op  = QLabel("—")
        self.lbl_te_op  = QLabel("—")
        self.lbl_lam_op = QLabel("—")
        self.lbl_tmax   = QLabel("—")
        for lbl in [self.lbl_id_op, self.lbl_iq_op,
                    self.lbl_te_op, self.lbl_lam_op, self.lbl_tmax]:
            lbl.setFont(QFont("Courier", 9))
        form.addRow("id* [A]",    self.lbl_id_op)
        form.addRow("iq* [A]",    self.lbl_iq_op)
        form.addRow("Te [Nm]",    self.lbl_te_op)
        form.addRow("λ_max [Wb]", self.lbl_lam_op)
        form.addRow("Tmax [Nm]",  self.lbl_tmax)
        vl.addLayout(form)
        vl.addStretch()

        root.addWidget(ctrl)

        # ---- plot (right of sim tab) ----
        fig = Figure(tight_layout=False)
        self.ax = fig.add_axes([0.12, 0.10, 0.82, 0.82])
        self.canvas = MplCanvas(fig)
        self._fig = fig
        self._bg_artists = []
        self._dyn_artists = []
        self._ID_bg = self._IQ_bg = self._LAM_bg = self._TE_bg = None
        root.addWidget(self.canvas, stretch=1)

    def _on_change(self):
        self.lbl_rpm.setText(f"{self.sl_rpm.value()} rpm")
        self.lbl_tref.setText(f"{self.sl_tref.value() * 0.1:.1f} Nm")
        self.sim_changed.emit()

    def get_rpm(self):
        return float(self.sl_rpm.value())

    def get_tref(self):
        return self.sl_tref.value() * 0.1

    def init_bg(self, p_, Id_at_Tmax, Iq_at_Tmax):
        for a in self._bg_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._bg_artists.clear()

        self.ax.cla()

        id_vec = np.linspace(-p_["Imax"] * 1.1, p_["Imax"] * 0.1, 40)
        iq_vec = np.linspace(0, p_["Imax"] * 1.1, 40)
        self._ID_bg, self._IQ_bg = np.meshgrid(id_vec, iq_vec)
        I_bg  = np.sqrt(self._ID_bg**2 + self._IQ_bg**2)
        LAM_d = p_["psi_f"] + p_["Ld"] * self._ID_bg
        LAM_q = p_["Lq"] * self._IQ_bg
        self._LAM_bg = np.sqrt(LAM_d**2 + LAM_q**2)
        self._TE_bg  = 1.5 * p_["pole_pairs"] * (
            p_["psi_f"] * self._IQ_bg
            + (p_["Ld"] - p_["Lq"]) * self._ID_bg * self._IQ_bg)

        # MTPA
        delta = p_["Lq"] - p_["Ld"]
        I_sweep = np.linspace(0.0, p_["Imax"], 60)
        if abs(delta) < 1e-12:
            id_mtpa = np.zeros_like(I_sweep)
            iq_mtpa = I_sweep.copy()
        else:
            k = p_["psi_f"] / delta
            id_mtpa = (k - np.sqrt(k**2 + 8.0 * I_sweep**2)) / 4.0
            iq_mtpa = np.sqrt(np.maximum(I_sweep**2 - id_mtpa**2, 0.0))
        v = (np.isfinite(id_mtpa) & np.isfinite(iq_mtpa)
             & (id_mtpa <= 1e-9) & (iq_mtpa >= -1e-9))

        cf = self.ax.contourf(self._ID_bg, self._IQ_bg, I_bg,
                              levels=[0, p_["Imax"]], colors=['#AED6F1'], alpha=0.3)
        ct = self.ax.contour(self._ID_bg, self._IQ_bg, I_bg,
                             levels=[p_["Imax"]], colors=['#2E86C1'], linewidths=1.5)
        lm, = self.ax.plot(id_mtpa[v], iq_mtpa[v], 'orange', lw=1.8, label="MTPA")
        # filter NaN from trajectory so the line is never cut off
        v_t = np.isfinite(Id_at_Tmax) & np.isfinite(Iq_at_Tmax)
        lt, = self.ax.plot(Id_at_Tmax[v_t], Iq_at_Tmax[v_t],
                           'purple', lw=1.8, label="Optimal traj.")

        self.ax.set_xlabel(r"$i_d$ [A]")
        self.ax.set_ylabel(r"$i_q$ [A]")
        self.ax.set_xlim(id_vec[0], id_vec[-1])
        self.ax.set_ylim(0, iq_vec[-1])
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)

        self._bg_artists.extend([cf, ct, lm, lt])
        self._point_op, = self.ax.plot([], [], 'ro', ms=10,
                                       mec='darkred', mew=2, zorder=5)
        self.canvas.draw()

    def redraw(self, p_, lam_ref, Tref_cmd, Tref, id_op, iq_op, Tmax_at_lam):
        for a in self._dyn_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._dyn_artists.clear()

        if self._ID_bg is None:
            return

        cf = self.ax.contourf(self._ID_bg, self._IQ_bg, self._LAM_bg,
                              levels=[0, lam_ref], colors=['#A9DFBF'], alpha=0.4)
        ct = self.ax.contour(self._ID_bg, self._IQ_bg, self._LAM_bg,
                             levels=[lam_ref], colors=['#1E8449'], linewidths=1.5)
        self._dyn_artists.extend([cf, ct])

        if Tref_cmd > 0:
            c1 = self.ax.contour(self._ID_bg, self._IQ_bg, self._TE_bg,
                                 levels=[Tref_cmd], colors=['red'],
                                 linewidths=1.5, linestyles='--')
            self._dyn_artists.append(c1)
        if Tref > 0:
            c2 = self.ax.contour(self._ID_bg, self._IQ_bg, self._TE_bg,
                                 levels=[Tref], colors=['red'], linewidths=2.5)
            self._dyn_artists.append(c2)

        if hasattr(self, '_point_op') and self._point_op is not None:
            self._point_op.set_data([id_op], [iq_op])

        handles, labels = self.ax.get_legend_handles_labels()
        valid = [h for h in handles if not h.get_label().startswith('_')]
        if valid:
            self.ax.legend(loc="upper right", fontsize=8, handles=valid)

        self.canvas.draw_idle()

    def update_results(self, id_op, iq_op, te_op, lam_ref, Tmax_at_lam):
        self.lbl_id_op.setText(f"{id_op:.3f} A")
        self.lbl_iq_op.setText(f"{iq_op:.3f} A")
        self.lbl_te_op.setText(f"{te_op:.3f} Nm")
        self.lbl_lam_op.setText(f"{lam_ref:.5f} Wb")
        self.lbl_tmax.setText(f"{Tmax_at_lam:.3f} Nm")


# =========================
# Left param panel
# =========================
class DescLineEdit(QLineEdit):
    """QLineEdit that emits a description string when focused."""
    focused = pyqtSignal(str)
    def __init__(self, val, desc, parent=None):
        super().__init__(val, parent)
        self._desc = desc
        self.setFixedHeight(26)
    def focusInEvent(self, e):
        super().focusInEvent(e)
        self.focused.emit(self._desc)


class ParamPanel(QWidget):
    rebuild_requested = pyqtSignal(dict, float)
    mode_changed = pyqtSignal()

    _DESCS = {
        "vdc":   "Vdc [V] — 직류 링크(배터리) 전압",
        "imax":  "Imax [A] — 최대 허용 상전류. id-iq 평면의 전류 제한원 반지름",
        "psif":  "ψf [Wb] — 영구자석이 만드는 자속 (PM flux linkage)",
        "ld":    "Ld [H] — d축 인덕턴스 (자속 방향, 약계자에 관련)",
        "lq":    "Lq [H] — q축 인덕턴스 (토크 방향, SPM 이면 Ld=Lq)",
        "pp":    "극쌍수 — 모터 극쌍수 (pole pairs). 전기적 속도 = 극쌍수 × 기계적 속도",
        "alpha": "α — 인버터 출력 전압 이용률 (Vmax = α × Vdc). 공간벡터 변조 시 1/√3 ≈ 0.577",
        "ct":    "동토크 제어 ON → λ_max × T_ratio 2D LUT로 지령 토크를 정밀 추적\n"
                 "동토크 제어 OFF → λ_max 1D 룩업으로 MTPA/MTPV 최대토크 점만 추적",
    }

    def __init__(self, p_init, Vdc_init, parent=None):
        super().__init__(parent)
        self.setFixedWidth(240)
        self._build_ui(p_init, Vdc_init)

    def _show_desc(self, text):
        self.desc_lbl.setText(text)

    def _build_ui(self, p_init, Vdc_init):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        title = QLabel("모터 파라미터")
        title.setFont(QFont("Arial", 11, QFont.Bold))
        layout.addWidget(title)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        layout.addWidget(sep)

        form = QFormLayout()
        form.setSpacing(6)

        def de(val, key, dec=4):
            e = DescLineEdit(str(round(val, dec)), self._DESCS[key])
            e.focused.connect(self._show_desc)
            return e

        self.e_vdc   = de(Vdc_init,          "vdc", 1)
        self.e_imax  = de(p_init["Imax"],    "imax", 1)
        self.e_psif  = de(p_init["psi_f"],   "psif", 4)
        self.e_ld    = de(p_init["Ld"],      "ld",   4)
        self.e_lq    = de(p_init["Lq"],      "lq",   4)
        self.e_pp    = de(p_init["pole_pairs"],"pp",  0)
        self.e_alpha = de(p_init["alpha"],   "alpha",3)

        form.addRow("Vdc [V]",       self.e_vdc)
        form.addRow("Imax [A]",      self.e_imax)
        form.addRow("ψf [Wb]",       self.e_psif)
        form.addRow("Ld [H]",        self.e_ld)
        form.addRow("Lq [H]",        self.e_lq)
        form.addRow("극쌍수",         self.e_pp)
        form.addRow("α (Vmax/Vdc)",  self.e_alpha)
        layout.addLayout(form)

        self.btn = QPushButton("LUT 생성")
        self.btn.setFixedHeight(38)
        self.btn.setStyleSheet(
            "QPushButton { background:#1565C0; color:white; border-radius:5px; font-weight:bold; font-size:11px; }"
            "QPushButton:hover { background:#0D47A1; }"
            "QPushButton:disabled { background:#90A4AE; color:#CFD8DC; }"
        )
        self.btn.clicked.connect(self._on_click)
        layout.addWidget(self.btn)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        self.progress.setFixedHeight(10)
        self.progress.setTextVisible(False)
        layout.addWidget(self.progress)

        self.status_lbl = QLabel("")
        self.status_lbl.setAlignment(Qt.AlignCenter)
        self.status_lbl.setStyleSheet("color: #555; font-size: 10px;")
        layout.addWidget(self.status_lbl)

        # Constant-torque checkbox
        self.chk_const_torque = QCheckBox("동토크 제어")
        self.chk_const_torque.setChecked(True)
        self.chk_const_torque.stateChanged.connect(self._on_ct_changed)
        layout.addWidget(self.chk_const_torque)

        layout.addStretch()

        # Inline description label — pinned to bottom
        desc_title = QLabel("설명")
        desc_title.setStyleSheet("color:#888; font-size:10px;")
        layout.addWidget(desc_title)

        self.desc_lbl = QLabel("")
        self.desc_lbl.setWordWrap(True)
        self.desc_lbl.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.desc_lbl.setStyleSheet(
            "color:#222; font-size:11px; background:#F5F5F5;"
            " border:1px solid #DDD; border-radius:3px; padding:5px;")
        self.desc_lbl.setMinimumHeight(70)
        layout.addWidget(self.desc_lbl)

    def _on_ct_changed(self):
        self._show_desc(self._DESCS["ct"])
        self.mode_changed.emit()

    def is_const_torque(self):
        return self.chk_const_torque.isChecked()

    def _on_click(self):
        try:
            p = {
                "pole_pairs": int(self.e_pp.text()),
                "Ld":         float(self.e_ld.text()),
                "Lq":         float(self.e_lq.text()),
                "psi_f":      float(self.e_psif.text()),
                "Imax":       float(self.e_imax.text()),
                "alpha":      float(self.e_alpha.text()),
            }
            Vdc = float(self.e_vdc.text())
        except ValueError:
            self.status_lbl.setText("⚠ 잘못된 입력값")
            return
        self.btn.setEnabled(False)
        self.progress.setVisible(True)
        self.status_lbl.setText("계산 중…")
        self.rebuild_requested.emit(p, Vdc)

    def on_done(self):
        self.btn.setEnabled(True)
        self.progress.setVisible(False)
        self.status_lbl.setText("✔ 완료")


# =========================
# Main window
# =========================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IPMM LUT 툴")
        self.resize(1200, 740)

        self.p_  = {"pole_pairs": 4, "Ld": 0.004, "Lq": 0.008,
                    "psi_f": 0.01, "Imax": 20.0, "alpha": 1/3}
        self.Vdc = 48.0
        self.lut_data = None
        self._worker  = None

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # ---- Left: param panel ----
        self.param_panel = ParamPanel(self.p_, self.Vdc)
        self.param_panel.rebuild_requested.connect(self._on_rebuild_requested)
        self.param_panel.mode_changed.connect(self._on_mode_changed)
        root.addWidget(self.param_panel)

        # ---- Separator ----
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFrameShadow(QFrame.Sunken)
        root.addWidget(sep)

        # ---- Right: tab widget ----
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.setEnabled(False)   # disabled until LUT built
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #B0BEC5;
                border-radius: 3px;
            }
            QTabBar::tab {
                background: #ECEFF1;
                border: 1px solid #B0BEC5;
                padding: 6px 18px;
                font-size: 11px;
            }
            QTabBar::tab:selected {
                background: #1565C0;
                color: white;
                font-weight: bold;
            }
            QTabBar::tab:hover:!selected { background: #BBDEFB; }
        """)

        self.tmax_tab = TmaxTab()
        self.lut_tab  = LutTab()
        self.sim_tab  = SimTab()

        self.tabs.addTab(self.tmax_tab, "최대토크")
        self.tabs.addTab(self.lut_tab,  "LUT")
        self.tabs.addTab(self.sim_tab,  "시뮬레이션")

        self.sim_tab.sim_changed.connect(self._refresh_sim)

        root.addWidget(self.tabs, stretch=1)

    # ------------------------------------------------------------------
    def _on_rebuild_requested(self, p_new, Vdc_new):
        self.p_  = p_new
        self.Vdc = Vdc_new
        self.tabs.setEnabled(False)
        self._worker = LutWorker(self.p_)
        self._worker.finished.connect(self._on_lut_done)
        self._worker.start()

    def _on_lut_done(self, data):
        self.lut_data = data
        self.param_panel.on_done()
        self.tabs.setEnabled(True)

        self.tmax_tab.update_plot(data["lam_grid"], data["Tmax_LUT"], p_=self.p_, Vdc=self.Vdc)
        self.lut_tab.update_plot(data["lam_grid"], data["Tratio_grid"],
                                 data["Id_LUT_2D"], data["Iq_LUT_2D"])
        self.sim_tab.init_bg(self.p_, data["Id_at_Tmax"], data["Iq_at_Tmax"])
        self._refresh_sim()

    def _on_mode_changed(self):
        """Called when the constant-torque checkbox is toggled."""
        if self.lut_data is None:
            return
        d = self.lut_data
        is_ct = self.param_panel.is_const_torque()
        # Update LUT tab
        if is_ct:
            self.lut_tab.update_plot(d["lam_grid"], d["Tratio_grid"],
                                     d["Id_LUT_2D"], d["Iq_LUT_2D"])
        else:
            self.lut_tab.update_trajectory(d["lam_grid"],
                                           d["Id_at_Tmax"], d["Iq_at_Tmax"])
        self._refresh_sim()

    def _refresh_sim(self):
        if self.lut_data is None:
            return
        d = self.lut_data
        rpm      = self.sim_tab.get_rpm()
        Tref_cmd = self.sim_tab.get_tref()
        is_ct    = self.param_panel.is_const_torque()

        lam_ref     = part1_lambda_max_ff(rpm, self.Vdc, self.p_)
        Tmax_at_lam = float(np.interp(lam_ref, d["lam_grid"], d["Tmax_LUT"]))

        if is_ct:
            Tref    = float(np.clip(Tref_cmd, 0.0, Tmax_at_lam * 0.999))
            T_ratio = Tref / max(Tmax_at_lam, 1e-6)
            pt    = np.array([[lam_ref, T_ratio]])
            id_op = d["interp_id"](pt).item()
            iq_op = d["interp_iq"](pt).item()
        else:
            # MTPA/MTPV: 1D lookup, always at max torque point
            Tref  = Tmax_at_lam
            v     = np.isfinite(d["Id_at_Tmax"]) & np.isfinite(d["Iq_at_Tmax"])
            id_op = float(np.interp(lam_ref, d["lam_grid"][v], d["Id_at_Tmax"][v]))
            iq_op = float(np.interp(lam_ref, d["lam_grid"][v], d["Iq_at_Tmax"][v]))

        te_op = Te(id_op, iq_op, self.p_)

        self.sim_tab.update_results(id_op, iq_op, te_op, lam_ref, Tmax_at_lam)
        self.sim_tab.redraw(self.p_, lam_ref, Tref_cmd, Tref,
                            id_op, iq_op, Tmax_at_lam)


# =========================
# Entry point
# =========================
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Arial", 9))
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

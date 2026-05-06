import sys
import warnings
import json
import csv
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
    QComboBox, QDialog, QScrollArea, QDialogButtonBox,
    QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem,
    QButtonGroup, QStackedWidget
)

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QCursor, QFontInfo

import os
# ----- High-DPI support -----
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
os.environ["QT_SCALE_FACTOR"] = "1"
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

# Scale helpers — computed once after QApplication is created.
# _S(px)  : scale a pixel value relative to 96 dpi baseline
# _FS(pt) : scale a font point size
_DPI_SCALE = 1.0   # updated in main()

def _S(px: int) -> int:
    """Scale pixel dimension for the current DPI."""
    return max(1, round(px * _DPI_SCALE))

def _FS(pt: int) -> int:
    """Scale font point size for the current DPI."""
    return max(6, round(pt * _DPI_SCALE))


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


def solve_zero_torque_point_for_lam(lam_ref, p_):
    """
    T = 0 조건에서 lam_mag <= lam_ref를 만족하는 Id/Iq를 반환.
    기본은 Id=0, Iq=0.
    만약 lam_ref < psi_f이면, Iq=0으로 두고
    Id = (lam_ref - psi_f) / Ld 로 약자속 Id를 계산.
    전류 제한 Imax를 넘으면 np.nan, np.nan 반환.
    """
    psi_f = p_["psi_f"]
    Ld = p_["Ld"]
    Imax = p_["Imax"]
    
    if psi_f <= lam_ref:
        return 0.0, 0.0
    else:
        id0 = (lam_ref - psi_f) / Ld
        iq0 = 0.0
        if abs(id0) <= Imax:
            return id0, iq0
        else:
            return np.nan, np.nan


# =========================
# Background worker thread
# =========================
class LutWorker(QThread):
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


# =========================
# Tab Overlay (Initial / Calculating)
# =========================
class TabOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("""
            TabOverlay {
                background-color: #FFFFFF;
            }
        """)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addStretch(2)

        # Card
        card = QWidget()
        card.setObjectName("card")
        card.setStyleSheet(f"""
            QWidget#card {{
                background: white;
                border-radius: {_S(12)}px;
                border: 1px solid #D0DCF0;
            }}
        """)
        card.setFixedWidth(_S(380))
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(_S(32), _S(32), _S(32), _S(32))
        card_layout.setSpacing(_S(18))

        # Main message
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setWordWrap(True)
        self.label.setStyleSheet(
            f"font-size: {_FS(13)}px; font-weight: bold; color: #1A237E; line-height: 1.5;"
        )
        card_layout.addWidget(self.label)

        # Sub message
        self.sub_lbl = QLabel()
        self.sub_lbl.setAlignment(Qt.AlignCenter)
        self.sub_lbl.setWordWrap(True)
        self.sub_lbl.setStyleSheet(
            f"font-size: {_FS(10)}px; color: #607D8B; font-weight: normal;"
        )
        card_layout.addWidget(self.sub_lbl)

        # Progress container
        self.prog_container = QWidget()
        pc = QVBoxLayout(self.prog_container)
        pc.setContentsMargins(0, 0, 0, 0)
        pc.setSpacing(_S(6))

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFixedHeight(_S(10))
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet(f"""
            QProgressBar {{
                border: none;
                border-radius: {_S(6)}px;
                background: #E8EAF6;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3949AB, stop:1 #1A237E);
                border-radius: {_S(6)}px;
            }}
        """)
        pc.addWidget(self.progress)

        # Percentage label
        self.pct_lbl = QLabel("0%")
        self.pct_lbl.setAlignment(Qt.AlignCenter)
        self.pct_lbl.setStyleSheet(
            f"font-size: {_FS(11)}px; font-weight: bold; color: #1A237E;"
        )
        pc.addWidget(self.pct_lbl)

        card_layout.addWidget(self.prog_container)
        self.prog_container.setVisible(False)

        outer.addWidget(card, 0, Qt.AlignCenter)
        outer.addStretch(3)

    def show_idle(self):
        self.label.setText("좌측 패널에 모터 파라미터를 입력하고\n[GENERATE LUT] 버튼을 눌러주세요.")
        self.sub_lbl.setText("입력된 조건에 따라 전동기 특성 맵(LUT)이 생성됩니다.")
        self.prog_container.setVisible(False)
        self.show()
        self.raise_()

    def show_calculating(self):
        self.label.setText("최적화 연산 수행 중...")
        self.sub_lbl.setText("입력된 파라미터를 기반으로 고정밀 LUT 데이터를 계산하고 있습니다.")
        self.progress.setValue(0)
        self.pct_lbl.setText("0%")
        self.prog_container.setVisible(True)
        self.show()
        self.raise_()

    def set_progress(self, pct):
        self.progress.setValue(pct)
        self.pct_lbl.setText(f"{pct}%")

    def update_geometry(self, rect):
        self.setGeometry(rect)


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

class BaseHelpDialog(QDialog):
    """도움말 다이얼로그 베이스 클래스 (미니멀 디자인)"""
    def __init__(self, title, color, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(_S(500), _S(450))
        self.color = color
        
        self.setStyleSheet(f"""
            QDialog {{ background: #FFFFFF; }}
            QLabel#title {{ font-size: {_FS(13)}px; font-weight: bold; color: white; }}
            QLabel#section {{ font-size: {_FS(10.5)}px; font-weight: bold; color: {color};
                             border-bottom: 2px solid {color}; padding-bottom: 2px;
                             margin-top: 15px; margin-bottom: 5px; }}
            QLabel#body {{ font-size: {_FS(10.5)}px; color: #37474F; line-height: 1.5; }}
            QLabel#formula {{ font-family: 'Consolas', 'Courier New'; font-size: {_FS(12)}px;
                             background: #F8F9FA; border-left: 4px solid {color};
                             padding: 10px; color: #263238; }}
        """)
        
        self.outer = QVBoxLayout(self)
        self.outer.setContentsMargins(0, 0, 0, 10)
        
        # Header
        header = QWidget()
        header.setFixedHeight(_S(50))
        header.setStyleSheet(f"background: {color};")
        hl = QHBoxLayout(header)
        title_lbl = QLabel(title)
        title_lbl.setObjectName("title")
        hl.addWidget(title_lbl)
        hl.addStretch()
        self.outer.addWidget(header)
        
        # Content Area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        self.content_container = QWidget()
        self.vl = QVBoxLayout(self.content_container)
        self.vl.setContentsMargins(20, 10, 20, 20)
        scroll.setWidget(self.content_container)
        self.outer.addWidget(scroll)
        
        # Close Button
        btn_box = QDialogButtonBox(QDialogButtonBox.Close)
        btn_box.setContentsMargins(15, 0, 15, 0)
        btn_box.rejected.connect(self.accept)
        self.outer.addWidget(btn_box)

    def add_section(self, title):
        l = QLabel(title); l.setObjectName("section"); self.vl.addWidget(l); return l
    def add_body(self, text):
        l = QLabel(text); l.setObjectName("body"); l.setWordWrap(True); self.vl.addWidget(l); return l
    def add_formula(self, text):
        l = QLabel(text); l.setObjectName("formula"); l.setWordWrap(True); self.vl.addWidget(l); return l

class FormulaHelpDialog(BaseHelpDialog):
    def __init__(self, parent=None):
        super().__init__("수식 설명 — 최대토크 & 출력", "#1A237E", parent)
        self.add_section("1. 전압 제한과 최대 자속 (&lambda;<sub>max</sub>)")
        self.add_body("인버터 최대 출력 전압과 회전 속도에 의해 최대 허용 자속이 결정됩니다.")
        self.add_formula(
            "<i>V</i><sub>max</sub> = &alpha; &times; <i>V</i><sub>dc</sub><br>"
            "&lambda;<sub>max</sub> = <i>V</i><sub>max</sub> / &omega;<sub>e</sub>  [Wb]"
        )
        self.add_section("2. 최대토크 (Max Torque)")
        self.add_body("전류 제한과 자속 제한 내에서 토크를 극대화하는 (id, iq)를 수치 최적화로 구합니다.")
        self.add_formula(
            "<i>T<sub>e</sub></i> = 1.5 &times; <i>P</i> &times; [ &psi;<sub>f</sub> &times; <i>i<sub>q</sub></i> + (<i>L<sub>d</sub></i> &minus; <i>L<sub>q</sub></i>) &times; <i>i<sub>d</sub></i> &times; <i>i<sub>q</sub></i> ]"
        )
        self.vl.addStretch()

class LutHelpDialog(BaseHelpDialog):
    def __init__(self, parent=None):
        super().__init__("도움말 — LUT 생성 및 제어", "#2E7D32", parent)
        self.add_section("1. 전 영역 LUT 생성")
        self.add_body("지령 토크 비율(T_ratio)에 대해 동손을 최소화하는 운전점을 검색합니다.")
        self.add_formula(
            "Minimize &radic;(<i>i<sub>d</sub></i><sup>2</sup> + <i>i<sub>q</sub></i><sup>2</sup>)<br>"
            "Subject to: <i>T<sub>e</sub></i>(<i>i<sub>d</sub></i>, <i>i<sub>q</sub></i>) = <i>T<sub>ref</sub></i>"
        )
        self.vl.addStretch()

class SimHelpDialog(BaseHelpDialog):
    def __init__(self, parent=None):
        super().__init__("도움말 — 실시간 시뮬레이션", "#455A64", parent)
        self.add_section("업데이트 중...")
        self.add_body("시뮬레이션 관련 수식 및 상세 설명은 다음 업데이트에서 제공될 예정입니다.")
        self.vl.addStretch()

class TmaxTab(QWidget):
    """최대토크 탭"""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 4)
        layout.setSpacing(6)

        self.mode = "Flux Linkage [Wb]"



        # ── 결과 요약 카드 행 ──────────────────────────────────
        card_row = QHBoxLayout()
        card_row.setSpacing(8)

        card_ss = (
            "QWidget { background:#FFFFFF; border:1px solid #CFD8DC;"
            f" border-radius:{_S(12)}px; }}"
        )
        val_main_ss = f"font-size:{_FS(18)}px; font-weight:bold; color:#1A237E; border:none;" 
        val_unit_ss = f"font-size:{_FS(10.5)}px; font-weight:500; color:#607D8B; border:none;"
        lbl_ss  = f"font-size:{_FS(9)}px; color:#546E7A; font-weight:bold; border:none; text-transform:uppercase; letter-spacing:0.5px;"

        def make_card(title):
            w = QWidget()
            w.setStyleSheet(card_ss)
            hl_main = QHBoxLayout(w)
            hl_main.setContentsMargins(_S(16), _S(8), _S(16), _S(8))
            hl_main.setSpacing(_S(12))
            
            lbl = QLabel(title)
            lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            lbl.setStyleSheet(lbl_ss)
            
            hl_main.addWidget(lbl)
            hl_main.addStretch()
            
            val_main = QLabel("—")
            val_main.setStyleSheet(val_main_ss)
            val_unit = QLabel("")
            val_unit.setStyleSheet(val_unit_ss)
            
            hl_main.addWidget(val_main)
            hl_main.addWidget(val_unit)
            
            return w, val_main, val_unit

        c1, self.card_t_val, self.card_t_unit = make_card("MAX TORQUE")
        c2, self.card_p_val, self.card_p_unit = make_card("MAX POWER")

        for c in [c1, c2]:
            card_row.addWidget(c, 1)
        layout.addLayout(card_row)



        fig = Figure(tight_layout=True)
        # 그래프 배경색 설정
        fig.patch.set_facecolor('#F8F9FA')
        self.ax_t = fig.add_subplot(211)
        self.ax_p = fig.add_subplot(212, sharex=self.ax_t)
        self.canvas = MplCanvas(fig)
        
        # 캔버스 주변 여백 및 배경 스타일
        canvas_layout = QHBoxLayout()
        canvas_layout.setContentsMargins(0, 4, 0, 0)
        canvas_layout.addWidget(self.canvas)
        layout.addLayout(canvas_layout)

        self.lam_grid = None
        self.Tmax_LUT = None
        self.p_ = None
        self.Vdc = None

    def update_plot(self, lam_grid, Tmax_LUT, p_=None, Vdc=None):
        self.lam_grid = lam_grid
        self.Tmax_LUT = Tmax_LUT
        self.p_ = p_
        self.Vdc = Vdc
        self._update_summary()
        self._redraw()

    def _update_summary(self):
        if self.lam_grid is None or self.p_ is None:
            return
        Vmax = self.p_["alpha"] * self.Vdc
        pp   = self.p_["pole_pairs"]
        omega_mech = Vmax / (np.maximum(self.lam_grid, 1e-9) * pp)
        power_w = self.Tmax_LUT * omega_mech

        valid = np.isfinite(self.Tmax_LUT) & np.isfinite(power_w)
        Tmax_peak = float(np.nanmax(self.Tmax_LUT[valid])) if valid.any() else float("nan")
        Pmax_peak = float(np.nanmax(power_w[valid])) if valid.any() else float("nan")
        rpm_max   = self.p_.get("rpm_max", float("nan"))
        alpha     = self.p_["alpha"]
        vlim_label = "SVPWM" if abs(alpha - 1/3**0.5) < 0.01 else "Conservative"

        if np.isfinite(Tmax_peak):
            self.card_t_val.setText(f"{Tmax_peak:.1f}")
            self.card_t_unit.setText("Nm")
        else:
            self.card_t_val.setText("—")
            self.card_t_unit.setText("")

        if np.isfinite(Pmax_peak):
            self.card_p_val.setText(f"{Pmax_peak/1000:.1f}")
            self.card_p_unit.setText("kW")
        else:
            self.card_p_val.setText("—")
            self.card_p_unit.setText("")

    def set_xaxis_mode(self, mode):
        self.mode = mode
        self._redraw()

    def _redraw(self):
        if self.lam_grid is None or self.Tmax_LUT is None:
            return

        self.ax_t.clear()
        self.ax_p.clear()
        
        # 그래프 스타일: 현대적이고 전문적인 느낌 (은은한 그리드, 부드러운 색상)
        for ax in [self.ax_t, self.ax_p]:
            ax.set_facecolor('#FFFFFF')
            # 그리드 투명도 높여 은은하게
            ax.grid(True, alpha=0.1, linestyle='-', color='#455A64')
            for spine in ax.spines.values():
                spine.set_edgecolor('#CFD8DC')
                spine.set_linewidth(1.0)
            ax.tick_params(colors='#546E7A', labelsize=8)

        Vmax = self.p_["alpha"] * self.Vdc
        pp = self.p_["pole_pairs"]
        
        omega_mech = Vmax / (np.maximum(self.lam_grid, 1e-9) * pp)
        power_w = self.Tmax_LUT * omega_mech
        power_kw = power_w / 1000.0

        if "Speed" in self.mode and self.p_ is not None and self.Vdc is not None:
            x_data = (omega_mech * 60.0) / (2.0 * np.pi)
            x_label = "Speed [rpm]"
            idx = np.argsort(x_data)
            self.ax_t.plot(x_data[idx], self.Tmax_LUT[idx], color='#E53935', marker='o', ms=2, lw=1.5, label="Torque [Nm]")
            self.ax_p.plot(x_data[idx], power_kw[idx], color='#1A237E', marker='o', ms=2, lw=1.5, label="Power [kW]")
        else:
            x_data = self.lam_grid
            x_label = "Flux Linkage Limit [Wb]"
            self.ax_t.plot(x_data, self.Tmax_LUT, color='#E53935', marker='o', ms=2, lw=1.5, label="Torque [Nm]")
            self.ax_p.plot(x_data, power_kw, color='#1A237E', marker='o', ms=2, lw=1.5, label="Power [kW]")

        self.ax_t.set_ylabel("Torque [Nm]", fontsize=9, fontweight='normal', color='#263238')
        self.ax_t.legend(fontsize=8, frameon=True, framealpha=0.95, facecolor='white', edgecolor='#ECEFF1')

        self.ax_p.set_xlabel(x_label, fontsize=9, fontweight='normal', color='#263238')
        self.ax_p.set_ylabel("Power [kW]", fontsize=9, fontweight='normal', color='#263238')
        self.ax_p.legend(fontsize=8, frameon=True, framealpha=0.95, facecolor='white', edgecolor='#ECEFF1')

        self.canvas.draw()



class LutTab(QWidget):
    """LUT 3D surface 및 테이블 탭"""
    lut_imported = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.mode = "Flux Linkage [Wb]"
        self.p_ = None
        self.Vdc = None
        
        # (이전 버튼 위치에서 제거)

        # ---- Segmented Toggle Switch ----
        seg_layout = QHBoxLayout()
        seg_layout.setAlignment(Qt.AlignCenter)
        seg_layout.setContentsMargins(0, _S(10), 0, _S(10))
        seg_layout.setSpacing(0)
        
        self.btn_group = QButtonGroup(self)
        self.btn_3d = QPushButton("3D 그래프")
        self.btn_tbl = QPushButton("데이터 테이블")
        
        self.btn_3d.setObjectName("segBtnLeft")
        self.btn_tbl.setObjectName("segBtnRight")
        
        r = _S(16)
        # 탭 전체에 적용되는 스타일시트 (Pill 형태 구현)
        self.setStyleSheet(f"""
            QPushButton#segBtnLeft {{
                border-top-left-radius: {r}px; border-bottom-left-radius: {r}px;
                border-top-right-radius: 0px; border-bottom-right-radius: 0px;
                border-right: none;
            }}
            QPushButton#segBtnRight {{
                border-top-right-radius: {r}px; border-bottom-right-radius: {r}px;
                border-top-left-radius: 0px; border-bottom-left-radius: 0px;
                border-left: none;
            }}
            QPushButton#segBtnLeft:checked, QPushButton#segBtnRight:checked {{
                background: #1A237E; color: white; font-weight: bold;
                border: 1px solid #1A237E;
            }}
            QPushButton#segBtnLeft:!checked, QPushButton#segBtnRight:!checked {{
                background: #FFFFFF; color: #546E7A; font-weight: bold;
                border: 1px solid #CFD8DC;
            }}
        """)
        
        for idx, b in enumerate([self.btn_3d, self.btn_tbl]):
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setFixedHeight(_S(32))
            b.setFixedWidth(_S(150))
            self.btn_group.addButton(b, idx)
            seg_layout.addWidget(b)
            
        self.btn_3d.setChecked(True)
        self.btn_group.idClicked.connect(lambda idx: self.stack.setCurrentIndex(idx))
        layout.addLayout(seg_layout)

        # ---- Stacked Content ----
        self.stack = QStackedWidget()
        
        # Plot Container
        self.plot_container = QWidget()
        pl = QVBoxLayout(self.plot_container)
        pl.setContentsMargins(0, 0, 0, 0)
        self._fig = Figure(figsize=(_S(12), _S(7)))
        self.canvas = MplCanvas(self._fig)
        pl.addWidget(self.canvas)
        self.stack.addWidget(self.plot_container)
        
        # Table Container
        self.table_container = QWidget()
        tl = QVBoxLayout(self.table_container)
        tl.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet(f"font-size: {_FS(9)}px;")
        tl.addWidget(self.table)
        self.stack.addWidget(self.table_container)
        
        layout.addWidget(self.stack, 1)
        
        # ---- Bottom Action Bar ----
        footer_layout = QHBoxLayout()
        footer_layout.setContentsMargins(_S(10), _S(10), _S(10), _S(10))
        footer_layout.setSpacing(_S(12))
        
        self.btn_import = QPushButton("LUT Import")
        self.btn_export = QPushButton("LUT Export")
        
        btn_ss = (
            f"QPushButton {{ background:transparent; color:#1A237E; border:1px solid #1A237E;"
            f" border-radius:{_S(6)}px; font-weight:bold; font-size:{_FS(10.5)}px; padding: {_S(6)}px {_S(14)}px; }}"
            "QPushButton:hover { background:#E8F0FE; }"
            "QPushButton:pressed { background:#D2E3FC; }"
        )
        
        for b in [self.btn_import, self.btn_export]:
            b.setFixedHeight(_S(32))
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(btn_ss)
            footer_layout.addWidget(b)
            
        self.btn_import.clicked.connect(self._on_import_clicked)
        self.btn_export.clicked.connect(self._on_export_clicked)
        
        footer_layout.addStretch()
        layout.addLayout(footer_layout)
        
        self._init_3d_axes()
        
        # Data storage
        self.lam_grid = None
        self.Tratio_grid = None
        self.Id_LUT_2D = None
        self.Iq_LUT_2D = None

    def _init_3d_axes(self):
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        self._fig.clear()
        self.ax_id = self._fig.add_subplot(121, projection='3d')
        self.ax_iq = self._fig.add_subplot(122, projection='3d')

    def update_plot(self, lam_grid, Tratio_grid, Id_LUT_2D, Iq_LUT_2D, p_=None, Vdc=None):
        self.lam_grid = lam_grid
        self.Tratio_grid = Tratio_grid
        self.Id_LUT_2D = Id_LUT_2D
        self.Iq_LUT_2D = Iq_LUT_2D
        if p_ is not None: self.p_ = p_
        if Vdc is not None: self.Vdc = Vdc
        
        self._redraw_plot()
        self._update_table()

    def set_xaxis_mode(self, mode):
        self.mode = mode
        if self.Id_LUT_2D is not None:
            self._redraw_plot()
            self._update_table()
        elif getattr(self, 'Id_at_Tmax', None) is not None:
            self._redraw_trajectory()

    def _redraw_plot(self):
        if self.lam_grid is None: return
        self._init_3d_axes()
        
        y_label = "lam_max [Wb]"
        y_data = self.lam_grid
        Id_data = self.Id_LUT_2D
        Iq_data = self.Iq_LUT_2D
        
        if "Speed" in self.mode and self.p_ is not None and self.Vdc is not None:
            Vmax = self.p_["alpha"] * self.Vdc
            pp = self.p_["pole_pairs"]
            omega_mech = Vmax / (np.maximum(self.lam_grid, 1e-9) * pp)
            speed_rpm = (omega_mech * 60.0) / (2.0 * np.pi)
            y_data = speed_rpm[::-1] # Reverse to make it ascending
            Id_data = Id_data[::-1, :]
            Iq_data = Iq_data[::-1, :]
            y_label = "Speed [rpm]"
            
        T, L = np.meshgrid(self.Tratio_grid, y_data)
        Id_masked = np.ma.masked_invalid(Id_data)
        Iq_masked = np.ma.masked_invalid(Iq_data)

        self.ax_id.plot_surface(T, L, Id_masked, cmap='coolwarm',
                                edgecolor='none', alpha=0.92)
        self.ax_id.set_title("Id LUT [A]", fontsize=9)
        self.ax_id.set_xlabel("T_ratio", fontsize=7, labelpad=2)
        self.ax_id.set_ylabel(y_label, fontsize=7, labelpad=2)
        self.ax_id.set_zlabel("Id [A]", fontsize=7, labelpad=2)
        self.ax_id.tick_params(labelsize=6)

        self.ax_iq.plot_surface(T, L, Iq_masked, cmap='viridis',
                                edgecolor='none', alpha=0.92)
        self.ax_iq.set_title("Iq LUT [A]", fontsize=9)
        self.ax_iq.set_xlabel("T_ratio", fontsize=7, labelpad=2)
        self.ax_iq.set_ylabel(y_label, fontsize=7, labelpad=2)
        self.ax_iq.set_zlabel("Iq [A]", fontsize=7, labelpad=2)
        self.ax_iq.tick_params(labelsize=6)
        self.canvas.draw()

    def _update_table(self):
        if self.lam_grid is None: return
        
        y_label = "lam_max [Wb]"
        y_data = self.lam_grid
        Id_data = self.Id_LUT_2D
        Iq_data = self.Iq_LUT_2D
        
        if "Speed" in self.mode and self.p_ is not None and self.Vdc is not None:
            Vmax = self.p_["alpha"] * self.Vdc
            pp = self.p_["pole_pairs"]
            omega_mech = Vmax / (np.maximum(self.lam_grid, 1e-9) * pp)
            y_data = (omega_mech * 60.0) / (2.0 * np.pi)
            y_label = "Speed [rpm]"

        # We'll show lam_max/Speed, T_ratio, Id, Iq columns
        rows = []
        for i, lam in enumerate(y_data):
            for j, ratio in enumerate(self.Tratio_grid):
                rows.append([lam, ratio, Id_data[i, j], Iq_data[i, j]])
        
        self.table.setRowCount(len(rows))
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels([y_label, "T_ratio", "Id [A]", "Iq [A]"])
        
        for r, data in enumerate(rows):
            for c, val in enumerate(data):
                item = QTableWidgetItem(f"{val:.6f}" if np.isfinite(val) else "NaN")
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(r, c, item)
        
        self.table.resizeColumnsToContents()

    def _on_import_clicked(self):
        path, _ = QFileDialog.getOpenFileName(self, "LUT 파일 열기", "", "LUT Files (*.csv *.json)")
        if not path: return
        
        try:
            if path.endswith('.json'):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                lam_grid = np.array(data["lam_grid"])
                Tratio_grid = np.array(data["Tratio_grid"])
                Id_LUT_2D = np.array(data["Id_LUT_2D"])
                Iq_LUT_2D = np.array(data["Iq_LUT_2D"])
            else:
                # Simple CSV parsing: lam, ratio, id, iq
                raw = np.loadtxt(path, delimiter=',', skiprows=1)
                lams = np.unique(raw[:, 0])
                ratios = np.unique(raw[:, 1])
                Nl, Nr = len(lams), len(ratios)
                if Nl * Nr != len(raw):
                    raise ValueError("CSV 데이터가 정규 그리드 형식이 아닙니다.")
                
                lam_grid = lams
                Tratio_grid = ratios
                Id_LUT_2D = raw[:, 2].reshape((Nl, Nr))
                Iq_LUT_2D = raw[:, 3].reshape((Nl, Nr))
            
            # Interpolators need to be rebuilt in MainWindow
            imported_data = {
                "lam_grid": lam_grid,
                "Tratio_grid": Tratio_grid,
                "Id_LUT_2D": Id_LUT_2D,
                "Iq_LUT_2D": Iq_LUT_2D,
            }
            self.lut_imported.emit(imported_data)
            QMessageBox.information(self, "성공", f"LUT 데이터를 성공적으로 가져왔습니다.\n(그리드: {len(lam_grid)}x{len(Tratio_grid)})")
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"파일을 읽는 중 오류가 발생했습니다:\n{str(e)}")

    def _on_export_clicked(self):
        if self.lam_grid is None:
            QMessageBox.warning(self, "경고", "내보낼 데이터가 없습니다. 먼저 LUT를 생성하세요.")
            return
            
        path, filter = QFileDialog.getSaveFileName(self, "LUT 파일 저장", "motor_lut", "CSV Files (*.csv);;JSON Files (*.json)")
        if not path: return
        
        try:
            if "JSON" in filter:
                if not path.endswith('.json'): path += '.json'
                data = {
                    "lam_grid": self.lam_grid.tolist(),
                    "Tratio_grid": self.Tratio_grid.tolist(),
                    "Id_LUT_2D": self.Id_LUT_2D.tolist(),
                    "Iq_LUT_2D": self.Iq_LUT_2D.tolist(),
                }
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            else:
                if not path.endswith('.csv'): path += '.csv'
                with open(path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["lam_max", "T_ratio", "Id", "Iq"])
                    for i, lam in enumerate(self.lam_grid):
                        for j, ratio in enumerate(self.Tratio_grid):
                            writer.writerow([lam, ratio, self.Id_LUT_2D[i, j], self.Iq_LUT_2D[i, j]])
            
            QMessageBox.information(self, "성공", "파일을 성공적으로 저장했습니다.")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"파일 저장 중 오류가 발생했습니다:\n{str(e)}")



    def update_trajectory(self, lam_grid, Id_at_Tmax, Iq_at_Tmax, p_=None, Vdc=None):
        """MTPA/MTPV mode: show MTPV surface extruded along T_ratio (3D)."""
        self.lam_grid = lam_grid
        self.Id_at_Tmax = Id_at_Tmax
        self.Iq_at_Tmax = Iq_at_Tmax
        if p_ is not None: self.p_ = p_
        if Vdc is not None: self.Vdc = Vdc
        self._redraw_trajectory()

    def _redraw_trajectory(self):
        v = np.isfinite(self.Id_at_Tmax) & np.isfinite(self.Iq_at_Tmax)
        lam_v  = self.lam_grid[v]
        Id_v   = self.Id_at_Tmax[v]
        Iq_v   = self.Iq_at_Tmax[v]
        
        y_label = "lam_max [Wb]"
        y_data = lam_v
        
        if "Speed" in self.mode and self.p_ is not None and self.Vdc is not None:
            Vmax = self.p_["alpha"] * self.Vdc
            pp = self.p_["pole_pairs"]
            omega_mech = Vmax / (np.maximum(lam_v, 1e-9) * pp)
            speed_rpm = (omega_mech * 60.0) / (2.0 * np.pi)
            y_data = speed_rpm[::-1] # Reverse to make it ascending
            Id_v = Id_v[::-1]
            Iq_v = Iq_v[::-1]
            y_label = "Speed [rpm]"
            
        t_fake = np.linspace(0.0, 1.0, 4)
        T2d, L2d = np.meshgrid(t_fake, y_data)
        Id_surf  = np.tile(Id_v[:, None], (1, 4))
        Iq_surf  = np.tile(Iq_v[:, None], (1, 4))

        self._init_3d_axes()
        self.ax_id.plot_surface(T2d, L2d, Id_surf, cmap='coolwarm',
                                edgecolor='none', alpha=0.92)
        self.ax_id.set_title("Id at Tmax [A] (MTPV)", fontsize=9)
        self.ax_id.set_xlabel("(T_ratio)", fontsize=7, labelpad=2)
        self.ax_id.set_ylabel(y_label, fontsize=7, labelpad=2)
        self.ax_id.set_zlabel("Id [A]", fontsize=7, labelpad=2)
        self.ax_id.tick_params(labelsize=6)

        self.ax_iq.plot_surface(T2d, L2d, Iq_surf, cmap='viridis',
                                edgecolor='none', alpha=0.92)
        self.ax_iq.set_title("Iq at Tmax [A] (MTPV)", fontsize=9)
        self.ax_iq.set_xlabel("(T_ratio)", fontsize=7, labelpad=2)
        self.ax_iq.set_ylabel(y_label, fontsize=7, labelpad=2)
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
        ctrl.setFixedWidth(_S(220))
        vl = QVBoxLayout(ctrl)
        vl.setContentsMargins(_S(6), _S(20), _S(6), _S(20))
        vl.setSpacing(_S(16))
        
        vl.addStretch()

        # 1) 동작점 결과 (상단 배치)
        res_title = QLabel("동작점 결과")
        res_title.setFont(QFont("Segoe UI", _FS(9), QFont.Bold))
        res_title.setStyleSheet("color:#546E7A; margin-bottom:2px;")
        vl.addWidget(res_title)

        res_box = QWidget()
        res_box.setStyleSheet(f"""
            QWidget {{ background: #F5F7FA; border: 1px solid #CFD8DC; border-radius: {_S(8)}px; }}
            QLabel {{ background: transparent; border: none; font-size: {_FS(10.5)}px; }}
        """)
        rl = QVBoxLayout(res_box)
        rl.setContentsMargins(_S(12), _S(12), _S(12), _S(12))
        rl.setSpacing(_S(8))

        self.lbl_id_op  = QLabel("—")
        self.lbl_iq_op  = QLabel("—")
        self.lbl_te_op  = QLabel("—")
        self.lbl_lam_op = QLabel("—")
        self.lbl_tmax   = QLabel("—")
        
        def add_res_row(label, val_lbl):
            row = QHBoxLayout()
            lbl_name = QLabel(label)
            lbl_name.setStyleSheet("color: #546E7A; font-weight: 500;")
            row.addWidget(lbl_name)
            row.addStretch()
            val_lbl.setFont(QFont("Courier", _FS(11), QFont.Bold))
            val_lbl.setStyleSheet("color:#1A237E;")
            row.addWidget(val_lbl)
            rl.addLayout(row)

        add_res_row("id* [A]",    self.lbl_id_op)
        add_res_row("iq* [A]",    self.lbl_iq_op)
        add_res_row("Te [Nm]",    self.lbl_te_op)
        add_res_row("λ_max [Wb]", self.lbl_lam_op)
        add_res_row("Tmax [Nm]",  self.lbl_tmax)
        vl.addWidget(res_box)

        vl.addSpacing(_S(12))

        # 2) 슬라이더 행 (하단 배치)
        slider_row = QHBoxLayout()
        slider_row.setSpacing(_S(12))

        # RPM Slider Group
        grp_rpm = QGroupBox("RPM")
        gl = QVBoxLayout(grp_rpm)
        gl.setContentsMargins(_S(8), _S(10), _S(8), _S(10))
        self.lbl_rpm = QLabel("3000\nrpm")
        self.lbl_rpm.setAlignment(Qt.AlignCenter)
        self.lbl_rpm.setStyleSheet(f"font-weight:bold; font-size:{_FS(10.5)}px; color:#1A237E;")
        self.sl_rpm = QSlider(Qt.Vertical)
        self.sl_rpm.setRange(0, 6000)
        self.sl_rpm.setValue(3000)
        self.sl_rpm.setFixedWidth(_S(34))  # 명시적 너비 확보
        self.sl_rpm.setFixedHeight(_S(180))
        self.sl_rpm.setTickInterval(1000)
        self.sl_rpm.setTickPosition(QSlider.TicksLeft)
        self.sl_rpm.valueChanged.connect(self._on_change)
        gl.addWidget(self.lbl_rpm)
        gl.addSpacing(_S(4))
        gl.addWidget(self.sl_rpm, 0, Qt.AlignCenter)
        slider_row.addWidget(grp_rpm)

        # Torque Slider Group
        grp_t = QGroupBox("Torque")
        gl2 = QVBoxLayout(grp_t)
        gl2.setContentsMargins(_S(8), _S(10), _S(8), _S(10))
        self.lbl_tref = QLabel("2.0\nNm")
        self.lbl_tref.setAlignment(Qt.AlignCenter)
        self.lbl_tref.setStyleSheet(f"font-weight:bold; font-size:{_FS(10.5)}px; color:#1A237E;")
        self.sl_tref = QSlider(Qt.Vertical)
        self.sl_tref.setRange(0, 500)
        self.sl_tref.setValue(20)
        self.sl_tref.setFixedWidth(_S(34)) # 명시적 너비 확보
        self.sl_tref.setFixedHeight(_S(180))
        self.sl_tref.setTickInterval(100)
        self.sl_tref.setTickPosition(QSlider.TicksRight)
        self.sl_tref.valueChanged.connect(self._on_change)
        gl2.addWidget(self.lbl_tref)
        gl2.addSpacing(_S(4))
        gl2.addWidget(self.sl_tref, 0, Qt.AlignCenter)
        slider_row.addWidget(grp_t)

        # 슬라이더 디자인 (상/하단 색상 방향 반전 보정)
        slider_qss = f"""
            QSlider:vertical {{
                width: {_S(34)}px;
            }}
            QSlider::groove:vertical {{
                background: #CFD8DC;
                width: {_S(8)}px;
                border-radius: {_S(4)}px;
                margin: 0 {_S(12)}px;
            }}
            QSlider::handle:vertical {{
                background: #1A237E;
                height: {_S(16)}px;
                width: {_S(24)}px;
                margin: 0 {_S(-8)}px;
                border-radius: {_S(4)}px;
            }}
            QSlider::sub-page:vertical {{
                background: #CFD8DC;
                border-radius: {_S(4)}px;
                margin: 0 {_S(12)}px;
                width: {_S(8)}px;
            }}
            QSlider::add-page:vertical {{
                background: #1A237E;
                border-radius: {_S(4)}px;
                margin: 0 {_S(12)}px;
                width: {_S(8)}px;
            }}
        """
        self.sl_rpm.setStyleSheet(slider_qss)
        self.sl_tref.setStyleSheet(slider_qss)

        vl.addLayout(slider_row)
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
        self.lbl_rpm.setText(f"{self.sl_rpm.value()}\nrpm")
        self.lbl_tref.setText(f"{self.sl_tref.value() * 0.1:.1f}\nNm")
        self.sim_changed.emit()

    def get_rpm(self):
        return float(self.sl_rpm.value())

    def get_tref(self):
        return self.sl_tref.value() * 0.1

    def set_max_rpm(self, rpm_max):
        """Update RPM slider range based on ParamPanel settings."""
        self.sl_rpm.setRange(0, int(rpm_max))
        self.sl_rpm.setTickInterval(max(500, int(rpm_max // 6)))
        if self.sl_rpm.value() > rpm_max:
            self.sl_rpm.setValue(int(rpm_max))
        self._on_change()

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
        self.lbl_id_op.setText(f"{id_op:.2f}")
        self.lbl_iq_op.setText(f"{iq_op:.2f}")
        self.lbl_te_op.setText(f"{te_op:.2f}")
        self.lbl_lam_op.setText(f"{lam_ref:.4f}")
        self.lbl_tmax.setText(f"{Tmax_at_lam:.2f}")

class ParamPanel(QWidget):
    rebuild_requested = pyqtSignal(dict, float)
    mode_changed = pyqtSignal()
    xaxis_changed = pyqtSignal(str)
    _ALPHA_PRESETS = [
        ("0.5 (SPWM)", 0.5),
        ("0.577 (SVPWM)", 0.577)
    ]

    def _section_label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"font-size:{_FS(9)}px; font-weight:bold; color:#FFFFFF;"
            f" background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1A237E, stop:1 transparent);"
            f" letter-spacing:1px; padding:{_S(6)}px {_S(10)}px; border-radius:{_S(6)}px;"
        )
        return lbl

    def _field_row(self, label_text, sub_text, widget, tip=""):
        """레이블(우측 정렬) + 보조설명 + 위젯 레이아웃 (툴팁 포함)"""
        row = QHBoxLayout()
        row.setSpacing(_S(8))
        
        # 레이블 뭉치 (VBox)
        lbl_vbox = QVBoxLayout()
        lbl_vbox.setSpacing(0)
        
        lbl = QLabel(label_text)
        lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lbl.setStyleSheet(f"font-size:{_FS(10.5)}px; font-weight:bold; color:#263238;")
        
        sub = QLabel(sub_text)
        sub.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        sub.setStyleSheet(f"font-size:{_FS(8.5)}px; color:#607D8B;")
        
        lbl_vbox.addWidget(lbl)
        lbl_vbox.addWidget(sub)
        
        lbl_container = QWidget()
        lbl_container.setLayout(lbl_vbox)
        lbl_container.setFixedWidth(_S(100))
        

            
        row.addWidget(lbl_container)
        row.addWidget(widget, 1)
        return row

    def _make_sep(self):
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color:#CFD8DC; margin:0px;")
        sep.setFixedHeight(1)
        return sep

    def __init__(self, p_init, Vdc_init):
        super().__init__()
        self.setFixedWidth(_S(320))
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Scrollable content area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border:none; background:transparent; }")

        content = QWidget()
        content.setStyleSheet("background:transparent;")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(_S(16), _S(8), _S(16), _S(16))
        layout.setSpacing(_S(4))

        field_ss = (
            f"QLineEdit {{ border:1px solid #CFD8DC; border-radius:{_S(6)}px;"
            f" padding:{_S(7)}px {_S(12)}px; background:#FFFFFF;"
            f" font-size:{_FS(10.5)}px; color:#263238; }}"
            "QLineEdit:focus { border-color:#1A237E; background:#F8FAFF; border-width:1.5px; }"
        )
        combo_ss = (
            f"QComboBox {{ border:1px solid #CFD8DC; border-radius:{_S(6)}px;"
            f" padding:{_S(6)}px {_S(12)}px; background:#FFFFFF;"
            f" font-size:{_FS(10.5)}px; color:#263238; }}"
            f"QComboBox::drop-down {{ border:none; width:{_S(22)}px; }}"
            "QComboBox:focus { border-color:#1A237E; border-width:1.5px; }"
        )

        def mk(val, dec=4):
            e = QLineEdit(str(round(val, dec)))
            e.setFixedHeight(_S(34))
            e.setStyleSheet(field_ss)
            return e

        def mk_combo():
            c = QComboBox()
            c.setFixedHeight(_S(34))
            c.setStyleSheet(combo_ss)
            return c

        # ── 1. POWER & INVERTER
        layout.addWidget(self._section_label("POWER & INVERTER"))
        self.e_vdc  = mk(Vdc_init, 1)
        self.e_imax = mk(p_init["Imax"], 1)
        layout.addLayout(self._field_row("Vdc [V]", "DC Link Voltage", self.e_vdc, "DC 링크 배터리 전압"))
        layout.addLayout(self._field_row("Imax [A]", "Max Current", self.e_imax, "인버터 최대 허용 전류 (Peak)"))
        self.combo_alpha = mk_combo()
        for label, _ in self._ALPHA_PRESETS: self.combo_alpha.addItem(label)
        self.combo_alpha.setCurrentIndex(0)
        layout.addLayout(self._field_row("Voltage Limit", "Modulation Mode", self.combo_alpha, "전압 변조 방식 및 이용률"))

        layout.addSpacing(_S(12)) # 섹션 간 간격

        # ── 2. MOTOR PARAMETERS
        layout.addWidget(self._section_label("MOTOR PARAMETERS"))
        self.e_psif = mk(p_init["psi_f"], 4)
        self.e_ld   = mk(p_init["Ld"], 4)
        self.e_lq   = mk(p_init["Lq"], 4)
        self.e_pp   = mk(p_init["pole_pairs"], 0)
        layout.addLayout(self._field_row("ψf [Wb]", "PM Flux", self.e_psif, "영구자석 자속 결합량"))
        layout.addLayout(self._field_row("Ld [H]", "d-axis Ind.", self.e_ld, "d축 인덕턴스"))
        layout.addLayout(self._field_row("Lq [H]", "q-axis Ind.", self.e_lq, "q축 인덕턴스"))
        layout.addLayout(self._field_row("Pole Pairs", "Poles", self.e_pp, "모터 극쌍수"))

        layout.addSpacing(_S(12)) # 섹션 간 간격

        # ── 3. OUTPUT SETTINGS
        layout.addWidget(self._section_label("OUTPUT SETTINGS"))
        self.e_rpmmax = mk(p_init.get("rpm_max", 6000.0), 0)
        layout.addLayout(self._field_row("Max Speed", "RPM Limit", self.e_rpmmax, "분석 수행 최대 회전수 [rpm]"))
        self.e_ngrid = mk(p_init.get("n_grid", 50), 0)
        layout.addLayout(self._field_row("Grid Size (N)", "Resolution", self.e_ngrid, "생성할 LUT 배열 해상도 (N x N)"))

        layout.addSpacing(_S(12)) # 섹션 간 간격

        # ── 4. VISUALIZATION
        layout.addWidget(self._section_label("VISUALIZATION"))
        self.combo_mode = mk_combo()
        self.combo_mode.addItems(["2D LUT (동토크)", "1D LUT (최대토크)"])
        self.combo_mode.currentIndexChanged.connect(self._on_ct_changed)
        layout.addLayout(self._field_row("Control Mode", "LUT Logic", self.combo_mode, "전류 맵 생성 방식 선택"))
        self.combo_x = mk_combo()
        self.combo_x.addItems(["Flux Linkage [Wb]", "Speed [rpm]"])
        self.combo_x.setCurrentIndex(1)
        self.combo_x.currentIndexChanged.connect(self._on_xaxis_changed)
        layout.addLayout(self._field_row("X-axis", "Basis Unit", self.combo_x, "그래프 가로축 기준 단위"))

        layout.addStretch()
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

        # ── Footer: GENERATE 버튼
        footer = QWidget()
        footer.setStyleSheet("background:transparent; border-top:none;")
        fl = QVBoxLayout(footer)
        fl.setContentsMargins(_S(20), 0, _S(20), _S(20))
        
        hl_def = QHBoxLayout()
        hl_def.addStretch()
        self.btn_def = QPushButton("↺ Set to default")
        self.btn_def.setCursor(Qt.PointingHandCursor)
        self.btn_def.setStyleSheet(
            f"QPushButton {{ background:transparent; color:#78909C; font-size:{_FS(9)}px; font-weight:bold; border:none; text-decoration:underline; }}"
            "QPushButton:hover { color:#1A237E; }"
        )
        self.btn_def.clicked.connect(self._on_default_clicked)
        hl_def.addWidget(self.btn_def)
        fl.addLayout(hl_def)
        
        self.btn = QPushButton("GENERATE LUT")
        self.btn.setFixedHeight(_S(48))
        self.btn.setStyleSheet(
            f"QPushButton {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1A237E, stop:1 #3949AB); color:#FFFFFF;"
            f" border-radius:{_S(6)}px; font-weight:bold; font-size:{_FS(11.5)}px;"
            f" border:none; letter-spacing:1.5px; }}"
            "QPushButton:hover { background:#0D47A1; }"
            "QPushButton:pressed { background:#002171; padding-top: 3px; padding-left: 3px; }"
            "QPushButton:disabled { background:#B0BEC5; color:#ECEFF1; }"
        )
        self.btn.clicked.connect(self._on_click)
        fl.addWidget(self.btn)
        outer.addWidget(footer)

    def _on_ct_changed(self):
        self.mode_changed.emit()

    def _on_xaxis_changed(self):
        self.xaxis_changed.emit(self.combo_x.currentText())

    def _on_default_clicked(self):
        self.e_vdc.setText("48.0")
        self.e_imax.setText("20.0")
        self.combo_alpha.setCurrentIndex(0)
        self.e_psif.setText("0.01")
        self.e_ld.setText("0.004")
        self.e_lq.setText("0.008")
        self.e_pp.setText("4")
        self.e_rpmmax.setText("4000")
        self.e_ngrid.setText("20")
        self.combo_mode.setCurrentIndex(0)
        self.combo_x.setCurrentIndex(1)

    def get_xaxis_mode(self):
        return self.combo_x.currentText()

    def is_const_torque(self):
        return self.combo_mode.currentIndex() == 0

    def get_alpha(self):
        return self._ALPHA_PRESETS[self.combo_alpha.currentIndex()][1]

    def _on_click(self):
        try:
            p = {
                "pole_pairs": int(self.e_pp.text()),
                "Ld":         float(self.e_ld.text()),
                "Lq":         float(self.e_lq.text()),
                "psi_f":      float(self.e_psif.text()),
                "Imax":       float(self.e_imax.text()),
                "alpha":      self.get_alpha(),
                "rpm_max":    float(self.e_rpmmax.text()),
                "n_grid":     int(self.e_ngrid.text()),
            }
            Vdc = float(self.e_vdc.text())
            p["Vdc"] = Vdc
        except ValueError:
            QMessageBox.warning(self, "입력 오류", "잘못된 입력값이 있습니다.")
            return
        self.btn.setEnabled(False)
        self.rebuild_requested.emit(p, Vdc)

    def on_done(self):
        self.btn.setEnabled(True)


# =========================
# Main window
# =========================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IPMM LUT 툴")
        self.resize(_S(1200), _S(740))

        self.p_  = {"pole_pairs": 4, "Ld": 0.004, "Lq": 0.008,
                    "psi_f": 0.01, "Imax": 20.0, "alpha": 0.5, "rpm_max": 4000.0, "n_grid": 20}
        self.Vdc = 48.0
        self.lut_data = None
        self._worker  = None

        self._build_ui()

    def _build_ui(self):
        self.setStyleSheet(f"""
            * {{ font-family: "Segoe UI", "NanumSquare"; }}
            QMainWindow {{ background-color: #ECEFF1; }}
            QLabel {{ font-size: {_FS(10.5)}px; color: #263238; }}
            QComboBox, QLineEdit {{
                border: 1px solid #CFD8DC; border-radius: {_S(6)}px; padding: {_S(6)}px {_S(10)}px;
                background-color: #FFFFFF; font-size: {_FS(10.5)}px; color: #263238;
            }}
            QComboBox:hover, QLineEdit:hover {{ border-color: #B0BEC5; }}
            QComboBox:focus, QLineEdit:focus {{ border-color: #1A237E; background-color: #FFFFFF; }}
            QComboBox::drop-down {{ border: none; width: {_S(22)}px; }}

            QTabWidget::pane {{
                border: 1px solid #CFD8DC; background: white;
                border-radius: {_S(12)}px;
                margin-top: -1px;
            }}
            QTabWidget::tab-bar {{ left: {_S(10)}px; }}
            QTabBar::tab {{
                background: #CFD8DC;
                border: 1px solid #B0BEC5;
                border-bottom: none;
                border-top-left-radius: {_S(8)}px;
                border-top-right-radius: {_S(8)}px;
                padding: {_S(5)}px {_S(18)}px;
                min-width: {_S(90)}px;
                font-size: {_FS(9)}px;
                color: #546E7A;
                font-weight: bold;
                letter-spacing: 0.5px;
                margin-right: {_S(4)}px;
                margin-top: {_S(4)}px;
            }}
            QTabBar::tab:selected {{
                background: white;
                color: #1A237E;
                font-weight: bold;
                border: 1px solid #CFD8DC;
                border-bottom: 2px solid white;
                margin-top: 0px;
                padding-bottom: {_S(8)}px;
            }}
            QTabBar::tab:hover:!selected {{ 
                background: #B0BEC5; 
                color: #263238; 
            }}
        """)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # ---- Left: param panel ----
        self.param_panel = ParamPanel(self.p_, self.Vdc)
        self.param_panel.rebuild_requested.connect(self._on_rebuild_requested)
        self.param_panel.mode_changed.connect(self._on_mode_changed)
        self.param_panel.xaxis_changed.connect(self._on_xaxis_changed)
        root.addWidget(self.param_panel)

        # ---- Right: tab widget ----
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.setEnabled(False)   # disabled until LUT built

        btn_help = QPushButton("❓ 설명")
        btn_help.setCursor(Qt.PointingHandCursor)
        btn_help.setFixedHeight(_S(26))
        btn_help.setStyleSheet(
            f"QPushButton {{ background:#1A237E; color:white; border-radius:{_S(13)}px;"
            f" font-size:{_FS(10)}px; padding: 0 {_S(16)}px; margin-right: {_S(10)}px; margin-bottom: {_S(6)}px; border:none; font-weight:bold; }}"
            "QPushButton:hover { background:#3949AB; }"
        )
        btn_help.clicked.connect(self._show_help)
        self.tabs.setCornerWidget(btn_help, Qt.TopRightCorner)
        self.btn_help = btn_help
        self.tabs.currentChanged.connect(self._on_tab_changed)
        
        # Initial call
        self._on_tab_changed(0)


        self.tmax_tab = TmaxTab()
        self.lut_tab  = LutTab()
        self.lut_tab.lut_imported.connect(self._on_lut_imported)
        self.sim_tab  = SimTab()

        self.tabs.addTab(self.tmax_tab, "GRAPH")
        self.tabs.addTab(self.lut_tab,  "LUT")
        self.tabs.addTab(self.sim_tab,  "SIMULATION")
        self.sim_tab.sim_changed.connect(self._refresh_sim)
        root.addWidget(self.tabs, stretch=1)

        # Overlay for tabs — parented to central widget, covers tabs area
        self.overlay = TabOverlay(central)
        self.overlay.show_idle()
        # Ensure correct initial position after layout has settled
        central.layout().activate()
        self.overlay.update_geometry(self.tabs.geometry())

    def showEvent(self, event):
        """Force overlay position once the window is actually shown."""
        super().showEvent(event)
        if hasattr(self, 'overlay') and hasattr(self, 'tabs'):
            self.overlay.update_geometry(self.tabs.geometry())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Position the overlay exactly over the tabs area
        # We use mapToParent to get the correct coordinates within 'central'
        if hasattr(self, 'overlay') and hasattr(self, 'tabs'):
            self.overlay.update_geometry(self.tabs.geometry())

    def _on_xaxis_changed(self, mode):
        self.tmax_tab.set_xaxis_mode(mode)
        self.lut_tab.set_xaxis_mode(mode)

    # ------------------------------------------------------------------
    def _on_rebuild_requested(self, p_new, Vdc_new):
        self.p_  = p_new
        self.Vdc = Vdc_new
        self.tabs.setEnabled(False)
        
        # Show "Calculating" overlay
        self.overlay.show_calculating()
        self.overlay.update_geometry(self.tabs.geometry())

        self._worker = LutWorker(self.p_)
        self._worker.progress.connect(self.overlay.set_progress)
        self._worker.finished.connect(self._on_lut_done)
        self._worker.start()

    def _on_lut_done(self, data):
        self.lut_data = data
        self.param_panel.on_done()
        self.tabs.setEnabled(True)
        
        # Hide overlay
        self.overlay.hide()

        self.tmax_tab.update_plot(data["lam_grid"], data["Tmax_LUT"], p_=self.p_, Vdc=self.Vdc)
        self.lut_tab.update_plot(data["lam_grid"], data["Tratio_grid"],
                                 data["Id_LUT_2D"], data["Iq_LUT_2D"], p_=self.p_, Vdc=self.Vdc)
        self.sim_tab.init_bg(self.p_, data["Id_at_Tmax"], data["Iq_at_Tmax"])
        self.sim_tab.set_max_rpm(self.p_["rpm_max"])
        self._refresh_sim()

    def _on_lut_imported(self, data):
        """Called when a LUT file is imported from the LUT tab."""
        lam_grid = data["lam_grid"]
        Tratio_grid = data["Tratio_grid"]
        Id_2D = data["Id_LUT_2D"]
        Iq_2D = data["Iq_LUT_2D"]
        
        # Reconstruct missing parts: Tmax_LUT is basically the last column (ratio=1.0)
        # but since our Tratio_grid might be slightly different, we'll take the max or last.
        # Here we assume the last index in Tratio_grid corresponds to the max torque point.
        Tmax_LUT = np.array([Te(Id_2D[i, -1], Iq_2D[i, -1], self.p_) for i in range(len(lam_grid))])
        Id_at_Tmax = Id_2D[:, -1]
        Iq_at_Tmax = Iq_2D[:, -1]
        
        # Build interpolators
        interp_id = RegularGridInterpolator(
            (lam_grid, Tratio_grid), Id_2D,
            method='linear', bounds_error=False, fill_value=None)
        interp_iq = RegularGridInterpolator(
            (lam_grid, Tratio_grid), Iq_2D,
            method='linear', bounds_error=False, fill_value=None)
            
        full_data = {
            **data,
            "Tmax_LUT": Tmax_LUT,
            "Id_at_Tmax": Id_at_Tmax,
            "Iq_at_Tmax": Iq_at_Tmax,
            "interp_id": interp_id,
            "interp_iq": interp_iq,
        }
        
        self.lut_data = full_data
        self.tabs.setEnabled(True)
        self.overlay.hide()
        
        # Update all tabs
        self.tmax_tab.update_plot(lam_grid, Tmax_LUT, p_=self.p_, Vdc=self.Vdc)
        self.lut_tab.update_plot(lam_grid, Tratio_grid, Id_2D, Iq_2D, p_=self.p_, Vdc=self.Vdc)
        self.sim_tab.init_bg(self.p_, Id_at_Tmax, Iq_at_Tmax)
        self.sim_tab.set_max_rpm(self.p_["rpm_max"])
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
                                     d["Id_LUT_2D"], d["Iq_LUT_2D"], p_=self.p_, Vdc=self.Vdc)
        else:
            self.lut_tab.update_trajectory(d["lam_grid"],
                                           d["Id_at_Tmax"], d["Iq_at_Tmax"], p_=self.p_, Vdc=self.Vdc)
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

    def _on_tab_changed(self, index):
        """Update help button color based on tab theme."""
        themes = [
            ("#1A237E", "#3949AB"), # Blue
            ("#2E7D32", "#43A047"), # Green
            ("#455A64", "#607D8B"), # Slate
        ]
        color, hover = themes[index] if index < len(themes) else themes[0]
        self.btn_help.setStyleSheet(
            f"QPushButton {{ background:{color}; color:white; border-radius:{_S(13)}px;"
            f" font-size:{_FS(10)}px; padding: 0 {_S(16)}px; margin-right: {_S(10)}px; margin-bottom: {_S(6)}px; border:none; font-weight:bold; }}"
            f"QPushButton:hover {{ background:{hover}; }}"
        )

    def _show_help(self):
        idx = self.tabs.currentIndex()
        if idx == 0:
            dlg = FormulaHelpDialog(self)
        elif idx == 1:
            dlg = LutHelpDialog(self)
        else:
            dlg = SimHelpDialog(self)
        dlg.exec_()


# =========================
# Entry point
# =========================
def main():
    global _DPI_SCALE
    app = QApplication(sys.argv)

    # Compute DPI scale factor relative to 96 dpi baseline
    screen = app.primaryScreen()
    logical_dpi = screen.logicalDotsPerInch()
    _DPI_SCALE = max(0.75, min(logical_dpi / 96.0, 3.0)) * 1.15  # 전체 크기 15% 확대

    app.setStyle("Fusion")
    
    # 영문 Segoe UI, 한글 NanumSquare 조합을 위해 폰트 설정 최적화
    # QFontInfo를 통한 체크보다 스타일시트 우선순위 활용
    app.setFont(QFont("Segoe UI", _FS(10)))
    QFont.insertSubstitution("Segoe UI", "NanumSquare")

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

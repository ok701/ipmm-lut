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

from PyQt5.QtGui import QIcon

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QTabWidget, QLabel, QLineEdit, QPushButton, QFormLayout,
    QSizePolicy, QGroupBox, QSlider, QFrame, QProgressBar, QCheckBox,
    QComboBox, QDialog, QScrollArea, QDialogButtonBox,
    QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem,
    QButtonGroup, QStackedWidget
)

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QCursor, QFontInfo, QFontDatabase, QPainter, QColor, QPen
import re

import os
# ----- High-DPI support -----
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
os.environ["QT_SCALE_FACTOR"] = "1"
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

# Scale helpers (Imported from core)
from core.utils import _S, _FS, _ui_font
import core.utils as utils


# -------------------------
# UI helpers
# -------------------------


# -------------------------
# Motor model (Imported from core)
# -------------------------
from core.motor_model import (
    Te, lam_d, lam_q, lam_mag, part1_lambda_max_ff,
    solve_Tmax_for_lammax, build_part3_LUT,
    solve_min_current_for_T_lam, solve_zero_torque_point_for_lam
)


# -------------------------
# Background worker (Imported from core)
# -------------------------
from core.worker import LutWorker


# -------------------------
# UI Components & Tabs (Imported from ui)
# -------------------------
from desktop.components.overlays import TabOverlay
from desktop.components.canvas import MplCanvas
from desktop.components.sliders import CircularSlider
from desktop.dialogs.help_dialogs import (
    FormulaHelpDialog, LutHelpDialog, SimHelpDialog
)
# -------------------------
# UI Panels & Tabs (Imported from desktop)
# -------------------------
from desktop.panels.param_panel import ParamPanel
from desktop.tabs.tmax_tab import TmaxTab
from desktop.tabs.lut_tab import LutTab
from desktop.tabs.sim_tab import SimTab


# =========================
# Main window
# =========================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IPMM LUT Generator")
        self.setWindowIcon(QIcon("icon.ico"))
        self.resize(_S(1200), _S(740))

        self.p_  = {"pole_pairs": 4, "Ld": 0.004, "Lq": 0.008,
                    "psi_f": 0.01, "Imax": 20.0, "alpha": 0.5, "rpm_max": 4000.0, "n_grid": 20}
        self.Vdc = 48.0
        self.lut_data = None
        self._worker  = None

        self._build_ui()

    def _build_ui(self):
        self.setStyleSheet(f"""

            QMainWindow {{ background-color: #ECEFF1; }}
            QLabel {{ font-size: {_ui_font(11)}px; color: #263238; }}
            QComboBox, QLineEdit {{
                border: 1px solid #CFD8DC; border-radius: {_S(6)}px; padding: {_S(6)}px {_S(10)}px;
                background-color: #FFFFFF; font-size: {_ui_font(11)}px; color: #263238;
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
                padding: {_S(4)}px {_S(14)}px;
                min-width: {_S(78)}px;
                font-size: {_ui_font(11)}px;
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
                padding-bottom: {_S(6)}px;
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

        btn_help = QPushButton("?")
        btn_help.setCursor(Qt.PointingHandCursor)
        btn_help.setFixedSize(_S(44), _S(44))
        btn_help.setMinimumSize(_S(44), _S(44))
        btn_help.setMaximumSize(_S(44), _S(44))
        btn_help.setStyleSheet(
            f"QPushButton {{"
            f" background: #111111; color:white; border-radius:{_S(22)}px;"
            f" min-width:{_S(44)}px; max-width:{_S(44)}px; min-height:{_S(44)}px; max-height:{_S(44)}px;"
            f" font-size:{_FS(16)}px; padding: 0px; margin: 0px;"
            f" border: 1px solid #111111; font-weight:bold; }}"
            "QPushButton:hover { background: #333333; }"
            "QPushButton:pressed { background: #000000; }"
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
        finite_tmax = np.isfinite(data["Tmax_LUT"])
        if finite_tmax.any():
            self.sim_tab.set_max_tref(float(np.nanmax(data["Tmax_LUT"][finite_tmax])) * 2.0)
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
        finite_tmax = np.isfinite(Tmax_LUT)
        if finite_tmax.any():
            self.sim_tab.set_max_tref(float(np.nanmax(Tmax_LUT[finite_tmax])) * 2.0)
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
        HELP_SIZE = _S(30)
        HELP_RADIUS = _S(15)

        self.btn_help.setFixedSize(HELP_SIZE, HELP_SIZE)
        self.btn_help.setMinimumSize(HELP_SIZE, HELP_SIZE)
        self.btn_help.setMaximumSize(HELP_SIZE, HELP_SIZE)

        self.btn_help.setStyleSheet(
            f"QPushButton {{"
            f" background:{color}; color:white;"
            f" border-radius:{HELP_RADIUS}px;"
            f" min-width:{HELP_SIZE}px; max-width:{HELP_SIZE}px;"
            f" min-height:{HELP_SIZE}px; max-height:{HELP_SIZE}px;"
            f" font-size:{_FS(16)}px;"
            f" padding: 0px;"
            f" margin-right:{_S(10)}px;"
            f" margin-bottom:{_S(4)}px;"
            f" border:none;"
            f" font-weight:bold;"
            f" }}"
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
    app = QApplication(sys.argv)
    # Compute DPI scale factor relative to 96 dpi baseline
    screen = app.primaryScreen()
    logical_dpi = screen.logicalDotsPerInch()
    utils._DPI_SCALE = max(0.75, min(logical_dpi / 96.0, 3.0)) * 1.15  # 전체 크기 15% 확대

    app.setStyle("Fusion")

    # 기본 앱 폰트 (영문 우선: Inter) — Pretendard/Noto Sans KR will act as Hangul fallbacks
    app.setFont(QFont("Inter", _ui_font(11)))

    # Global font-family stack (applies to QWidget equivalents: body, buttons, inputs, textareas, selects)
    app.setStyleSheet(
        """
        QWidget, QLabel, QPushButton, QLineEdit, QTextEdit, QPlainTextEdit,
        QComboBox, QTableWidget, QMenu, QMenuBar, QStatusBar {
            font-family: "Inter", "Pretendard", "Noto Sans KR", system-ui, sans-serif;
        }
        """
    )

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

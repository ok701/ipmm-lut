import numpy as np
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QGroupBox, QSlider
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from matplotlib.figure import Figure
from ui.components.canvas import MplCanvas
from ui.components.sliders import CircularSlider
from core.utils import _S, _FS, _ui_font

class SimTab(QWidget):
    """시뮬레이션 탭"""
    sim_changed = pyqtSignal()
    _TREF_STEP_NM = 0.1

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(6)

        # ---- controls (left of sim tab) ----
        ctrl = QWidget()
        ctrl.setFixedWidth(_S(260))
        vl = QVBoxLayout(ctrl)
        vl.setContentsMargins(_S(6), _S(20), _S(20), _S(20))
        vl.setSpacing(_S(16))
        
        vl.addStretch()

        # 1) Operating Point
        res_box = QWidget()
        res_box.setStyleSheet(f"""
            QWidget {{ background: #F7F7F7; border: 1px solid #BDBDBD; border-radius: {_S(10)}px; }}
            QLabel {{ background: transparent; border: none; font-size: {_ui_font(11)}px; color: #111111; }}
        """)
        rl = QVBoxLayout(res_box)
        rl.setContentsMargins(_S(12), _S(11), _S(12), _S(11))
        rl.setSpacing(_S(7))

        self.lbl_id_op  = QLabel("—")
        self.lbl_iq_op  = QLabel("—")
        self.lbl_te_op  = QLabel("—")
        self.lbl_lam_op = QLabel("—")
        self.lbl_tmax   = QLabel("—")
        
        def add_res_row(label, val_lbl):
            row = QHBoxLayout()
            lbl_name = QLabel(label)
            lbl_name.setStyleSheet(f"color: #111111; font-size: {_ui_font(11)}px; font-weight: 700;")
            row.addWidget(lbl_name)
            row.addStretch()
            val_lbl.setFont(QFont("Consolas", _ui_font(11), QFont.Bold))
            val_lbl.setStyleSheet("color:#111111;")
            row.addWidget(val_lbl)
            rl.addLayout(row)

        add_res_row("id* [A]",    self.lbl_id_op)
        add_res_row("iq* [A]",    self.lbl_iq_op)
        add_res_row("Te [Nm]",    self.lbl_te_op)
        add_res_row("λ_max [Wb]", self.lbl_lam_op)
        add_res_row("Tmax [Nm]",  self.lbl_tmax)
        vl.addWidget(res_box)

        vl.addSpacing(_S(12))

        # 2) 슬라이더 행
        slider_row = QHBoxLayout()
        slider_row.setSpacing(_S(12))

        # RPM Slider Group
        grp_rpm = QGroupBox()
        grp_rpm.setStyleSheet(f"QGroupBox {{ color:#111111; font-size:{_ui_font(11)}px; font-weight:bold; border:1px solid #BDBDBD; border-radius:{_S(10)}px; }}")
        gl = QVBoxLayout(grp_rpm)
        gl.setContentsMargins(_S(8), _S(8), _S(8), _S(8))
        self.lbl_rpm = QLabel("3000\nSpeed")
        self.lbl_rpm.setAlignment(Qt.AlignCenter)
        self.lbl_rpm.setStyleSheet(f"font-weight:bold; font-size:{_ui_font(11)}px; color:#111111;")
        self.sl_rpm = CircularSlider(Qt.Vertical)
        self.sl_rpm.setRange(0, 6000)
        self.sl_rpm.setValue(3000)
        self.sl_rpm.setFixedWidth(_S(42))
        self.sl_rpm.setFixedHeight(_S(200))
        self.sl_rpm.setTickInterval(1000)
        self.sl_rpm.setTickPosition(QSlider.TicksLeft)
        self.sl_rpm.valueChanged.connect(self._on_change)
        gl.addWidget(self.lbl_rpm)
        gl.addSpacing(_S(4))
        gl.addWidget(self.sl_rpm, 0, Qt.AlignCenter)
        slider_row.addWidget(grp_rpm)

        # Torque Slider Group
        grp_t = QGroupBox()
        grp_t.setStyleSheet(f"QGroupBox {{ color:#111111; font-size:{_ui_font(11)}px; font-weight:bold; border:1px solid #BDBDBD; border-radius:{_S(10)}px; }}")
        gl2 = QVBoxLayout(grp_t)
        gl2.setContentsMargins(_S(8), _S(8), _S(8), _S(8))
        self.lbl_tref = QLabel("2.0\nNm")
        self.lbl_tref.setAlignment(Qt.AlignCenter)
        self.lbl_tref.setStyleSheet(f"font-weight:bold; font-size:{_ui_font(11)}px; color:#111111;")
        self.sl_tref = CircularSlider(Qt.Vertical)
        self.sl_tref.setRange(0, 500)
        self.sl_tref.setValue(20)
        self.sl_tref.setFixedWidth(_S(42))
        self.sl_tref.setFixedHeight(_S(200))
        self.sl_tref.setTickInterval(100)
        self.sl_tref.setTickPosition(QSlider.TicksRight)
        self.sl_tref.valueChanged.connect(self._on_change)
        gl2.addWidget(self.lbl_tref)
        gl2.addSpacing(_S(4))
        gl2.addWidget(self.sl_tref, 0, Qt.AlignCenter)
        slider_row.addWidget(grp_t)

        vl.addLayout(slider_row)
        vl.addStretch()

        # ---- plot ----
        fig = Figure(tight_layout=False)
        self.ax = fig.add_axes([0.12, 0.10, 0.82, 0.82])
        self.canvas = MplCanvas(fig)
        self._fig = fig
        self._bg_artists = []
        self._dyn_artists = []
        self._ID_bg = self._IQ_bg = self._LAM_bg = self._TE_bg = None
        root.addWidget(self.canvas, stretch=1)
        root.addWidget(ctrl)

    def _on_change(self):
        self.lbl_rpm.setText(f"{self.sl_rpm.value()}\nrpm")
        self.lbl_tref.setText(f"{self.sl_tref.value() * 0.1:.1f}\nNm")
        self.sim_changed.emit()

    def get_rpm(self):
        return float(self.sl_rpm.value())

    def get_tref(self):
        return self.sl_tref.value() * self._TREF_STEP_NM

    def set_max_tref(self, tref_max_nm):
        tref_max_nm = max(0.0, float(tref_max_nm))
        max_steps = max(1, int(round(tref_max_nm / self._TREF_STEP_NM)))
        was_blocked = self.sl_tref.blockSignals(True)
        try:
            self.sl_tref.setRange(0, max_steps)
            self.sl_tref.setTickInterval(max(1, max_steps // 6))
            if self.sl_tref.value() > max_steps:
                self.sl_tref.setValue(max_steps)
        finally:
            self.sl_tref.blockSignals(was_blocked)
        self.lbl_tref.setText(f"{self.sl_tref.value() * self._TREF_STEP_NM:.1f}\nNm")

    def set_max_rpm(self, rpm_max):
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

        cf = self.ax.contourf(self._ID_bg, self._IQ_bg, I_bg,
                               levels=[0, p_["Imax"]], colors=['#AED6F1'], alpha=0.3)
        ct = self.ax.contour(self._ID_bg, self._IQ_bg, I_bg,
                     levels=[p_["Imax"]], colors=['#2E86C1'], linewidths=0.9)
        
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

        lm, = self.ax.plot(id_mtpa[v], iq_mtpa[v], 'orange', lw=0.9, label="MTPA")
        
        v_t = np.isfinite(Id_at_Tmax) & np.isfinite(Iq_at_Tmax)
        lt, = self.ax.plot(Id_at_Tmax[v_t], Iq_at_Tmax[v_t],
                           'purple', lw=0.9, label="Field-Weakening & MTPV")

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
                             levels=[lam_ref], colors=['#1E8449'], linewidths=0.9)
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

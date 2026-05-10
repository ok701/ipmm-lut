import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from desktop.components.canvas import MplCanvas
from core.utils import _S, _FS

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
        val_unit_ss = f"font-size:{_FS(11)}px; font-weight:500; color:#607D8B; border:none;"
        lbl_ss  = f"font-size:{_FS(11)}px; color:#546E7A; font-weight:bold; border:none; text-transform:uppercase; letter-spacing:0.5px;"

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
        
        # 그래프 스타일
        for ax in [self.ax_t, self.ax_p]:
            ax.set_facecolor('#FFFFFF')
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

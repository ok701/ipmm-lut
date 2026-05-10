import json
import csv
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QStackedWidget, QTableWidget, QTableWidgetItem, QFileDialog,
    QMessageBox, QButtonGroup
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from matplotlib.figure import Figure
from desktop.components.canvas import MplCanvas
from core.utils import _S, _FS, _ui_font

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
            QPushButton#segBtnLeft, QPushButton#segBtnRight {{
                font-size: {_ui_font(11)}px;
                padding-top: {_S(4)}px;
                padding-bottom: {_S(4)}px;
            }}
        """)
        
        for idx, b in enumerate([self.btn_3d, self.btn_tbl]):
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setFixedHeight(_S(32))
            b.setFixedWidth(_S(150))
            b.setFont(QFont("Inter", _ui_font(11), QFont.Bold))
            self.btn_group.addButton(b, idx)
            seg_layout.addWidget(b)
        
        self.btn_3d.setChecked(True)
        self.btn_group.buttonClicked[int].connect(self._on_stack_change)
        layout.addLayout(seg_layout)

        self.stack = QStackedWidget()
        layout.addWidget(self.stack)

        # 1) 3D Plot Container
        self.plot_container = QWidget()
        pl = QVBoxLayout(self.plot_container)
        fig = Figure() # Remove tight_layout to prevent jumping
        fig.patch.set_facecolor('#F8F9FA')
        # Use subplots_adjust for stable, independent spacing
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.95, wspace=0.1)
        
        self.ax_id = fig.add_subplot(121, projection='3d')
        self.ax_iq = fig.add_subplot(122, projection='3d')
        self.canvas = MplCanvas(fig)
        pl.addWidget(self.canvas)
        self.stack.addWidget(self.plot_container)

        # 2) Table Container
        self.table_container = QWidget()
        tl = QVBoxLayout(self.table_container)
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet(f"""
            QTableWidget {{ 
                background-color: white; border: 1px solid #CFD8DC; gridline-color: #ECEFF1;
                font-size: {_ui_font(10)}px;
            }}
            QHeaderView::section {{
                background-color: #F8F9FA; padding: 4px; border: 1px solid #CFD8DC;
                font-weight: bold; color: #455A64;
            }}
        """)
        tl.addWidget(self.table)
        
        btn_row = QHBoxLayout()
        self.btn_export = QPushButton("CSV로 내보내기")
        self.btn_import = QPushButton("JSON 불러오기")
        for b in [self.btn_export, self.btn_import]:
            b.setCursor(Qt.PointingHandCursor)
            b.setFixedHeight(_S(36))
            b.setStyleSheet(f"""
                QPushButton {{
                    background: #FFFFFF; border: 1.5px solid #1A237E; color: #1A237E;
                    border-radius: {_S(6)}px; font-weight: bold; font-size: {_FS(11)}px;
                    padding: 0 {_S(16)}px;
                }}
                QPushButton:hover {{ background: #E8EAF6; }}
            """)
        self.btn_export.clicked.connect(self._export_csv)
        self.btn_import.clicked.connect(self._import_json)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_import)
        btn_row.addWidget(self.btn_export)
        tl.addLayout(btn_row)
        
        self.stack.addWidget(self.table_container)

        self.lam_grid = None
        self.Tratio_grid = None
        self.Id_LUT_2D = None
        self.Iq_LUT_2D = None
        self.Id_at_Tmax = None
        self.Iq_at_Tmax = None

    def _on_stack_change(self, idx):
        self.stack.setCurrentIndex(idx)

    def _init_3d_axes(self):
        self.ax_id.clear()
        self.ax_iq.clear()
        for ax in [self.ax_id, self.ax_iq]:
            ax.set_facecolor('#F8F9FA')
            ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False

    def update_plot(self, lam_grid, Tratio_grid, Id_2D, Iq_2D, p_=None, Vdc=None):
        self.lam_grid    = lam_grid
        self.Tratio_grid = Tratio_grid
        self.Id_LUT_2D   = Id_2D
        self.Iq_LUT_2D   = Iq_2D
        if p_ is not None: self.p_ = p_
        if Vdc is not None: self.Vdc = Vdc

        self._redraw_3d()
        self._update_table()

    def set_xaxis_mode(self, mode):
        self.mode = mode
        self._redraw_3d()
        self._update_table()

    def _redraw_3d(self):
        if self.lam_grid is None: return
        self._init_3d_axes()
        
        y_label = "lam_max [Wb]"
        y_data = self.lam_grid
        
        if "Speed" in self.mode and self.p_ is not None and self.Vdc is not None:
            Vmax = self.p_["alpha"] * self.Vdc
            pp = self.p_["pole_pairs"]
            y_data = (Vmax / (np.maximum(self.lam_grid, 1e-9) * pp)) * 60 / (2*np.pi)
            y_label = "Speed [rpm]"

        T2d, L2d = np.meshgrid(self.Tratio_grid, y_data)
        
        self.ax_id.plot_surface(T2d, L2d, self.Id_LUT_2D, cmap='coolwarm',
                                edgecolor='none', alpha=0.92)
        self.ax_id.set_title("Id LUT [A]", fontsize=9, pad=10)
        self.ax_id.set_xlabel("T_ratio", fontsize=7)
        self.ax_id.set_ylabel(y_label, fontsize=7)
        self.ax_id.tick_params(labelsize=6)

        self.ax_iq.plot_surface(T2d, L2d, self.Iq_LUT_2D, cmap='viridis',
                                edgecolor='none', alpha=0.92)
        self.ax_iq.set_title("Iq LUT [A]", fontsize=9, pad=10)
        self.ax_iq.set_xlabel("T_ratio", fontsize=7)
        self.ax_iq.set_ylabel(y_label, fontsize=7)
        self.ax_iq.tick_params(labelsize=6)
        
        self.canvas.draw()

    def _update_table(self):
        if self.lam_grid is None: return
        rows = len(self.lam_grid)
        cols = len(self.Tratio_grid)
        self.table.setRowCount(rows * cols)
        self.table.setColumnCount(4)
        
        y_label = "lam_max"
        y_vals = self.lam_grid
        if "Speed" in self.mode and self.p_ is not None and self.Vdc is not None:
            Vmax = self.p_["alpha"] * self.Vdc
            pp = self.p_["pole_pairs"]
            y_vals = (Vmax / (np.maximum(self.lam_grid, 1e-9) * pp)) * 60 / (2*np.pi)
            y_label = "Speed"

        self.table.setHorizontalHeaderLabels([y_label, "T_ratio", "Id [A]", "Iq [A]"])
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                self.table.setItem(idx, 0, QTableWidgetItem(f"{y_vals[i]:.4f}"))
                self.table.setItem(idx, 1, QTableWidgetItem(f"{self.Tratio_grid[j]:.3f}"))
                self.table.setItem(idx, 2, QTableWidgetItem(f"{self.Id_LUT_2D[i,j]:.3f}"))
                self.table.setItem(idx, 3, QTableWidgetItem(f"{self.Iq_LUT_2D[i,j]:.3f}"))
                idx += 1
        self.table.resizeColumnsToContents()

    def _export_csv(self):
        if self.lam_grid is None: return
        path, _ = QFileDialog.getSaveFileName(self, "CSV 내보내기", "motor_lut.csv", "CSV Files (*.csv)")
        if not path: return
        try:
            with open(path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(["lam_max", "T_ratio", "Id", "Iq"])
                for i in range(len(self.lam_grid)):
                    for j in range(len(self.Tratio_grid)):
                        writer.writerow([self.lam_grid[i], self.Tratio_grid[j], self.Id_LUT_2D[i,j], self.Iq_LUT_2D[i,j]])
            QMessageBox.information(self, "알림", "내보내기가 완료되었습니다.")
        except Exception as e:
            QMessageBox.critical(self, "에러", f"파일 저장 중 오류 발생: {e}")

    def _import_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "JSON 불러오기", "", "JSON Files (*.json)")
        if not path: return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                d = json.load(f)
            data = {
                "lam_grid":    np.array(d["lam_grid"]),
                "Tratio_grid": np.array(d["Tratio_grid"]),
                "Id_LUT_2D":   np.array(d["Id_LUT_2D"]),
                "Iq_LUT_2D":   np.array(d["Iq_LUT_2D"]),
                "Tmax_LUT":    np.array(d["Tmax_LUT"]),
                "Id_at_Tmax":  np.array(d["Id_at_Tmax"]),
                "Iq_at_Tmax":  np.array(d["Iq_at_Tmax"]),
            }
            from scipy.interpolate import RegularGridInterpolator
            data["interp_id"] = RegularGridInterpolator(
                (data["lam_grid"], data["Tratio_grid"]), data["Id_LUT_2D"],
                method='linear', bounds_error=False, fill_value=None)
            data["interp_iq"] = RegularGridInterpolator(
                (data["lam_grid"], data["Tratio_grid"]), data["Iq_LUT_2D"],
                method='linear', bounds_error=False, fill_value=None)

            self.update_plot(data["lam_grid"], data["Tratio_grid"],
                             data["Id_LUT_2D"], data["Iq_LUT_2D"])
            # Update trajectories as well
            self.update_trajectory(data["lam_grid"], data["Id_at_Tmax"], data["Iq_at_Tmax"])
            self.lut_imported.emit(data)
            QMessageBox.information(self, "알림", "데이터를 성공적으로 불러왔습니다.")
        except Exception as e:
            QMessageBox.critical(self, "에러", f"파일 로드 중 오류 발생: {e}")

    def update_trajectory(self, lam_grid, Id_at_Tmax, Iq_at_Tmax, p_=None, Vdc=None):
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

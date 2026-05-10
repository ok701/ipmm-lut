from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox,
    QScrollArea, QFrame, QPushButton, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from core.utils import _S, _FS, _ui_font

class ParamPanel(QWidget):
    rebuild_requested = pyqtSignal(dict, float)
    mode_changed = pyqtSignal()
    xaxis_changed = pyqtSignal(str)
    _ALPHA_PRESETS = [
        ("0.5 (SPWM)", 0.5),
        ("0.577 (SVPWM)", 0.577)
    ]

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
            f" font-size:{_FS(11)}px; color:#263238; }}"
            "QLineEdit:focus { border-color:#1A237E; background:#F8FAFF; border-width:1.5px; }"
        )
        combo_ss = (
            f"QComboBox {{ border:1px solid #CFD8DC; border-radius:{_S(6)}px;"
            f" padding:{_S(6)}px {_S(12)}px; background:#FFFFFF;"
            f" font-size:{_FS(11)}px; color:#263238; }}"
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

        layout.addSpacing(_S(12))

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

        layout.addSpacing(_S(12))

        # ── 3. OUTPUT SETTINGS
        layout.addWidget(self._section_label("OUTPUT SETTINGS"))
        self.e_rpmmax = mk(p_init.get("rpm_max", 6000.0), 0)
        layout.addLayout(self._field_row("Max Speed", "RPM Limit", self.e_rpmmax, "분석 수행 최대 회전수 [rpm]"))
        self.e_ngrid = mk(p_init.get("n_grid", 50), 0)
        layout.addLayout(self._field_row("Grid Size (N)", "Resolution", self.e_ngrid, "생성할 LUT 배열 해상도 (N x N)"))

        layout.addSpacing(_S(12))

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
            f"QPushButton {{ background:transparent; color:#78909C; font-size:{_FS(11)}px; font-weight:bold; border:none; text-decoration:underline; }}"
            "QPushButton:hover { color:#1A237E; }"
        )
        self.btn_def.clicked.connect(self._on_default_clicked)
        hl_def.addWidget(self.btn_def)
        fl.addLayout(hl_def)
        
        self.btn = QPushButton("GENERATE LUT")
        self.btn.setFixedHeight(_S(48))
        self.btn.setFont(QFont("Inter", _ui_font(14), QFont.Bold))
        self.btn.setStyleSheet(
            f"QPushButton {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1A237E, stop:1 #3949AB); color:#FFFFFF;"
            f" border-radius:{_S(6)}px; font-weight:bold; font-size:{_FS(14)}px;"
            f" border:none; letter-spacing:1.5px; }}"
            "QPushButton:hover { background:#0D47A1; }"
            "QPushButton:pressed { background:#002171; padding-top: 3px; padding-left: 3px; }"
            "QPushButton:disabled { background:#B0BEC5; color:#ECEFF1; }"
        )
        self.btn.clicked.connect(self._on_click)
        fl.addWidget(self.btn)
        outer.addWidget(footer)

    def _section_label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"font-size:{_FS(11)}px; font-weight:normal; color:#FFFFFF;"
            f" background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1A237E, stop:1 transparent);"
            f" letter-spacing:1px; padding:{_S(5)}px {_S(10)}px; border-radius:{_S(6)}px; margin-bottom:{_S(1)}px;"
        )
        return lbl

    def _field_row(self, label_text, sub_text, widget, tip=""):
        row = QHBoxLayout()
        row.setSpacing(_S(8))
        row.setContentsMargins(0, 0, 0, 0)
        
        lbl_vbox = QVBoxLayout()
        lbl_vbox.setSpacing(0)
        lbl_vbox.setContentsMargins(0, 0, 0, 0)
        
        lbl = QLabel(label_text)
        lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lbl.setStyleSheet(f"font-size:{_FS(11)}px; font-weight:bold; color:#263238;")
        
        sub = QLabel(sub_text)
        sub.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        sub.setStyleSheet(f"font-size:{_FS(11)}px; color:#607D8B;")
        
        lbl_vbox.addWidget(lbl)
        lbl_vbox.addWidget(sub)
        
        lbl_container = QWidget()
        lbl_container.setLayout(lbl_vbox)
        lbl_container.setFixedWidth(_S(100))
        if tip:
            lbl_container.setToolTip(tip)
            
        row.addWidget(lbl_container)
        row.addWidget(widget, 1)
        return row

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

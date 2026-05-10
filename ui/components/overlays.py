from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from core.utils import _S, _FS, _ui_font

class TabOverlay(QWidget):
    """Initial and Calculating overlay for tabs."""
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
        card_layout.setSpacing(_S(12))

        # Main message
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setWordWrap(True)
        self.label.setFont(QFont("", _ui_font(14), QFont.Bold))
        self.label.setStyleSheet(
            f"font-size: {_FS(14)}px; font-weight: bold; color: #1A237E; line-height: 1.5;"
        )
        card_layout.addWidget(self.label)

        # Sub message
        self.sub_lbl = QLabel()
        self.sub_lbl.setAlignment(Qt.AlignCenter)
        self.sub_lbl.setWordWrap(True)
        self.sub_lbl.setStyleSheet(
            f"font-size: {_FS(11)}px; color: #607D8B; font-weight: normal; margin-top: 0px;"
        )
        card_layout.addWidget(self.sub_lbl)

        # Progress container
        self.prog_container = QWidget()
        pc = QVBoxLayout(self.prog_container)
        pc.setContentsMargins(0, 0, 0, 0)
        pc.setSpacing(_S(4))

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
        self.pct_lbl.setFont(QFont("", _ui_font(11), QFont.Bold))
        self.pct_lbl.setStyleSheet(
            f"font-size: {_FS(11)}px; font-weight: bold; color: #1A237E;"
        )
        pc.addWidget(self.pct_lbl)

        card_layout.addWidget(self.prog_container)
        self.prog_container.setVisible(False)

        outer.addWidget(card, 0, Qt.AlignCenter)
        outer.addStretch(3)

    def show_idle(self):
        self.label.setText("GENERATE LUT를 눌러주세요.")
        self.sub_lbl.setText("입력된 조건에 따라 전동기 특성 맵이 생성됩니다.")
        self.prog_container.setVisible(False)
        self.show()
        self.raise_()

    def show_calculating(self):
        self.label.setText("최적화 연산 수행 중...")
        self.sub_lbl.setText("입력된 파라미터를 기반으로 계산하고 있습니다.")
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

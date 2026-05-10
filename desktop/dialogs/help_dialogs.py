from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QWidget, QScrollArea, QFrame, QPushButton
from PyQt5.QtCore import Qt
from core.utils import _S, _FS

class BaseHelpDialog(QDialog):
    """도움말 다이얼로그 베이스 클래스 (미니멀 디자인)"""
    def __init__(self, title, color, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(_S(500), _S(450))
        self.color = color
        
        self.setStyleSheet(f"""
            QDialog {{ background: #FFFFFF; }}
            QLabel#title {{ font-size: {_FS(14)}px; font-weight: bold; color: white; }}
            QLabel#section {{ font-size: {_FS(11)}px; font-weight: bold; color: {color};
                             border-bottom: 2px solid {color}; padding-bottom: 2px;
                             margin-top: 15px; margin-bottom: 5px; }}
            QLabel#body {{ font-size: {_FS(11)}px; color: #37474F; line-height: 1.5; }}
            QLabel#formula {{ font-family: "Inter", "Pretendard", "Noto Sans KR", system-ui, sans-serif; font-size: {_FS(12)}px;
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
        close_btn = QPushButton("닫기")
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setFixedHeight(_S(32))
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background: #F0F2F5; color: #37474F; border: none;
                border-radius: {_S(6)}px; font-weight: bold; font-size: {_FS(11)}px;
                padding: 0 {_S(20)}px;
            }}
            QPushButton:hover {{ background: #E2E6EA; }}
        """)
        close_btn.clicked.connect(self.accept)
        
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(15, 0, 15, 0)
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        
        self.outer.addLayout(btn_layout)

    def add_section(self, title):
        l = QLabel(title); l.setObjectName("section"); self.vl.addWidget(l); return l
    def add_body(self, text):
        l = QLabel(text); l.setObjectName("body"); l.setWordWrap(True); self.vl.addWidget(l); return l
    def add_formula(self, text):
        l = QLabel(text); l.setObjectName("formula"); l.setWordWrap(True); self.vl.addWidget(l); return l

class FormulaHelpDialog(BaseHelpDialog):
    def __init__(self, parent=None):
        super().__init__("수식 설명 — 최대토크 & 출력", "#1A237E", parent)

        self.add_section("1. 전압 제한과 최대 허용 자속")
        self.add_body(
            "회전 속도가 증가하면 역기전력이 커지므로, 인버터 전압 한계 안에서 "
            "허용 가능한 최대 자속 λmax를 먼저 계산합니다. "
            "이 값은 해당 속도에서 전류 벡터가 넘지 말아야 할 자속 제한입니다."
        )
        self.add_formula(
            "<i>V</i><sub>max</sub> = &alpha; &times; <i>V</i><sub>dc</sub><br>"
            "&omega;<sub>mech</sub> = rpm &times; 2&pi; / 60<br>"
            "&omega;<sub>e</sub> = pole_pairs &times; &omega;<sub>mech</sub><br>"
            "&lambda;<sub>max</sub> = <i>V</i><sub>max</sub> / &omega;<sub>e</sub>"
        )

        self.add_section("2. d-q축 자속 계산")
        self.add_body(
            "각 전류 조합 id, iq에 대해 d축 자속과 q축 자속을 계산하고, "
            "두 성분의 벡터 크기를 전체 자속 크기로 사용합니다."
        )
        self.add_formula(
            "&lambda;<sub>d</sub> = &psi;<sub>f</sub> + L<sub>d</sub> i<sub>d</sub><br>"
            "&lambda;<sub>q</sub> = L<sub>q</sub> i<sub>q</sub><br>"
            "|&lambda;| = &radic;(&lambda;<sub>d</sub><sup>2</sup> + "
            "&lambda;<sub>q</sub><sup>2</sup>)"
        )

        self.add_section("3. 전자기 토크 계산")
        self.add_body(
            "토크는 영구자석 토크 성분과 릴럭턴스 토크 성분을 함께 고려해 계산합니다. "
            "따라서 iq만 증가시키는 방식이 아니라, id와 iq의 조합에 따라 토크가 달라집니다."
        )
        self.add_formula(
            "T<sub>e</sub> = 1.5 &times; pole_pairs &times; "
            "[ &psi;<sub>f</sub> i<sub>q</sub> + "
            "(L<sub>d</sub> - L<sub>q</sub>) i<sub>d</sub> i<sub>q</sub> ]"
        )

        self.add_section("4. 최대토크 최적화")
        self.add_body(
            "각 λmax에서 전류 제한과 자속 제한을 동시에 만족하는 id, iq 중 "
            "토크 Te가 가장 큰 지점을 수치 최적화로 찾습니다. "
            "코드에서는 -Te를 최소화하여 Te 최대화 문제를 풉니다."
        )
        self.add_formula(
            "Maximize: T<sub>e</sub>(i<sub>d</sub>, i<sub>q</sub>)<br>"
            "Subject to:<br>"
            "i<sub>d</sub><sup>2</sup> + i<sub>q</sub><sup>2</sup> &le; I<sub>max</sub><sup>2</sup><br>"
            "|&lambda;(i<sub>d</sub>, i<sub>q</sub>)| &le; &lambda;<sub>max</sub><br>"
            "-I<sub>max</sub> &le; i<sub>d</sub> &le; 0, "
            "0 &le; i<sub>q</sub> &le; I<sub>max</sub>"
        )

        self.add_section("5. 여러 초기값을 사용하는 이유")
        self.add_body(
            "비선형 최적화는 시작점에 따라 다른 해에 수렴할 수 있습니다. "
            "따라서 여러 초기 전류 조합에서 최적화를 반복 수행한 뒤, "
            "제약 조건을 만족하면서 가장 큰 토크를 내는 해를 선택합니다."
        )

        self.add_section("6. 최대 출력 계산")
        self.add_body(
            "최대 출력은 각 속도에서 얻은 최대토크와 기계각속도의 곱으로 계산합니다. "
            "그래프에서는 W 값을 kW로 변환해 표시합니다."
        )
        self.add_formula(
            "P<sub>max</sub> = T<sub>max</sub> &times; &omega;<sub>mech</sub><br>"
            "P<sub>max,kW</sub> = P<sub>max</sub> / 1000"
        )

        self.vl.addStretch()

class LutHelpDialog(BaseHelpDialog):
    def __init__(self, parent=None):
        super().__init__("도움말 — LUT 생성 및 제어", "#2E7D32", parent)

        self.add_section("1. LUT의 목적")
        self.add_body(
            "LUT는 주어진 자속 제한 λmax와 토크 비율 T_ratio에 대해 "
            "최적의 d축 전류 id와 q축 전류 iq를 미리 계산해 저장한 표입니다. "
            "실시간 제어에서는 매번 최적화를 수행하지 않고, 이 LUT를 보간하여 "
            "전류 지령을 빠르게 얻습니다."
        )

        self.add_section("2. λmax 그리드 생성")
        self.add_body(
            "먼저 운전 가능한 자속 범위를 여러 개의 λmax 지점으로 나눕니다. "
            "최소 자속은 최고속도 rpm_max에서 필요한 전압 제한을 기준으로 정하고, "
            "최대 자속은 전류 한계 Imax에서 가능한 자속 크기를 기준으로 정합니다."
        )
        self.add_formula(
            "V<sub>max</sub> = &alpha; &times; V<sub>dc</sub><br>"
            "&lambda;<sub>min</sub> &approx; "
            "V<sub>max</sub> / (&omega;<sub>max</sub> &times; pole_pairs)<br>"
            "&lambda;<sub>grid</sub> = linspace(&lambda;<sub>lower</sub>, "
            "&lambda;<sub>upper</sub>, N)"
        )

        self.add_section("3. 각 λmax에서 최대토크 계산")
        self.add_body(
            "각 λmax 지점마다 전류 제한과 자속 제한을 동시에 만족하는 범위에서 "
            "가장 큰 토크를 낼 수 있는 id, iq를 먼저 찾습니다. "
            "이 결과가 Tmax_LUT, Id_at_Tmax, Iq_at_Tmax로 저장됩니다."
        )
        self.add_formula(
            "T<sub>max</sub>(&lambda;<sub>max</sub>) = "
            "max T<sub>e</sub>(i<sub>d</sub>, i<sub>q</sub>)<br>"
            "Subject to:<br>"
            "i<sub>d</sub><sup>2</sup> + i<sub>q</sub><sup>2</sup> "
            "&le; I<sub>max</sub><sup>2</sup><br>"
            "|&lambda;(i<sub>d</sub>, i<sub>q</sub>)| "
            "&le; &lambda;<sub>max</sub>"
        )

        self.add_section("4. 토크 비율 T_ratio 적용")
        self.add_body(
            "각 λmax에서 최대토크를 구한 뒤, 0부터 거의 1까지의 토크 비율 "
            "T_ratio를 곱해 목표토크 Tref를 만듭니다. "
            "즉 LUT의 각 열은 최대토크 대비 몇 퍼센트 토크를 낼 것인지를 의미합니다."
        )
        self.add_formula(
            "T<sub>ratio</sub> = 0.0 ~ 0.999<br>"
            "T<sub>ref</sub> = T<sub>ratio</sub> &times; "
            "T<sub>max</sub>(&lambda;<sub>max</sub>)"
        )

        self.add_section("5. 최소전류 운전점 탐색")
        self.add_body(
            "각 λmax와 T_ratio 조합에 대해, 목표토크 Tref를 정확히 만족하면서 "
            "전류 크기가 가장 작은 id, iq를 찾습니다. "
            "전류 크기를 최소화하면 동손 I²R을 줄이는 운전점이 됩니다."
        )
        self.add_formula(
            "Minimize: i<sub>d</sub><sup>2</sup> + i<sub>q</sub><sup>2</sup><br>"
            "Subject to:<br>"
            "T<sub>e</sub>(i<sub>d</sub>, i<sub>q</sub>) = T<sub>ref</sub><br>"
            "|&lambda;(i<sub>d</sub>, i<sub>q</sub>)| "
            "&le; &lambda;<sub>max</sub><br>"
            "i<sub>d</sub><sup>2</sup> + i<sub>q</sub><sup>2</sup> "
            "&le; I<sub>max</sub><sup>2</sup>"
        )

        self.add_section("6. T_ratio = 0일 때의 처리")
        self.add_body(
            "토크 지령이 0이면 iq는 0으로 두고, 자속 제한을 만족하는 id를 계산합니다. "
            "자속 제한이 충분히 크면 id = 0, iq = 0을 사용합니다. "
            "반대로 λmax가 영구자석 자속보다 작으면 음의 id를 넣어 자속을 낮춥니다."
        )
        self.add_formula(
            "If &lambda;<sub>max</sub> &ge; &psi;<sub>f</sub>: "
            "i<sub>d</sub> = 0, i<sub>q</sub> = 0<br>"
            "If &lambda;<sub>max</sub> &lt; &psi;<sub>f</sub>: "
            "i<sub>d</sub> = "
            "(&lambda;<sub>max</sub> - &psi;<sub>f</sub>) / L<sub>d</sub>, "
            "i<sub>q</sub> = 0"
        )

        self.add_section("7. 2D LUT 저장")
        self.add_body(
            "계산된 최적 전류는 λmax 방향과 T_ratio 방향의 2차원 배열로 저장됩니다. "
            "Id_LUT_2D에는 d축 전류, Iq_LUT_2D에는 q축 전류가 저장됩니다."
        )
        self.add_formula(
            "Id_LUT_2D[&lambda;<sub>index</sub>, T<sub>ratio,index</sub>] "
            "= i<sub>d</sub><sup>*</sup><br>"
            "Iq_LUT_2D[&lambda;<sub>index</sub>, T<sub>ratio,index</sub>] "
            "= i<sub>q</sub><sup>*</sup>"
        )

        self.add_section("8. 보간을 통한 실시간 사용")
        self.add_body(
            "실제 시뮬레이션이나 제어에서는 λmax와 T_ratio가 LUT 격자점과 "
            "정확히 일치하지 않을 수 있습니다. "
            "그래서 RegularGridInterpolator를 사용해 주변 격자값을 선형 보간하고, "
            "현재 운전점에 맞는 id*, iq*를 계산합니다."
        )
        self.add_formula(
            "i<sub>d</sub><sup>*</sup> = interp_id("
            "&lambda;<sub>max</sub>, T<sub>ratio</sub>)<br>"
            "i<sub>q</sub><sup>*</sup> = interp_iq("
            "&lambda;<sub>max</sub>, T<sub>ratio</sub>)"
        )

        self.vl.addStretch()

class SimHelpDialog(BaseHelpDialog):
    def __init__(self, parent=None):
        super().__init__("도움말 — 실시간 시뮬레이션", "#455A64", parent)

        self.add_section("1. 시뮬레이션의 목적")
        self.add_body(
            "시뮬레이션 탭은 사용자가 설정한 회전속도 rpm과 토크 지령 Tref_cmd에 대해 "
            "현재 운전점의 최적 전류 지령 id*, iq*를 계산하고, "
            "전류 제한 영역, 자속 제한 영역, 토크 등고선 위에 운전점을 표시합니다."
        )

        self.add_section("2. 속도에 따른 자속 제한 계산")
        self.add_body(
            "먼저 현재 rpm에서 인버터 전압 한계로부터 허용 가능한 최대 자속 λmax를 계산합니다. "
            "속도가 높아질수록 λmax는 작아지며, 이 값이 현재 운전점의 자속 제한으로 사용됩니다."
        )
        self.add_formula(
            "V<sub>max</sub> = &alpha; &times; V<sub>dc</sub><br>"
            "&omega;<sub>mech</sub> = rpm &times; 2&pi; / 60<br>"
            "&omega;<sub>e</sub> = pole_pairs &times; &omega;<sub>mech</sub><br>"
            "&lambda;<sub>max</sub> = V<sub>max</sub> / &omega;<sub>e</sub>"
        )

        self.add_section("3. 현재 자속 제한에서 가능한 최대토크")
        self.add_body(
            "계산된 λmax를 기준으로 Tmax_LUT를 보간하여 현재 속도에서 낼 수 있는 "
            "최대토크 Tmax를 구합니다. 이 값은 토크 지령을 제한하는 기준이 됩니다."
        )
        self.add_formula(
            "T<sub>max,current</sub> = interp("
            "&lambda;<sub>max</sub>, Tmax_LUT)"
        )

        self.add_section("4. 정토크 모드")
        self.add_body(
            "정토크 모드에서는 사용자가 입력한 토크 지령 Tref_cmd를 사용합니다. "
            "단, 현재 λmax에서 가능한 최대토크를 넘지 않도록 Tref를 제한합니다. "
            "그 뒤 Tref를 Tmax로 나누어 LUT 입력값인 T_ratio를 계산합니다."
        )
        self.add_formula(
            "T<sub>ref</sub> = clip("
            "T<sub>ref_cmd</sub>, 0, 0.999 &times; T<sub>max,current</sub>)<br>"
            "T<sub>ratio</sub> = T<sub>ref</sub> / T<sub>max,current</sub>"
        )

        self.add_section("5. LUT 보간으로 전류 지령 계산")
        self.add_body(
            "정토크 모드에서는 λmax와 T_ratio를 입력으로 사용하여 "
            "Id_LUT_2D, Iq_LUT_2D를 보간합니다. "
            "이렇게 얻은 값이 현재 운전점의 최적 전류 지령 id*, iq*입니다."
        )
        self.add_formula(
            "i<sub>d</sub><sup>*</sup> = interp_id("
            "&lambda;<sub>max</sub>, T<sub>ratio</sub>)<br>"
            "i<sub>q</sub><sup>*</sup> = interp_iq("
            "&lambda;<sub>max</sub>, T<sub>ratio</sub>)"
        )

        self.add_section("6. MTPA / MTPV 모드")
        self.add_body(
            "정토크 모드가 아닐 경우, 시뮬레이션은 항상 현재 λmax에서의 최대토크 운전점을 사용합니다. "
            "이때 id*, iq*는 Id_at_Tmax, Iq_at_Tmax 데이터를 1차원 보간하여 계산합니다."
        )
        self.add_formula(
            "T<sub>ref</sub> = T<sub>max,current</sub><br>"
            "i<sub>d</sub><sup>*</sup> = interp("
            "&lambda;<sub>max</sub>, Id_at_Tmax)<br>"
            "i<sub>q</sub><sup>*</sup> = interp("
            "&lambda;<sub>max</sub>, Iq_at_Tmax)"
        )

        self.add_section("7. 실제 토크 계산 및 표시")
        self.add_body(
            "계산된 id*, iq*를 다시 토크 식에 대입하여 실제 발생 토크 Te를 계산합니다. "
            "결과창에는 id*, iq*, Te, λmax, Tmax가 표시됩니다."
        )
        self.add_formula(
            "T<sub>e</sub> = 1.5 &times; pole_pairs &times; "
            "[ &psi;<sub>f</sub> i<sub>q</sub><sup>*</sup> + "
            "(L<sub>d</sub> - L<sub>q</sub>) "
            "i<sub>d</sub><sup>*</sup> i<sub>q</sub><sup>*</sup> ]"
        )

        self.add_section("8. 그래프에서 표시되는 의미")
        self.add_body(
            "그래프에는 전류 제한 원, MTPA 궤적, 약자속/MTPV 궤적, "
            "현재 자속 제한 영역, 토크 등고선, 그리고 현재 운전점이 함께 표시됩니다. "
            "이를 통해 현재 지령이 전류 제한과 자속 제한 안에서 가능한 운전점인지 확인할 수 있습니다."
        )

        self.vl.addStretch()

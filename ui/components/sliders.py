from PyQt5.QtWidgets import QSlider
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QColor, QPen
from core.utils import _S

class CircularSlider(QSlider):
    """Custom vertical slider with a circular handle."""
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.setAttribute(Qt.WA_Hover, True)
        self.setMouseTracking(True)
        self.setTracking(True)

    def _value_from_pos(self, pos_y):
        minimum = self.minimum()
        maximum = self.maximum()
        if maximum <= minimum:
            return minimum

        top_margin = _S(18)
        bottom_margin = _S(18)
        groove_top = top_margin
        groove_bottom = max(groove_top + 1, self.height() - bottom_margin)
        clamped_y = max(groove_top, min(int(pos_y), groove_bottom))
        ratio = (groove_bottom - clamped_y) / float(groove_bottom - groove_top)
        if self.invertedAppearance():
            ratio = 1.0 - ratio
        return int(round(minimum + ratio * (maximum - minimum)))

    def _handle_center_y(self):
        minimum = self.minimum()
        maximum = self.maximum()
        if maximum == minimum:
            ratio = 0.0
        else:
            ratio = (self.value() - minimum) / float(maximum - minimum)

        top_margin = _S(18)
        bottom_margin = _S(18)
        groove_top = top_margin
        groove_bottom = max(groove_top + 1, self.height() - bottom_margin)
        if self.invertedAppearance():
            return groove_top + ratio * (groove_bottom - groove_top)
        return groove_bottom - ratio * (groove_bottom - groove_top)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        groove_width = _S(8)
        groove_radius = groove_width // 2
        top_margin = _S(18)
        bottom_margin = _S(18)
        handle_radius = _S(14)
        center_x = self.rect().center().x()
        groove_top = top_margin
        groove_bottom = max(groove_top + 1, self.height() - bottom_margin)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#9E9E9E"))
        painter.drawRoundedRect(
            center_x - groove_width // 2,
            groove_top,
            groove_width,
            groove_bottom - groove_top,
            groove_radius,
            groove_radius,
        )

        handle_center_y = self._handle_center_y()

        painter.setBrush(QColor("#1A237E"))
        painter.setPen(QPen(QColor("#1A237E"), 1))
        painter.drawEllipse(center_x - handle_radius, int(handle_center_y) - handle_radius, handle_radius * 2, handle_radius * 2)
        painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setSliderDown(True)
            self.setValue(self._value_from_pos(event.pos().y()))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.setValue(self._value_from_pos(event.pos().y()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setSliderDown(False)
            self.setValue(self._value_from_pos(event.pos().y()))
            event.accept()
            return
        super().mouseReleaseEvent(event)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QSizePolicy

class MplCanvas(FigureCanvas):
    """Matplotlib canvas class for embedding plots in PyQt."""
    def __init__(self, fig):
        super().__init__(fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

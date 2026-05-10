# UI and Scaling Utilities

# Scale helpers — computed once after QApplication is created.
# _S(px)  : scale a pixel value relative to 96 dpi baseline
# _FS(pt) : scale a font point size
_DPI_SCALE = 1.0   # Updated in main() via core.utils._DPI_SCALE

def _S(px: int) -> int:
    """Scale pixel dimension for the current DPI."""
    return max(1, round(px * _DPI_SCALE))

def _FS(pt: int) -> int:
    """Scale font point size for the current DPI."""
    return max(6, round(pt * _DPI_SCALE))

def _ui_font(pt: int) -> int:
    """Centralized UI font scale target."""
    return _FS(pt)

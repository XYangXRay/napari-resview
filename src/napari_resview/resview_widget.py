#!/usr/bin/env python3
"""
ResView (ResviewDockWidget) — napari dock widget with YAML persistence

Fix for: TypeError ... __init__() missing 1 required positional argument: 'viewer'
- viewer is now OPTIONAL in __init__
- if viewer is None, we try napari.current_viewer()
"""

import contextlib
import logging
import os
import pathlib
import re
from typing import Any

import napari
import numpy as np
import yaml
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    FileEdit,
    FloatSpinBox,
    Label,
    LineEdit,
    PushButton,
    SpinBox,
    TextEdit,
)
from napari.utils.notifications import show_error
from napari.viewer import Viewer
from qtpy import QtCore, QtGui, QtWidgets

from .data_io import RSMDataLoader, write_rsm_volume_to_vtr
from .data_viz import IntensityNapariViewer, RSMNapariViewer
from .rsm3d import RSMBuilder

# Initialize basic logging to help debugging (file + console). Adjust path if you want.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# -----------------------------------------------------------------------------
# App styling / icon
# -----------------------------------------------------------------------------

APP_ICON_PATH = (pathlib.Path(__file__).parent / "resview_icon.png").resolve()

APP_QSS = """
QGroupBox {
    border: 1px solid #d9d9d9;
    border-radius: 8px;
    margin-top: 12px;
    background: #ffffff;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 6px 10px;
    color: #2c3e50;
    font-size: 18px;
    font-weight: 700;
    letter-spacing: 0.2px;
}
QLabel { color: #34495e; }
QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QDoubleSpinBox, QSpinBox {
    border: 1px solid #d4d7dd;
    border-radius: 6px;
    padding: 4px 6px;
    background: #ffffff;
}
QPushButton {
    background: #eef2f7;
    border: 1px solid #d4d7dd;
    border-radius: 8px;
    padding: 6px 10px;
    font-weight: 600;
}
QPushButton:hover { background: #e6ebf3; }
QPushButton:pressed { background: #dfe5ee; }
"""

RUN_ALL_QSS = """
QPushButton#RunAllPrimary {
    background: #ff9800;
    color: #ffffff;
    border: 2px solid #e68900;
    border-radius: 10px;
    padding: 14px 20px;
    font-size: 18px;
    font-weight: 800;
}
QPushButton#RunAllPrimary:hover { background: #ffa726; }
QPushButton#RunAllPrimary:pressed { background: #fb8c00; }
"""


def load_app_icon() -> QtGui.QIcon | None:
    if APP_ICON_PATH.is_file():
        icon = QtGui.QIcon(str(APP_ICON_PATH))
        return icon if not icon.isNull() else None
    return None


def try_apply_app_icon() -> QtGui.QIcon | None:
    """Return app icon if available. Do not re-apply stylesheet here — centralize styling."""
    icon = load_app_icon()
    return icon


# -----------------------------------------------------------------------------
# YAML persistence
# -----------------------------------------------------------------------------

DEFAULTS_ENV = "RSM3D_DEFAULTS_YAML"
os.environ.setdefault(
    DEFAULTS_ENV,
    str(pathlib.Path(__file__).with_name("rsm3d_defaults.yaml").resolve()),
)


def yaml_path() -> str:
    p = os.environ.get(DEFAULTS_ENV, "").strip()
    if p:
        return os.path.abspath(os.path.expanduser(p))
    return os.path.join(os.path.expanduser("~"), ".rsm3d_defaults.yaml")


def ensure_yaml(path: str) -> None:
    """Create YAML file if missing. Only create parent dir when it's non-empty."""
    if os.path.isfile(path):
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    seed: dict[str, Any] = {
        "data": {
            "spec_file": None,
            "data_file": None,
            "scans": "",
            "only_hkl": None,
        },
        "ExperimentSetup": {
            "distance": None,
            "pitch": None,
            "ycenter": None,
            "xcenter": None,
            "xpixels": None,
            "ypixels": None,
            "energy": None,
            "wavelength": None,
        },
        "Crystal": {"ub": None},
        "build": {
            "ub_includes_2pi": None,
            "center_is_one_based": None,
            "sample_axes": "",
            "detector_axes": "",
        },
        "crop": {
            "enable": None,
            "y_min": None,
            "y_max": None,
            "x_min": None,
            "x_max": None,
        },
        "regrid": {
            "space": None,
            "grid_shape": "",
            "fuzzy": None,
            "fuzzy_width": None,
            "normalize": None,
        },
        "view": {
            "log_view": None,
            "cmap": None,
            "rendering": None,
            "contrast_lo": None,
            "contrast_hi": None,
        },
        "export": {"vtr_path": None},
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(seed, f, sort_keys=False)
    except Exception as e:
        logging.exception("Failed to create defaults YAML: %s", e)
        # Don't propagate; the app can continue and later show a message if needed.


def load_yaml(path: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        return doc or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as e:
        logging.warning("Failed to load YAML %s: %s", path, e)
        return {}


def save_yaml(path: str, doc: dict[str, Any]) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(doc, f, sort_keys=False)
    except (OSError, UnicodeEncodeError, yaml.YAMLError) as e:
        logging.exception("Failed to write YAML: %s", e)
        show_error(f"Failed to write YAML: {e}")


def as_path_str(v: Any) -> str:
    if v is None:
        return ""
    try:
        return os.fspath(v)
    except TypeError:
        return str(v)


# -----------------------------------------------------------------------------
# Parsing / formatting
# -----------------------------------------------------------------------------


def format_ub_matrix(ub: Any) -> str:
    if ub is None:
        return ""
    try:
        arr = np.asarray(ub, dtype=float)
    except (TypeError, ValueError):
        return str(ub)
    if arr.ndim == 0:
        return f"{arr.item():.6g}"
    if arr.ndim == 1:
        return " ".join(f"{v:.6g}" for v in arr)
    if arr.ndim == 2:
        return "\n".join(" ".join(f"{v:.6g}" for v in row) for row in arr)
    # For unexpected higher dimensions, return a concise repr.
    return np.array2string(arr, precision=6, separator=" ")


def parse_ub_matrix(text: str) -> np.ndarray | None:
    stripped = (text or "").strip()
    if not stripped:
        return None
    rows: list[list[float]] = []
    for line in stripped.splitlines():
        parts = [p for p in re.split(r"[,\s]+", line.strip()) if p]
        if parts:
            rows.append([float(p) for p in parts])
    if not rows:
        return None
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("UB rows must have equal length.")
    return np.array(rows, dtype=float)


def parse_scan_list(text: str) -> list[int]:
    if not text or not text.strip():
        return []
    out: set[int] = set()
    for part in re.split(r"[,\s]+", text.strip()):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = a.strip(), b.strip()
            if a.isdigit() and b.isdigit():
                lo, hi = int(a), int(b)
                if lo > hi:
                    lo, hi = hi, lo
                out.update(range(lo, hi + 1))
            else:
                raise ValueError(f"Bad scan range: '{part}'")
        else:
            if part.isdigit():
                out.add(int(part))
            else:
                raise ValueError(f"Bad scan id: '{part}'")
    return sorted(out)


def parse_axes_list(text: str) -> list[str]:
    if not text:
        return []
    return [p.strip() for p in re.split(r"[,\s]+", text) if p.strip()]


def parse_grid_shape(text: str) -> tuple[int | None, int | None, int | None]:
    if text is None:
        return (None, None, None)
    s = text.strip()
    if not s:
        return (None, None, None)
    parts = [p.strip() for p in s.split(",")]
    if len(parts) == 1:
        parts += ["*", "*"]
    if len(parts) != 3:
        raise ValueError("Grid must be 'x,y,z' (y/z may be '*').")

    def one(p: str) -> int | None:
        if p in ("*", "", None):
            return None
        if not p.isdigit():
            raise ValueError(f"Grid size must be integer or '*', got '{p}'")
        v = int(p)
        if v <= 0:
            raise ValueError("Grid sizes must be > 0")
        return v

    return tuple(one(p) for p in parts)  # type: ignore[return-value]


# -----------------------------------------------------------------------------
# UI helpers
# -----------------------------------------------------------------------------


def hsep(height: int = 10) -> Label:
    w = Label(value="")
    try:
        w.native.setFrameShape(QtWidgets.QFrame.HLine)
        w.native.setFrameShadow(QtWidgets.QFrame.Sunken)
        w.native.setLineWidth(1)
        w.native.setFixedHeight(height)
    except AttributeError:
        pass
    return w


def q_hsep(height: int = 10) -> QtWidgets.QWidget:
    line = QtWidgets.QFrame()
    line.setFrameShape(QtWidgets.QFrame.HLine)
    line.setFrameShadow(QtWidgets.QFrame.Sunken)
    line.setLineWidth(1)
    line.setFixedHeight(height)
    return line


def make_group(
    title: str, inner_widget: QtWidgets.QWidget
) -> QtWidgets.QGroupBox:
    box = QtWidgets.QGroupBox(title)
    lay = QtWidgets.QVBoxLayout(box)
    lay.setContentsMargins(12, 12, 12, 12)
    lay.setSpacing(8)
    lay.addWidget(inner_widget)
    return box


def make_scroll(inner: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
    wrapper = QtWidgets.QWidget()
    v = QtWidgets.QVBoxLayout(wrapper)
    v.setContentsMargins(8, 8, 8, 8)
    v.setSpacing(8)
    v.addWidget(inner)
    sc = QtWidgets.QScrollArea()
    sc.setWidgetResizable(True)
    sc.setFrameShape(QtWidgets.QFrame.NoFrame)
    sc.setWidget(wrapper)
    return sc


def set_file_button_symbol(
    fe: FileEdit, symbol: str = "📂"
) -> QtWidgets.QPushButton | None:
    native = getattr(fe, "native", None)
    if native is None:
        return None
    for btn in native.findChildren(QtWidgets.QPushButton):
        # some FileEdit implementations use a pair of buttons; we use the first one.
        with contextlib.suppress(Exception):
            btn.setText(symbol)
            btn.setMinimumWidth(32)
            btn.setMaximumWidth(36)
            btn.setCursor(QtCore.Qt.PointingHandCursor)
        return btn
    return None
    return None


def attach_dual_picker(fe: FileEdit, button: QtWidgets.QPushButton) -> None:
    menu = QtWidgets.QMenu(button)
    act_file = menu.addAction("Pick File…")
    act_dir = menu.addAction("Pick Folder…")

    def pick_file() -> None:
        start = as_path_str(fe.value).strip() or os.path.expanduser("~")
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            button, "Select file", start
        )
        if path:
            fe.value = path

    def pick_dir() -> None:
        start = as_path_str(fe.value).strip() or os.path.expanduser("~")
        path = QtWidgets.QFileDialog.getExistingDirectory(
            button, "Select folder", start
        )
        if path:
            fe.value = path

    act_file.triggered.connect(pick_file)
    act_dir.triggered.connect(pick_dir)

    def on_click() -> None:
        menu.exec_(button.mapToGlobal(QtCore.QPoint(0, button.height())))

    button.clicked.connect(on_click)


def attach_dir_picker(fe: FileEdit, button: QtWidgets.QPushButton) -> None:
    """Attach a direct folder picker: clicking the button opens directory dialog with a Select Folder button."""

    def pick_dir() -> None:
        start = as_path_str(fe.value).strip() or os.path.expanduser("~")
        path = QtWidgets.QFileDialog.getExistingDirectory(
            button, "Select folder", start
        )
        if path:
            fe.value = path
        else:
            # Use napari's show_error to avoid raw prints
            show_error("No folder selected.")

    button.clicked.connect(pick_dir)


# -----------------------------------------------------------------------------
# Dock widget
# -----------------------------------------------------------------------------


class ResviewDockWidget(QtWidgets.QWidget):
    """ResView UI as a napari dock widget (tabs: Data/Build/View)."""

    def __init__(
        self,
        viewer: Viewer | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)

        # ---- FIX: viewer injection is optional
        if viewer is None:
            try:
                viewer = napari.current_viewer()
            except RuntimeError:
                viewer = None

        if viewer is None:
            raise TypeError(
                "ResviewDockWidget requires a napari Viewer. "
                "If you are registering the widget in a napari plugin/manifest, "
                "register a factory that receives `viewer`, or allow this class to "
                "be instantiated while a napari viewer is active."
            )

        self.viewer: Viewer = viewer

        # Style + icon (apply stylesheet once)
        icon = try_apply_app_icon()
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.setStyleSheet(APP_QSS + RUN_ALL_QSS)

        # YAML init
        self._yaml_path = yaml_path()
        ensure_yaml(self._yaml_path)
        self._ydoc: dict[str, Any] = load_yaml(self._yaml_path)

        # Runtime state
        self._state: dict[str, Any] = {
            "loader": None,
            "builder": None,
            "grid": None,
            "edges": None,
            "intensity_viewer": None,
            "ub": None,
        }

        # ---------------------------------------------------------------------
        # Build the UI (3 tabs)
        # ---------------------------------------------------------------------

        # --- Data tab
        self.spec_file_w = FileEdit(mode="r", label="SPEC file")
        self.data_file_w = FileEdit(mode="r", label="DATA folder")
        # only folder selection is allowed for DATA
        self.scans_w = LineEdit(label="Scans (e.g. 17, 18-22, 30)")
        self.scans_w.tooltip = "Comma/range list. Examples: 17, 18-22, 30"
        self.only_hkl_w = CheckBox(label="Only HKL scans")

        _ = set_file_button_symbol(self.spec_file_w, "📂")
        data_btn = set_file_button_symbol(self.data_file_w, "📂")
        if data_btn is not None:
            attach_dir_picker(self.data_file_w, data_btn)

        title_params = Label(value="<b>Experiment Setup</b>")
        self.distance_w = FloatSpinBox(
            label="Distance (m)", min=-1e9, max=1e9, step=1e-6
        )
        self.pitch_w = FloatSpinBox(
            label="Pitch (m)", min=-1e9, max=1e9, step=1e-9
        )
        self.ypixels_w = SpinBox(
            label="Detector H (px)", min=0, max=10_000_000, step=1
        )
        self.xpixels_w = SpinBox(
            label="Detector W (px)", min=0, max=10_000_000, step=1
        )
        self.ycenter_w = SpinBox(
            label="BeamCenter H (px)", min=0, max=10_000_000, step=1
        )
        self.xcenter_w = SpinBox(
            label="BeamCenter W (px)", min=0, max=10_000_000, step=1
        )
        self.energy_w = FloatSpinBox(
            label="Energy (keV)", min=-1e9, max=1e9, step=1e-3
        )
        self.wavelength_w = FloatSpinBox(
            label="Wavelength (Å)", min=1e-6, max=1e6, step=1e-3
        )
        self.wavelength_w.tooltip = "Leave empty to derive from energy. < 1e-3 is meters → converted to Å."

        title_crystal = Label(value="<b>Crystal</b>")
        self.ub_matrix_w = TextEdit(label="UB (matrix)")
        self.ub_matrix_w.tooltip = (
            "Rows separated by newlines; values separated by spaces or commas."
        )
        self.ub_matrix_w.value = "1 0 0\n0 1 0\n0 0 1"
        with contextlib.suppress(AttributeError):
            self.ub_matrix_w.native.setMinimumHeight(80)

        self.btn_load = PushButton(text="📂 Load Data")
        self.btn_intensity = PushButton(text="📈 View Intensity")

        btn_row1 = QtWidgets.QWidget()
        row1 = QtWidgets.QHBoxLayout(btn_row1)
        row1.setContentsMargins(0, 0, 0, 0)
        row1.setSpacing(8)
        row1.addWidget(self.btn_load.native)
        row1.addWidget(self.btn_intensity.native)
        row1.addStretch(1)

        col1 = Container(
            layout="vertical",
            widgets=[
                self.spec_file_w,
                self.data_file_w,
                self.scans_w,
                self.only_hkl_w,
                hsep(),
                title_params,
                self.distance_w,
                self.pitch_w,
                self.ypixels_w,
                self.xpixels_w,
                self.ycenter_w,
                self.xcenter_w,
                self.energy_w,
                self.wavelength_w,
                hsep(),
                title_crystal,
                self.ub_matrix_w,
            ],
        )
        g1 = make_group("Data", col1.native)
        g1.layout().addStretch(1)
        g1.layout().addWidget(btn_row1)
        tab_data = make_scroll(g1)

        # --- Build tab
        title_build = Label(value="<b>RSM Builder</b>")
        self.sample_axes_w = LineEdit(label="Sample axes")
        self.sample_axes_w.value = "x+, y+, z-"
        self.detector_axes_w = LineEdit(label="Detector axes")
        self.detector_axes_w.value = "x+"
        self.ub_2pi_w = CheckBox(label="UB includes 2π")
        self.center_one_based_w = CheckBox(label="1-based center")

        title_regrid = Label(value="<b>Grid Settings</b>")
        self.space_w = ComboBox(label="Space", choices=["hkl", "q"])
        self.grid_shape_w = LineEdit(label="Grid (x,y,z), '*' allowed")
        self.grid_shape_w.tooltip = (
            "Examples: 200,*,* or 256,256,256 or just 200"
        )
        self.fuzzy_w = CheckBox(label="Fuzzy gridder")
        self.fuzzy_width_w = FloatSpinBox(
            label="Width (fuzzy)", min=0.0, max=1e9, step=0.01
        )
        self.normalize_w = ComboBox(label="Normalize", choices=["mean", "sum"])

        title_crop = Label(value="<b>Crop Settings</b>")
        self.crop_enable_w = CheckBox(label="Crop before regrid")
        self.y_min_w = SpinBox(label="y_min", min=0, max=10_000_000, step=1)
        self.y_max_w = SpinBox(label="y_max", min=0, max=10_000_000, step=1)
        self.x_min_w = SpinBox(label="x_min", min=0, max=10_000_000, step=1)
        self.x_max_w = SpinBox(label="x_max", min=0, max=10_000_000, step=1)

        self.btn_build = PushButton(text="🔧 Build RSM Map")
        self.btn_regrid = PushButton(text="🧮 Regrid")

        btn_row2 = QtWidgets.QWidget()
        row2 = QtWidgets.QHBoxLayout(btn_row2)
        row2.setContentsMargins(0, 0, 0, 0)
        row2.setSpacing(8)
        row2.addWidget(self.btn_build.native)
        row2.addWidget(self.btn_regrid.native)
        row2.addStretch(1)

        col2 = Container(
            layout="vertical",
            widgets=[
                title_build,
                self.ub_2pi_w,
                self.center_one_based_w,
                self.sample_axes_w,
                self.detector_axes_w,
                hsep(),
                title_regrid,
                self.space_w,
                self.grid_shape_w,
                self.fuzzy_w,
                self.fuzzy_width_w,
                self.normalize_w,
                hsep(),
                title_crop,
                self.crop_enable_w,
                self.y_min_w,
                self.y_max_w,
                self.x_min_w,
                self.x_max_w,
                hsep(),
            ],
        )
        g2 = make_group("Build", col2.native)
        g2.layout().addStretch(1)
        g2.layout().addWidget(btn_row2)
        tab_build = make_scroll(g2)

        # --- View tab
        title_view = Label(value="<b>Napari Viewer</b>")
        self.log_view_w = CheckBox(label="Log view")
        self.cmap_w = ComboBox(
            label="Colormap",
            choices=["viridis", "inferno", "magma", "plasma", "cividis"],
        )
        self.rendering_w = ComboBox(
            label="Rendering", choices=["attenuated_mip", "mip", "translucent"]
        )
        self.contrast_lo_w = FloatSpinBox(
            label="Contrast low (%)", min=0.0, max=100.0, step=0.1
        )
        self.contrast_hi_w = FloatSpinBox(
            label="Contrast high (%)", min=0.0, max=100.0, step=0.1
        )

        self.status_w = TextEdit(value="")
        try:
            self.status_w.native.setReadOnly(True)
            self.status_w.native.setMinimumHeight(220)
        except AttributeError:
            pass

        self.progress = QtWidgets.QProgressBar()
        self.progress.setMinimum(0)
        self.progress.setMaximum(100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)

        self.export_vtr_w = FileEdit(mode="w", label="Output VTK (.vtr)")
        _ = set_file_button_symbol(self.export_vtr_w, "📂")

        self.btn_view = PushButton(text="🔭 View RSM")
        self.btn_export = PushButton(text="💾 Export to VTK")

        # Create a plain Qt QPushButton for "Run All" with direct inline styling
        # This bypasses magicgui and QSS issues to ensure the button displays correctly.
        self.btn_run_all_native = QtWidgets.QPushButton("Run All")
        self.btn_run_all_native.setMinimumHeight(50)
        self.btn_run_all_native.setMinimumWidth(200)
        self.btn_run_all_native.setFont(QtGui.QFont("", 14, QtGui.QFont.Bold))
        # Apply inline stylesheet directly to the button (not via QSS)
        self.btn_run_all_native.setStyleSheet("""
            background-color: #ff9800;
            color: white;
            border: 2px solid #e68900;
            border-radius: 8px;
            padding: 8px 12px;
            font-weight: bold;
            font-size: 14px;
        """)
        # Load and set icon if available
        with contextlib.suppress(Exception):
            if icon is not None:
                self.btn_run_all_native.setIcon(icon)
                self.btn_run_all_native.setIconSize(QtCore.QSize(20, 20))
        # Ensure the button is connected only once to avoid duplicate execution
        with contextlib.suppress(Exception):
            self.btn_run_all_native.clicked.disconnect()
        self.btn_run_all_native.clicked.connect(self.on_run_all)

        left_bottom = QtWidgets.QWidget()
        vleft = QtWidgets.QVBoxLayout(left_bottom)
        vleft.setContentsMargins(0, 0, 0, 0)
        vleft.setSpacing(8)
        vleft.addWidget(self.btn_view.native)
        vleft.addWidget(self.btn_export.native)

        btn_row3 = QtWidgets.QWidget()
        row3 = QtWidgets.QHBoxLayout(btn_row3)
        row3.setContentsMargins(0, 0, 0, 0)
        row3.setSpacing(12)
        row3.addWidget(left_bottom)
        row3.addStretch(1)
        row3.addWidget(self.btn_run_all_native)

        col3 = Container(
            layout="vertical",
            widgets=[
                title_view,
                self.log_view_w,
                self.cmap_w,
                self.rendering_w,
                self.contrast_lo_w,
                self.contrast_hi_w,
                hsep(),
            ],
        )

        g3 = make_group("View", col3.native)
        g3_lay = g3.layout()
        g3_lay.addWidget(q_hsep())
        g3_lay.addWidget(QtWidgets.QLabel("<b>Status / Output</b>"))
        g3_lay.addWidget(self.status_w.native)
        g3_lay.addWidget(self.progress)
        g3_lay.addWidget(q_hsep())
        g3_lay.addWidget(QtWidgets.QLabel("<b>Export</b>"))
        g3_lay.addWidget(self.export_vtr_w.native)
        g3_lay.addStretch(1)
        g3_lay.addWidget(q_hsep())
        g3_lay.addWidget(btn_row3)

        tab_view = make_scroll(g3)

        # Tabs container
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(tab_data, "Data")
        tabs.addTab(tab_build, "Build")
        tabs.addTab(tab_view, "View")

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)
        outer.addWidget(tabs)

        self.status_bar = QtWidgets.QStatusBar()
        outer.addWidget(self.status_bar)

        # YAML <-> UI binding
        self._widget_map: dict[str, dict[str, Any]] = {
            "data": {
                "spec_file": self.spec_file_w,
                "data_file": self.data_file_w,
                "scans": self.scans_w,
                "only_hkl": self.only_hkl_w,
            },
            "ExperimentSetup": {
                "distance": self.distance_w,
                "pitch": self.pitch_w,
                "ycenter": self.ycenter_w,
                "xcenter": self.xcenter_w,
                "xpixels": self.xpixels_w,
                "ypixels": self.ypixels_w,
                "energy": self.energy_w,
                "wavelength": self.wavelength_w,
            },
            "Crystal": {"ub": self.ub_matrix_w},
            "build": {
                "ub_includes_2pi": self.ub_2pi_w,
                "center_is_one_based": self.center_one_based_w,
                "sample_axes": self.sample_axes_w,
                "detector_axes": self.detector_axes_w,
            },
            "crop": {
                "enable": self.crop_enable_w,
                "y_min": self.y_min_w,
                "y_max": self.y_max_w,
                "x_min": self.x_min_w,
                "x_max": self.x_max_w,
            },
            "regrid": {
                "space": self.space_w,
                "grid_shape": self.grid_shape_w,
                "fuzzy": self.fuzzy_w,
                "fuzzy_width": self.fuzzy_width_w,
                "normalize": self.normalize_w,
            },
            "view": {
                "log_view": self.log_view_w,
                "cmap": self.cmap_w,
                "rendering": self.rendering_w,
                "contrast_lo": self.contrast_lo_w,
                "contrast_hi": self.contrast_hi_w,
            },
            "export": {"vtr_path": self.export_vtr_w},
        }

        self._apply_yaml_to_widgets()
        self._connect_widget_changes()
        # Ensure DATA folder picker opens a sensible directory by default
        if not as_path_str(self.data_file_w.value).strip():
            self.data_file_w.value = os.path.expanduser("~")

        # Button handlers
        self.btn_load.clicked.connect(self.on_load)
        self.btn_intensity.clicked.connect(self.on_view_intensity)
        self.btn_build.clicked.connect(self.on_build)
        self.btn_regrid.clicked.connect(self.on_regrid)
        self.btn_view.clicked.connect(self.on_view)
        self.btn_export.clicked.connect(self.on_export_vtk)

    # -------------------------------------------------------------------------
    # YAML helpers (no blind exceptions)
    # -------------------------------------------------------------------------

    def _apply_yaml_to_widgets(self) -> None:
        ydoc = self._ydoc
        for section in self._widget_map:
            ydoc.setdefault(section, {})

        def set_widget(widget: Any, value: Any) -> None:
            if value is None:
                return
            try:
                if isinstance(widget, FloatSpinBox):
                    widget.value = float(value)
                elif isinstance(widget, SpinBox):
                    widget.value = int(value)
                elif isinstance(widget, CheckBox):
                    widget.value = bool(value)
                elif isinstance(widget, ComboBox):
                    sval = str(value)
                    if sval in widget.choices:
                        widget.value = sval
                elif isinstance(widget, (LineEdit, TextEdit, FileEdit)):
                    widget.value = str(value)
            except (TypeError, ValueError, AttributeError):
                # skip problematic values
                return

        for section, mapping in self._widget_map.items():
            vals = ydoc.get(section, {}) or {}
            for key, widget in mapping.items():
                set_widget(widget, vals.get(key))

        # Only set a default UB in YAML if it is completely missing
        if (
            "Crystal" not in ydoc
            or ydoc.get("Crystal", {}).get("ub", None) is None
        ):
            # do not clobber user's YAML if UB is present but empty string handled above
            self._ydoc.setdefault("Crystal", {})["ub"] = self.ub_matrix_w.value
            save_yaml(self._yaml_path, self._ydoc)

    def _widget_value_for_yaml(
        self, widget: Any, section: str, key: str
    ) -> Any:
        if section == "ExperimentSetup" and key == "wavelength":
            txt = str(widget.value).strip()
            if txt.lower() in {"", "none", "null"}:
                return None
            try:
                return float(txt)
            except ValueError:
                return txt

        if section == "Crystal" and key == "ub":
            txt = str(widget.value).strip()
            return txt or None

        if isinstance(widget, FloatSpinBox):
            return float(widget.value)
        if isinstance(widget, SpinBox):
            return int(widget.value)
        if isinstance(widget, CheckBox):
            return bool(widget.value)
        if isinstance(widget, ComboBox):
            return str(widget.value)
        if isinstance(widget, (LineEdit, TextEdit, FileEdit)):
            return str(widget.value)
        return widget.value

    def _connect_widget_changes(self) -> None:
        def on_changed(section: str, key: str, widget: Any) -> None:
            self._ydoc.setdefault(section, {})
            self._ydoc[section][key] = self._widget_value_for_yaml(
                widget, section, key
            )
            save_yaml(self._yaml_path, self._ydoc)

        for section, mapping in self._widget_map.items():
            for key, widget in mapping.items():
                # ensure we capture current widget in the lambda using defaults
                widget.changed.connect(
                    lambda *_, s=section, k=key, w=widget: on_changed(s, k, w)
                )

    # -------------------------------------------------------------------------
    # UI status helpers
    # -------------------------------------------------------------------------

    def pump(self, ms: int = 0) -> None:
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, ms)

    def status(self, msg: str) -> None:
        try:
            self.status_w.native.append(msg)
        except AttributeError:
            self.status_w.value = (self.status_w.value or "") + (
                ("\n" if self.status_w.value else "") + msg
            )
        self.status_bar.showMessage(msg, 3000)
        logging.info("Status: %s", msg)

    def set_progress(self, value: int | None, *, busy: bool = False) -> None:
        if busy:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 100)
            try:
                self.progress.setValue(int(value or 0))
            except (TypeError, ValueError):
                self.progress.setValue(0)

    def set_busy(self, b: bool) -> None:
        """
        Enable/disable both magicgui widgets (which often expose .native)
        and plain Qt widgets. More explicit than blind AttributeError suppression.
        """
        widgets = (
            self.btn_load,
            self.btn_intensity,
            self.btn_build,
            self.btn_regrid,
            self.btn_view,
            self.btn_export,
        )
        for btn in widgets:
            # magicgui widget: disable .native if present
            with contextlib.suppress(Exception):
                if hasattr(btn, "native") and btn.native is not None:
                    btn.native.setEnabled(not b)
                    continue
            # fallback: try to call set_enabled or setEnabled if available
            with contextlib.suppress(Exception):
                set_enabled = getattr(btn, "set_enabled", None)
                if callable(set_enabled):
                    set_enabled(not b)  # type: ignore[call-arg]
                    continue
                set_enabled_qt = getattr(btn, "setEnabled", None)
                if callable(set_enabled_qt):
                    set_enabled_qt(not b)

        # handle the plain Qt Run All button
        with contextlib.suppress(Exception):
            self.btn_run_all_native.setEnabled(not b)

    # -------------------------------------------------------------------------
    # Actions (BLE001-safe)
    # -------------------------------------------------------------------------

    def on_view_intensity(self) -> None:
        loader = self._state.get("loader")
        if loader is None:
            show_error("Load data first.")
            return

        try:
            _a, _b, df = loader.load()
            frames = list(df.intensity)
        except (OSError, ValueError, RuntimeError, TypeError, KeyError) as e:
            logging.exception("Intensity load error")
            show_error(f"Intensity load error: {e}")
            return

        try:
            viewer_local = IntensityNapariViewer(
                frames,
                name="Intensity",
                log_view=True,
                contrast_percentiles=(1.0, 99.8),
                cmap="inferno",
                rendering="attenuated_mip",
                add_timeseries=True,
                add_volume=True,
                scale_tzyx=(1.0, 1.0, 1.0),
                pad_value=np.nan,
            ).launch(viewer=self.viewer)
            self._state["intensity_viewer"] = viewer_local
        except (RuntimeError, ValueError, TypeError) as e:
            logging.exception("Failed to open intensity viewer")
            show_error(f"Failed to open intensity viewer: {e}")

    def on_load(self) -> None:
        spec = as_path_str(self.spec_file_w.value).strip()
        dpath = as_path_str(self.data_file_w.value).strip()

        try:
            scans = parse_scan_list((self.scans_w.value or "").strip())
        except ValueError as e:
            show_error(str(e))
            return

        if not spec or not os.path.isfile(spec):
            show_error("Select a valid SPEC file.")
            return
        if not os.path.isdir(dpath):
            show_error("Select a valid DATA folder.")
            return
        if not scans:
            show_error("Enter at least one scan (e.g. '17, 18-22').")
            return

        self.set_busy(True)
        self.set_progress(None, busy=True)
        self.status(f"Loading scans {scans}…")
        # If load becomes long, move into a QThread worker
        try:
            tiff_arg = dpath
            loader = RSMDataLoader(
                spec,
                yaml_path(),
                tiff_arg,
                selected_scans=scans,
                process_hklscan_only=bool(self.only_hkl_w.value),
            )
            load_result = loader.load()

            ub_from_load = None
            if isinstance(load_result, tuple) and len(load_result) >= 2:
                ub_from_load = load_result[1]
            if ub_from_load is None:
                ub_from_load = getattr(loader, "ub", None)

            self._state["ub"] = ub_from_load
            self.ub_matrix_w.value = format_ub_matrix(ub_from_load)

            self._state["loader"] = loader
            self._state["builder"] = None
            self._state["grid"] = None
            self._state["edges"] = None

            self.set_progress(25, busy=False)
            self.status("Data loaded.")
        except (OSError, ValueError, RuntimeError, TypeError, KeyError) as e:
            logging.exception("Load error")
            show_error(f"Load error: {e}")
            self.set_progress(0, busy=False)
            self.status(f"Load failed: {e}")
        finally:
            self.set_busy(False)

    def on_build(self) -> None:
        if self._state.get("loader") is None:
            show_error("Load data first.")
            return

        self.set_busy(True)
        self.set_progress(None, busy=True)
        self.status("Computing Q/HKL/intensity…")

        try:
            b = RSMBuilder(
                self._state["loader"],
                sample_axes=parse_axes_list(self.sample_axes_w.value),
                detector_axes=parse_axes_list(self.detector_axes_w.value),
                ub_includes_2pi=bool(self.ub_2pi_w.value),
                center_is_one_based=bool(self.center_one_based_w.value),
            )
            # If compute_full is heavy, run it in a worker thread.
            b.compute_full(verbose=False)
            self._state["builder"] = b
            self._state["grid"] = None
            self._state["edges"] = None
            self.set_progress(50, busy=False)
            self.status("RSM map built.")
        except (ValueError, RuntimeError, TypeError, KeyError) as e:
            logging.exception("Build error")
            show_error(f"Build error: {e}")
            self.set_progress(40, busy=False)
            self.status(f"Build failed: {e}")
        finally:
            self.set_busy(False)

    def on_regrid(self) -> None:
        b = self._state.get("builder")
        if b is None:
            show_error("Build the RSM map first.")
            return

        try:
            gx, gy, gz = parse_grid_shape(self.grid_shape_w.value)
        except ValueError as e:
            show_error(str(e))
            return
        if gx is None:
            show_error("Grid X (first value) is required (e.g., 200,*,*).")
            return

        do_crop = bool(self.crop_enable_w.value)
        ymin, ymax = int(self.y_min_w.value), int(self.y_max_w.value)
        xmin, xmax = int(self.x_min_w.value), int(self.x_max_w.value)

        self.set_busy(True)
        self.set_progress(None, busy=True)
        self.status(
            f"Regridding to {str(self.space_w.value).upper()} grid {(gx, gy, gz)}…"
        )

        try:
            if do_crop:
                if ymin >= ymax or xmin >= xmax:
                    raise ValueError(
                        "Crop bounds must satisfy y_min < y_max and x_min < x_max."
                    )
                loader = self._state.get("loader")
                if loader is None:
                    raise RuntimeError(
                        "Internal error: loader missing; run Build again."
                    )

                b = RSMBuilder(
                    loader,
                    sample_axes=parse_axes_list(self.sample_axes_w.value),
                    detector_axes=parse_axes_list(self.detector_axes_w.value),
                    ub_includes_2pi=bool(self.ub_2pi_w.value),
                    center_is_one_based=bool(self.center_one_based_w.value),
                )
                b.compute_full(verbose=False)
                b.crop_by_positions(y_bound=(ymin, ymax), x_bound=(xmin, xmax))

            kwargs: dict[str, Any] = {
                "space": self.space_w.value,
                "grid_shape": (gx, gy, gz),
                "fuzzy": bool(self.fuzzy_w.value),
                "normalize": self.normalize_w.value,
                "stream": True,
            }
            if (
                bool(self.fuzzy_w.value)
                and float(self.fuzzy_width_w.value or 0) > 0
            ):
                kwargs["width"] = float(self.fuzzy_width_w.value)

            grid, edges = b.regrid_xu(**kwargs)
            self._state["grid"], self._state["edges"] = grid, edges

            self.set_progress(75, busy=False)
            self.status("Regrid completed.")
        except (ValueError, RuntimeError, TypeError, KeyError) as e:
            logging.exception("Regrid error")
            show_error(f"Regrid error: {e}")
            self.set_progress(60, busy=False)
            self.status(f"Regrid failed: {e}")
        finally:
            self.set_busy(False)

    def on_view(self) -> None:
        if self._state.get("grid") is None or self._state.get("edges") is None:
            show_error("Regrid first.")
            return

        try:
            lo = float(self.contrast_lo_w.value)
            hi = float(self.contrast_hi_w.value)
            if not (0 <= lo < hi <= 100):
                raise ValueError(
                    "Contrast % must satisfy 0 ≤ low < high ≤ 100"
                )
        except ValueError as e:
            show_error(str(e))
            return

        self.set_progress(None, busy=True)
        self.status("Opening RSM viewer…")

        try:
            viz = RSMNapariViewer(
                self._state["grid"],
                self._state["edges"],
                space=self.space_w.value,
                name="RSM3D",
                log_view=bool(self.log_view_w.value),
                contrast_percentiles=(lo, hi),
                cmap=self.cmap_w.value,
                rendering=self.rendering_w.value,
            )
            viewer_local = viz.launch(viewer=self.viewer)
            self._state["rsm_viewer"] = viewer_local
            self.set_progress(100, busy=False)
            self.status("RSM viewer opened.")
        except (RuntimeError, ValueError, TypeError) as e:
            logging.exception("View error")
            show_error(f"View error: {e}")
            self.set_progress(80, busy=False)
            self.status(f"View failed: {e}")

    def on_export_vtk(self) -> None:
        if self._state.get("grid") is None or self._state.get("edges") is None:
            show_error("Regrid first, then export.")
            return

        out_path = as_path_str(self.export_vtr_w.value).strip()
        if not out_path:
            show_error("Choose an output .vtr file path.")
            return
        if not out_path.lower().endswith(".vtr"):
            out_path += ".vtr"

        self.set_busy(True)
        self.set_progress(None, busy=True)
        self.status(f"Exporting VTK (.vtr) → {out_path}")

        try:
            write_rsm_volume_to_vtr(
                self._state["grid"],
                self._state["edges"],
                out_path,
                binary=False,
                compress=True,
            )
            self.set_progress(100, busy=False)
            self.status(f"Exported: {out_path}")
        except (OSError, RuntimeError, ValueError, TypeError) as e:
            logging.exception("Export error")
            show_error(f"Export error: {e}")
            self.set_progress(0, busy=False)
            self.status(f"Export failed: {e}")
        finally:
            self.set_busy(False)

    def on_run_all(self) -> None:
        # disable the run button immediately to prevent re-entry
        self.btn_run_all_native.setEnabled(False)
        try:
            self.set_progress(0, busy=False)
            self.status("Running pipeline (Load → Build → Regrid → View)…")

            self.on_load()
            if self._state.get("loader") is None:
                return

            self.on_build()
            if self._state.get("builder") is None:
                return

            self.on_regrid()
            if (
                self._state.get("grid") is None
                or self._state.get("edges") is None
            ):
                return

            self.on_view()
            self.status("Run All completed.")
        finally:
            self.btn_run_all_native.setEnabled(True)


# -----------------------------------------------------------------------------
# Optional: npe2 dock widget provider
# -----------------------------------------------------------------------------


def napari_experimental_provide_dock_widget():
    # Returning the class is still fine; with viewer optional, both patterns work.
    return ResviewDockWidget


# #!/usr/bin/env python3
# """
# ResView (ResviewDockWidget) — napari dock widget with YAML persistence

# Fix for: TypeError ... __init__() missing 1 required positional argument: 'viewer'
# - viewer is now OPTIONAL in __init__
# - if viewer is None, we try napari.current_viewer()

# UI:
# - 3 vertical tabs: Data / Build / View
# """

# from __future__ import annotations

# import contextlib
# import os
# import pathlib
# import re
# from typing import Any

# import napari
# import numpy as np
# import yaml
# from magicgui.widgets import (
#     CheckBox,
#     ComboBox,
#     Container,
#     FileEdit,
#     FloatSpinBox,
#     Label,
#     LineEdit,
#     PushButton,
#     SpinBox,
#     TextEdit,
# )
# from napari.utils.notifications import show_error
# from napari.viewer import Viewer
# from qtpy import QtCore, QtGui, QtWidgets

# from .data_io import RSMDataLoader, write_rsm_volume_to_vtr
# from .data_viz import IntensityNapariViewer, RSMNapariViewer
# from .rsm3d import RSMBuilder

# # -----------------------------------------------------------------------------
# # App styling / icon
# # -----------------------------------------------------------------------------

# APP_ICON_PATH = (pathlib.Path(__file__).parent / "resview_icon.png").resolve()

# APP_QSS = """
# QGroupBox {
#     border: 1px solid #d9d9d9;
#     border-radius: 8px;
#     margin-top: 12px;
#     background: #ffffff;
#     font-weight: 600;
# }
# QGroupBox::title {
#     subcontrol-origin: margin;
#     subcontrol-position: top left;
#     padding: 6px 10px;
#     color: #2c3e50;
#     font-size: 18px;
#     font-weight: 700;
#     letter-spacing: 0.2px;
# }
# QLabel { color: #34495e; }
# QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QDoubleSpinBox, QSpinBox {
#     border: 1px solid #d4d7dd;
#     border-radius: 6px;
#     padding: 4px 6px;
#     background: #ffffff;
# }
# QPushButton {
#     background: #eef2f7;
#     border: 1px solid #d4d7dd;
#     border-radius: 8px;
#     padding: 6px 10px;
#     font-weight: 600;
# }
# QPushButton:hover { background: #e6ebf3; }
# QPushButton:pressed { background: #dfe5ee; }
# """

# RUN_ALL_QSS = """
# QPushButton#RunAllPrimary {
#     background: #ff9800;
#     color: #ffffff;
#     border: 2px solid #e68900;
#     border-radius: 10px;
#     padding: 14px 20px;
#     font-size: 18px;
#     font-weight: 800;
# }
# QPushButton#RunAllPrimary:hover { background: #ffa726; }
# QPushButton#RunAllPrimary:pressed { background: #fb8c00; }
# """


# def load_app_icon() -> QtGui.QIcon | None:
#     if APP_ICON_PATH.is_file():
#         icon = QtGui.QIcon(str(APP_ICON_PATH))
#         return icon if not icon.isNull() else None
#     return None


# def try_apply_app_icon() -> None:
#     icon = load_app_icon()
#     if icon is None:
#         return
#     app = QtWidgets.QApplication.instance()
#     if app is not None:
#         app.setStyleSheet(APP_QSS + RUN_ALL_QSS)


# # -----------------------------------------------------------------------------
# # YAML persistence
# # -----------------------------------------------------------------------------

# DEFAULTS_ENV = "RSM3D_DEFAULTS_YAML"
# os.environ.setdefault(
#     DEFAULTS_ENV,
#     str(pathlib.Path(__file__).with_name("rsm3d_defaults.yaml").resolve()),
# )


# def yaml_path() -> str:
#     p = os.environ.get(DEFAULTS_ENV, "").strip()
#     if p:
#         return os.path.abspath(os.path.expanduser(p))
#     return os.path.join(os.path.expanduser("~"), ".rsm3d_defaults.yaml")


# def ensure_yaml(path: str) -> None:
#     if os.path.isfile(path):
#         return
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     seed: dict[str, Any] = {
#         "data": {
#             "spec_file": None,
#             "data_file": None,
#             "scans": "",
#             "only_hkl": None,
#         },
#         "ExperimentSetup": {
#             "distance": None,
#             "pitch": None,
#             "ycenter": None,
#             "xcenter": None,
#             "xpixels": None,
#             "ypixels": None,
#             "energy": None,
#             "wavelength": None,
#         },
#         "Crystal": {"ub": None},
#         "build": {
#             "ub_includes_2pi": None,
#             "center_is_one_based": None,
#             "sample_axes": "",
#             "detector_axes": "",
#         },
#         "crop": {
#             "enable": None,
#             "y_min": None,
#             "y_max": None,
#             "x_min": None,
#             "x_max": None,
#         },
#         "regrid": {
#             "space": None,
#             "grid_shape": "",
#             "fuzzy": None,
#             "fuzzy_width": None,
#             "normalize": None,
#         },
#         "view": {
#             "log_view": None,
#             "cmap": None,
#             "rendering": None,
#             "contrast_lo": None,
#             "contrast_hi": None,
#         },
#         "export": {"vtr_path": None},
#     }
#     with open(path, "w", encoding="utf-8") as f:
#         yaml.safe_dump(seed, f, sort_keys=False)


# def load_yaml(path: str) -> dict[str, Any]:
#     try:
#         with open(path, encoding="utf-8") as f:
#             doc = yaml.safe_load(f)
#         return doc or {}
#     except (OSError, UnicodeDecodeError, yaml.YAMLError):
#         return {}


# def save_yaml(path: str, doc: dict[str, Any]) -> None:
#     try:
#         with open(path, "w", encoding="utf-8") as f:
#             yaml.safe_dump(doc, f, sort_keys=False)
#     except (OSError, UnicodeEncodeError, yaml.YAMLError) as e:
#         show_error(f"Failed to write YAML: {e}")


# def as_path_str(v: Any) -> str:
#     if v is None:
#         return ""
#     try:
#         return os.fspath(v)
#     except TypeError:
#         return str(v)


# # -----------------------------------------------------------------------------
# # Parsing / formatting
# # -----------------------------------------------------------------------------


# def format_ub_matrix(ub: Any) -> str:
#     if ub is None:
#         return ""
#     try:
#         arr = np.asarray(ub, dtype=float)
#     except (TypeError, ValueError):
#         return str(ub)
#     if arr.ndim == 0:
#         return f"{arr.item():.6g}"
#     if arr.ndim == 1:
#         return " ".join(f"{v:.6g}" for v in arr)
#     return "\n".join(" ".join(f"{v:.6g}" for v in row) for row in arr)


# def parse_ub_matrix(text: str) -> np.ndarray | None:
#     stripped = (text or "").strip()
#     if not stripped:
#         return None
#     rows: list[list[float]] = []
#     for line in stripped.splitlines():
#         parts = [p for p in re.split(r"[,\s]+", line.strip()) if p]
#         if parts:
#             rows.append([float(p) for p in parts])
#     if not rows:
#         return None
#     width = len(rows[0])
#     if any(len(row) != width for row in rows):
#         raise ValueError("UB rows must have equal length.")
#     return np.array(rows, dtype=float)


# def parse_scan_list(text: str) -> list[int]:
#     if not text or not text.strip():
#         return []
#     out: set[int] = set()
#     for part in re.split(r"[,\s]+", text.strip()):
#         if not part:
#             continue
#         if "-" in part:
#             a, b = part.split("-", 1)
#             a, b = a.strip(), b.strip()
#             if a.isdigit() and b.isdigit():
#                 lo, hi = int(a), int(b)
#                 if lo > hi:
#                     lo, hi = hi, lo
#                 out.update(range(lo, hi + 1))
#             else:
#                 raise ValueError(f"Bad scan range: '{part}'")
#         else:
#             if part.isdigit():
#                 out.add(int(part))
#             else:
#                 raise ValueError(f"Bad scan id: '{part}'")
#     return sorted(out)


# def parse_axes_list(text: str) -> list[str]:
#     if not text:
#         return []
#     return [p.strip() for p in re.split(r"[,\s]+", text) if p.strip()]


# def parse_grid_shape(text: str) -> tuple[int | None, int | None, int | None]:
#     if text is None:
#         return (None, None, None)
#     s = text.strip()
#     if not s:
#         return (None, None, None)
#     parts = [p.strip() for p in s.split(",")]
#     if len(parts) == 1:
#         parts += ["*", "*"]
#     if len(parts) != 3:
#         raise ValueError("Grid must be 'x,y,z' (y/z may be '*').")

#     def one(p: str) -> int | None:
#         if p in ("*", "", None):
#             return None
#         if not p.isdigit():
#             raise ValueError(f"Grid size must be integer or '*', got '{p}'")
#         v = int(p)
#         if v <= 0:
#             raise ValueError("Grid sizes must be > 0")
#         return v

#     return tuple(one(p) for p in parts)  # type: ignore[return-value]


# # -----------------------------------------------------------------------------
# # UI helpers
# # -----------------------------------------------------------------------------


# def hsep(height: int = 10) -> Label:
#     w = Label(value="")
#     try:
#         w.native.setFrameShape(QtWidgets.QFrame.HLine)
#         w.native.setFrameShadow(QtWidgets.QFrame.Sunken)
#         w.native.setLineWidth(1)
#         w.native.setFixedHeight(height)
#     except AttributeError:
#         pass
#     return w


# def q_hsep(height: int = 10) -> QtWidgets.QWidget:
#     line = QtWidgets.QFrame()
#     line.setFrameShape(QtWidgets.QFrame.HLine)
#     line.setFrameShadow(QtWidgets.QFrame.Sunken)
#     line.setLineWidth(1)
#     line.setFixedHeight(height)
#     return line


# def make_group(
#     title: str, inner_widget: QtWidgets.QWidget
# ) -> QtWidgets.QGroupBox:
#     box = QtWidgets.QGroupBox(title)
#     lay = QtWidgets.QVBoxLayout(box)
#     lay.setContentsMargins(12, 12, 12, 12)
#     lay.setSpacing(8)
#     lay.addWidget(inner_widget)
#     return box


# def make_scroll(inner: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
#     wrapper = QtWidgets.QWidget()
#     v = QtWidgets.QVBoxLayout(wrapper)
#     v.setContentsMargins(8, 8, 8, 8)
#     v.setSpacing(8)
#     v.addWidget(inner)
#     sc = QtWidgets.QScrollArea()
#     sc.setWidgetResizable(True)
#     sc.setFrameShape(QtWidgets.QFrame.NoFrame)
#     sc.setWidget(wrapper)
#     return sc


# def set_file_button_symbol(
#     fe: FileEdit, symbol: str = "📂"
# ) -> QtWidgets.QPushButton | None:
#     try:
#         for btn in fe.native.findChildren(QtWidgets.QPushButton):
#             btn.setText(symbol)
#             btn.setMinimumWidth(32)
#             btn.setMaximumWidth(36)
#             btn.setCursor(QtCore.Qt.PointingHandCursor)
#             return btn
#     except AttributeError:
#         return None
#     return None


# def attach_dual_picker(fe: FileEdit, button: QtWidgets.QPushButton) -> None:
#     menu = QtWidgets.QMenu(button)
#     act_file = menu.addAction("Pick File…")
#     act_dir = menu.addAction("Pick Folder…")

#     def pick_file() -> None:
#         start = as_path_str(fe.value).strip() or os.path.expanduser("~")
#         path, _ = QtWidgets.QFileDialog.getOpenFileName(
#             button, "Select file", start
#         )
#         if path:
#             fe.value = path

#     def pick_dir() -> None:
#         start = as_path_str(fe.value).strip() or os.path.expanduser("~")
#         path = QtWidgets.QFileDialog.getExistingDirectory(
#             button, "Select folder", start
#         )
#         if path:
#             fe.value = path
#             print(f"Folder selected: {path}")  # Debugging log
#         else:
#             print("No folder selected.")  # Debugging log

#     act_file.triggered.connect(pick_file)
#     act_dir.triggered.connect(pick_dir)

#     def on_click() -> None:
#         menu.exec_(button.mapToGlobal(QtCore.QPoint(0, button.height())))

#     button.clicked.connect(on_click)


# def attach_dir_picker(fe: FileEdit, button: QtWidgets.QPushButton) -> None:
#     """Attach a direct folder picker: clicking the button opens directory dialog with a Select Folder button."""

#     def pick_dir() -> None:
#         start = as_path_str(fe.value).strip() or os.path.expanduser("~")
#         dialog = QtWidgets.QFileDialog(button, "Select folder", start)
#         dialog.setFileMode(QtWidgets.QFileDialog.Directory)
#         dialog.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
#         dialog.setLabelText(QtWidgets.QFileDialog.Accept, "Select Folder")
#         dialog.setLabelText(QtWidgets.QFileDialog.Reject, "Cancel")

#         # Ensure the dialog is fully initialized and refreshed
#         dialog.open()
#         dialog.repaint()

#         if dialog.exec_() == QtWidgets.QFileDialog.Accepted:
#             selected_folder = dialog.selectedFiles()[0]
#             fe.value = selected_folder  # Explicitly set the selected folder
#         else:
#             show_error("No folder selected. Please try again.")

#     button.clicked.connect(pick_dir)


# # -----------------------------------------------------------------------------
# # Dock widget
# # -----------------------------------------------------------------------------


# class ResviewDockWidget(QtWidgets.QWidget):
#     """ResView UI as a napari dock widget (tabs: Data/Build/View)."""

#     def __init__(
#         self,
#         viewer: Viewer | None = None,
#         parent: QtWidgets.QWidget | None = None,
#     ):
#         super().__init__(parent)

#         # ---- FIX: viewer injection is optional
#         if viewer is None:
#             try:
#                 viewer = napari.current_viewer()
#             except RuntimeError:
#                 viewer = None

#         if viewer is None:
#             raise TypeError(
#                 "ResviewDockWidget requires a napari Viewer. "
#                 "If you are registering the widget in a napari plugin/manifest, "
#                 "register a factory that receives `viewer`, or allow this class to "
#                 "be instantiated while a napari viewer is active."
#             )

#         self.viewer: Viewer = viewer

#         # Style + icon
#         try_apply_app_icon()
#         app = QtWidgets.QApplication.instance()
#         if app is not None:
#             app.setStyleSheet(APP_QSS + RUN_ALL_QSS)

#         # YAML init
#         self._yaml_path = yaml_path()
#         ensure_yaml(self._yaml_path)
#         self._ydoc: dict[str, Any] = load_yaml(self._yaml_path)

#         # Runtime state
#         self._state: dict[str, Any] = {
#             "loader": None,
#             "builder": None,
#             "grid": None,
#             "edges": None,
#             "intensity_viewer": None,
#             "ub": None,
#         }

#         # ---------------------------------------------------------------------
#         # Build the UI (3 tabs)
#         # ---------------------------------------------------------------------

#         # --- Data tab
#         self.spec_file_w = FileEdit(mode="r", label="SPEC file")
#         self.data_file_w = FileEdit(mode="r", label="DATA folder")
#         # only folder selection is allowed for DATA
#         self.scans_w = LineEdit(label="Scans (e.g. 17, 18-22, 30)")
#         self.scans_w.tooltip = "Comma/range list. Examples: 17, 18-22, 30"
#         self.only_hkl_w = CheckBox(label="Only HKL scans")

#         _ = set_file_button_symbol(self.spec_file_w, "📂")
#         data_btn = set_file_button_symbol(self.data_file_w, "📂")
#         if data_btn is not None:
#             attach_dir_picker(self.data_file_w, data_btn)

#         title_params = Label(value="<b>Experiment Setup</b>")
#         self.distance_w = FloatSpinBox(
#             label="Distance (m)", min=-1e9, max=1e9, step=1e-6
#         )
#         self.pitch_w = FloatSpinBox(
#             label="Pitch (m)", min=-1e9, max=1e9, step=1e-9
#         )
#         self.ypixels_w = SpinBox(
#             label="Detector H (px)", min=0, max=10_000_000, step=1
#         )
#         self.xpixels_w = SpinBox(
#             label="Detector W (px)", min=0, max=10_000_000, step=1
#         )
#         self.ycenter_w = SpinBox(
#             label="BeamCenter H (px)", min=0, max=10_000_000, step=1
#         )
#         self.xcenter_w = SpinBox(
#             label="BeamCenter W (px)", min=0, max=10_000_000, step=1
#         )
#         self.energy_w = FloatSpinBox(
#             label="Energy (keV)", min=-1e9, max=1e9, step=1e-3
#         )
#         self.wavelength_w = FloatSpinBox(
#             label="Wavelength (Å)", min=1e-6, max=1e6, step=1e-3
#         )
#         self.wavelength_w.tooltip = "Leave empty to derive from energy. < 1e-3 is meters → converted to Å."

#         title_crystal = Label(value="<b>Crystal</b>")
#         self.ub_matrix_w = TextEdit(label="UB (matrix)")
#         self.ub_matrix_w.tooltip = (
#             "Rows separated by newlines; values separated by spaces or commas."
#         )
#         self.ub_matrix_w.value = "1 0 0\n0 1 0\n0 0 1"
#         with contextlib.suppress(AttributeError):
#             self.ub_matrix_w.native.setMinimumHeight(80)

#         self.btn_load = PushButton(text="📂 Load Data")
#         self.btn_intensity = PushButton(text="📈 View Intensity")

#         btn_row1 = QtWidgets.QWidget()
#         row1 = QtWidgets.QHBoxLayout(btn_row1)
#         row1.setContentsMargins(0, 0, 0, 0)
#         row1.setSpacing(8)
#         row1.addWidget(self.btn_load.native)
#         row1.addWidget(self.btn_intensity.native)
#         row1.addStretch(1)

#         col1 = Container(
#             layout="vertical",
#             widgets=[
#                 self.spec_file_w,
#                 self.data_file_w,
#                 self.scans_w,
#                 self.only_hkl_w,
#                 hsep(),
#                 title_params,
#                 self.distance_w,
#                 self.pitch_w,
#                 self.ypixels_w,
#                 self.xpixels_w,
#                 self.ycenter_w,
#                 self.xcenter_w,
#                 self.energy_w,
#                 self.wavelength_w,
#                 hsep(),
#                 title_crystal,
#                 self.ub_matrix_w,
#             ],
#         )
#         g1 = make_group("Data", col1.native)
#         g1.layout().addStretch(1)
#         g1.layout().addWidget(btn_row1)
#         tab_data = make_scroll(g1)

#         # --- Build tab
#         title_build = Label(value="<b>RSM Builder</b>")
#         self.sample_axes_w = LineEdit(label="Sample axes")
#         self.sample_axes_w.value = "x+, y+, z-"
#         self.detector_axes_w = LineEdit(label="Detector axes")
#         self.detector_axes_w.value = "x+"
#         self.ub_2pi_w = CheckBox(label="UB includes 2π")
#         self.center_one_based_w = CheckBox(label="1-based center")

#         title_regrid = Label(value="<b>Grid Settings</b>")
#         self.space_w = ComboBox(label="Space", choices=["hkl", "q"])
#         self.grid_shape_w = LineEdit(label="Grid (x,y,z), '*' allowed")
#         self.grid_shape_w.tooltip = (
#             "Examples: 200,*,* or 256,256,256 or just 200"
#         )
#         self.fuzzy_w = CheckBox(label="Fuzzy gridder")
#         self.fuzzy_width_w = FloatSpinBox(
#             label="Width (fuzzy)", min=0.0, max=1e9, step=0.01
#         )
#         self.normalize_w = ComboBox(label="Normalize", choices=["mean", "sum"])

#         title_crop = Label(value="<b>Crop Settings</b>")
#         self.crop_enable_w = CheckBox(label="Crop before regrid")
#         self.y_min_w = SpinBox(label="y_min", min=0, max=10_000_000, step=1)
#         self.y_max_w = SpinBox(label="y_max", min=0, max=10_000_000, step=1)
#         self.x_min_w = SpinBox(label="x_min", min=0, max=10_000_000, step=1)
#         self.x_max_w = SpinBox(label="x_max", min=0, max=10_000_000, step=1)

#         self.btn_build = PushButton(text="🔧 Build RSM Map")
#         self.btn_regrid = PushButton(text="🧮 Regrid")

#         btn_row2 = QtWidgets.QWidget()
#         row2 = QtWidgets.QHBoxLayout(btn_row2)
#         row2.setContentsMargins(0, 0, 0, 0)
#         row2.setSpacing(8)
#         row2.addWidget(self.btn_build.native)
#         row2.addWidget(self.btn_regrid.native)
#         row2.addStretch(1)

#         col2 = Container(
#             layout="vertical",
#             widgets=[
#                 title_build,
#                 self.ub_2pi_w,
#                 self.center_one_based_w,
#                 self.sample_axes_w,
#                 self.detector_axes_w,
#                 hsep(),
#                 title_regrid,
#                 self.space_w,
#                 self.grid_shape_w,
#                 self.fuzzy_w,
#                 self.fuzzy_width_w,
#                 self.normalize_w,
#                 hsep(),
#                 title_crop,
#                 self.crop_enable_w,
#                 self.y_min_w,
#                 self.y_max_w,
#                 self.x_min_w,
#                 self.x_max_w,
#                 hsep(),
#             ],
#         )
#         g2 = make_group("Build", col2.native)
#         g2.layout().addStretch(1)
#         g2.layout().addWidget(btn_row2)
#         tab_build = make_scroll(g2)

#         # --- View tab
#         title_view = Label(value="<b>Napari Viewer</b>")
#         self.log_view_w = CheckBox(label="Log view")
#         self.cmap_w = ComboBox(
#             label="Colormap",
#             choices=["viridis", "inferno", "magma", "plasma", "cividis"],
#         )
#         self.rendering_w = ComboBox(
#             label="Rendering", choices=["attenuated_mip", "mip", "translucent"]
#         )
#         self.contrast_lo_w = FloatSpinBox(
#             label="Contrast low (%)", min=0.0, max=100.0, step=0.1
#         )
#         self.contrast_hi_w = FloatSpinBox(
#             label="Contrast high (%)", min=0.0, max=100.0, step=0.1
#         )

#         self.status_w = TextEdit(value="")
#         try:
#             self.status_w.native.setReadOnly(True)
#             self.status_w.native.setMinimumHeight(220)
#         except AttributeError:
#             pass

#         self.progress = QtWidgets.QProgressBar()
#         self.progress.setMinimum(0)
#         self.progress.setMaximum(100)
#         self.progress.setValue(0)
#         self.progress.setTextVisible(True)

#         self.export_vtr_w = FileEdit(mode="w", label="Output VTK (.vtr)")
#         _ = set_file_button_symbol(self.export_vtr_w, "📂")

#         self.btn_view = PushButton(text="🔭 View RSM")
#         self.btn_export = PushButton(text="💾 Export to VTK")

#         # Create a plain Qt QPushButton for "Run All" with direct inline styling
#         # This bypasses magicgui and QSS issues to ensure the button displays correctly.
#         self.btn_run_all_native = QtWidgets.QPushButton("Run All")
#         self.btn_run_all_native.setMinimumHeight(50)
#         self.btn_run_all_native.setMinimumWidth(200)
#         self.btn_run_all_native.setFont(QtGui.QFont("", 14, QtGui.QFont.Bold))
#         # Apply inline stylesheet directly to the button (not via QSS)
#         self.btn_run_all_native.setStyleSheet("""
#             background-color: #ff9800;
#             color: white;
#             border: 2px solid #e68900;
#             border-radius: 8px;
#             padding: 8px 12px;
#             font-weight: bold;
#             font-size: 14px;
#         """)
#         # Load and set icon if available
#         with contextlib.suppress(Exception):
#             icon = load_app_icon()
#             if icon is not None:
#                 self.btn_run_all_native.setIcon(icon)
#                 self.btn_run_all_native.setIconSize(QtCore.QSize(20, 20))
#         # Button connection handled later to ensure single connection

#         # Ensure the button is connected only once to avoid duplicate execution
#         with contextlib.suppress(TypeError):
#             self.btn_run_all_native.clicked.disconnect()
#         self.btn_run_all_native.clicked.connect(self.on_run_all)

#         left_bottom = QtWidgets.QWidget()
#         vleft = QtWidgets.QVBoxLayout(left_bottom)
#         vleft.setContentsMargins(0, 0, 0, 0)
#         vleft.setSpacing(8)
#         vleft.addWidget(self.btn_view.native)
#         vleft.addWidget(self.btn_export.native)

#         btn_row3 = QtWidgets.QWidget()
#         row3 = QtWidgets.QHBoxLayout(btn_row3)
#         row3.setContentsMargins(0, 0, 0, 0)
#         row3.setSpacing(12)
#         row3.addWidget(left_bottom)
#         row3.addStretch(1)
#         row3.addWidget(self.btn_run_all_native)

#         col3 = Container(
#             layout="vertical",
#             widgets=[
#                 title_view,
#                 self.log_view_w,
#                 self.cmap_w,
#                 self.rendering_w,
#                 self.contrast_lo_w,
#                 self.contrast_hi_w,
#                 hsep(),
#             ],
#         )

#         g3 = make_group("View", col3.native)
#         g3_lay = g3.layout()
#         g3_lay.addWidget(q_hsep())
#         g3_lay.addWidget(QtWidgets.QLabel("<b>Status / Output</b>"))
#         g3_lay.addWidget(self.status_w.native)
#         g3_lay.addWidget(self.progress)
#         g3_lay.addWidget(q_hsep())
#         g3_lay.addWidget(QtWidgets.QLabel("<b>Export</b>"))
#         g3_lay.addWidget(self.export_vtr_w.native)
#         g3_lay.addStretch(1)
#         g3_lay.addWidget(q_hsep())
#         g3_lay.addWidget(btn_row3)

#         tab_view = make_scroll(g3)

#         # Tabs container
#         tabs = QtWidgets.QTabWidget()
#         tabs.addTab(tab_data, "Data")
#         tabs.addTab(tab_build, "Build")
#         tabs.addTab(tab_view, "View")

#         outer = QtWidgets.QVBoxLayout(self)
#         outer.setContentsMargins(6, 6, 6, 6)
#         outer.setSpacing(6)
#         outer.addWidget(tabs)

#         self.status_bar = QtWidgets.QStatusBar()
#         outer.addWidget(self.status_bar)

#         # YAML <-> UI binding
#         self._widget_map: dict[str, dict[str, Any]] = {
#             "data": {
#                 "spec_file": self.spec_file_w,
#                 "data_file": self.data_file_w,
#                 "scans": self.scans_w,
#                 "only_hkl": self.only_hkl_w,
#             },
#             "ExperimentSetup": {
#                 "distance": self.distance_w,
#                 "pitch": self.pitch_w,
#                 "ycenter": self.ycenter_w,
#                 "xcenter": self.xcenter_w,
#                 "xpixels": self.xpixels_w,
#                 "ypixels": self.ypixels_w,
#                 "energy": self.energy_w,
#                 "wavelength": self.wavelength_w,
#             },
#             "Crystal": {"ub": self.ub_matrix_w},
#             "build": {
#                 "ub_includes_2pi": self.ub_2pi_w,
#                 "center_is_one_based": self.center_one_based_w,
#                 "sample_axes": self.sample_axes_w,
#                 "detector_axes": self.detector_axes_w,
#             },
#             "crop": {
#                 "enable": self.crop_enable_w,
#                 "y_min": self.y_min_w,
#                 "y_max": self.y_max_w,
#                 "x_min": self.x_min_w,
#                 "x_max": self.x_max_w,
#             },
#             "regrid": {
#                 "space": self.space_w,
#                 "grid_shape": self.grid_shape_w,
#                 "fuzzy": self.fuzzy_w,
#                 "fuzzy_width": self.fuzzy_width_w,
#                 "normalize": self.normalize_w,
#             },
#             "view": {
#                 "log_view": self.log_view_w,
#                 "cmap": self.cmap_w,
#                 "rendering": self.rendering_w,
#                 "contrast_lo": self.contrast_lo_w,
#                 "contrast_hi": self.contrast_hi_w,
#             },
#             "export": {"vtr_path": self.export_vtr_w},
#         }

#         self._apply_yaml_to_widgets()
#         self._connect_widget_changes()
#         # Ensure DATA folder picker opens a sensible directory by default
#         if not as_path_str(self.data_file_w.value).strip():
#             self.data_file_w.value = os.path.expanduser("~")

#         # Button handlers
#         self.btn_load.clicked.connect(self.on_load)
#         self.btn_intensity.clicked.connect(self.on_view_intensity)
#         self.btn_build.clicked.connect(self.on_build)
#         self.btn_regrid.clicked.connect(self.on_regrid)
#         self.btn_view.clicked.connect(self.on_view)
#         self.btn_export.clicked.connect(self.on_export_vtk)

#     # -------------------------------------------------------------------------
#     # YAML helpers (no blind exceptions)
#     # -------------------------------------------------------------------------

#     def _apply_yaml_to_widgets(self) -> None:
#         ydoc = self._ydoc
#         for section in self._widget_map:
#             ydoc.setdefault(section, {})

#         def set_widget(widget: Any, value: Any) -> None:
#             if value is None:
#                 return
#             try:
#                 if isinstance(widget, FloatSpinBox):
#                     widget.value = float(value)
#                 elif isinstance(widget, SpinBox):
#                     widget.value = int(value)
#                 elif isinstance(widget, CheckBox):
#                     widget.value = bool(value)
#                 elif isinstance(widget, ComboBox):
#                     sval = str(value)
#                     if sval in widget.choices:
#                         widget.value = sval
#                 elif isinstance(widget, (LineEdit, TextEdit, FileEdit)):
#                     widget.value = str(value)
#             except (TypeError, ValueError, AttributeError):
#                 return

#         for section, mapping in self._widget_map.items():
#             vals = ydoc.get(section, {}) or {}
#             for key, widget in mapping.items():
#                 set_widget(widget, vals.get(key))

#         ub_txt = str(ydoc.get("Crystal", {}).get("ub") or "").strip()
#         if not ub_txt:
#             self._ydoc["Crystal"]["ub"] = self.ub_matrix_w.value
#             save_yaml(self._yaml_path, self._ydoc)

#     def _widget_value_for_yaml(
#         self, widget: Any, section: str, key: str
#     ) -> Any:
#         if section == "ExperimentSetup" and key == "wavelength":
#             txt = str(widget.value).strip()
#             if txt.lower() in {"", "none", "null"}:
#                 return None
#             try:
#                 return float(txt)
#             except ValueError:
#                 return txt

#         if section == "Crystal" and key == "ub":
#             txt = str(widget.value).strip()
#             return txt or None

#         if isinstance(widget, FloatSpinBox):
#             return float(widget.value)
#         if isinstance(widget, SpinBox):
#             return int(widget.value)
#         if isinstance(widget, CheckBox):
#             return bool(widget.value)
#         if isinstance(widget, ComboBox):
#             return str(widget.value)
#         if isinstance(widget, (LineEdit, TextEdit, FileEdit)):
#             return str(widget.value)
#         return widget.value

#     def _connect_widget_changes(self) -> None:
#         def on_changed(section: str, key: str, widget: Any) -> None:
#             self._ydoc.setdefault(section, {})
#             self._ydoc[section][key] = self._widget_value_for_yaml(
#                 widget, section, key
#             )
#             save_yaml(self._yaml_path, self._ydoc)

#         for section, mapping in self._widget_map.items():
#             for key, widget in mapping.items():
#                 widget.changed.connect(
#                     lambda *_, s=section, k=key, w=widget: on_changed(s, k, w)
#                 )

#     # -------------------------------------------------------------------------
#     # UI status helpers
#     # -------------------------------------------------------------------------

#     def pump(self, ms: int = 0) -> None:
#         QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, ms)

#     def status(self, msg: str) -> None:
#         try:
#             self.status_w.native.append(msg)
#         except AttributeError:
#             self.status_w.value = (self.status_w.value or "") + (
#                 ("\n" if self.status_w.value else "") + msg
#             )
#         self.status_bar.showMessage(msg, 3000)

#     def set_progress(self, value: int | None, *, busy: bool = False) -> None:
#         if busy:
#             self.progress.setRange(0, 0)
#         else:
#             self.progress.setRange(0, 100)
#             self.progress.setValue(int(value or 0))

#     def set_busy(self, b: bool) -> None:
#         for btn in (
#             self.btn_load,
#             self.btn_intensity,
#             self.btn_build,
#             self.btn_regrid,
#             self.btn_view,
#             self.btn_export,
#             self.btn_run_all_native,
#         ):
#             with contextlib.suppress(AttributeError):
#                 btn.native.setEnabled(not b)

#     # -------------------------------------------------------------------------
#     # Actions (BLE001-safe)
#     # -------------------------------------------------------------------------

#     def on_view_intensity(self) -> None:
#         loader = self._state.get("loader")
#         if loader is None:
#             show_error("Load data first.")
#             return

#         try:
#             _a, _b, df = loader.load()
#             frames = list(df.intensity)
#         except (OSError, ValueError, RuntimeError, TypeError, KeyError) as e:
#             show_error(f"Intensity load error: {e}")
#             return

#         try:
#             viewer_local = IntensityNapariViewer(
#                 frames,
#                 name="Intensity",
#                 log_view=True,
#                 contrast_percentiles=(1.0, 99.8),
#                 cmap="inferno",
#                 rendering="attenuated_mip",
#                 add_timeseries=True,
#                 add_volume=True,
#                 scale_tzyx=(1.0, 1.0, 1.0),
#                 pad_value=np.nan,
#             ).launch(viewer=self.viewer)
#             self._state["intensity_viewer"] = viewer_local
#         except (RuntimeError, ValueError, TypeError) as e:
#             show_error(f"Failed to open intensity viewer: {e}")

#     def on_load(self) -> None:
#         spec = as_path_str(self.spec_file_w.value).strip()
#         dpath = as_path_str(self.data_file_w.value).strip()

#         try:
#             scans = parse_scan_list((self.scans_w.value or "").strip())
#         except ValueError as e:
#             show_error(str(e))
#             return

#         if not spec or not os.path.isfile(spec):
#             show_error("Select a valid SPEC file.")
#             return
#         if not os.path.isdir(dpath):
#             show_error("Select a valid DATA folder.")
#             return
#         if not scans:
#             show_error("Enter at least one scan (e.g. '17, 18-22').")
#             return

#         self.set_busy(True)
#         self.set_progress(None, busy=True)
#         self.status(f"Loading scans {scans}…")
#         self.pump(50)

#         try:
#             tiff_arg = dpath
#             loader = RSMDataLoader(
#                 spec,
#                 yaml_path(),
#                 tiff_arg,
#                 selected_scans=scans,
#                 process_hklscan_only=bool(self.only_hkl_w.value),
#             )
#             load_result = loader.load()

#             ub_from_load = None
#             if isinstance(load_result, tuple) and len(load_result) >= 2:
#                 ub_from_load = load_result[1]
#             if ub_from_load is None:
#                 ub_from_load = getattr(loader, "ub", None)

#             self._state["ub"] = ub_from_load
#             self.ub_matrix_w.value = format_ub_matrix(ub_from_load)

#             self._state["loader"] = loader
#             self._state["builder"] = None
#             self._state["grid"] = None
#             self._state["edges"] = None

#             self.set_progress(25, busy=False)
#             self.status("Data loaded.")
#         except (OSError, ValueError, RuntimeError, TypeError, KeyError) as e:
#             show_error(f"Load error: {e}")
#             self.set_progress(0, busy=False)
#             self.status(f"Load failed: {e}")
#         finally:
#             self.set_busy(False)

#     def on_build(self) -> None:
#         if self._state.get("loader") is None:
#             show_error("Load data first.")
#             return

#         self.set_busy(True)
#         self.set_progress(None, busy=True)
#         self.status("Computing Q/HKL/intensity…")
#         self.pump(50)

#         try:
#             b = RSMBuilder(
#                 self._state["loader"],
#                 sample_axes=parse_axes_list(self.sample_axes_w.value),
#                 detector_axes=parse_axes_list(self.detector_axes_w.value),
#                 ub_includes_2pi=bool(self.ub_2pi_w.value),
#                 center_is_one_based=bool(self.center_one_based_w.value),
#             )
#             b.compute_full(verbose=False)
#             self._state["builder"] = b
#             self._state["grid"] = None
#             self._state["edges"] = None
#             self.set_progress(50, busy=False)
#             self.status("RSM map built.")
#         except (ValueError, RuntimeError, TypeError, KeyError) as e:
#             show_error(f"Build error: {e}")
#             self.set_progress(40, busy=False)
#             self.status(f"Build failed: {e}")
#         finally:
#             self.set_busy(False)

#     def on_regrid(self) -> None:
#         b = self._state.get("builder")
#         if b is None:
#             show_error("Build the RSM map first.")
#             return

#         try:
#             gx, gy, gz = parse_grid_shape(self.grid_shape_w.value)
#         except ValueError as e:
#             show_error(str(e))
#             return
#         if gx is None:
#             show_error("Grid X (first value) is required (e.g., 200,*,*).")
#             return

#         do_crop = bool(self.crop_enable_w.value)
#         ymin, ymax = int(self.y_min_w.value), int(self.y_max_w.value)
#         xmin, xmax = int(self.x_min_w.value), int(self.x_max_w.value)

#         self.set_busy(True)
#         self.set_progress(None, busy=True)
#         self.status(
#             f"Regridding to {self.space_w.value.upper()} grid {(gx, gy, gz)}…"
#         )
#         self.pump(50)

#         try:
#             if do_crop:
#                 if ymin >= ymax or xmin >= xmax:
#                     raise ValueError(
#                         "Crop bounds must satisfy y_min < y_max and x_min < x_max."
#                     )
#                 loader = self._state.get("loader")
#                 if loader is None:
#                     raise RuntimeError(
#                         "Internal error: loader missing; run Build again."
#                     )

#                 b = RSMBuilder(
#                     loader,
#                     sample_axes=parse_axes_list(self.sample_axes_w.value),
#                     detector_axes=parse_axes_list(self.detector_axes_w.value),
#                     ub_includes_2pi=bool(self.ub_2pi_w.value),
#                     center_is_one_based=bool(self.center_one_based_w.value),
#                 )
#                 b.compute_full(verbose=False)
#                 b.crop_by_positions(y_bound=(ymin, ymax), x_bound=(xmin, xmax))

#             kwargs: dict[str, Any] = {
#                 "space": self.space_w.value,
#                 "grid_shape": (gx, gy, gz),
#                 "fuzzy": bool(self.fuzzy_w.value),
#                 "normalize": self.normalize_w.value,
#                 "stream": True,
#             }
#             if (
#                 bool(self.fuzzy_w.value)
#                 and float(self.fuzzy_width_w.value or 0) > 0
#             ):
#                 kwargs["width"] = float(self.fuzzy_width_w.value)

#             grid, edges = b.regrid_xu(**kwargs)
#             self._state["grid"], self._state["edges"] = grid, edges

#             self.set_progress(75, busy=False)
#             self.status("Regrid completed.")
#         except (ValueError, RuntimeError, TypeError, KeyError) as e:
#             show_error(f"Regrid error: {e}")
#             self.set_progress(60, busy=False)
#             self.status(f"Regrid failed: {e}")
#         finally:
#             self.set_busy(False)

#     def on_view(self) -> None:
#         if self._state.get("grid") is None or self._state.get("edges") is None:
#             show_error("Regrid first.")
#             return

#         try:
#             lo = float(self.contrast_lo_w.value)
#             hi = float(self.contrast_hi_w.value)
#             if not (0 <= lo < hi <= 100):
#                 raise ValueError(
#                     "Contrast % must satisfy 0 ≤ low < high ≤ 100"
#                 )
#         except ValueError as e:
#             show_error(str(e))
#             return

#         self.set_progress(None, busy=True)
#         self.status("Opening RSM viewer…")
#         self.pump(50)

#         try:
#             viz = RSMNapariViewer(
#                 self._state["grid"],
#                 self._state["edges"],
#                 space=self.space_w.value,
#                 name="RSM3D",
#                 log_view=bool(self.log_view_w.value),
#                 contrast_percentiles=(lo, hi),
#                 cmap=self.cmap_w.value,
#                 rendering=self.rendering_w.value,
#             )
#             viewer_local = viz.launch(viewer=self.viewer)
#             self._state["rsm_viewer"] = viewer_local
#             self.set_progress(100, busy=False)
#             self.status("RSM viewer opened.")
#         except (RuntimeError, ValueError, TypeError) as e:
#             show_error(f"View error: {e}")
#             self.set_progress(80, busy=False)
#             self.status(f"View failed: {e}")

#     def on_export_vtk(self) -> None:
#         if self._state.get("grid") is None or self._state.get("edges") is None:
#             show_error("Regrid first, then export.")
#             return

#         out_path = as_path_str(self.export_vtr_w.value).strip()
#         if not out_path:
#             show_error("Choose an output .vtr file path.")
#             return
#         if not out_path.lower().endswith(".vtr"):
#             out_path += ".vtr"

#         self.set_busy(True)
#         self.set_progress(None, busy=True)
#         self.status(f"Exporting VTK (.vtr) → {out_path}")
#         self.pump(50)

#         try:
#             write_rsm_volume_to_vtr(
#                 self._state["grid"],
#                 self._state["edges"],
#                 out_path,
#                 binary=False,
#                 compress=True,
#             )
#             self.set_progress(100, busy=False)
#             self.status(f"Exported: {out_path}")
#         except (OSError, RuntimeError, ValueError, TypeError) as e:
#             show_error(f"Export error: {e}")
#             self.set_progress(0, busy=False)
#             self.status(f"Export failed: {e}")
#         finally:
#             self.set_busy(False)

#     def on_run_all(self) -> None:
#         self.btn_run_all_native.setEnabled(False)
#         try:
#             self.set_progress(0, busy=False)
#             self.status("Running pipeline (Load → Build → Regrid → View)…")

#             self.on_load()
#             if self._state.get("loader") is None:
#                 return

#             self.on_build()
#             if self._state.get("builder") is None:
#                 return

#             self.on_regrid()
#             if (
#                 self._state.get("grid") is None
#                 or self._state.get("edges") is None
#             ):
#                 return

#             self.on_view()
#             self.status("Run All completed.")
#         finally:
#             self.btn_run_all_native.setEnabled(True)


# # -----------------------------------------------------------------------------
# # Optional: npe2 dock widget provider
# # -----------------------------------------------------------------------------


# def napari_experimental_provide_dock_widget():
#     # Returning the class is still fine; with viewer optional, both patterns work.
#     return ResviewDockWidget

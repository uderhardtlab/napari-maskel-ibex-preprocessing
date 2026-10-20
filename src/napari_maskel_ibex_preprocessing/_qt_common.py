"""GUI parts shared by the plugin windows: logo, run-button style, status messages from the
worker thread, region from a Shapes layer, running a function in a background thread."""
from pathlib import Path
import numpy as np
from qtpy.QtCore import QObject, Signal, Qt
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import QLabel
from napari.layers import Image, Labels, Shapes
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_error, show_info

NONE = "(none)"
LOGO = Path(__file__).parent / "resources" / "logo_preprocessing.png"
LOGO_WIDTH = 380                     # px; large enough that "IBEX preprocessing" is readable
GREEN, DARK_GREEN = "#648218", "#32410c"   # magnifying glass / (old) text colour of the logo
RUN_STYLE = (f"QPushButton {{ background-color: {GREEN}; color: white; font-weight: bold; "
             f"border-radius: 4px; padding: 5px; }}"
             f"QPushButton:hover {{ background-color: #74961c; }}"
             f"QPushButton:disabled {{ background-color: {DARK_GREEN}; color: #9aa58a; }}")


class Messenger(QObject):
    """Carries status messages from the worker thread to the GUI thread (queued signal)."""
    message = Signal(str)


def logo_label():
    """The logo (transparent background, readable on napari's dark theme)."""
    logo = QLabel()
    pix = QPixmap(str(LOGO))
    if not pix.isNull():
        logo.setPixmap(pix.scaledToWidth(LOGO_WIDTH, Qt.SmoothTransformation))
    logo.setAlignment(Qt.AlignCenter)
    return logo


def image_names(viewer):
    """Channel layers (not results of this plugin)."""
    return [l.name for l in viewer.layers if isinstance(l, Image)
            and not l.metadata.get("maskel_derived")] or [NONE]


def mask_names(viewer):
    """Any Labels or Image layer that could be a binary mask."""
    return [l.name for l in viewer.layers if isinstance(l, (Labels, Image))] or [NONE]


def shape_names(viewer):
    return [NONE] + [l.name for l in viewer.layers if isinstance(l, Shapes)]


def region_from_shapes(shapes, image):
    """(y0, y1, x0, x1) in full-resolution pixels of `image` from the LAST shape of a Shapes
    layer (its bounding box in the XY plane)."""
    if shapes is None or len(shapes.data) == 0:
        raise ValueError("draw a rectangle in a Shapes layer and choose it as region")
    sp = np.asarray(image.scale, float)
    verts = np.asarray(shapes.data[-1], float)
    world = np.array([shapes.data_to_world(v) for v in verts])
    px = (world[:, -2:] - np.asarray(image.translate)[-2:]) / sp[-2:]
    ny, nx = image.data.shape[-2:]
    y0, x0 = np.clip(np.floor(px.min(0)), 0, [ny, nx]).astype(int)
    y1, x1 = np.clip(np.ceil(px.max(0)), 0, [ny, nx]).astype(int)
    if y1 - y0 < 64 or x1 - x0 < 64:
        raise ValueError("region too small (needs at least 64 x 64 px)")
    return int(y0), int(y1), int(x0), int(x1)


def run_in_thread(owner, fn, done):
    """Runs fn(log) in a background thread, then done(result) in the GUI thread. `owner` needs
    the attributes _busy, buttons (magicgui buttons) and _msg (Messenger), and _status(text)."""
    if owner._busy:
        show_info("busy - wait for the running step to finish")
        return
    owner._busy = True
    for b in owner.buttons:
        b.enabled = False
    worker = thread_worker(fn, start_thread=False)(owner._msg.message.emit)

    def finish(*_):
        owner._busy = False
        for b in owner.buttons:
            b.enabled = True

    def on_err(e):
        owner._status(f"ERROR: {e}")
        show_error(str(e))

    worker.returned.connect(done)
    worker.errored.connect(on_err)
    worker.finished.connect(finish)
    worker.start()

"""Cx43 window: Cx43-positive volume near the fibroblast mask, object table export.

Inputs are LAYERS: the Cx43 channel and a fibroblast mask (e.g. "fibroblast mask FINAL [full]"
of the first window, or a mask loaded from a TIFF; full-resolution voxel grid of the channel).
"Run on region" uses the last rectangle of the chosen Shapes layer (all z), "Run on full
volume" the whole image; both in a background thread. See core/cx43.py for the method.
"""
from pathlib import Path
import numpy as np
from qtpy.QtWidgets import QWidget, QVBoxLayout, QScrollArea
from magicgui.widgets import Container, ComboBox, FloatSpinBox, PushButton, Label, FileEdit
from napari.utils.notifications import show_error

from .core.params import defaults
from .core.io import VolumeSource
from .core.cx43 import compute_cx43, write_table
from ._qt_common import (NONE, RUN_STYLE, Messenger, logo_label, image_names, mask_names,
                         shape_names, region_from_shapes, run_in_thread)


def _fspin(label, value, lo, hi, step=0.1, tip=""):
    return FloatSpinBox(label=label, value=value, min=lo, max=hi, step=step, tooltip=tip)


class Cx43Widget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.results = {}                                # per scope: "region" / "full"
        self._busy = False
        self._msg = Messenger()
        self._msg.message.connect(self._status)
        p = defaults()
        v = self.viewer
        self.w_cx = ComboBox(label="Cx43 channel", choices=lambda *_: image_names(v))
        self.w_mask = ComboBox(label="fibroblast mask", choices=lambda *_: mask_names(v),
                               tooltip="binary mask on the full-resolution grid of the channel")
        self.w_region = ComboBox(label="region (Shapes layer)", choices=lambda *_: shape_names(v))
        self.w_sy = _fspin("smoothing σ y (px)", p["cx43_smooth_px"][1], 0, 20, 0.1,
                           "pixels: tied to the line step in Y of the acquisition")
        self.w_sx = _fspin("smoothing σ x (px)", p["cx43_smooth_px"][2], 0, 20, 0.1)
        self.w_bg = _fspin("background window (µm)", p["cx43_bg_um"], 1, 200, 0.5)
        self.w_thr = _fspin("threshold T (grey values above background)", p["cx43_threshold"],
                            0, 255, 0.5)
        self.w_dist = _fspin("max. distance to fibroblast mask (µm)", p["cx43_max_dist_um"],
                             0, 100, 0.5)
        self.b_reg = PushButton(text="4: Cx43 – run on region")
        self.b_full = PushButton(text="4: Cx43 – run on full volume")
        self.w_dir = FileEdit(label="output folder", mode="d")
        self.b_csv = PushButton(text="Export object table CSV")
        self.status = Label(value="Choose the Cx43 channel and the fibroblast mask.")

        def head(t):
            return Label(value=f"<b>{t}</b>")

        self.container = Container(widgets=[
            head("Data"), self.w_cx, self.w_mask, self.w_region,
            head("4: Cx43-positive volume"), self.w_sy, self.w_sx, self.w_bg, self.w_thr,
            self.w_dist, self.b_reg, self.b_full,
            head("Export"), self.w_dir, self.b_csv, head("Status"), self.status])
        self.container.native.layout().insertWidget(0, logo_label())
        for b in (self.b_reg, self.b_full):
            b.native.setStyleSheet(RUN_STYLE)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.container.native)
        QVBoxLayout(self).addWidget(scroll)

        self.buttons = [self.b_reg, self.b_full, self.b_csv]
        self.b_reg.changed.connect(lambda: self._run("region"))
        self.b_full.changed.connect(lambda: self._run("full"))
        self.b_csv.changed.connect(self._export)
        for ev in (v.layers.events.inserted, v.layers.events.removed):
            ev.connect(self._refresh)
        self._refresh()

    # ------------------------------------------------------------------ helpers
    def _refresh(self, *_):
        for w in (self.w_cx, self.w_mask, self.w_region):
            w.reset_choices()
        p = defaults()
        srcs = {self.viewer.layers[n].metadata.get("maskel_source")
                for n in image_names(self.viewer) if n != NONE} - {None}
        new = srcs - getattr(self, "_seen", set())
        if new:                                          # new sample: default Cx43 channel
            self._seen = getattr(self, "_seen", set()) | new
            for n in image_names(self.viewer):
                md = self.viewer.layers[n].metadata if n != NONE else {}
                if md.get("maskel_source") in new and md.get("maskel_channel") == p["cx43_channel"]:
                    self.w_cx.value = n
                    if str(self.w_dir.value) in ("", "."):
                        self.w_dir.value = Path(md["maskel_source"]).parent
        # default mask: the full fibroblast mask of the first window (else its region result),
        # unless a fibroblast mask is already chosen
        names = mask_names(self.viewer)
        if "fibroblast" not in str(self.w_mask.value):
            for cand in ("fibroblast mask FINAL [full]", "fibroblast mask FINAL [region]"):
                if cand in names:
                    self.w_mask.value = cand
                    break

    def _status(self, text):
        self.status.value = text

    def _layer(self, name):
        return self.viewer.layers[name] if name not in (None, NONE) and name in self.viewer.layers else None

    def _params(self):
        p = defaults()
        p.update(cx43_smooth_px=[0.0, self.w_sy.value, self.w_sx.value], cx43_bg_um=self.w_bg.value,
                 cx43_threshold=self.w_thr.value, cx43_max_dist_um=self.w_dist.value)
        return p

    @staticmethod
    def _mask_on(region_src, img, mask_layer):
        """The mask layer sampled on the pixel grid of the region of `img`. The mask must have
        the voxel size of the channel; its translate may be offset by whole voxels (e.g. a
        region result). Parts of the region not covered by the mask layer are False."""
        sp = np.asarray(img.scale, float)
        if not np.allclose(np.asarray(mask_layer.scale, float), sp, rtol=1e-3):
            raise ValueError("the fibroblast mask must have the same voxel size as the Cx43 "
                             "channel (full-resolution grid)")
        off = (np.asarray(mask_layer.translate, float) - np.asarray(img.translate, float)) / sp
        off_i = np.round(off).astype(int)
        if not np.allclose(off, off_i, atol=0.05):
            raise ValueError("the fibroblast mask is not aligned to the voxel grid of the channel")
        nz, ny, nx = region_src.shape
        y0, _, x0, _ = region_src.region
        mz, my, mx = mask_layer.data.shape
        out = np.zeros((nz, ny, nx), bool)
        # overlap in channel pixel coordinates
        za, zb = max(0, off_i[0]), min(nz, off_i[0] + mz)
        ya, yb = max(y0, off_i[1]), min(y0 + ny, off_i[1] + my)
        xa, xb = max(x0, off_i[2]), min(x0 + nx, off_i[2] + mx)
        if za >= zb or ya >= yb or xa >= xb:
            raise ValueError("the fibroblast mask does not overlap this region")
        part = np.asarray(mask_layer.data[za - off_i[0]:zb - off_i[0], ya - off_i[1]:yb - off_i[1],
                                          xa - off_i[2]:xb - off_i[2]])
        out[za:zb, ya - y0:yb - y0, xa - x0:xb - x0] = part > 0
        return out

    # ------------------------------------------------------------------ run
    def _run(self, scope):
        try:
            img = self._layer(self.w_cx.value)
            mlay = self._layer(self.w_mask.value)
            if img is None or mlay is None:
                raise ValueError("choose the Cx43 channel and the fibroblast mask")
            region = None
            if scope == "region":
                region = region_from_shapes(self._layer(self.w_region.value), img)
            src = VolumeSource([img.data], img.scale, region)
            if scope == "full" and mlay.data.shape != img.data.shape:
                raise ValueError("for the full volume, the fibroblast mask must cover the whole "
                                 f"image (shape {img.data.shape}); choose the [full] mask")
        except Exception as e:
            show_error(str(e))
            return
        p = self._params()

        def fn(log):
            log(f"Cx43 on {scope} {src.region} ...")
            mask = self._mask_on(src, img, mlay)
            res = compute_cx43(src.volume(0), mask, src.sp, p, origin_yx=src.region[0::2], log=log)
            res.update(region=src.region, sp=src.sp)
            return res

        def done(res):
            self.results[scope] = res
            sp = res["sp"]
            y0, _, x0, _ = res["region"]
            tr = (0.0, y0 * sp[1], x0 * sp[2])
            tag = f"[{scope}]"
            meta = dict(maskel_derived=True)
            for name, data, kind, kw in (
                    (f"Cx43 contrast {tag}", res["contrast"], "image",
                     dict(colormap="magma", contrast_limits=[-10, 60], visible=False)),
                    (f"distance to fibroblast mask (µm) {tag}",
                     np.where(np.isfinite(res["distance"]), res["distance"], 999).astype(np.float32),
                     "image", dict(colormap="viridis", contrast_limits=[0, 10], visible=False)),
                    (f"Cx43 positive {tag}", res["labels"], "labels", dict(opacity=0.7))):
                if name in self.viewer.layers:
                    self.viewer.layers.remove(name)
                add = self.viewer.add_labels if kind == "labels" else self.viewer.add_image
                add(data, name=name, scale=sp, translate=tr, metadata=meta, **kw)
            self._status(f"Cx43 done ({scope}): {res['n_objects']} objects, "
                         f"{res['volume_total_um3']:.0f} µm³ positive within "
                         f"{p['cx43_max_dist_um']:g} µm of the mask "
                         f"({res['volume_in_mask_um3']:.0f} µm³ inside)")

        run_in_thread(self, fn, done)

    # ------------------------------------------------------------------ export
    def _export(self):
        scope = "full" if "full" in self.results else ("region" if "region" in self.results else None)
        if scope is None:
            show_error("run the Cx43 step first")
            return
        d = Path(str(self.w_dir.value))
        if str(self.w_dir.value) in ("", "."):
            show_error("choose an output folder")
            return
        img = self._layer(self.w_cx.value)
        src = img.metadata.get("maskel_source") if img is not None else None
        stem = Path(src).stem if src else "sample"
        path = d / f"{stem}_cx43_objects{'' if scope == 'full' else '_region'}.csv"
        try:
            d.mkdir(parents=True, exist_ok=True)
            write_table(path, self.results[scope]["table"])
        except Exception as e:
            show_error(str(e))
            return
        self._status(f"written {path} ({scope} result, {self.results[scope]['n_objects']} objects)")

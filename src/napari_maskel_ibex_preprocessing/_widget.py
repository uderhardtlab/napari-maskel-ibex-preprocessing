"""Dock widget: muscle surface -> CD34 mask -> distance filter -> TIFF export.

Every step can run on a REGION (the last rectangle of a chosen Shapes layer; full z range) for
fast parameter tuning, or on the FULL VOLUME. Computations run in a background thread; the
status line shows progress. Results appear as layers named "... [region]" / "... [full]".
Export needs the full-volume results (the TIFFs have the original image shape).
"""
from pathlib import Path
import numpy as np
import dask.array as da
from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import QWidget, QVBoxLayout, QScrollArea
from magicgui.widgets import (Container, ComboBox, FloatSpinBox, SpinBox, CheckBox, PushButton,
                              Label, FileEdit)
from napari.layers import Image, Shapes
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_error, show_info

from .core.params import defaults
from .core.io import VolumeSource, write_mask_tif, new_steps_file, save_step
from .core.muscle import measure_orientation, compute_surface
from .core.cd34 import compute_cd34, h_for_region
from .core.submeso import compute_submeso

NONE = "(none)"


class _Messenger(QObject):
    """Carries status messages from the worker thread to the GUI thread (queued signal)."""
    message = Signal(str)


def _fspin(label, value, lo, hi, step=0.1, tip=""):
    return FloatSpinBox(label=label, value=value, min=lo, max=hi, step=step, tooltip=tip)


class PreprocessingWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.results = {"surface": {}, "cd34": {}, "final": {}}   # per scope: "region" / "full"
        self._busy = False
        self._msg = _Messenger()
        self._msg.message.connect(self._status)
        p = defaults()

        # ---- data
        self.w_muscle = ComboBox(label="muscle channel (TOM20)", choices=self._image_names)
        self.w_cd34 = ComboBox(label="CD34 channel", choices=self._image_names)
        self.w_region = ComboBox(label="region (Shapes layer)", choices=self._shape_names,
                                 tooltip="Last rectangle of this Shapes layer = region for 'Run "
                                         "on region' (all z). Draw it in the 2D view.")
        # ---- step 1
        self.w_ds = SpinBox(label="block size (px)", value=p["layers_ds"], min=1, max=16)
        self.w_tz = _fspin("TOM20 smoothing σ z (µm)", p["tom_sigma_um"][0], 0, 50)
        self.w_ty = _fspin("TOM20 smoothing σ y (µm)", p["tom_sigma_um"][1], 0, 50)
        self.w_tx = _fspin("TOM20 smoothing σ x (µm)", p["tom_sigma_um"][2], 0, 50)
        self.w_mauto = CheckBox(label="muscle threshold automatic (lowest 3-class Otsu)", value=True)
        self.w_mthr = _fspin("muscle threshold (if not automatic)", 12.0, 0, 255, 0.5)
        self.w_mvol = _fspin("min. muscle object (µm³)", p["muscle_min_volume_um3"], 0, 1e7, 500)
        self.w_line = _fspin("fibre filter line length (µm)", p["fibre_line_um"], 0, 500, 1)
        self.w_aauto = CheckBox(label="measure fibre direction automatically", value=True)
        self.w_angle = _fspin("fibre direction (°, x right, y down)", 0.0, -90, 90, 0.5)
        self.w_close = _fspin("surface closing radius (µm)", p["close_radius_um"], 0, 100, 0.1)
        self.b1r = PushButton(text="1  Muscle surface: run on region")
        self.b1f = PushButton(text="1  Muscle surface: run on full volume")
        # ---- step 2
        self.w_tis = _fspin("tissue smoothing σ (µm)", p["tissue_smooth_um"], 0, 50)
        self.w_cz = _fspin("CD34 smoothing σ z (µm)", p["cd34_smooth_um"][0], 0, 20)
        self.w_cy = _fspin("CD34 smoothing σ y (µm)", p["cd34_smooth_um"][1], 0, 20)
        self.w_cx = _fspin("CD34 smoothing σ x (µm)", p["cd34_smooth_um"][2], 0, 20)
        self.w_bg = _fspin("CD34 background σ (µm)", p["bg_sigma_um"], 0, 200, 1)
        self.w_cvol = _fspin("min. CD34 object (µm³)", p["cd34_min_volume_um3"], 0, 1e6, 5)
        self.w_k = _fspin("hysteresis low threshold: k · SD", p["hyst_k_sd"], 0, 50, 0.5)
        self.w_blk = _fspin("noise block size (µm)", p["hyst_block_um"], 10, 5000, 10)
        self.w_blo = _fspin("noise band from d = (µm)", p["hyst_noise_band_um"][0], -200, 200, 1)
        self.w_bhi = _fspin("noise band to d = (µm)", p["hyst_noise_band_um"][1], -200, 200, 1)
        self.b2r = PushButton(text="2  CD34 mask: run on region")
        self.b2f = PushButton(text="2  CD34 mask: run on full volume")
        # ---- step 3
        self.w_cut = _fspin("distance cutoff: keep d > (µm)", p["cutoff_um"], -200, 200, 1)
        self.w_comp = _fspin("min. CD34 component (µm³)", p["min_component_um3"], 0, 1e8, 50)
        self.b3r = PushButton(text="3  Distance filter: run on region")
        self.b3f = PushButton(text="3  Distance filter: run on full volume")
        # ---- export
        self.w_dir = FileEdit(label="output folder", mode="d")
        self.w_dthr = _fspin("distance TIFF: 1 where d >", 0.0, -500, 500, 0.5)
        self.w_dunit = ComboBox(label="unit of this threshold", choices=["µm", "z slices"])
        self.w_comp_tif = CheckBox(label="ZIP-compress TIFFs", value=True)
        self.b4d = PushButton(text="Export distance mask TIFF (full volume)")
        self.b4m = PushButton(text="Export fibroblast mask TIFF (full volume)")
        self.b4s = PushButton(text="Save all results as steps .h5")
        self.status = Label(value="Load a converted .h5 (File > Open), choose the channels.")

        def head(t):
            return Label(value=f"<b>{t}</b>")

        self.container = Container(widgets=[
            head("Data"), self.w_muscle, self.w_cd34, self.w_region,
            head("1  Muscle surface (TOM20)"), self.w_ds, self.w_tz, self.w_ty, self.w_tx,
            self.w_mauto, self.w_mthr, self.w_mvol, self.w_line, self.w_aauto, self.w_angle,
            self.w_close, self.b1r, self.b1f,
            head("2  CD34 mask"), self.w_tis, self.w_cz, self.w_cy, self.w_cx, self.w_bg,
            self.w_cvol, self.w_k, self.w_blk, self.w_blo, self.w_bhi, self.b2r, self.b2f,
            head("3  Distance filter"), self.w_cut, self.w_comp, self.b3r, self.b3f,
            head("Export"), self.w_dir, self.w_dthr, self.w_dunit, self.w_comp_tif,
            self.b4d, self.b4m, self.b4s,
            head("Status"), self.status])
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.container.native)
        lay = QVBoxLayout(self)
        lay.addWidget(scroll)

        self.buttons = [self.b1r, self.b1f, self.b2r, self.b2f, self.b3r, self.b3f,
                        self.b4d, self.b4m, self.b4s]
        self.b1r.changed.connect(lambda: self._step1("region"))
        self.b1f.changed.connect(lambda: self._step1("full"))
        self.b2r.changed.connect(lambda: self._step2("region"))
        self.b2f.changed.connect(lambda: self._step2("full"))
        self.b3r.changed.connect(lambda: self._step3("region"))
        self.b3f.changed.connect(lambda: self._step3("full"))
        self.b4d.changed.connect(self._export_distance)
        self.b4m.changed.connect(self._export_mask)
        self.b4s.changed.connect(self._save_steps)
        self.w_muscle.changed.connect(self._default_folder)
        for ev in (self.viewer.layers.events.inserted, self.viewer.layers.events.removed):
            ev.connect(self._refresh)
        self._refresh()

    # ------------------------------------------------------------------ helpers
    def _image_names(self, *_):
        return [l.name for l in self.viewer.layers if isinstance(l, Image)
                and not l.metadata.get("maskel_derived")] or [NONE]

    def _shape_names(self, *_):
        return [NONE] + [l.name for l in self.viewer.layers if isinstance(l, Shapes)]

    def _refresh(self, *_):
        for w in (self.w_muscle, self.w_cd34, self.w_region):
            w.reset_choices()
        # defaults when a NEW sample appears (reader metadata): muscle = channel of
        # DEFAULTS["muscle_channel"], CD34 = DEFAULTS["cd34_channel"]; later user choices stay
        p = defaults()
        srcs = {self.viewer.layers[n].metadata.get("maskel_source")
                for n in self._image_names() if n != NONE}
        srcs.discard(None)
        new = srcs - getattr(self, "_seen_sources", set())
        if new:
            self._seen_sources = getattr(self, "_seen_sources", set()) | new
            src = sorted(new)[-1]
            for w, idx in ((self.w_muscle, p["muscle_channel"]), (self.w_cd34, p["cd34_channel"])):
                for n in self._image_names():
                    md = self.viewer.layers[n].metadata if n != NONE else {}
                    if md.get("maskel_source") == src and md.get("maskel_channel") == idx:
                        w.value = n
        self._default_folder()

    def _default_folder(self, *_):
        lay = self._layer(self.w_muscle.value)
        if lay is not None and lay.metadata.get("maskel_source") and \
                str(self.w_dir.value) in ("", "."):
            self.w_dir.value = Path(lay.metadata["maskel_source"]).parent

    def _layer(self, name):
        return self.viewer.layers[name] if name not in (None, NONE) and name in self.viewer.layers else None

    def _status(self, text):
        self.status.value = text

    def _params(self):
        p = defaults()
        p.update(layers_ds=int(self.w_ds.value),
                 tom_sigma_um=[self.w_tz.value, self.w_ty.value, self.w_tx.value],
                 muscle_threshold=None if self.w_mauto.value else self.w_mthr.value,
                 muscle_min_volume_um3=self.w_mvol.value, fibre_line_um=self.w_line.value,
                 fibre_theta_deg=None if self.w_aauto.value else self.w_angle.value,
                 close_radius_um=self.w_close.value, tissue_smooth_um=self.w_tis.value,
                 cd34_smooth_um=[self.w_cz.value, self.w_cy.value, self.w_cx.value],
                 bg_sigma_um=self.w_bg.value, cd34_min_volume_um3=self.w_cvol.value,
                 hyst_k_sd=self.w_k.value, hyst_block_um=self.w_blk.value,
                 hyst_noise_band_um=[self.w_blo.value, self.w_bhi.value],
                 cutoff_um=self.w_cut.value, min_component_um3=self.w_comp.value)
        return p

    def _channels(self):
        """All channel layers of the same sample as the muscle channel (reader metadata), in
        channel order; without metadata: all non-derived image layers of the same shape."""
        m = self._layer(self.w_muscle.value)
        if m is None:
            raise ValueError("choose the muscle channel")
        src = m.metadata.get("maskel_source")
        lays = [l for l in self.viewer.layers if isinstance(l, Image)
                and not l.metadata.get("maskel_derived") and l.data.shape == m.data.shape
                and (src is None or l.metadata.get("maskel_source") == src)]
        lays.sort(key=lambda l: l.metadata.get("maskel_channel", 0))
        return lays, m

    def _source(self, scope):
        lays, m = self._channels()
        sp = np.asarray(m.scale, float)
        region = None
        if scope == "region":
            sh = self._layer(self.w_region.value)
            if sh is None or len(sh.data) == 0:
                raise ValueError("draw a rectangle in a Shapes layer and choose it as region")
            verts = np.asarray(sh.data[-1], float)
            world = np.array([sh.data_to_world(v) for v in verts])
            px = (world[:, -2:] - np.asarray(m.translate)[-2:]) / sp[-2:]
            ny, nx = m.data.shape[1:]
            y0, x0 = np.clip(np.floor(px.min(0)), 0, [ny, nx]).astype(int)
            y1, x1 = np.clip(np.ceil(px.max(0)), 0, [ny, nx]).astype(int)
            if y1 - y0 < 64 or x1 - x0 < 64:
                raise ValueError("region too small (needs at least 64 x 64 px)")
            region = (y0, y1, x0, x1)
        return VolumeSource([l.data for l in lays], sp, region, [l.name for l in lays]), lays, m

    def _show(self, name, data, kind, scale, translate, **kw):
        """Add or replace a result layer."""
        meta = dict(maskel_derived=True)
        if name in self.viewer.layers:
            self.viewer.layers.remove(name)
        if kind == "labels":
            return self.viewer.add_labels(data, name=name, scale=scale, translate=translate,
                                          metadata=meta, **kw)
        return self.viewer.add_image(data, name=name, scale=scale, translate=translate,
                                     metadata=meta, **kw)

    def _run(self, fn, done):
        """Runs fn(log) in a background thread, then done(result) in the GUI thread."""
        if self._busy:
            show_info("busy - wait for the running step to finish")
            return
        self._busy = True
        for b in self.buttons:
            b.enabled = False
        log = self._msg.message.emit
        worker = thread_worker(fn, start_thread=False)(log)

        def finish(*_):
            self._busy = False
            for b in self.buttons:
                b.enabled = True

        def on_err(e):
            self._status(f"ERROR: {e}")
            show_error(str(e))

        worker.returned.connect(done)
        worker.errored.connect(on_err)
        worker.finished.connect(finish)
        worker.start()

    # ------------------------------------------------------------------ step 1
    def _step1(self, scope):
        try:
            src, lays, m = self._source(scope)
        except Exception as e:
            show_error(str(e))
            return
        p = self._params()
        ch = lays.index(m)

        def fn(log):
            log(f"step 1 on {scope} {src.region} ...")
            theta = p["fibre_theta_deg"]
            orient = None
            if theta is None:
                orient = measure_orientation(src, ch, p, log)
                theta = orient["theta0_deg"]
            res = compute_surface(src, ch, theta, p, log)
            res.update(orientation=orient, full_shape=src.full_shape, sp=src.sp)
            return res

        def done(res):
            self.results["surface"][scope] = res
            for k in ("cd34", "final"):                 # later steps of this scope are stale
                self.results[k].pop(scope, None)
            self.w_angle.value = round(res["theta0_deg"], 1)
            ds, sp = res["ds"], res["sp"]
            y0, y1, x0, x1 = res["region"]
            tr = (0.0, (y0 + (ds - 1) / 2) * sp[1], (x0 + (ds - 1) / 2) * sp[2])
            sc = (sp[0], sp[1] * ds, sp[2] * ds)
            tag = f"[{scope}]"
            self._show(f"TOM20 smoothed {tag}", res["tom_smoothed"], "image", sc, tr,
                       colormap="gray", visible=False)
            self._show(f"muscle mask {tag}", res["muscle"].astype(np.uint8), "labels", sc, tr,
                       opacity=0.4, visible=False)
            zi = np.arange(res["muscle"].shape[0])[:, None, None]
            self._show(f"muscle surface {tag}", (zi == res["h"][None]).astype(np.uint8), "image",
                       sc, tr, colormap="cyan", blending="additive", contrast_limits=[0, 1])
            self._show(f"height map h (z slice) {tag}", res["h"], "image", sc[1:], tr[1:],
                       colormap="viridis", visible=False)
            self._show(f"signed distance d (µm) {tag}", self._lazy_distance(res), "image",
                       sp, (0.0, y0 * sp[1], x0 * sp[2]), colormap="PiYG",
                       contrast_limits=[-20, 20], visible=False)
            self._status(f"step 1 done ({scope}): fibre direction {res['theta0_deg']:.1f}°, "
                         f"muscle threshold {res['threshold']:.2f}")

        self._run(fn, done)

    def _lazy_distance(self, surf):
        """d = (z - h) * dz at full resolution for the region of `surf`, computed lazily."""
        y0, y1, x0, x1 = surf["region"]
        h = h_for_region(surf, surf["region"])
        nz = surf["muscle"].shape[0]
        z = da.arange(nz, chunks=1, dtype=np.float32)[:, None, None]
        hh = da.from_array(h, chunks=h.shape)[None]
        return (z - hh) * np.float32(surf["sp"][0])

    def _surface_for(self, scope, region):
        """Height map for `region`: from the surface of the same scope, else from the full one."""
        cands = [self.results["surface"].get(k) for k in dict.fromkeys((scope, "full"))]
        cands = [s for s in cands if s is not None]
        if not cands:
            raise ValueError("run step 1 (muscle surface) first")
        err = None
        for s in cands:
            try:
                return h_for_region(s, region)
            except ValueError as e:
                err = e
        raise err

    # ------------------------------------------------------------------ step 2
    def _step2(self, scope):
        try:
            src, lays, _ = self._source(scope)
            h_full = self._surface_for(scope, src.region)
            ch = lays.index(self._layer(self.w_cd34.value))
        except Exception as e:
            show_error(str(e))
            return
        p = self._params()

        def fn(log):
            log(f"step 2 on {scope} {src.region} ...")
            res = compute_cd34(src, ch, h_full, p, log)
            res.update(h_full=h_full, sp=src.sp)
            return res

        def done(res):
            self.results["cd34"][scope] = res
            self.results["final"].pop(scope, None)
            sp = res["sp"]
            y0, _, x0, _ = res["region"]
            tr = (0.0, y0 * sp[1], x0 * sp[2])
            tag = f"[{scope}]"
            self._show(f"CD34 minus background {tag}", res["s"], "image", sp, tr,
                       colormap="magma", contrast_limits=[-5, 40], visible=False)
            self._show(f"CD34 Otsu seeds {tag}", res["seeds"].astype(np.uint8), "labels", sp, tr,
                       visible=False)
            self._show(f"CD34 mask {tag}", res["mask"].astype(np.uint8), "labels", sp, tr,
                       opacity=0.5)
            self._status(f"step 2 done ({scope}): CD34 Otsu {res['otsu']:.2f}, low threshold "
                         f"{res['low_map'].min():.2f}..{res['low_map'].max():.2f}")

        self._run(fn, done)

    # ------------------------------------------------------------------ step 3
    def _step3(self, scope):
        cd = self.results["cd34"].get(scope)
        if cd is None:
            show_error(f"run step 2 on the {scope} first")
            return
        p = self._params()

        def fn(log):
            log(f"step 3 on {scope} ...")
            res = compute_submeso(cd["mask"], cd["h_full"], cd["sp"], p, log)
            res.update(region=cd["region"], sp=cd["sp"])
            return res

        def done(res):
            self.results["final"][scope] = res
            sp = res["sp"]
            y0, _, x0, _ = res["region"]
            tr = (0.0, y0 * sp[1], x0 * sp[2])
            tag = f"[{scope}]"
            self._show(f"removed small components {tag}", res["removed_small"].astype(np.uint8),
                       "labels", sp, tr, visible=False)
            self._show(f"removed by distance cutoff {tag}", res["removed_by_cut"].astype(np.uint8),
                       "labels", sp, tr, visible=False)
            self._show(f"fibroblast mask FINAL {tag}", res["final"].astype(np.uint8), "image",
                       sp, tr, colormap="green", blending="additive", contrast_limits=[0, 1])
            self._status(f"step 3 done ({scope}): {int(res['final'].sum())} voxels in the "
                         "fibroblast mask")

        self._run(fn, done)

    # ------------------------------------------------------------------ export
    def _stem(self):
        m = self._layer(self.w_muscle.value)
        src = m.metadata.get("maskel_source") if m is not None else None
        return Path(src).stem if src else "sample"

    def _outdir(self):
        d = Path(str(self.w_dir.value))
        if str(self.w_dir.value) in ("", "."):
            raise ValueError("choose an output folder")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _export_distance(self):
        surf = self.results["surface"].get("full")
        if surf is None:
            show_error("run step 1 on the full volume first (the TIFF has the original image shape)")
            return
        try:
            out = self._outdir()
        except Exception as e:
            show_error(str(e))
            return
        thr, unit, comp = self.w_dthr.value, self.w_dunit.value, self.w_comp_tif.value
        stem = self._stem()

        def fn(log):
            sp = surf["sp"]
            thr_um = thr * sp[0] if unit == "z slices" else thr
            nz, ny, nx = surf["full_shape"]
            h = h_for_region(surf, (0, ny, 0, nx))
            mask = np.empty((nz, ny, nx), bool)
            for z in range(nz):
                mask[z] = (z - h) * sp[0] > thr_um
            tag = f"{thr:g}{'sl' if unit == 'z slices' else 'um'}"
            path = out / f"{stem}_distance_gt{tag}.tif"
            log(f"writing {path.name} ...")
            write_mask_tif(path, mask, sp, comp)
            return path

        self._run(fn, lambda path: self._status(f"written {path}"))

    def _export_mask(self):
        fin = self.results["final"].get("full")
        if fin is None:
            show_error("run steps 1-3 on the full volume first")
            return
        try:
            out = self._outdir()
        except Exception as e:
            show_error(str(e))
            return
        comp, stem = self.w_comp_tif.value, self._stem()

        def fn(log):
            path = out / f"{stem}_fibroblast_mask.tif"
            log(f"writing {path.name} ...")
            write_mask_tif(path, fin["final"], fin["sp"], comp)
            return path

        self._run(fn, lambda path: self._status(f"written {path}"))

    def _save_steps(self):
        try:
            out = self._outdir()
        except Exception as e:
            show_error(str(e))
            return
        res, p, stem = self.results, self._params(), self._stem()

        def fn(log):
            written = []
            for scope in ("region", "full"):
                s, c, f = (res[k].get(scope) for k in ("surface", "cd34", "final"))
                if s is None and c is None and f is None:
                    continue
                any_res = s or c or f
                sp, region = any_res["sp"], any_res["region"]
                path = out / f"{stem}_preprocessing_{scope}.h5"
                with new_steps_file(path, sp, p, region) as o:
                    q = dict(sp=sp, log=log)
                    if s is not None:
                        ds = s["ds"]
                        org = s["region"][0::2]
                        save_step(o, "1a_tom20_smoothed_ds", s["tom_smoothed"], "image", ds=ds, origin_yx=org, **q)
                        save_step(o, "1b_muscle_ds", s["muscle"], "labels", ds=ds, origin_yx=org, **q)
                        save_step(o, "1c_h_z", s["h"], "image", ds=ds, origin_yx=org, **q)
                    if c is not None:
                        org = c["region"][0::2]
                        save_step(o, "2a_cd34_minus_background", c["s"], "image", origin_yx=org, **q)
                        save_step(o, "2b_cd34_mask", c["mask"], "labels", origin_yx=org, **q)
                    if f is not None:
                        org = f["region"][0::2]
                        save_step(o, "3a_fibroblast_mask_FINAL", f["final"], "labels", origin_yx=org, **q)
                written.append(path)
            if not written:
                raise ValueError("nothing computed yet")
            return written

        self._run(fn, lambda paths: self._status("written " + ", ".join(str(x) for x in paths)))

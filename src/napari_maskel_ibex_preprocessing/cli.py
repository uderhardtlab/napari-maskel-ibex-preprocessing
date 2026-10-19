"""Batch runner: the whole preprocessing for one sample or every sample in a folder, with the
default parameters (core/params.py) - the same functions as the napari widget.

  0  .ims -> .h5 (only if the input is an .ims without .h5)
  1  fibre direction, muscle mask and surface      -> <stem>_layers.h5
  2  tissue, CD34 Otsu + hysteresis mask           -> <stem>_cd34.h5
  3  distance filter -> fibroblast mask            -> <stem>_submeso_cd34.h5
  4  TIFFs (ZIP-compressed, original image shape)  -> <stem>_tif/
       <stem>_fibroblast_mask.tif   final mask
       <stem>_distance_gt<T>.tif    1 where d > T (T from --dist-threshold, default 0 um)
  5  orientation check: TOM20 (muscle) peak must lie at LOWER z than the CD34 peak
A sample is skipped if its fibroblast mask TIFF already exists (--force recomputes). One failing
sample does not stop the others; console output also goes to <stem>_preprocessing_log.txt.

Usage: maskel-ibex-batch <sample.ims | sample.h5 | folder> [--force] [--uncompressed]
                         [--dist-threshold T] [--dist-unit um|slices] [--dry-run]
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
from .core.params import defaults
from .core.io import open_h5, check_h5, new_steps_file, save_step, write_mask_tif
from .core.ims import ims_to_h5
from .core.muscle import measure_orientation, compute_surface
from .core.cd34 import compute_cd34, h_for_region
from .core.submeso import compute_submeso
from .core.filters import h_fullres

OUTPUT_SUFFIXES = ("_layers.h5", "_cd34.h5", "_submeso_cd34.h5", "_cd34steps.h5",
                   "_cd34_hyst.h5", "_cx43.h5", "_cx43pos.h5")


def find_samples(folder):
    h5s = [f for f in folder.glob("*.h5") if not f.name.endswith(OUTPUT_SUFFIXES)]
    ims = [f for f in folder.glob("*.ims") if not f.with_suffix(".h5").exists()]
    return sorted(h5s + ims)


def process(inp, opt, log):
    t0 = time.time()
    h5 = inp.with_suffix(".h5")
    if inp.suffix.lower() == ".ims" and not h5.exists():
        ims_to_h5(inp, log)
    check_h5(h5)
    stem = h5.stem
    tifdir = h5.with_name(f"{stem}_tif")
    final_tif = tifdir / f"{stem}_fibroblast_mask.tif"
    if final_tif.exists() and not opt["force"]:
        log(f"{final_tif.name} exists - skipped (use --force to recompute)")
        return
    p = defaults()
    src, f = open_h5(h5)
    try:
        sp = src.sp
        # 1. muscle surface
        orient = measure_orientation(src, p["muscle_channel"], p, log)
        surf = compute_surface(src, p["muscle_channel"], orient["theta0_deg"], p, log)
        ds = surf["ds"]
        with new_steps_file(h5.with_name(f"{stem}_layers.h5"), sp, dict(p, orientation=orient)) as out:
            for name, key, kind in (("1a_valid_ds", "valid_ds", "labels"),
                                    ("1c_tom20_smoothed_ds", "tom_smoothed", "image"),
                                    ("2a_muscle_thresholded_ds", "muscle_thresholded", "labels"),
                                    ("2b_muscle_size_filtered_ds", "muscle_size_filtered", "labels"),
                                    ("2c_muscle_fibres_ds", "muscle", "labels"),
                                    ("3a_h_raw_z", "h_raw", "image"),
                                    ("3c_h_filled_z", "h_filled", "image"),
                                    ("3d_h_clean_z", "h", "image")):
                save_step(out, name, surf[key], kind, sp, ds=ds, log=lambda *_: None)
        # 2. CD34
        nz, ny, nx = src.shape
        h_full = h_for_region(surf, src.region)
        cd = compute_cd34(src, p["cd34_channel"], h_full, p, log)
        with new_steps_file(h5.with_name(f"{stem}_cd34.h5"), sp, p) as out:
            save_step(out, "1c_tissue", cd["tissue"], "labels", sp, log=lambda *_: None)
            save_step(out, "4c_cd34_minus_background", cd["s"], "image", sp, log=lambda *_: None)
            save_step(out, "4d_cd34_seeds_otsu", cd["seeds"], "labels", sp, log=lambda *_: None)
            save_step(out, "5a_noise_sd_map", cd["noise_sd_map"], "image", sp, log=lambda *_: None)
            save_step(out, "5b_low_threshold_map", cd["low_map"], "image", sp, log=lambda *_: None)
            save_step(out, "6_cd34_mask", cd["mask"], "labels", sp, log=lambda *_: None)
        mask = cd["mask"]
        del cd
        # 3. distance filter
        sub = compute_submeso(mask, h_full, sp, p, log)
        with new_steps_file(h5.with_name(f"{stem}_submeso_cd34.h5"), sp, p) as out:
            save_step(out, "3a_removed_small_components", sub["removed_small"], "labels", sp, log=lambda *_: None)
            save_step(out, "4a_removed_by_cut", sub["removed_by_cut"], "labels", sp, log=lambda *_: None)
            save_step(out, "4b_submesothelial_cd34_FINAL", sub["final"], "labels", sp, log=lambda *_: None)
        # 4. TIFFs
        tifdir.mkdir(exist_ok=True)
        thr_um = opt["dist_threshold"] * (sp[0] if opt["dist_unit"] == "slices" else 1.0)
        dmask = np.empty((nz, ny, nx), bool)
        for z in range(nz):
            dmask[z] = (z - h_full) * sp[0] > thr_um
        tag = f"{opt['dist_threshold']:g}{'sl' if opt['dist_unit'] == 'slices' else 'um'}"
        write_mask_tif(tifdir / f"{stem}_distance_gt{tag}.tif", dmask, sp, opt["compress"])
        write_mask_tif(final_tif, sub["final"], sp, opt["compress"])
        log(f"TIFFs written to {tifdir.name}")
        # 5. orientation check
        prof = {}
        for name, c in (("muscle", p["muscle_channel"]), ("cd34", p["cd34_channel"])):
            prof[name] = int(np.argmax([src.slice(c, z).mean() for z in range(nz)]))
        ok = prof["muscle"] < prof["cd34"]
        log(f"orientation: mean-intensity peak muscle z {prof['muscle']}, CD34 z {prof['cd34']} -> "
            + ("OK" if ok else "WARNING: muscle not below CD34 - mesothelium may not be at high z; "
                               "do not use this result"))
    finally:
        f.close()
    log(f"{stem}: done in {(time.time() - t0) / 60:.1f} min")


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    opt = dict(force="--force" in a, compress="--uncompressed" not in a, dist_threshold=0.0,
               dist_unit="um", dry="--dry-run" in a)
    for key, conv in (("--dist-threshold", float), ("--dist-unit", str)):
        if key in a:
            i = a.index(key)
            opt[key[2:].replace("-", "_")] = conv(a[i + 1])
            del a[i:i + 2]
    args = [x for x in a if not x.startswith("--")]
    if len(args) != 1 or opt["dist_unit"] not in ("um", "slices"):
        raise SystemExit(__doc__)
    inp = Path(args[0]).resolve()
    samples = find_samples(inp) if inp.is_dir() else [inp]
    print(f"{len(samples)} sample(s):\n" + "\n".join(f"  {s.name}" for s in samples))
    if opt["dry"]:
        return
    failed = []
    for k, s in enumerate(samples, 1):
        logf = s.with_name(f"{s.with_suffix('').name}_preprocessing_log.txt")
        with open(logf, "a", encoding="utf-8") as fh:
            def log(msg):
                print(msg, flush=True)
                fh.write(f"{msg}\n")
                fh.flush()
            log(f"\n##### {k}/{len(samples)} {s} {time.ctime()} options {json.dumps(opt)}")
            try:
                process(s, opt, log)
            except Exception as e:                      # one failing sample does not stop the rest
                log(f"!!!!! FAILED: {type(e).__name__}: {e}")
                failed.append(s.name)
    print(f"\n{len(samples) - len(failed)} ok, {len(failed)} failed" +
          "".join(f"\n  FAILED: {n}" for n in failed))


if __name__ == "__main__":
    main()

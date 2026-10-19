"""Tests of the core algorithms on a small synthetic volume:
muscle fibres (bright, along 30 deg) at low z, a thin CD34 sheet with processes above them,
plus noise. Checks that the fibre direction, the surface height and the final mask behave
as designed."""
import numpy as np
from napari_maskel_ibex_preprocessing.core.params import defaults
from napari_maskel_ibex_preprocessing.core.io import VolumeSource, write_mask_tif
from napari_maskel_ibex_preprocessing.core.muscle import measure_orientation, compute_surface
from napari_maskel_ibex_preprocessing.core.cd34 import compute_cd34, h_for_region
from napari_maskel_ibex_preprocessing.core.submeso import compute_submeso

SP = np.array([2.0, 0.46, 0.46])
NZ, NY, NX = 24, 768, 768
SURF_Z = 10                                         # top of the muscle layer (slice)


def synthetic(seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:NY, 0:NX]
    th = np.radians(30.0)
    across = -xx * np.sin(th) + yy * np.cos(th)      # coordinate across the fibres (px)
    fibre = (np.mod(across, 60) < 45)                # 45 px fibres, 15 px gaps
    bright = np.mod(across, 120) < 60                # bright and dim fibres alternate
    tom = np.zeros((NZ, NY, NX), np.float32)
    tom[:SURF_Z + 1] = np.where(fibre, np.where(bright, 60, 25), 3)   # muscle up to SURF_Z
    cd34 = np.zeros_like(tom)
    sheet = (np.mod(xx, 40) < 3) | (np.mod(yy, 40) < 3)   # network of thin processes
    cd34[SURF_Z + 2] = np.where(sheet, 80, 0)
    # connective tissue above the muscle: faint autofluorescence in every channel (as in the
    # real data), so that the tissue mask includes the submesothelial layer
    tissue_bg = np.zeros_like(tom)
    tissue_bg[:SURF_Z + 6] = 25
    chans = []
    for i, base in enumerate((tom, np.zeros_like(tom), np.zeros_like(tom), cd34,
                              np.zeros_like(tom), np.zeros_like(tom))):
        bg = tissue_bg * (0.12 if i == 0 else 1.0)  # connective tissue is dark in TOM20
        a = base + bg + rng.normal(5, 2, base.shape) + 3
        chans.append(np.clip(a, 1, 255).astype(np.uint8))
    return chans


def test_pipeline(tmp_path):
    p = defaults()
    src = VolumeSource(synthetic(), SP)
    orient = measure_orientation(src, 0, p, log=lambda *_: None)
    assert abs(orient["theta0_deg"] - 30.0) < 3, orient
    surf = compute_surface(src, 0, orient["theta0_deg"], p, log=lambda *_: None)
    assert abs(np.median(surf["h"]) - SURF_Z) <= 1                # top muscle slice
    assert np.percentile(surf["h"], 5) >= SURF_Z - 1               # gaps closed
    h_full = h_for_region(surf, src.region)
    cd = compute_cd34(src, 3, h_full, p, log=lambda *_: None)
    assert cd["mask"][SURF_Z + 2].mean() > 0.05                    # the sheet is found
    sub = compute_submeso(cd["mask"], h_full, SP, p, log=lambda *_: None)
    assert sub["final"][:SURF_Z - 2].sum() == 0                    # nothing deep in the muscle
    assert sub["final"][SURF_Z + 2].sum() > 0.9 * cd["mask"][SURF_Z + 2].sum()
    path = write_mask_tif(tmp_path / "m.tif", sub["final"], SP, compress=True)
    assert path.exists()


def test_region_alignment():
    src = VolumeSource(synthetic(), SP, region=(101, 400, 203, 500))
    assert src.region[0] % 4 == 0 and src.region[2] % 4 == 0
    assert src.shape == (NZ, 400 - src.region[0], 500 - src.region[2])

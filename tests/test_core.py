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


def test_macrophages_and_hair():
    """A macrophage cluster on the muscle (bright in TOM20 and CD11b, 4 slices high) and a hair
    lying 3 slices above the muscle (bright in TOM20 only, long and thin along the fibres)
    must not lift the surface: h stays at the top muscle slice."""
    p = defaults()
    chans = synthetic()
    tom, cd11b = chans[0].astype(np.float32), chans[4].astype(np.float32)
    yy, xx = np.mgrid[0:NY, 0:NX]
    blob = (yy - 300) ** 2 + (xx - 300) ** 2 < 60 ** 2           # ~28 um radius
    for z in range(SURF_Z + 1, SURF_Z + 5):
        tom[z][blob] = 80
        cd11b[z][blob] = 120
    th = np.radians(30.0)
    along = xx * np.cos(th) + yy * np.sin(th)
    across = -xx * np.sin(th) + yy * np.cos(th)
    hair = (np.abs(across + 150) < 8) & (np.abs(along - 450) < 150)   # far from the blob
    for z in range(SURF_Z + 4, SURF_Z + 7):
        tom[z][hair] = 80
    chans[0], chans[4] = (np.clip(a, 0, 255).astype(np.uint8) for a in (tom, cd11b))
    # the synthetic CD11b has a uniform tissue background and no macrophages: the rule must
    # skip itself instead of marking every column
    s0 = compute_surface(VolumeSource(synthetic(), SP), 0, 30.0, p, log=lambda *_: None, mac_ch=4)
    assert s0["macrophage_skipped"] and abs(np.median(s0["h"]) - SURF_Z) <= 1
    # real CD11b: dim background, bright cells
    rng = np.random.default_rng(2)
    cd11b = rng.normal(3, 1, cd11b.shape).astype(np.float32)
    for z in range(SURF_Z + 1, SURF_Z + 5):
        cd11b[z][blob] = 120
    chans[4] = np.clip(cd11b, 0, 255).astype(np.uint8)
    src = VolumeSource(chans, SP)
    surf = compute_surface(src, 0, 30.0, p, log=lambda *_: None, mac_ch=4)
    h = h_for_region(surf, src.region)
    assert abs(np.median(h[blob]) - SURF_Z) <= 1, np.median(h[blob])
    assert np.percentile(h[hair], 95) <= SURF_Z + 1, np.percentile(h[hair], 95)
    # without the two rules both lift the surface
    p.update(muscle_min_component_fraction=0.0)
    h0 = h_for_region(compute_surface(src, 0, 30.0, p, log=lambda *_: None), src.region)
    assert np.median(h0[blob]) >= SURF_Z + 3 and np.median(h0[hair]) >= SURF_Z + 4


def test_large_components():
    """Components < 1 % of the largest are removed, larger separate pieces are kept."""
    from napari_maskel_ibex_preprocessing.core.muscle import keep_large_components
    m = np.zeros((10, 100, 100), bool)
    m[:, :50, :] = True                                  # largest: 50000 voxels
    m[:5, 60:80, :10] = True                             # 1000 voxels = 2 %   -> kept
    m[0, 90:95, 90:95] = True                            # 25 voxels = 0.05 %  -> removed
    k = keep_large_components(m, 0.01)
    assert k[:, :50].all() and k[:5, 60:80, :10].all() and not k[0, 90:95, 90:95].any()
    assert keep_large_components(m, 0.0).sum() == m.sum()
    assert keep_large_components(m, 1.0).sum() == 50000


def test_cx43():
    """Bright 3 x 3 px puncta: one row on top of a fibroblast sheet, one row 10 slices (20 um)
    above it. Only the puncta within cx43_max_dist_um (2 um) of the mask are positive."""
    from napari_maskel_ibex_preprocessing.core.cx43 import compute_cx43
    p = defaults()
    rng = np.random.default_rng(1)
    nz, ny, nx = 24, 256, 256
    raw = rng.normal(10, 2, (nz, ny, nx)).astype(np.float32)
    mask = np.zeros((nz, ny, nx), bool)
    mask[10, :, :] = True                                 # fibroblast sheet in slice 10
    for k, x in enumerate(range(30, 230, 40)):
        raw[10, 100:103, x:x + 3] += 60                   # on the sheet  -> positive
        raw[20, 160:163, x:x + 3] += 60                   # 20 um above   -> excluded
    res = compute_cx43(raw, mask, SP, p, log=lambda *_: None)
    assert res["n_objects"] == 5, res["n_objects"]
    t = res["table"]
    assert np.all(np.abs(t[:, 1] - 10 * SP[0]) < 1e-6)    # all at z = slice 10
    assert np.all(t[:, 9] == 1.0)                         # fully inside the mask


def test_region_alignment():
    src = VolumeSource(synthetic(), SP, region=(101, 400, 203, 500))
    assert src.region[0] % 4 == 0 and src.region[2] % 4 == 0
    assert src.shape == (NZ, 400 - src.region[0], 500 - src.region[2])

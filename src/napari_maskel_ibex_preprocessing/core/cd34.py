"""Step 2: tissue mask and CD34 fibroblast mask (hysteresis with a local noise threshold).

  valid    = > 0 in any channel; tissue = Otsu on log(1 + Gaussian 5 um of the summed channels,
             2 x 2 block means) back at full resolution & valid
  s        = CD34 normalized Gaussian sigma (1, 0.5, 0.5) um  minus  background (2 x 2 block
             means, normalized Gaussian 15 um): grey values above the local background
  seeds    = (s > Otsu(s in tissue)) & tissue; otsu_mask = seeds without objects < 20 um^3
  noise    = tissue outside otsu_mask, within the band -10 <= d <= +30 um around the muscle
             surface (the deep muscle with little CD34 would otherwise lower the estimate)
  low(y,x) = median + 4 * robust SD of the noise per ~272 um block (all z), bilinear between
             block centres
  mask     = 26-connected components of (s > low) & tissue that contain a seed voxel, objects
             < 20 um^3 removed
NOTE for a region (preview): Otsu thresholds and noise statistics are computed from the region
only, so the result can differ from the same region of a full-volume run.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from .filters import (S26, gauss, gauss_n, block_mean, upsample, remove_small, robust,
                      h_fullres)


def tissue_mask(src, p, log=print):
    """valid and tissue (full resolution, region of src)."""
    ds = p["tissue_ds"]
    shape = src.shape
    valid = np.zeros(shape, bool)
    total = None
    for c in range(src.n_channels):
        raw = src.volume(c)
        valid |= raw > 0
        b = block_mean(raw, ds)
        total = b if total is None else total + b
        log(f"tissue: channel {c + 1}/{src.n_channels}")
    t = np.log1p(gauss(total, p["tissue_smooth_um"], src.sp * np.array([1, ds, ds])))
    thr = threshold_otsu(t)
    tissue = upsample(t > thr, ds, shape) & valid
    return valid, tissue, float(thr)


def compute_cd34(src, ch, h_full, p, log=print):
    """CD34 mask in the region of src. h_full: height map (z slice index) at full resolution
    for exactly this region (2D, shape src.shape[1:]). Returns a dict of the steps."""
    sp = src.sp
    ds = p["tissue_ds"]
    shape = src.shape
    nz, ny, nx = shape
    valid, tissue, t_thr = tissue_mask(src, p, log)

    log("CD34: smoothing and background")
    raw = src.volume(ch)
    valid_ds = block_mean(valid, ds) >= 0.5
    raw_ds = block_mean(raw, ds) * valid_ds
    bg_ds = gauss_n(raw_ds, p["bg_sigma_um"], valid_ds, sp * np.array([1, ds, ds]))
    del raw_ds
    s = gauss_n(raw, p["cd34_smooth_um"], valid, sp)
    del raw, valid
    for z in range(nz):
        s[z] -= upsample(bg_ds[z], ds, shape[1:])
    otsu = float(threshold_otsu(s[tissue]))
    seeds = (s > otsu) & tissue
    otsu_mask = remove_small(seeds, int(np.ceil(p["cd34_min_volume_um3"] / sp.prod())))

    log("CD34: local noise threshold")
    lo_um, hi_um = p["hyst_noise_band_um"]
    noise = tissue & ~otsu_mask
    for z in range(nz):
        dz = (z - h_full) * sp[0]
        noise[z] &= (dz >= lo_um) & (dz <= hi_um)
    nby = max(1, int(round(ny * sp[1] / p["hyst_block_um"])))
    nbx = max(1, int(round(nx * sp[2] / p["hyst_block_um"])))
    ye = np.linspace(0, ny, nby + 1).astype(int)
    xe = np.linspace(0, nx, nbx + 1).astype(int)
    med_b = np.full((nby, nbx), np.nan)
    sd_b = np.full((nby, nbx), np.nan)
    for i in range(nby):
        for j in range(nbx):
            blk = (slice(None), slice(ye[i], ye[i + 1]), slice(xe[j], xe[j + 1]))
            v = s[blk][noise[blk]]
            if v.size >= p["hyst_min_noise_vox"]:
                med_b[i, j], sd_b[i, j] = robust(v)
    if np.isnan(med_b).all():
        raise RuntimeError("no block has enough noise voxels - region too small, or check the "
                           "muscle surface / tissue mask")
    bad = np.isnan(med_b)
    if bad.any():
        idx = ndi.distance_transform_edt(bad, return_distances=False, return_indices=True)
        med_b, sd_b = med_b[tuple(idx)], sd_b[tuple(idx)]
    cy = (ye[:-1] + ye[1:] - 1) / 2
    cx = (xe[:-1] + xe[1:] - 1) / 2
    gy, gx = np.meshgrid(np.interp(np.arange(ny), cy, np.arange(nby)),
                         np.interp(np.arange(nx), cx, np.arange(nbx)), indexing="ij")
    med_map = ndi.map_coordinates(med_b, [gy, gx], order=1).astype(np.float32)
    sd_map = ndi.map_coordinates(sd_b, [gy, gx], order=1).astype(np.float32)
    del gy, gx
    low = med_map + np.float32(p["hyst_k_sd"]) * sd_map

    log("CD34: hysteresis")
    cand = (s > low[None]) & tissue
    lab, n = ndi.label(cand, structure=S26)
    keep = np.zeros(n + 1, bool)
    keep[np.unique(lab[seeds])] = True
    keep[0] = False
    hyst = keep[lab]
    del lab
    mask = remove_small(hyst, int(np.ceil(p["cd34_min_volume_um3"] / sp.prod())))
    log(f"CD34: tissue Otsu {t_thr:.3f}; CD34 Otsu {otsu:.2f}; low threshold "
        f"{low.min():.2f}..{low.max():.2f}; mask {mask[tissue].mean():.1%} of tissue "
        f"(Otsu mask {otsu_mask[tissue].mean():.1%})")
    return dict(tissue=tissue, s=s, seeds=seeds, otsu_mask=otsu_mask, otsu=otsu,
                noise_sd_map=sd_map, low_map=low, candidate=cand, hysteresis=hyst, mask=mask,
                block_median=med_b, block_sd=sd_b, region=src.region)


def h_for_region(surface, region):
    """Height map at full resolution for `region` (y0, y1, x0, x1) from a surface result that
    covers it (its own region and block grid)."""
    sy0, sy1, sx0, sx1 = surface["region"]
    y0, y1, x0, x1 = region
    if y0 < sy0 or x0 < sx0 or y1 > sy1 or x1 > sx1:
        raise ValueError("the muscle surface (step 1) does not cover this region - run step 1 "
                         "on the full volume or on a region containing this one")
    return h_fullres(surface["h"], y1 - y0, x1 - x0, surface["ds"], y0 - sy0, x0 - sx0)

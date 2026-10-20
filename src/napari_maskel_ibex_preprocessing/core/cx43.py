"""Step 4: Cx43-positive volume near the fibroblast mask (simple adaptive threshold).

Validated on LSM 2026-07-20_B6-12w-naive_co1-s1-r1 (threshold T = 12 grey values chosen by
Anna in napari; finds all 39 of her annotated real puncta, 4 of 17 false examples hit).
  smoothed   = Gaussian sigma (Z, Y, X) = (0, 1.0, 0.7) PIXELS (= 0, 0.46, 0.32 um at
               0.461 um; wider in Y because of the line step of 2 px in Y of the acquisition)
  background = median over ~14.8 um in XY per slice, computed on 4 x 4 px block means (median
               of 8 x 8 blocks) and interpolated bilinearly back (the background is smooth;
               a full-resolution median took minutes per slice)
  contrast   = smoothed - background (grey values above the local background)
  near       = anisotropic Euclidean distance to the fibroblast mask <= max_dist (um)
  positive   = contrast > T & near; 26-connected objects (may contain merged puncta - the
               positive VOLUME is the quantity of interest)
Per object: centroid (um, full image), voxels, volume, mean / max contrast, mean raw value,
fraction of its voxels inside the fibroblast mask, distance of the centroid to the mask.
"""
import numpy as np
from scipy import ndimage as ndi
from .filters import S26

TABLE_COLUMNS = ["label", "z_um", "y_um", "x_um", "n_voxels", "volume_um3", "mean_contrast",
                 "max_contrast", "mean_raw", "fraction_in_fibroblast_mask",
                 "centroid_dist_to_mask_um"]


def background_median(sm, zs, block, n_blocks):
    """Median background per slice on a block grid, bilinearly interpolated back."""
    nz, ny, nx = sm.shape
    nyb, nxb = ny // block, nx // block
    bg = np.zeros_like(sm)
    for z in zs:
        blk = sm[z, :nyb * block, :nxb * block].reshape(nyb, block, nxb, block).mean((1, 3))
        med = ndi.median_filter(blk, size=n_blocks)
        bg[z] = ndi.zoom(med, (ny / nyb, nx / nxb), order=1)[:ny, :nx]
    return bg


def compute_cx43(raw, fib_mask, sp, p, origin_yx=(0, 0), log=print):
    """raw: Cx43 (Z, Y, X) of the region; fib_mask: fibroblast mask (bool) of the same region;
    origin_yx: region origin in full-image pixels (for the table coordinates)."""
    sp = np.asarray(sp, float)
    raw = np.asarray(raw, np.float32)
    nz = raw.shape[0]
    # restriction: within max_dist of the fibroblast mask; the distance transform only needs
    # the z range of the mask +- max_dist (farther slices cannot be within max_dist)
    zs_mask = np.nonzero(fib_mask.any((1, 2)))[0]
    if zs_mask.size == 0:
        raise ValueError("the fibroblast mask is empty in this region")
    dz_pad = int(np.ceil(p["cx43_max_dist_um"] / sp[0]))
    za, zb = max(zs_mask.min() - dz_pad, 0), min(zs_mask.max() + dz_pad + 1, nz)
    log(f"Cx43: distance to the fibroblast mask (slices {za}-{zb - 1})")
    dist = np.full(raw.shape, np.inf, np.float32)
    dist[za:zb] = ndi.distance_transform_edt(~fib_mask[za:zb], sampling=sp)
    near = dist <= p["cx43_max_dist_um"]

    log("Cx43: smoothing and background")
    sm = ndi.gaussian_filter(raw, np.asarray(p["cx43_smooth_px"], float))
    block = int(p["cx43_bg_block_px"])
    n_blocks = max(1, int(round(p["cx43_bg_um"] / (block * sp[1]))))
    zs = np.nonzero(near.any((1, 2)))[0]
    bg = background_median(sm, zs, block, n_blocks)
    sm -= bg                                         # in place: contrast = smoothed - background
    contrast = sm
    del sm, bg

    log("Cx43: threshold and objects")
    pos = (contrast > p["cx43_threshold"]) & near
    lab, n = ndi.label(pos, structure=S26)
    zz, yy, xx = np.nonzero(lab)
    l = lab[zz, yy, xx]

    def osum(w):
        return np.bincount(l, weights=w, minlength=n + 1)[1:]

    nvox = np.bincount(l, minlength=n + 1)[1:].astype(float)
    cz = osum(zz * sp[0]) / np.maximum(nvox, 1)
    cy = osum((yy + origin_yx[0]) * sp[1]) / np.maximum(nvox, 1)
    cx = osum((xx + origin_yx[1]) * sp[2]) / np.maximum(nvox, 1)
    c = contrast[zz, yy, xx]
    cmax = np.full(n, -np.inf)
    np.maximum.at(cmax, l - 1, c)
    inmask = osum(fib_mask[zz, yy, xx].astype(float)) / np.maximum(nvox, 1)
    iz = np.clip(np.round(cz / sp[0]).astype(int), 0, nz - 1)
    iy = np.clip(np.round(cy / sp[1]).astype(int) - origin_yx[0], 0, raw.shape[1] - 1)
    ix = np.clip(np.round(cx / sp[2]).astype(int) - origin_yx[1], 0, raw.shape[2] - 1)
    table = np.column_stack([np.arange(1, n + 1), cz, cy, cx, nvox, nvox * sp.prod(),
                             osum(c) / np.maximum(nvox, 1), cmax,
                             osum(raw[zz, yy, xx]) / np.maximum(nvox, 1), inmask,
                             dist[iz, iy, ix]])
    vol_in = float((pos & fib_mask).sum() * sp.prod())
    vol_all = float(pos.sum() * sp.prod())
    log(f"Cx43: {n} objects, positive volume {vol_all:.0f} um^3 within "
        f"{p['cx43_max_dist_um']:g} um of the mask ({vol_in:.0f} um^3 inside it); background "
        f"window {n_blocks} x {block} px")
    return dict(contrast=contrast, labels=lab.astype(np.int32), distance=dist, table=table,
                n_objects=n, volume_in_mask_um3=vol_in, volume_total_um3=vol_all)


def write_table(path, table):
    """Object table -> CSV (columns TABLE_COLUMNS)."""
    fmt = ["%d", "%.3f", "%.3f", "%.3f", "%d", "%.4f", "%.3f", "%.3f", "%.3f", "%.4f", "%.3f"]
    np.savetxt(path, table, delimiter=",", header=",".join(TABLE_COLUMNS), comments="", fmt=fmt)
    return path

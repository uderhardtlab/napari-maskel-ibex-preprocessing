"""Step 1: muscle mask from TOM20 and the outer muscle surface h(y, x).

Geometry (validated): en face sheet, mesothelium at HIGH z, muscle at low z, muscle everywhere
in the field of view. Everything on a ds x ds XY block grid (ds = 4: 1.845 um for 0.461 um px):
  valid    = > 0 in any channel; per block: mean over valid pixels, block valid if >= 50 %
  smoothed = TOM20 normalized Gaussian, sigma (2, 5, 5) um
  muscle   = smoothed > lowest 3-class multi-Otsu threshold (dim fibres included) & valid;
             26-connected objects < 5000 um^3 removed; fibre filter = binary opening with a
             30 um line along the fibre direction (outside the valid image counts as muscle for
             the erosion) - removes round artefacts even when they touch a fibre
  h_raw    = highest muscle z per column; empty columns take the nearest column's value
  h        = grey closing of h_raw with a 9.2 um disk (fills gaps between fibres <= ~18 um)
  d        = (z - h) * dz um, > 0 = mesothelium side (signed vertical distance)
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_multiotsu
from .filters import gauss_n, remove_small, disk, line


def measure_orientation(src, ch, p, log=print):
    """In-plane fibre direction theta0 (deg; x right, y down, from +x towards +y) from the
    structure tensor of the TOM20 max projection (block means over ds x ds px, Gaussian
    2 um; J smoothed with rho = 20 um). Global direction = circular mean of 2 * theta weighted
    by coherence * (Jxx + Jyy). Returns a dict with theta0 and the spread of local deviations."""
    ds = p["layers_ds"]
    px = src.sp[1] * ds
    nz, ny, nx = src.shape
    nyd, nxd = ny // ds, nx // ds
    mip = np.zeros((nyd, nxd), np.float32)
    for z in range(nz):
        a = src.slice(ch, z)[:nyd * ds, :nxd * ds].astype(np.float32)
        mip = np.maximum(mip, a.reshape(nyd, ds, nxd, ds).mean((1, 3)))
    I = ndi.gaussian_filter(mip, p["orient_smooth_um"] / px)
    Iy, Ix = np.gradient(I, px)
    rho = p["orient_rho_um"] / px
    Jxx, Jyy, Jxy = (ndi.gaussian_filter(a, rho) for a in (Ix * Ix, Iy * Iy, Ix * Iy))
    theta = np.degrees(0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)) + 90.0
    theta = (theta + 90.0) % 180.0 - 90.0
    tr = Jxx + Jyy
    coh = np.where(tr > 0, np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy**2) / np.maximum(tr, 1e-12), 0)
    w = coh * tr * (mip > 0)
    t2 = np.radians(2 * theta)
    theta0 = float(np.degrees(np.arctan2((w * np.sin(t2)).sum(), (w * np.cos(t2)).sum())) / 2)
    dev = (theta - theta0 + 90) % 180 - 90
    sel = w > np.percentile(w[w > 0], 25) if (w > 0).any() else w > 0
    q = np.percentile(np.abs(dev[sel]), [50, 90, 95, 99]) if sel.any() else np.full(4, np.nan)
    tol = float(np.degrees(np.arcsin(min(20.0 / p["fibre_line_um"], 1.0))))
    res = dict(theta0_deg=theta0, abs_dev_pct_50_90_95_99=q.tolist(),
               median_coherence=float(np.median(coh[sel])) if sel.any() else float("nan"),
               tolerated_dev_deg=tol, ok=bool(q[2] < tol))
    log(f"fibre direction {theta0:.1f} deg; |deviation| 50/90/95/99 % = {np.round(q, 1).tolist()} deg"
        + ("" if res["ok"] else f" - WARNING: 95 % deviation exceeds {tol:.0f} deg"))
    return res


def fibre_filter(m, valid, length_px, theta_deg):
    """Binary opening with a line along the fibres: a voxel is kept only if a line segment of
    this length along the fibres, containing the voxel, lies entirely in the mask."""
    fp = line(length_px, theta_deg)
    er = ndi.binary_erosion(m | ~valid, structure=fp, border_value=1)
    return ndi.binary_dilation(er, structure=fp) & m


def top_z(m):
    """Highest True z per (y, x) column (float32); NaN where the column is empty."""
    zi = np.arange(m.shape[0], dtype=np.float32)[:, None, None]
    return np.where(m.any(0), np.where(m, zi, -1).max(0), np.nan).astype(np.float32)


def compute_surface(src, ch, theta0, p, log=print):
    """Muscle mask and surface on the block grid of `src` (its region). Returns a dict with the
    intermediate arrays (block grid, 'ds' = block size) and the height map 'h' (z slice index
    of the top muscle voxel per block, float32)."""
    ds = p["layers_ds"]
    sp = src.sp
    spd = sp * np.array([1, ds, ds])
    nz, ny, nx = src.shape
    nyd, nxd = ny // ds, nx // ds
    Y, X = nyd * ds, nxd * ds
    tom = np.zeros((nz, nyd, nxd), np.float32)
    valid_ds = np.zeros((nz, nyd, nxd), bool)
    for z in range(nz):
        raws = [src.slice(c, z)[:Y, :X] for c in range(src.n_channels)]
        valid = np.any([r > 0 for r in raws], axis=0)
        nv = valid.reshape(nyd, ds, nxd, ds).sum((1, 3))
        valid_ds[z] = nv >= 0.5 * ds * ds
        s = (raws[ch] * valid).reshape(nyd, ds, nxd, ds).sum((1, 3), dtype=np.float32)
        tom[z] = np.where(valid_ds[z], s / np.maximum(nv, 1), 0)
        if z % 10 == 0:
            log(f"muscle: reading slice {z + 1}/{nz}")
    tom_s = gauss_n(tom, p["tom_sigma_um"], valid_ds, spd)
    vals = tom_s[valid_ds]
    if p.get("muscle_threshold") is None:
        thr = float(threshold_multiotsu(vals, classes=p["n_classes"])[0])
    else:
        thr = float(p["muscle_threshold"])
    m = (tom_s > thr) & valid_ds
    muscle0 = remove_small(m, int(np.ceil(p["muscle_min_volume_um3"] / spd.prod())))
    muscle = fibre_filter(muscle0, valid_ds, p["fibre_line_um"] / spd[1], theta0)
    has = muscle.any(0)
    h_raw = top_z(muscle)
    if has.any():
        idx = ndi.distance_transform_edt(~has, sampling=spd[1:], return_distances=False,
                                         return_indices=True)
        h_fill = h_raw[tuple(idx)]
    else:
        h_fill = np.zeros_like(h_raw)
    h = ndi.grey_closing(h_fill, footprint=disk(p["close_radius_um"] / spd[1]))
    log(f"muscle: threshold {thr:.2f}; muscle {muscle[valid_ds].mean():.1%} of valid voxels "
        f"(fibre filter removed {1 - muscle.sum() / max(muscle0.sum(), 1):.1%}); columns without "
        f"muscle {1 - has.mean():.2%}; h (z slice) 1/50/99 % = "
        + ", ".join(f"{v:.0f}" for v in np.percentile(h, [1, 50, 99])))
    return dict(ds=ds, valid_ds=valid_ds, tom_ds=tom, tom_smoothed=tom_s, threshold=thr,
                muscle_thresholded=m, muscle_size_filtered=muscle0, muscle=muscle,
                h_raw=h_raw, h_filled=h_fill, h=h, theta0_deg=theta0, region=src.region)


def distance_slice(h_full, z, dz):
    """Signed distance d = (z - h) * dz (um) of one slice; h_full = height map at full
    resolution (z slice index)."""
    return (np.float32(z) - h_full) * np.float32(dz)

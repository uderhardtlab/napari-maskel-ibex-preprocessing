"""Basic operations. All sigmas / radii are given in um and converted per axis with the voxel
size `sp` = (dz, dy, dx) in um."""
import numpy as np
from scipy import ndimage as ndi

S26 = np.ones((3, 3, 3), bool)          # 26-connectivity


def gauss(img, sigma_um, sp):
    """Gaussian filter with sigma in um per axis (scalar = isotropic), float32 output."""
    sigma_px = np.broadcast_to(np.asarray(sigma_um, float), (3,)) / np.asarray(sp, float)
    return ndi.gaussian_filter(img, sigma=sigma_px, output=np.float32)


def gauss_n(img, sigma_um, valid, sp):
    """Normalized convolution G(img * valid) / G(valid): averages over valid voxels only, so the
    zero padding of a stitched image does not darken the edges. img must be 0 where valid is
    False."""
    w = gauss(valid.astype(np.float32), sigma_um, sp)
    out = gauss(img, sigma_um, sp)
    np.divide(out, w, out=out, where=w > 1e-3)
    out[w <= 1e-3] = 0
    return out


def block_mean(a, f):
    """Mean over f x f blocks in Y and X (Z untouched). Crops Y, X to a multiple of f."""
    z, y, x = a.shape
    a = a[:, : y - y % f, : x - x % f].astype(np.float32)
    return a.reshape(z, y // f, f, x // f, f).mean(axis=(2, 4))


def upsample(m, f, shape):
    """Nearest-neighbour inverse of block_mean (last two axes), edge-padded to `shape`."""
    u = np.repeat(np.repeat(m, f, axis=-2), f, axis=-1)
    pad = [(0, s - us) for s, us in zip(shape, u.shape)]
    return np.pad(u, pad, mode="edge")


def remove_small(mask, min_voxels):
    """Removes 26-connected objects with fewer than min_voxels voxels."""
    lab, _ = ndi.label(mask, structure=S26)
    counts = np.bincount(lab.ravel())
    keep = counts >= min_voxels
    keep[0] = False
    return keep[lab]


def disk(r_px):
    """Flat disk footprint: all pixels with distance <= r_px from the centre."""
    n = int(np.floor(r_px))
    yy, xx = np.ogrid[-n:n + 1, -n:n + 1]
    return yy**2 + xx**2 <= r_px**2 + 1e-9


def line(length_px, theta_deg):
    """Flat line footprint of the given length (pixels) along theta (from +x towards +y,
    y down), shape (1, k, k): it lies in ONE z slice."""
    n = int(np.ceil(length_px / 2))
    t = np.arange(-length_px / 2, length_px / 2 + 1e-9, 0.25)
    fp = np.zeros((1, 2 * n + 1, 2 * n + 1), bool)
    th = np.radians(theta_deg)
    fp[0, np.round(t * np.sin(th)).astype(int) + n, np.round(t * np.cos(th)).astype(int) + n] = True
    return fp


def robust(v):
    """Median and robust SD (1.4826 * median absolute deviation) of the values v."""
    med = float(np.median(v))
    return med, float(1.4826 * np.median(np.abs(v - med)))


def h_fullres(h_ds, ny, nx, ds, y_off=0, x_off=0):
    """Height map of a ds x ds block grid at full resolution (nearest neighbour), for the
    full-resolution pixel rows y_off .. y_off+ny-1 and columns x_off .. x_off+nx-1 RELATIVE to
    the origin of the grid; pixels beyond the grid take the edge value."""
    return h_ds[np.ix_(np.clip((np.arange(ny) + y_off) // ds, 0, h_ds.shape[0] - 1),
                       np.clip((np.arange(nx) + x_off) // ds, 0, h_ds.shape[1] - 1))]

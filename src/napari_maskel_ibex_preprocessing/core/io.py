"""Volume sources, steps files and TIFF export."""
import json
from pathlib import Path
import numpy as np
import h5py
import tifffile

ALIGN = 4          # region origins are aligned to this many pixels (the coarsest block grid)


class VolumeSource:
    """Channels of one sample as (Z, Y, X) arrays, optionally restricted to an XY region.

    arrays  list of array-likes supporting [z, y0:y1, x0:x1] slicing (numpy, dask, h5py
            datasets), all of the same shape
    sp      voxel size (dz, dy, dx) in um
    region  (y0, y1, x0, x1) in full-resolution pixels or None for the whole image; the origin
            is moved down to a multiple of 4 px so that block grids of the region coincide with
            those of the whole image
    """

    def __init__(self, arrays, sp, region=None, names=None):
        self.arrays = list(arrays)
        self.sp = np.asarray(sp, float)
        self.full_shape = tuple(self.arrays[0].shape)
        nz, ny, nx = self.full_shape
        if region is None:
            y0, y1, x0, x1 = 0, ny, 0, nx
        else:
            y0, y1, x0, x1 = (int(v) for v in region)
            y0, x0 = max(0, y0 - y0 % ALIGN), max(0, x0 - x0 % ALIGN)
            y1, x1 = min(ny, max(y1, y0 + ALIGN)), min(nx, max(x1, x0 + ALIGN))
        self.region = (y0, y1, x0, x1)
        self.shape = (nz, y1 - y0, x1 - x0)
        self.names = names or [f"Channel_{i}" for i in range(len(self.arrays))]

    @property
    def is_full(self):
        y0, y1, x0, x1 = self.region
        return (y0, x0) == (0, 0) and (y1, x1) == self.full_shape[1:]

    @property
    def n_channels(self):
        return len(self.arrays)

    def slice(self, c, z):
        y0, y1, x0, x1 = self.region
        return np.asarray(self.arrays[c][z, y0:y1, x0:x1])

    def volume(self, c):
        y0, y1, x0, x1 = self.region
        return np.asarray(self.arrays[c][:, y0:y1, x0:x1])


def open_h5(path, region=None):
    """VolumeSource of a converted .h5 (datasets Channel_0, Channel_1, ... and attribute
    element_size_um on Channel_0). Returns (source, open h5py.File)."""
    f = h5py.File(path, "r")
    keys = sorted((k for k in f if k.startswith("Channel_")), key=lambda k: int(k.split("_")[1]))
    if not keys:
        raise ValueError(f"{path}: no datasets Channel_0, Channel_1, ...")
    sp = np.asarray(f[keys[0]].attrs["element_size_um"], float)
    names = [f"{k} ({f[k].attrs['channel_name']})" if "channel_name" in f[k].attrs else k
             for k in keys]
    return VolumeSource([f[k] for k in keys], sp, region, names), f


def check_h5(path, n_channels=6):
    """Raises ValueError if the .h5 is not a complete converted file."""
    with h5py.File(path, "r") as f:
        missing = [f"Channel_{c}" for c in range(n_channels) if f"Channel_{c}" not in f]
        if missing:
            raise ValueError(f"{Path(path).name} is incomplete (missing {', '.join(missing)}), "
                             "e.g. from an interrupted conversion")
        shapes = {f[f"Channel_{c}"].shape for c in range(n_channels)}
        if len(shapes) != 1 or len(next(iter(shapes))) != 3:
            raise ValueError(f"{Path(path).name}: channels must be 3D with equal shapes ({shapes})")
        if "element_size_um" not in f["Channel_0"].attrs:
            raise ValueError(f"{Path(path).name}: attribute element_size_um missing on Channel_0")


# ----------------------------------------------------------------------------- steps files
def new_steps_file(path, sp, params, region=None):
    out = h5py.File(path, "w")
    out.attrs["z_offset"] = 0
    out.attrs["element_size_um"] = np.asarray(sp, float)
    out.attrs["params"] = json.dumps(params, default=float)
    if region is not None:
        out.attrs["region_fullres_yx"] = list(region)
    return out


def save_step(out, name, arr, kind, sp, ds=1, origin_yx=(0, 0), contrast=None, log=print):
    """One intermediate step (conventions of view_results.py): kind 'image' -> float16 with a
    display range `contrast`; 'labels' -> bool as uint8, integers as they are. 3D (Z, Y, X) or
    2D (Y, X); `translate` = region origin + block centre offset (ds - 1) / 2 pixels."""
    sp = np.asarray(sp, float)
    if arr.dtype == bool:
        arr = arr.astype(np.uint8)
    elif kind == "image":
        if contrast is None:
            sub = np.asarray(arr[..., ::4, ::4], float)
            fin = sub[np.isfinite(sub)]
            contrast = np.percentile(fin, [0.5, 99.5]) if fin.size else [0, 1]
        arr = arr.astype(np.float16)
    chunks = tuple([1] * (arr.ndim - 2) + [min(512, n) for n in arr.shape[-2:]])
    d = out.create_dataset(name, data=arr, chunks=chunks, compression="gzip", compression_opts=4)
    yx_t = (np.asarray(origin_yx, float) + (ds - 1) / 2) * sp[1:]
    d.attrs["kind"] = kind
    d.attrs["scale"] = np.r_[sp[0], sp[1:] * ds] if arr.ndim == 3 else sp[1:] * ds
    d.attrs["translate"] = np.r_[0.0, yx_t] if arr.ndim == 3 else yx_t
    if contrast is not None:
        d.attrs["contrast"] = np.asarray(contrast, float)
    log(f"  saved {name}")


# ----------------------------------------------------------------------------- TIFF export
def write_mask_tif(path, mask, sp, compress=True):
    """Binary mask (Z, Y, X) -> ImageJ TIFF, uint8 0/255, voxel size in the metadata, ZIP
    (deflate) compression if `compress`. Read back and compared (lossless check)."""
    full = np.asarray(mask, bool).view(np.uint8) * np.uint8(255)
    tifffile.imwrite(path, full, imagej=True, compression="zlib" if compress else None,
                     compressionargs={"level": 6} if compress else None,
                     resolution=(1 / sp[2], 1 / sp[1]),
                     metadata={"spacing": float(sp[0]), "unit": "um", "axes": "ZYX"})
    if not np.array_equal(tifffile.imread(path), full):
        raise IOError(f"{path}: written TIFF differs from the mask")
    return path

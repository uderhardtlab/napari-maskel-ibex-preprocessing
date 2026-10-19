"""napari reader for converted light-sheet .h5 files (datasets Channel_0, Channel_1, ..., each
(Z, Y, X), attribute element_size_um = voxel size in um). Channels are opened lazily with dask
(one chunk = one z slice), with the correct anisotropic scale; the layer metadata records the
source file and channel index (used by the preprocessing widget)."""
from pathlib import Path
import numpy as np
import h5py
import dask.array as da

COLORMAPS = ["magenta", "yellow", "gray", "green", "cyan", "blue", "red", "bop orange"]


def get_reader(path):
    if isinstance(path, list):
        path = path[0]
    if not str(path).lower().endswith(".h5"):
        return None
    try:
        with h5py.File(path, "r") as f:
            if "Channel_0" not in f or f["Channel_0"].ndim != 3:
                return None
    except OSError:
        return None
    return read_h5


def read_h5(path):
    if isinstance(path, list):
        path = path[0]
    f = h5py.File(path, "r")                       # kept open while the layers exist
    keys = sorted((k for k in f if k.startswith("Channel_")), key=lambda k: int(k.split("_")[1]))
    sp = np.asarray(f[keys[0]].attrs.get("element_size_um", (1.0, 1.0, 1.0)), float)
    layers = []
    for i, k in enumerate(keys):
        d = f[k]
        data = da.from_array(d, chunks=(1,) + d.shape[1:], lock=True)
        mid = d[d.shape[0] // 2]
        nz = mid[mid > 0]
        lim = [0.0, float(np.percentile(nz, 99.8))] if nz.size else [0.0, 255.0]
        cname = d.attrs.get("channel_name", "")
        layers.append((data, dict(
            name=f"{k}" + (f" ({cname})" if cname else ""), scale=tuple(sp),
            colormap=COLORMAPS[i % len(COLORMAPS)], blending="additive", contrast_limits=lim,
            visible=i == 0,
            metadata=dict(maskel_source=str(Path(path).resolve()), maskel_channel=i,
                          voxel_size_um=sp.tolist())), "image"))
    return layers

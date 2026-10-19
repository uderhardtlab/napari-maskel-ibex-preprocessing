"""Imaris .ims -> .h5: copies full resolution (ResolutionLevel 0, TimePoint 0) into one 3D
dataset (Z, Y, X) per channel ('Channel_0', ...), cropped to the real image size
(DataSetInfo/Image X, Y, Z; Imaris pads the stored arrays). The .ims is only read; an existing
.h5 is never overwritten. Written to <stem>.h5.part and renamed when complete, so an
interrupted conversion never leaves an incomplete file that looks finished.

Command line: maskel-ibex-ims2h5 <sample.ims>
"""
import sys
from pathlib import Path
import numpy as np
import h5py

Z_BLOCK = 8                       # z slices copied per step (limits RAM)


def _attr_str(group, name):
    """Imaris stores attributes as arrays of single bytes, e.g. [b'5', b'6'] -> '56'."""
    return np.asarray(group.attrs[name]).tobytes().decode()


def ims_to_h5(src, log=print):
    src = Path(src)
    dst = src.with_suffix(".h5")
    if dst.exists():
        raise FileExistsError(f"{dst} already exists - nothing overwritten.")
    part = dst.with_name(dst.name + ".part")
    if part.exists():
        part.unlink()
    with h5py.File(src, "r") as s, h5py.File(part, "w") as d:
        info = s["DataSetInfo/Image"]
        nx, ny, nz = (int(_attr_str(info, k)) for k in ("X", "Y", "Z"))
        ext = {k: float(_attr_str(info, k)) for k in
               ("ExtMin0", "ExtMax0", "ExtMin1", "ExtMax1", "ExtMin2", "ExtMax2")}
        dx = (ext["ExtMax0"] - ext["ExtMin0"]) / nx
        dy = (ext["ExtMax1"] - ext["ExtMin1"]) / ny
        dz = (ext["ExtMax2"] - ext["ExtMin2"]) / nz
        level = s["DataSet/ResolutionLevel 0/TimePoint 0"]
        channels = sorted(level.keys(), key=lambda k: int(k.split()[-1]))
        log(f"Image: Z={nz}, Y={ny}, X={nx}, {len(channels)} channels; "
            f"voxel (Z, Y, X) = ({dz:.4f}, {dy:.4f}, {dx:.4f}) um")
        for ch in channels:
            data = level[ch]["Data"]
            out = d.create_dataset(ch.replace(" ", "_"), shape=(nz, ny, nx), dtype=data.dtype,
                                   chunks=(1, ny, nx), compression="gzip", compression_opts=4)
            for z0 in range(0, nz, Z_BLOCK):
                z1 = min(z0 + Z_BLOCK, nz)
                out[z0:z1] = data[z0:z1, :ny, :nx]
            out.attrs["element_size_um"] = (dz, dy, dx)
            src_ch = s.get(f"DataSetInfo/{ch}")
            if src_ch is not None and "Name" in src_ch.attrs:
                out.attrs["channel_name"] = _attr_str(src_ch, "Name")
            log(f"  {ch}: done")
    part.rename(dst)
    log(f"Written: {dst}")
    return dst


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: maskel-ibex-ims2h5 <sample.ims>")
    ims_to_h5(sys.argv[1])

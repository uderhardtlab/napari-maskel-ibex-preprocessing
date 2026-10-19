"""Step 3: submesothelial fibroblast mask = CD34 above the muscle surface, small pieces removed.

  components = 26-connected components of the WHOLE CD34 mask (connectivity decided before the
               cut, so pieces linked to the network only through voxels below the cutoff stay
               linked)
  kept       = components with volume >= 500 um^3
  final      = kept & (d > -4 um), d = (z - h) * dz: from one slice below the top muscle voxel
               upwards (the surface sits ~1-2 slices too high; CD34 is densest at d = 0, -2 um)
"""
import numpy as np
from scipy import ndimage as ndi
from .filters import S26


def compute_submeso(cd34_mask, h_full, sp, p, log=print):
    """cd34_mask (Z, Y, X) bool; h_full height map at full resolution for the same region."""
    sp = np.asarray(sp, float)
    nz = cd34_mask.shape[0]
    above = np.empty(cd34_mask.shape, bool)
    for z in range(nz):
        above[z] = (z - h_full) * sp[0] > p["cutoff_um"]
    lab, n = ndi.label(cd34_mask, structure=S26)
    vol = np.bincount(lab.ravel()) * sp.prod()
    keep = vol >= p["min_component_um3"]
    keep[0] = False
    kept = keep[lab]
    reach = np.bincount(lab[above].ravel(), minlength=n + 1) > 0
    reach[0] = False
    del lab
    final = kept & above
    removed_small = cd34_mask & ~kept & above
    removed_cut = kept & ~above
    tot = int((cd34_mask & above).sum())
    log(f"distance filter: {n} components, {int(reach.sum())} reach above d > "
        f"{p['cutoff_um']:+g} um, kept {int((keep & reach).sum())}; final {int(final.sum())} "
        f"voxels = {final.sum() / max(tot, 1):.2%} of CD34 above the cutoff")
    return dict(final=final, kept=kept, removed_small=removed_small, removed_by_cut=removed_cut)

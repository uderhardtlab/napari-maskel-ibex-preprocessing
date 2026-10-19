"""Default parameters. All lengths in um (converted to voxels per axis with the voxel size of
the sample); volumes in um^3. The values are those validated on
LSM 2026-07-20_B6-12w-naive_co1-s1-r1 (see the README for the reasoning behind each)."""
from copy import deepcopy

DEFAULTS = dict(
    # --- channels (indices into the h5 / the list of loaded channels)
    muscle_channel=0,                 # TOM20 (muscle fibres: mitochondria + autofluorescence)
    cd34_channel=3,                   # CD34 (fibroblasts)
    # --- step 1: muscle surface (computed on a ds x ds XY block grid)
    layers_ds=4,                      # XY block size in pixels
    orient_smooth_um=2.0,             # fibre direction: smoothing of the TOM20 max projection
    orient_rho_um=20.0,               # fibre direction: structure tensor window
    tom_sigma_um=[2.0, 5.0, 5.0],     # TOM20 smoothing (Z, Y, X)
    n_classes=3,                      # multi-Otsu classes; muscle = above the LOWEST threshold
    muscle_threshold=None,            # None = automatic (multi-Otsu); a number overrides it
    muscle_min_volume_um3=5000.0,     # smaller 3D muscle objects removed
    fibre_line_um=30.0,               # fibre filter: line length along the fibre direction
    fibre_theta_deg=None,             # None = measure; else fibre direction (x right, y down)
    close_radius_um=9.2,              # grey closing of h: fills gaps between fibres <= ~18 um
    # --- step 2: CD34 mask
    tissue_ds=2,                      # XY block size for tissue mask and CD34 background
    tissue_smooth_um=5.0,             # tissue mask: smoothing of the summed channels
    cd34_smooth_um=[1.0, 0.5, 0.5],   # CD34 smoothing (Z, Y, X)
    bg_sigma_um=15.0,                 # CD34 background (isotropic, on the 2x2 grid)
    cd34_min_volume_um3=20.0,         # objects smaller than this removed
    hyst_k_sd=4.0,                    # hysteresis low threshold = local median + k * local SD
    hyst_block_um=272.0,              # block size of the local noise statistics
    hyst_min_noise_vox=10000,         # blocks with fewer noise voxels take the nearest block
    hyst_noise_band_um=[-10.0, 30.0],  # noise only where -10 <= d <= +30 um
    # --- step 3: distance filter
    cutoff_um=-4.0,                   # keep CD34 with signed distance d > cutoff
    min_component_um3=500.0,          # keep 26-connected CD34 components >= this volume
)


def defaults():
    """A fresh copy of the default parameters."""
    return deepcopy(DEFAULTS)

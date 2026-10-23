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
    muscle_min_component_fraction=0.01,  # muscle components < this x the largest are removed
                                      # (objects on the tissue, e.g. a hair; 0 = keep all)
    macrophage_exclusion=True,        # columns with CD11b+ muscle at the top: h from neighbours
    macrophage_channel=4,             # CD11b
    macrophage_sigma_um=[1.0, 2.0, 2.0],  # CD11b smoothing (Z, Y, X) on the block grid
    macrophage_threshold=None,        # None = Otsu of all valid voxels; a number overrides it
    macrophage_top_slices=2,          # CD11b+ muscle within this many slices below the top
    macrophage_dilate_um=10.0,        # macrophage columns dilated in XY (~2 x TOM20 sigma XY)
    macrophage_max_column_fraction=0.2,  # more macrophage columns than this: rule skipped
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
    # --- step 4: Cx43 (separate window / batch step)
    cx43_channel=2,                   # Cx43
    cx43_smooth_px=[0.0, 1.0, 0.7],   # Gaussian sigma (Z, Y, X) in PIXELS (tied to the line
                                      # step of 2 px in Y of the acquisition, hence pixels)
    cx43_bg_block_px=4,               # background: block means of 4 x 4 px ...
    cx43_bg_um=14.8,                  # ... median over this window (8 x 8 blocks = 32 px)
    cx43_threshold=12.0,              # positive: grey values above the local background > T
    cx43_max_dist_um=2.0,             # only within this distance of the fibroblast mask
)


def defaults():
    """A fresh copy of the default parameters."""
    return deepcopy(DEFAULTS)

# napari-maskel-ibex-preprocessing

<p align="center">
  <img src="https://raw.githubusercontent.com/uderhardtlab/napari-maskel-ibex-preprocessing/main/src/napari_maskel_ibex_preprocessing/resources/logo_preprocessing.png" alt="MaSkel IBEX preprocessing logo" width="400">
</p>

[![PyPI](https://img.shields.io/pypi/v/napari-maskel-ibex-preprocessing.svg)](https://pypi.org/project/napari-maskel-ibex-preprocessing)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-maskel-ibex-preprocessing)](https://napari-hub.org/plugins/napari-maskel-ibex-preprocessing)
[![tests](https://github.com/uderhardtlab/napari-maskel-ibex-preprocessing/actions/workflows/tests.yml/badge.svg)](https://github.com/uderhardtlab/napari-maskel-ibex-preprocessing/actions/workflows/tests.yml)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://github.com/uderhardtlab/napari-maskel-ibex-preprocessing/blob/main/LICENSE)

napari plugin and batch runner for light-sheet data of mouse peritoneum. It does three things:

- finds the **outer surface of the muscle layer** from the TOM20 channel;
- builds a **CD34 fibroblast mask**;
- keeps only the CD34 above the muscle (the **submesothelial fibroblasts**) and exports binary TIFFs, e.g. for Imaris or as input for MASKEL.

The algorithms are in `napari_maskel_ibex_preprocessing.core` and use only numpy/scipy (no napari). Both the napari widget and the command-line runner use them.

## Installation

Into an existing napari environment (Python ≥ 3.9), either in napari via *Plugins > Install/Uninstall Plugins*, or:

```
pip install napari-maskel-ibex-preprocessing
```

Use `pip install "napari-maskel-ibex-preprocessing[napari]"` to install napari as well.

For development, from a clone of this repository:

```
pip install -e ".[testing]"
pytest tests
```

## Input

A converted `.h5` file with one dataset per channel, `Channel_0`, `Channel_1`, …, each `(Z, Y, X)`. `Channel_0` carries the attribute `element_size_um` (voxel size `(dz, dy, dx)` in µm).

An Imaris `.ims` file can be converted with `maskel-ibex-ims2h5 sample.ims`, or by the batch runner automatically.

Assumed geometry:
- the tissue is imaged en face, with the **mesothelium at high z** and the muscle at low z;
- muscle is present under the whole field of view;
- the fibres are roughly parallel.

## napari

1. Open the `.h5` (*File > Open*; choose this plugin's reader if asked). Every channel becomes an image layer with the correct voxel size.
2. Open *Plugins > MASKEL IBEX preprocessing > Muscle surface and CD34 filtering*.
3. Choose the **muscle channel** (TOM20) and the **CD34 channel**. For files from the reader, Channel_0 and Channel_3 are pre-selected.
4. For fast tuning, draw a **rectangle** in a Shapes layer (2D view) and choose that layer as the *region*. Every step has *Run on region* (seconds to a minute) and *Run on full volume* (minutes). Both run in the background.
   - **1 Muscle surface:** muscle mask, surface, height map `h` and the **signed distance map** `d = (z − h)·dz` (µm; > 0 = mesothelium side).
   - **2 CD34 mask:** CD34 mask, thresholded by hysteresis with a local, noise-based low threshold.
   - **3 Distance filter:** keeps CD34 components ≥ the minimum volume, then the part with `d >` cutoff → **fibroblast mask**.
5. **Export** (needs the full-volume results; TIFFs in the original image shape, uint8 0/255, voxel size in the metadata, optional ZIP compression):
   - *distance mask*: 1 where `d >` threshold (given in µm or in z slices), else 0;
   - *fibroblast mask*;
   - optionally all results as a steps `.h5`.

Results computed on a region use Otsu thresholds and noise statistics of that region only. They can therefore differ slightly from the same area of a full-volume run.

## Batch

```
maskel-ibex-batch D:\data\sample.ims
maskel-ibex-batch D:\data\round1 --dry-run
maskel-ibex-batch D:\data\round1 --dist-threshold -4 --dist-unit um
```

- **Input:** a file or a folder. A folder means every `.h5` / `.ims` in it.
- **Parameters:** the defaults below.
- **Output per sample:**
  - `<stem>_layers.h5`, `<stem>_cd34.h5`, `<stem>_submeso_cd34.h5`: intermediate steps;
  - `<stem>_tif\…_fibroblast_mask.tif` and `…_distance_gt<T>.tif`;
  - a log file.
- `--force` recomputes, `--uncompressed` writes plain TIFFs.

## Steps and default parameters (`core/params.py`)

| Step | Operation | Default |
|---|---|---|
| 1 | TOM20 block means (valid pixels only) | 4 × 4 px |
| 1 | normalized Gaussian | σ (2, 5, 5) µm |
| 1 | muscle = above the lowest 3-class multi-Otsu threshold; small objects removed | < 5000 µm³ |
| 1 | fibre filter: binary opening with a line along the measured fibre direction (structure tensor) | 30 µm |
| 1 | `h` = highest muscle z; empty columns from the nearest column; grey closing | disk 9.2 µm |
| 2 | tissue = Otsu on log of the summed channels (2 × 2 blocks) | smoothing 5 µm |
| 2 | CD34 normalized Gaussian minus background | σ (1, 0.5, 0.5) µm; background 15 µm |
| 2 | seeds = above Otsu within tissue | |
| 2 | noise = tissue outside the Otsu mask, in the band −10 ≤ d ≤ 30 µm; per block median + k · robust SD | blocks 272 µm, k = 4 |
| 2 | hysteresis: components of the low-threshold mask that contain a seed; small objects removed | < 20 µm³ |
| 3 | 26-connected components of the whole CD34 mask; keep those ≥ the minimum volume, then `d >` cutoff | 500 µm³; −4 µm |

The defaults were validated on `LSM 2026-07-20_B6-12w-naive_co1-s1-r1` (naive, round 1).

## Development and releases

- **Every push and pull request:** `.github/workflows/tests.yml` runs the tests on Windows, Linux and macOS with Python 3.9–3.12. It also builds the package and checks it with `twine check`.
- **Release:** set the new `version` in `pyproject.toml` and in `src/napari_maskel_ibex_preprocessing/__init__.py`, commit, then push a tag with the same version:
  ```
  git tag v0.1.0
  git push origin v0.1.0
  ```
  `.github/workflows/release.yml` builds the package, checks that the tag matches the version, and publishes to PyPI through trusted publishing.

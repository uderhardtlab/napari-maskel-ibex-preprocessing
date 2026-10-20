"""napari-maskel-ibex-preprocessing: muscle surface, CD34 fibroblast mask, distance filtering.

The algorithms live in `core` (numpy / scipy only, usable without napari); the napari reader
and dock widget, and the command-line batch runner `maskel-ibex-batch`, all use them.
"""
__version__ = "0.2.0"

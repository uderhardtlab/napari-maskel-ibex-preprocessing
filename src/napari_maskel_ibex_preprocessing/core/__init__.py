"""Algorithms (no napari dependency).

params   default parameters with units
filters  Gaussian / normalized convolution / block means / morphology helpers
io       volume sources (h5 file or in-memory / dask arrays, optional XY region), steps
         files, TIFF export
ims      Imaris .ims -> .h5 conversion
muscle   fibre orientation, muscle mask, muscle surface h(y, x), signed distance
cd34     tissue mask, CD34 Otsu mask, CD34 hysteresis mask (local noise threshold)
submeso  components >= minimum volume, cut at the distance cutoff -> fibroblast mask
"""

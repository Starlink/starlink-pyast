#!/home/dsb/bin/python3

from __future__ import print_function

import sys
import starlink.Ast as Ast
import starlink.Grf as Grf
import matplotlib.pyplot as plt

#  Check the header name was supplied on the command line
if len(sys.argv) < 2:
    print("Usage: plothead.py <basename>")
    sys.exit(2)

#  Attempt to open the header file
fin1 = open(sys.argv[1] + ".head")

#  Attempt to open an associated file holding attributes that control how
#  the FITS headers are interpreted. If succesful, read the whole file into
#  a single string.
try:
    fin2 = open(sys.argv[1] + ".fattr")
    fits_atts = fin2.read()
    fin2.close()
except (IOError):
    fits_atts = ""

#  Attempt to open an associated file holding attributes that control the
#  appearance of the plot. If successful, read the whole file into a
#  single string.
try:
    fin2 = open(sys.argv[1] + ".attr")
    plot_atts = fin2.read()
    fin2.close()
except (IOError):
    plot_atts = ""

#  Attempt to open an associated file holding the pixel bounds of the
#  area of the FITS array to be plotted.
try:
    fin2 = open(sys.argv[1] + ".box")
    box = [float(v) for v in fin2.read().strip().split()]
    fin2.close()
except (IOError):
    box = None

#  Read the header lines into a list, and store this list in a new
#  Ast.FitsChan, using the requested attributes to modify the
#  interpretation of the header.
fc = Ast.FitsChan(fin1.readlines(), None, fits_atts)

#  Close the header file.
fin1.close()

#  Create a FrameSet from the FITS headers in the FitsChan.
fs = fc.read()
if fs is None:
    print("Could not read WCS from headers in " + sys.argv[1] + ".head")
    sys.exit(2)

#  Exit if the header does not have exactly 2 pixel axes and 2 WCS axes.
if fs.Nin != 2 or fs.Nout != 2:
    print("The headers in " + sys.argv[1] + ".head do not describe a 2-D image")
    sys.exit(2)

#  Create a matplotlib plotting region. Ensure that the matplotlib axis
#  annotations are not drawn.
dx = 12.0
dy = 9.0
fig = plt.figure(figsize=(dx, dy))
ax = fig.add_axes((0, 0, 1, 1))
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

#  If the bounds of the pixel grid to be plotted were specified, use them
if box is not None:
    bbox = box

#  Otherwise, we map the entire FITS pixel grid onto this box. So get the
#  bounds of the pixel grid from the FitsChan. If the NAXIS1/2 keywords
#  are not in the FitsChan, use defaults of 500.
else:
    if "NAXIS1" in fc:
        naxis1 = fc["NAXIS1"]
    else:
        naxis1 = 500

    if "NAXIS2" in fc:
        naxis2 = fc["NAXIS2"]
    else:
        naxis2 = 500
    bbox = (0.5, 0.5, naxis1 + 0.5, naxis2 + 0.5)

#  Set the bounds (in matplotlib data coordinates) of the largest rectangle
#  that can be drawn on the matplotlib plotting area that has the same
#  aspect ratio as the FITS array.
fits_aspect_ratio = (bbox[3] - bbox[1]) / (bbox[2] - bbox[0])
if fits_aspect_ratio < 0.05 or fits_aspect_ratio > 20:
    fits_aspect_ratio = 1.0
grf_aspect_ratio = dy / dx
if grf_aspect_ratio > fits_aspect_ratio:
    hx = 0.5
    hy = 0.5 * fits_aspect_ratio / grf_aspect_ratio
else:
    hx = 0.5 * grf_aspect_ratio / fits_aspect_ratio
    hy = 0.5

#  Shrink the box to leave room for axis annotation.
hx *= 0.7
hy *= 0.7

gbox = (0.5 - hx, 0.5 - hy, 0.5 + hx, 0.5 + hy)

#  Create a drawing object that knows how to draw primitives (lines,
#  marks and strings) into the matplotlib plotting region.
grf = Grf.grf_matplotlib(ax)

#  Create the AST Plot, using the above object to draw the primitives. The
#  Plot is based on the FrameSet that describes the WCS read from the FITS
#  headers, so the plot knows how to convert from WCS coords to pixel
#  coords, and then to matplotlib data coords. Specify the plotting
#  attributes read from the "attr" file.
plot = Ast.Plot(fs, gbox, bbox, grf, plot_atts)

#  And finally, draw the annotated WCS axes and any coordinate grid
#  requested in the plotting attributes.
plot.grid()

#  Make the matplotlib plotting area visible
plt.show()

import matplotlib.pyplot as plt
import starlink.Ast
import starlink.Grf

#  Create a figure covering the whole work space
fig = plt.figure()

#  Create a plotting region ( matplotlib calls it an "Axes") covering the
#  whole figure. Ensure that the matplotlib axis annotations are not drawn.
ax = fig.add_subplot(111)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

#  Create a grf module that knows how to draw into the matplotlib plotting
#  region.
grf = starlink.Grf.grf_matplotlib(ax)

#  Find the coords at the centre of the plotting region, and its half-width
#  and half-height.
xlo, xhi = ax.get_xbound()
ylo, yhi = ax.get_ybound()
xcen = (xlo + xhi) / 2
ycen = (ylo + yhi) / 2
w = 0.5 * (xhi - xlo)
h = 0.5 * (yhi - ylo)

#  The AST Plot will cover half the matplotlib plotting region on ach axis.
gbox = (xcen - w / 2, ycen - h / 2, xcen + w / 2, ycen + h / 2)

bbox = (-100, -100, 100, 100)

#  Create the AST Plot describing coords from -100 to +100 on each axis,
#  linearly related to the matplotlib coords. Use thegrf module created
#  above.
plot = starlink.Ast.Plot(None, gbox, bbox, grf)

#  Draw a coordinate grid, using the matplotlib default graphics
#  properties.
plot.grid()

#  Set markers to use the default colour 1 provided by the grf module,
#  then plot a set of threee markers using marker type 5 (the default
#  type 5 marker is also defined in the grf module).
plot.Colour_Markers = 1
plot.mark([[-80, 0, 80], [80, 0, -80]], 5)

#  Draw a thick line.
plot.Width_Curves = 10
plot.curve([-80, -80], [80, 80])

#  Plot a string using the default font 0 provided by the grf module.
plot.Font_Strings = 0
plot.text("Hello", [0, 75], [1, 1], "CC")

#  Change the definition of font 0 within the grf module, and plot
#  another string, still using font 0.
grf.fonts[0] = {"family": 'monospace', "style": 'italic'}
plot.text("Hello", [0, 65], [-1, 1], "CC")

#  Display everything.
plt.show()

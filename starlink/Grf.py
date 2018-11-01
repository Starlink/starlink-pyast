import starlink.Ast as Ast
import matplotlib
import matplotlib.pyplot
import matplotlib.lines
import math

"""
This module provides classes that provide primitive drawing facilities
for the starlink.Ast.Plot class. Each class defined in this module
implements the drawing methods required by the Plot class, using a
different underlying graphics system.

Currently only one class is provided, which uses the matplotlib
package to provide the primitive rdawing capabilities.

For further information about the methods implemented by these
classes, see the file grf_pgplot.c included in the AST source
distribution.
"""


class grf_matplotlib(object):

    """
    When creating a grf_matplotlib, the supplied "axes" object should be an
    instance of the matplotlib Axes class (or a subclass).
    """

# ------------------------------------------------------------------------
    def __init__(self, axes):
        if isinstance(axes, matplotlib.axes.Axes):
            self.axes = axes
            self.renderer = None

#  Save the current axis scales.
            self.Scales()

#  Create a temporary text string and line from which we can determine
#  the default graphics properties.
            xl, xr = self.axes.get_xlim()
            yb, yt = self.axes.get_ylim()
            xc = 0.5 * (xl + xr)
            yc = 0.5 * (yt + yb)
            text = matplotlib.text.Text(xc, yc, "a")
            line = matplotlib.lines.Line2D([xc], [yc], marker="+")

#  Save the current default marker and text sizes.
            self.__deftsize = text.get_size()
            self.__defmsize = line.get_markersize()

#  Save the default text colour.
            defcol = text.get_color()

#  Save the default text font family and style.
            deffont = {"family": text.get_family(), "style": text.get_style()}

#  Save the default line style
            defstyle = line.get_linestyle()

#  A list used to convert AST integer marker types into matplotlib
#  character marker types.
            self.markers = ['s', '.', '+', '*', 'o', 'x', ',', '^', 'v', '<', '>',
                            'p', 'h', 'D']

#  A list used to convert AST integer line style types into corresponding matplotlib
#  properties. Ensure the first line style is the default.
            self.styles = [{"linestyle": defstyle}, {"linestyle": '-'},
                           {"linestyle": '--'}, {"linestyle": ':'},
                           {"linestyle": '-.'}]

#  A list used to convert AST integer font types into corresponding matplotlib
#  properties. Ensure the first font is the default.
            self.fonts = [deffont, {"family": 'serif', "style": 'normal'},
                          {"family": 'serif', "style": 'italic'},
                          {"family": 'sans-serif', "style": 'normal'},
                          {"family": 'sans-serif', "style": 'italic'},
                          {"family": 'monospace', "style": 'normal'},
                          {"family": 'monospace', "style": 'italic'}]

#  A list used to convert AST integer colours into corresponding matplotlib
#  properties. Ensure the first colour is the default.
            self.colours = [{"color": defcol}, {"color": '#ff0000'}, {"color": '#00ff00'},
                            {"color": '#0000ff'}, {"color": '#00ffff'},
                            {"color": '#ff00ff'}, {"color": '#ffff00'},
                            {"color": '#000000'}, {"color": '#a9a9a9'},
                            {"color": '#808080'}, {"color": '#d3d3d3'},
                            {"color": '#ffffff'}]

#  The current graphics attribute values as used by AST
            self.__attrs = {Ast.grfLINE: {Ast.grfSTYLE: 1, Ast.grfWIDTH: 1, Ast.grfSIZE: 1,
                                          Ast.grfFONT: 1, Ast.grfCOLOUR: 1},
                            Ast.grfMARK: {Ast.grfSTYLE: 1, Ast.grfWIDTH: 1, Ast.grfSIZE: 1,
                                          Ast.grfFONT: 1, Ast.grfCOLOUR: 1},
                            Ast.grfTEXT: {Ast.grfSTYLE: 1, Ast.grfWIDTH: 1, Ast.grfSIZE: 1,
                                          Ast.grfFONT: 1, Ast.grfCOLOUR: 1}}

#  The corresponding graphics properties used by matplotlib
            self.__props = {Ast.grfLINE: {"solid_capstyle": 'butt'},
                            Ast.grfMARK: {}, Ast.grfTEXT: {}}

#  Ensure the defaults are current.
            for attr in (Ast.grfCOLOUR, Ast.grfWIDTH, Ast.grfSIZE,
                         Ast.grfFONT, Ast.grfSTYLE):
                for prim in (Ast.grfTEXT, Ast.grfLINE, Ast.grfMARK):
                    self.Attr(attr, 1.0, prim)

#  Set new delimiters for graphical sky axis values, using appropriate escape
#  sequences to get he superscripts looking nice.
            Ast.tunec("hrdel", "%-%^85+%s70+h%>45+%+")
            Ast.tunec("mndel", "%-%^85+%s70+m%>45+%+")
            Ast.tunec("scdel", "%-%^85+%s70+s%>45+%+")
            Ast.tunec("dgdel", "%-%^90+%s60+o%>45+%+")
            Ast.tunec("amdel", "%-%^30+%s85+'%>45+%+")
            Ast.tunec("asdel", "%-%^30+%s85+\"%>45+%+")
            Ast.tunec("exdel", "10%-%^85+%s60+%>20+")

#  Initialise the correction vector for text.
            self._xcorr = 0.0
            self._ycorr = 0.0

#  Save the current character heights, and update the vertical offset
#  correction for text.
            self.Qch()

#  Report an error if the supplied object is not suitable
        else:
            m = axes.__class__.__module__
            c = axes.__class__.__name__
            if m != "builtins":
                c = m + "." + c
            raise TypeError("The supplied axes object is a " + c + ", it should "
                            "an instance of matplotlib.axes.Axes or a subclass")

# ------------------------------------------------------------------------
#  Some backends, such as TkAgg, have the get_renderer method, which #makes this
#  easy. Other backends do not have the get_renderer method, so we have a work
#  around to find the renderer.  Print the figure to a temporary file #object,
#  and then grab the renderer that was used. This trick is stolen from the
#  matplotlib backend_bases.py print_figure() method.

    def find_renderer(self, fig):

        if not self.renderer:
            if hasattr(fig, "canvas"):
                if hasattr(fig.canvas, "get_renderer"):
                    self.renderer = fig.canvas.get_renderer()
                else:
                    try:
                        import io
                        fig.canvas.print_pdf(io.BytesIO())
                        self.renderer = fig._cachedRenderer
                    except Exception:
                        pass

        if not self.renderer:
            raise AttributeError("No renderer available using matplotlib "
                                 "backend {0} - use a different backend".
                                 format(matplotlib.get_backend()))

        return(self.renderer)

# ------------------------------------------------------------------------
    def Attr(self, attr, value, prim):

        #  Save the old AST attribute value.
        oldval = self.__attrs[prim][attr]

#  Nothing more to do if the new value is AST__BAD
        if value != Ast.BAD:

            #  Save the old AST attribute value, and record the new value (if not .
            oldval = self.__attrs[prim][attr]
            self.__attrs[prim][attr] = value

#  Now need to update the matplotlib properties to make them reflect the
#  new AST value.

#  Style only applied to lines
            if attr == Ast.grfSTYLE:
                value = int(value) - 1
                if prim == Ast.grfLINE:
                    if value >= 0 and value < len(self.styles):
                        self.__props[prim].update(self.styles[value])

#  Width applies to lines, marks and texts
            elif attr == Ast.grfWIDTH:
                if prim == Ast.grfLINE:

                    #  Get bounds of plot in user coords
                    xl, xr = self.axes.get_xlim()
                    yb, yt = self.axes.get_ylim()

#  Transform to device coords
                    tr = self.axes.transData
                    xl, yb = tr.transform([xl, yb])
                    xr, yt = tr.transform([xr, yt])

#  Transform to inches
                    tr = self.axes.get_figure().dpi_scale_trans.inverted()
                    xl, yb = tr.transform([xl, yb])
                    xr, yt = tr.transform([xr, yt])

#  Find the length of the diagonal in inches, and convert to points.
                    diag = 72.0 * math.sqrt((xr - xl)**2 + (yt - yb)**2)

#  Find the number of points corresponding to an AST line width of 1.0,
#  ensure it is at least 1 point, and then scale it by the supplied AST
#  line width.
                    lw = max(0.0005 * diag, 1.0) * value
                    self.__props[prim].update({"linewidth": lw})

#  Similar for markers
                elif prim == Ast.grfMARK:
                    xl, xr = self.axes.get_xlim()
                    yb, yt = self.axes.get_ylim()
                    tr = self.axes.transData
                    xl, yb = tr.transform([xl, yb])
                    xr, yt = tr.transform([xr, yt])
                    tr = self.axes.get_figure().dpi_scale_trans.inverted()
                    xl, yb = tr.transform([xl, yb])
                    xr, yt = tr.transform([xr, yt])
                    diag = 72.0 * math.sqrt((xr - xl)**2 + (yt - yb)**2)
                    lw = max(0.0005 * diag, 1.0) * value
                    self.__props[prim].update({"markeredgewidth": lw})

#  Cannot control exact line width of texts, use weight instead.
                elif prim == Ast.grfTEXT:
                    if value < 0.0:
                        wgt = 0.0
                    elif value < 1.0:
                        wgt = 250.0 * value
                    elif value < 10.0:
                        wgt = (650.0 * value + 1600.0) / 9.0
                    else:
                        wgt = 900
                    self.__props[prim].update({"weight": wgt})

#  Size applies to marks and texts
            elif attr == Ast.grfSIZE:
                if prim == Ast.grfMARK:
                    self.__props[prim].update({"markersize": self.__defmsize * value})
                elif prim == Ast.grfTEXT:
                    self.__props[prim].update({"size": self.__deftsize * value})

#  Font only applies to texts
            elif attr == Ast.grfFONT:
                value = int(value) - 1
                if prim == Ast.grfTEXT:
                    if value >= 0 and value < len(self.fonts):
                        self.__props[prim].update(self.fonts[value])

#  Colour applied to them all
            elif attr == Ast.grfCOLOUR:
                value = int(value) - 1
                if value >= 0 and value < len(self.colours):
                    self.__props[prim].update(self.colours[value])

        return oldval

# ------------------------------------------------------------------------
    def BBuf(self):
        return

# ------------------------------------------------------------------------
    def Cap(self, cap, value):
        if cap == Ast.grfSCALES:
            return 1
        elif cap == Ast.grfMJUST:
            return 1
        elif cap == Ast.grfESC:
            return 0
        else:
            return 0

# ------------------------------------------------------------------------
    def EBuf(self):
        return

# ------------------------------------------------------------------------
    def Flush(self):
        return

# ------------------------------------------------------------------------
    def Line(self, n, x, y):
        self.axes.add_line(matplotlib.lines.Line2D(x, y,
                                                   **self.__props[Ast.grfLINE]))

# ------------------------------------------------------------------------
    def Mark(self, n, x, y, type):
        if type < 0 or type >= len(self.markers):
            marker = '.'
        else:
            marker = self.markers[type]

        props = self.__props[Ast.grfMARK].copy()
        props["linestyle"] = 'None'
        props["marker"] = marker
        self.axes.add_line(matplotlib.lines.Line2D(x, y, **props))

# ------------------------------------------------------------------------
    def Qch(self):
        xl, xr = self.axes.get_xlim()
        yb, yt = self.axes.get_ylim()
        x = 0.5 * (xl + xr)
        y = 0.5 * (yt - yb)

        a = self.TxExt("a", x, y, "CC", 0, 1)
        self._chh = max(a[4:]) - min(a[4:])

        a = self.TxExt("a", x, y, "CC", 1, 0)
        self._chv = max(a[:3]) - min(a[:3])

        self._xcorr = self._chv * 0.15
        self._ycorr = self._chh * 0.15

        return (self._chv, self._chh)

# ------------------------------------------------------------------------
    def Scales(self):
        xleft, xright = self.axes.get_xlim()
        ybot, ytop = self.axes.get_ylim()
        a = self.axes.transData.transform([(xleft, ybot), (xright, ytop)])
        self.__alpha = (a[1][0] - a[0][0]) / (xright - xleft)
        self.__beta = (a[1][1] - a[0][1]) / (ytop - ybot)
        return (self.__alpha, self.__beta)

# ------------------------------------------------------------------------
    def Text(self, text, x, y, just, upx, upy, boxprops={}):
        if just[0] == "T":
            va = "top"
        elif just[0] == "B":
            va = "baseline"
        elif just[0] == "M":
            va = "bottom"
        else:
            va = "center"
        if just[1] == "L":
            ha = "left"
        elif just[1] == "R":
            ha = "right"
        else:
            ha = "center"
        rot = math.atan2(-upx, upy) * Ast.DR2D

#  matplotlib always seems to plot each text string a little higher than
#  requested, sp correct the reference position by a small amount
#  determined empirically to produce visually better text positioning.
        uplen = math.sqrt(upx**2 + upy**2)
        if uplen > 0.0:
            x -= upx * self._xcorr / uplen
            y -= upy * self._ycorr / uplen

        props = self.__props[Ast.grfTEXT].copy()
        props["verticalalignment"] = va
        props["horizontalalignment"] = ha
        props["rotation"] = rot
        props["rotation_mode"] = "anchor"
        if len(boxprops) > 0:
            props["bbox"] = boxprops

        otext = matplotlib.text.Text(x, y, text, **props)
        self.axes.add_artist(otext)
        return otext

# ------------------------------------------------------------------------
    def TxExt(self, text, x, y, just, upx, upy):
        otext = self.Text(text, x, y, just, upx, upy, boxprops={"boxstyle": "square,pad=0.0"})
        renderer = self.find_renderer(self.axes.get_figure())
        otext.draw(renderer)
        pix_verts = otext.get_bbox_patch().get_verts()
        wcs_verts = otext.get_transform().inverted().transform(pix_verts)
        otext.remove()

        return (wcs_verts[0][0], wcs_verts[1][0], wcs_verts[2][0], wcs_verts[3][0],
                wcs_verts[0][1], wcs_verts[1][1], wcs_verts[2][1], wcs_verts[3][1])

# ------------------------------------------------------------------------
    def ColToInt(self, colour):
        result = -1

#  If integer, use as is.
        try:
            result = int(colour)

#  If not, convert the supplied string to an RGB triple, and then to a
#  html hex string.
        except ValueError:
            try:
                rgb = matplotlib.colors.colorConverter.to_rgb(colour)
                hex = matplotlib.colors.rgb2hex(rgb)

#  Check if this hex string is already in the list of known colours.
                index = -1
                for item in self.colours:
                    index += 1
                    if item['color'] == hex:
                        result = index
                        break

#  If not, add it to the end.
                if result == -1:
                    self.colours.append({"color": hex})
                    result = index + 1

#  Return -1 if the supplied colour is not legal.
            except ValueError:
                pass

#  GRF colours are zero-based, but AST colours are 1-based.
        if result != -1:
            result += 1

        return result

# ------------------------------------------------------------------------
    def IntToCol(self, colour):
        result = None

# Convert from 1-based AST values to zero based Grf values.
        colour = int(colour) - 1

#  Check it is in the range of the list of known colours (otherwise we
#  reyturn None).
        if colour >= 0 and colour < len(self.colours):

            #  Get the corresponding colour name (a html hex string).
            result = self.colours[colour]['color'].upper()

#  Replace the hex string with any corresponding standard colour name.
            for name, hex in matplotlib.colors.cnames.items():
                if hex == result:
                    result = name
                    break

#  Return the colour name.
        return result

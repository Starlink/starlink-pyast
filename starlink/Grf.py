import starlink.Ast as Ast
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

#------------------------------------------------------------------------
   def __init__(self,axes):
      if isinstance(axes,matplotlib.axes.Axes):
         self.axes = axes

#  A list used to convert AST integer marker types into matplotlib
#  character marker types.
         self.markers = ['s','.','+','*','o','x',',','^','v','<','>',
                         'p','h','D']

#  A list used to convert AST integer line style types into corresponding matplotlib
#  properties.
         self.styles = [ {"linestyle":'-'}, {"linestyle":'--'},
                         {"linestyle":':'}, {"linestyle":'-.'} ]

#  A list used to convert AST integer font types into corresponding matplotlib
#  properties.
         self.fonts = [ {"family":'serif',"style":'normal'},
                        {"family":'serif',"style":'italic'},
                        {"family":'sans-serif',"style":'normal'},
                        {"family":'sans-serif',"style":'italic'},
                        {"family":'monospace',"style":'normal'},
                        {"family":'monospace',"style":'italic'} ]

#  A list used to convert AST integer colours into corresponding matplotlib
#  properties.
         self.colours = [ {"color":'red'}, {"color":'green'},
                          {"color":'blue'}, {"color":'cyan'},
                          {"color":'magenta'}, {"color":'yellow'},
                          {"color":'black'}, {"color":'darkgrey'},
                          {"color":'grey'}, {"color":'lightgrey'},
                          {"color":'white'} ]

#  The current graphics attribute values as used by AST
         self.__attrs = { Ast.grfLINE:{Ast.grfSTYLE:1, Ast.grfWIDTH:1, Ast.grfSIZE:1,
                                       Ast.grfFONT:1, Ast.grfCOLOUR:1},
                          Ast.grfMARK:{Ast.grfSTYLE:1, Ast.grfWIDTH:1, Ast.grfSIZE:1,
                                       Ast.grfFONT:1, Ast.grfCOLOUR:1},
                          Ast.grfTEXT:{Ast.grfSTYLE:1, Ast.grfWIDTH:1, Ast.grfSIZE:1,
                                       Ast.grfFONT:1, Ast.grfCOLOUR:1}}

#  The corresponding graphics properties used by matplotlib
         self.__props = { Ast.grfLINE:{}, Ast.grfMARK:{}, Ast.grfTEXT:{}}

#  Save the current default marker and text sizes.
         xl,xr = self.axes.get_xlim()
         yb,yt = self.axes.get_ylim()
         xc = 0.5*(xl+xr);
         yc = 0.5*(yt+yb);
         self.__deftsize = matplotlib.text.Text( xc, yc, "a").get_size()
         self.__defmsize = matplotlib.lines.Line2D( [xc], [yc], marker="+").get_markersize()

#  Report an error if the supplied object is not suitable
      else:
         m = axes.__class__.__module__
         c = axes.__class__.__name__
         if m != "builtins":
            c = m + "." + c
         raise TypeError("The supplied axes object is a "+c+", it should "
                         "an instance of matplotlib.axes.Axes or a subclass")

#------------------------------------------------------------------------
   def Attr( self, attr, value, prim ):

#  Save the old AST attribute value.
      oldval = self.__attrs[prim][attr]

#  Nothing more to do if the new value is AST__BAD
      if value != Ast.BAD:

#  Save the old AST attribute value, and record the new value (if not .
         oldval = self.__attrs[prim][attr]
         self.__attrs[prim][attr] = value

#  Now need to update the matplotlib proprties to make them reflect the
#  new AST value.

#  Style only applied to lines
         if attr == Ast.grfSTYLE:
            value = int(value)
            if prim == Ast.grfLINE:
               if value >= 0 and value < len(self.styles):
                  self.__props[prim].update(self.styles[value])

#  Width applies to lines, marks and texts
         elif attr == Ast.grfWIDTH:
            if prim == Ast.grfLINE:

#  Get bounds of plot in user coords
               xl,xr = self.axes.get_xlim()
               yb,yt = self.axes.get_ylim()

#  Transform to device coords
               tr = self.axes.transData
               xl,yb = tr.transform([xl,yb])
               xr,yt = tr.transform([xr,yt])

#  Transform to inches
               tr = self.axes.get_figure().dpi_scale_trans.inverted()
               xl,yb = tr.transform([xl,yb])
               xr,yt = tr.transform([xr,yt])

#  Find the length of the diagonal in inches, and convert to points.
               diag = 72.0*math.sqrt( (xr-xl)**2 + (yt-yb)**2 )

#  Find the number of points corresponding to an AST line width of 1.0,
#  ensure it is at least 1 point, and then scale it by the supplied AST
#  line width.
               lw = max( 0.0005*diag, 1.0 )*value
               self.__props[prim].update({"linewidth":lw})

#  Similar for markers
            elif prim == Ast.grfMARK:
               xl,xr = self.axes.get_xlim()
               yb,yt = self.axes.get_ylim()
               tr = self.axes.transData
               xl,yb = tr.transform([xl,yb])
               xr,yt = tr.transform([xr,yt])
               tr = self.axes.get_figure().dpi_scale_trans.inverted()
               xl,yb = tr.transform([xl,yb])
               xr,yt = tr.transform([xr,yt])
               diag = 72.0*math.sqrt( (xr-xl)**2 + (yt-yb)**2 )
               lw = max( 0.0005*diag, 1.0 )*value
               self.__props[prim].update({"markeredgewidth":lw})

#  Cannot control exact line width of texts, use weight instead.
            elif prim == Ast.grfTEXT:
               if value < 0.0:
                  wgt = 0.0
               elif value < 1.0:
                  wgt = 400.0*value
               elif value < 10.0:
                  wgt = (500.0*value + 3100.0 )/9.0
               else:
                  wgt = 900
               self.__props[prim].update({"weight":wgt})

#  Size applies to marks and texts
         elif attr == Ast.grfSIZE:
            if prim == Ast.grfMARK:
               self.__props[prim].update({"markersize":self.__defmsize*value})
            elif prim == Ast.grfTEXT:
               self.__props[prim].update({"size":self.__deftsize*value})

#  Font only applies to texts
         elif attr == Ast.grfFONT:
            value = int(value)
            if prim == Ast.grfTEXT:
               if value >= 0 and value < len(self.fonts):
                  self.__props[prim].update(self.fonts[value])

#  Colour applied to them all
         elif attr == Ast.grfCOLOUR:
            value = int(value)
            if value >= 0 and value < len(self.colours):
               self.__props[prim].update(self.colours[value])


      return oldval

#------------------------------------------------------------------------
   def BBuf( self ):
      return

#------------------------------------------------------------------------
   def Cap( self, cap, value ):
      if cap == Ast.grfSCALES:
         return 1
      elif cap == Ast.grfMJUST:
         return 1;
      elif cap == Ast.grfESC:
         return 0;
      else:
         return 0

#------------------------------------------------------------------------
   def EBuf( self ):
      return

#------------------------------------------------------------------------
   def Flush( self ):
      return

#------------------------------------------------------------------------
   def Line( self, n, x, y ):
      self.axes.add_line( matplotlib.lines.Line2D( x, y,
                                                **self.__props[Ast.grfLINE] ) )

#------------------------------------------------------------------------
   def Mark( self, n, x, y, type ):
      if type < 0 or type >= len(self.markers):
         marker= '.'
      else:
         marker = self.markers[ type ]

      props = self.__props[Ast.grfMARK].copy()
      props["linestyle"] = 'None'
      props["marker"] = marker
      self.axes.add_line( matplotlib.lines.Line2D( x, y, **props ) )

#------------------------------------------------------------------------
   def Qch( self ):
      xl,xr = self.axes.get_xlim()
      yb,yt = self.axes.get_ylim()
      x = 0.5*(xl+xr)
      y = 0.5*(yt-yb)

      a = self.TxExt( "a", x, y, "CC", 0, 1 )
      chh = max(a[4:]) - min(a[4:])

      a = self.TxExt( "a", x, y, "CC", 1, 0 )
      chv = max(a[:3]) - min(a[:3])
      return (chv,chh)

#------------------------------------------------------------------------
   def Scales( self ):
      xleft,xright = self.axes.get_xlim()
      ybot,ytop = self.axes.get_ylim()
      a = self.axes.transData.transform([(xleft,ybot),(xright,ytop)])
      alpha = ( a[1][0] - a[0][0] ) / ( xright - xleft )
      beta = ( a[1][1] - a[0][1] ) / ( ytop - ybot )
      return (alpha,beta)

#------------------------------------------------------------------------
   def Text( self, text, x, y, just, upx, upy, boxprops={} ):
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
      rot = math.atan2( -upx, upy )*Ast.DR2D

      props = self.__props[Ast.grfTEXT].copy()
      props["verticalalignment"] = va
      props["horizontalalignment"] = ha
      props["rotation"] = rot
      if len(boxprops) > 0:
         props["bbox"] = boxprops

      otext = matplotlib.text.Text( x, y, text, **props )
      self.axes.add_artist( otext )
      return otext

#------------------------------------------------------------------------
   def TxExt( self, text, x, y, just, upx, upy ):
      otext = self.Text(  text, x, y, just, upx, upy, boxprops={"boxstyle":"square"} )
      renderer = self.axes.get_figure().canvas.get_renderer()
      otext.draw(renderer)
      pix_verts = otext.get_bbox_patch().get_verts()
      wcs_verts = otext.get_transform().inverted().transform(pix_verts)
      otext.remove()

      return (wcs_verts[0][0], wcs_verts[1][0], wcs_verts[2][0], wcs_verts[3][0],
              wcs_verts[0][1], wcs_verts[1][1], wcs_verts[2][1], wcs_verts[3][1])


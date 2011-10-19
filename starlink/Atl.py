import starlink.Ast as Ast
import starlink.Grf as Grf
import matplotlib.pyplot as plt
import pyfits

"""
This module provides function that wrap up sequences of PyAST calls
to perform commonly used operations. It requires the pyfits and
matplotlib libraries to be installed.
"""

def readfits( ffile, hdu=0 ):

   r"""Reads a FrameSet from a FITS file.

       readfits( ffile, hdu=0 )

       The header from the specified HDU of the specified FITS file
       is read, and an AST FrameSet describing the WCS information in
       the header is returned. None is returned if WCS information cannot
       be read from the header.

       "ffile" should be a reference to a FITS file, as returned by
       pyfits.open().

       "hdu" should be theinteger index of the required HDU (zero for the
       primary HDU).

   >>> import pyfits
   >>> import starlink.Atl as Atl
   >>>
   >>> ffile = pyfits.open( 'test.fit' )
   >>> frameset = Atl.readfits( ffile )
   >>> if frameset == None:
   >>>    print( "Cannot read WCS from test.fit" )

   """

   return Ast.FitsChan( ffile[hdu].header.ascardlist() ).read()




def plotframeset( axes, gbox, bbox, frameset, options="" ):

   r"""Plot an annotated coordinate grid in a matplotlib axes area.

       plot = plotframeset( axes, gbox, bbox, frameset, options="" )

       "axes" should be a matplot lib "Axes" object. The annotated axes
       normally produced by matplotlib will be removed, and axes will
       instead be drawn by the AST Plot class.

       "gbox" is a list of four values giving the bounds of the new
       annotated axes within the matplotlib Axes object. The supplied
       values should be in the order (xleft,ybottom,xright,ytop) and
       should be given in the matplotlib "axes" coordinate system.

       "bbox" is a list of four values giving the bounds of the new
       annotated axes within the coordinate system represented by the
       base Frame of the supplied FrameSet. The supplied values should
       be in the order (xleft,ybottom,xright,ytop).

       "frameset" should be an AST FrameSet such as returned by the
       Atl.readfits function. Its base Frame should be 2-dimensional.

       "options" is an optional string holding a comma-separated list
       of Plot attribute settings. These control the appearance of the
       annotated axes.

       The function returns a reference to the Plot that was used to draw
       the axes.

   >>> import pyfits
   >>> import starlink.Atl as Atl
   >>> import matplotlib.pyplot
   >>>
   >>> ffile = pyfits.open( 'test.fit' )
   >>> frameset = starlink.Atl.readfits( ffile )
   >>> if frameset != None:
   >>>    naxis1 = ffile[0].header['NAXIS1']
   >>>    naxis2 = ffile[0].header['NAXIS2']
   >>>    Atl.plotframeset( matplotlib.pyplot.figure().add_subplot(111),
   >>>                      [ 0.1, 0.1, 0.9, 0.9 ],
   >>>                      [ 0.5, 0.5, naxis1+0.5, naxis2+0.5 ], frameset )
   >>>    matplotlib.pyplot.show()
   """

   axes.xaxis.set_visible( False )
   axes.yaxis.set_visible( False )
   plot = Ast.Plot( frameset, gbox, bbox, Grf.grf_matplotlib( axes ), options )
   plot.grid()
   return plot



def plotfitswcs( axes, gbox, ffile, hdu=0, options="" ):

   r"""Read WCS from a FITS image and plot an annotated coordinate grid
       in a matplotlib axes area.

       The grid covers the entire image.

       plot = plotfitswcs( axes, gbox, ffile, hdu=0, options="" )

       "axes" should be a matplot lib "Axes" object. The annotated axes
       normally produced by matplotlib will be removed, and axes will
       instead be drawn by the AST Plot class.

       "gbox" is a list of four values giving the bounds of the new
       annotated axes within the matplotlib Axes object. The supplied
       values should be in the order (xleft,ybottom,xright,ytop) and
       should be given in the matplotlib "axes" coordinate system.

       "ffile" should be a reference to a FITS file, as returned by
       pyfits.open().

       "hdu" should be the integer index of the required HDU (zero for the
       primary HDU).

       "options" is an optional string holding a comma-separated list
       of Plot attribute settings. These control the appearance of the
       annotated axes.

       The function returns a reference to the Plot that was used to draw
       the axes.

   >>> import pyfits
   >>> import starlink.Atl as Atl
   >>> import matplotlib.pyplot
   >>>
   >>> ffile = pyfits.open( 'test.fit' )
   >>> Atl.plotfitswcs( matplotlib.pyplot.figure().add_subplot(111),
   >>>                  [ 0.1, 0.1, 0.9, 0.9 ], ffile )
   >>> matplotlib.pyplot.show()
   """

   frameset = readfits( ffile, hdu )
   naxis1 = ffile[ hdu ].header[ 'NAXIS1' ]
   naxis2 = ffile[ hdu ].header[ 'NAXIS2' ]
   return plotframeset( axes, gbox, [ 0.5, 0.5, naxis1+0.5, naxis2+0.5 ],
                        frameset, options )



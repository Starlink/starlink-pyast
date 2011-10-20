import starlink.Ast as Ast
import starlink.Grf as Grf
import matplotlib.pyplot as plt
import pyfits

"""
This module provides function that wrap up sequences of PyAST calls
to perform commonly used operations. It requires the pyfits and
matplotlib libraries to be installed.
"""

# ======================================================================
def readfitswcs( ffile, hdu=0 ):

   r"""Reads an AST FrameSet from a FITS file.

       The header from the specified HDU of the specified FITS file
       is read, and an AST FrameSet describing the WCS information in
       the header is returned. None is returned if WCS information cannot
       be read from the header.

       readfitswcs( ffile, hdu=0 )

       "ffile" should be a reference to a FITS file, as returned by
       pyfits.open().

       "hdu" should be theinteger index of the required HDU (zero for the
       primary HDU).

   >>> import pyfits
   >>> import starlink.Atl as Atl
   >>>
   >>> ffile = pyfits.open( 'test.fit' )
   >>> frameset = Atl.readfitswcs( ffile )
   >>> if frameset == None:
   >>>    print( "Cannot read WCS from test.fit" )

   """

   return Ast.FitsChan( Atl.PyFITSAdaptor(ffile[hdu])).read()




# ======================================================================
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
       Atl.readfitswcs function. Its base Frame should be 2-dimensional.

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
   >>> frameset = starlink.Atl.readfitswcs( ffile )
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



# ======================================================================
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

   frameset = readfitswcs( ffile, hdu )
   naxis1 = ffile[ hdu ].header[ 'NAXIS1' ]
   naxis2 = ffile[ hdu ].header[ 'NAXIS2' ]
   return plotframeset( axes, gbox, [ 0.5, 0.5, naxis1+0.5, naxis2+0.5 ],
                        frameset, options )


# ======================================================================
class PyFITSAdapter:
   """
   Adapter to allow use of pyfits HDU objects with starlink.Ast.FitsChan

   This class allows a pyfits HDU to be used as the source or sink object
   with a FitsChan.

   When used as a FitsChan source, the PyFITSAdapter will allow the
   FitsChan to read each of the cards in the associated PyFITS header,
   thus allowing the cards to be copied into the FitsChan. This happens
   when the newly created, empty FitsChan is used for the first time
   (i.e. when any of its methods is invoked), and subsequently whenever
   the FitsChan.readfits() method is invoked.

   When used as a FitsChan sink, the PyFITSAdapter will first empty the
   associated PyFITS header, and then allow the FitsChan to copy its own
   header card into the PyFITS header. This happens when the FitsChan is
   deleted or when the FitsChan.writefits() method is invoked.
   """

   def __init__(self,hdu):
      """
      Construct a PyFITSAdapter for a specified PyFITS HDU.

      Parameters:
         hdu: An element of the hdulist associated with a FITS file
              opened using pyfits.open(). If the entire hdulist is supplied,
              rather than an element of the hdulist, then the primary HDU
              (element zero) will be used.

      Examples:
              - To read WCS from the 'DATA' extension in FITS file 'test.fit':

              >>> import pyfits
              >>> import starlink.Ast as Ast
              >>> import starlink.Atl as Atl

              >>> hdulist = pyfits.open('test.fit')
              >>> fc = Ast.FitsChan( Atl.PyFITSAdapter( hdulist['DATA'] ) )
              >>> framset = fc.read()

              - To write a FrameSet to the primary HDU in FITS file
              'old.fit', using standard FITS-WCS keywords:

              >>> import pyfits
              >>> import starlink.Ast as Ast
              >>> import starlink.Atl as Atl

              >>> hdulist = pyfits.open('old.fit')
              >>> fc = Ast.FitsChan( None, Atl.PyFITSAdapter( hdulist ) )
              >>> if fc.write( framset ) == 0:
              >>>    print("Failed to convert FrameSet to FITS header")
      """

#  If the supplied object behaves like a sequence, use element zero (the
#  primary HDU). Otherwise use the supplied object.
      try:
         self.hdu = hdu[ 0 ]
      except TypeError:
         self.hdu = hdu

#  Initialise the index of the next card to read or write.
      self.index = 0


# -----------------------------------------------------------------
   def astsource(self):

      """
      This method is called by the FitsChan to obtain a single 80-character
      FITS header card. It iterates over all the cards in the PyFITS
      header, returning each one in turn. It then returns "None" to
      indicate that there are no more header cards to read.
      """

      if self.index < len( self.hdu.header.ascard ):
         result = self.hdu.header.ascard[ self.index ].ascardimage()
         self.index += 1

      else:
         result = None
         self.index = 0

      return result

# -----------------------------------------------------------------
   def astsink(self,card):

      """
      This method is called by the FitsChan to store a single 80-character
      FITS header card. It replaces the existing PyFITS header with a new
      empty header on the first call, and then appends the supplied card
      to the header.
      """

      if self.index == 0:
         self.hdu.header = pyfits.core.Header()

      self.hdu.header.ascard.append( pyfits.core.Card.fromstring(card),
                                     bottom=True )
      self.index += 1


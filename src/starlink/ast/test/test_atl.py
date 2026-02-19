import astropy.io.fits as pyfits
import matplotlib.pyplot

from starlink import Ast, Atl

#  Use pyfits to open a test files file
with pyfits.open("src/starlink/ast/test/cobe.fit") as ffile:
    #  Use matplotlib to plot an annotated grid of the WCS coords
    Atl.plotfitswcs(matplotlib.pyplot.figure(figsize=(8, 8)).add_subplot(111), [0.1, 0.1, 0.9, 0.9], ffile)
    matplotlib.pyplot.show()

    #  Create a FitsChan telling it to use the pyfits primary hdu as the
    #  external data source and sink. Note, we take the default value of
    #  "True" for the "clear" property when creating the PyFITSADapater,
    #  which means the PyFITS header will be cleared immediately before
    #  the FitsChan.writefits() method writes to it.
    adapter = Atl.PyFITSAdapter(ffile)
    fc = Ast.FitsChan(adapter, adapter)

    #  Read the FrameSet from the FitsChan. This will read all headers from
    #  the pyfits hdu into the FitsChan, create a FrameSet from the WCS
    #  headers, and remove all WCS-related headers from the FitsChan (but not
    #  the pyfits primary hdu as yet).
    fs = fc.read()

    #  Tell the FitsChan to write out the remaining headers to its external
    #  data sink.
    fc.writefits()

    #  Display the headers now in the pyfits primary hdu.
    print()
    print("The non-WCS cards in cobe.fit: ")
    for v in ffile[0].header.cards:
        print(v)

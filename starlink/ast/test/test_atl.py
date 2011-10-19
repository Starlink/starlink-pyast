import pyfits
import starlink.Atl as Atl
import matplotlib.pyplot

ffile = pyfits.open( 'starlink/ast/test/cobe.fit' )
Atl.plotfitswcs( matplotlib.pyplot.figure(figsize=(8,8)).add_subplot(111),
                 [ 0.1, 0.1, 0.9, 0.9 ], ffile )
matplotlib.pyplot.show()


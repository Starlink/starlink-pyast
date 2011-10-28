from __future__ import print_function

import pyfits
import starlink.Atl as Atl
import starlink.Ast as Ast
import matplotlib.pyplot

ffile = pyfits.open( 'starlink/ast/test/cobe.fit' )
Atl.plotfitswcs( matplotlib.pyplot.figure(figsize=(8,8)).add_subplot(111),
                 [ 0.1, 0.1, 0.9, 0.9 ], ffile )
matplotlib.pyplot.show()

fc = Ast.FitsChan( Atl.PyFITSAdapter(ffile), Atl.PyFITSAdapter(ffile) )

fs = fc.read()
fc.writefits()

print()
print("The non-WCS cards in cobe.fit: ")
for v in ffile[0].header.ascard:
   print(v)


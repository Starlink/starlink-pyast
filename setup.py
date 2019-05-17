
from __future__ import print_function

from distutils import ccompiler
from distutils.core import setup, Extension
import sys
import os
import numpy
import tarfile
import ctypes
from tools import make_exceptions, make_attributes

include_dirs = []

include_dirs.append(numpy.get_include())
include_dirs.append(os.path.join('.', 'starlink', 'include'))
include_dirs.append(os.path.join('.', 'ast'))

#  Create the support files needed for the build. These find the AST
#  source code using the environment variable AST_SOURCE, so set AST_SOURCE
#  to point to the AST source code directory distributed with PyAST.
os.environ["AST_SOURCE"] = os.path.join(os.getcwd(), 'ast')
make_exceptions.make_exceptions(os.path.join('starlink', 'ast'))
make_attributes.make_attributes(os.path.join('starlink', 'ast'))

#  Extract the AST documentation
if not os.path.exists('sun211.htx'):
    tar = tarfile.open('ast/sun211.htx_tar')
    tar.extractall()
    tar.close()

#  List the cminpack source files required by AST:
cminpack_c = ('enorm.c', 'lmder.c', 'qrfac.c', 'dpmpar.c', 'lmder1.c',
              'lmpar.c', 'qrsolv.c')

#  List the C source files for implemented AST classes:
ast_c = ('axis.c', 'box.c', 'channel.c', 'circle.c', 'cmpframe.c',
         'cmpmap.c', 'cmpregion.c', 'dsbspecframe.c', 'dssmap.c',
         'ellipse.c', 'error.c', 'fitschan.c', 'fluxframe.c', 'frame.c',
         'frameset.c', 'globals.c', 'grf3d.c', 'grf_2.0.c', 'grf_3.2.c',
         'grf_5.6.c', 'grismmap.c', 'interval.c', 'keymap.c',
         'loader.c', 'lutmap.c', 'mapping.c', 'mathmap.c', 'matrixmap.c',
         'memory.c', 'moc.c', 'mocchan.c', 'normmap.c', 'nullregion.c',
         'object.c', 'palwrap.c', 'pcdmap.c', 'permmap.c', 'plot.c',
         'pointlist.c', 'pointset.c', 'polygon.c', 'polymap.c',
         'prism.c', 'proj.c', 'ratemap.c', 'region.c', 'shiftmap.c',
         'skyaxis.c', 'skyframe.c', 'specfluxframe.c', 'specframe.c',
         'sphmap.c', 'stcschan.c', 'timeframe.c', 'timemap.c', 'tpn.c',
         'tranmap.c', 'unit.c', 'unitmap.c', 'wcsmap.c', 'wcstrig.c',
         'winmap.c', 'xml.c', 'xphmap.c', 'zoommap.c', 'specmap.c',
         'slamap.c', 'chebymap.c', 'unitnormmap.c', 'pyast_extra.c' )

#  List the erfa source files required by AST.
erfa_c = ('a2af.c', 'a2tf.c', 'ab.c', 'af2a.c', 'anp.c', 'anpm.c',
          'apcg.c', 'apcg13.c', 'apci.c', 'apci13.c', 'apco.c',
          'apco13.c', 'apcs.c', 'apcs13.c', 'aper.c', 'aper13.c',
          'apio.c', 'apio13.c', 'atci13.c', 'atciq.c', 'atciqn.c',
          'atciqz.c', 'atco13.c', 'atic13.c', 'aticq.c', 'aticqn.c',
          'atio13.c', 'atioq.c', 'atoc13.c', 'atoi13.c', 'atoiq.c',
          'bi00.c', 'bp00.c', 'bp06.c', 'bpn2xy.c', 'c2i00a.c',
          'c2i00b.c', 'c2i06a.c', 'c2ibpn.c', 'c2ixy.c', 'c2ixys.c',
          'c2s.c', 'c2t00a.c', 'c2t00b.c', 'c2t06a.c', 'c2tcio.c',
          'c2teqx.c', 'c2tpe.c', 'c2txy.c', 'cal2jd.c', 'cp.c', 'cpv.c',
          'cr.c', 'd2dtf.c', 'd2tf.c', 'dat.c', 'dtdb.c', 'dtf2d.c',
          'eceq06.c', 'ecm06.c', 'ee00.c', 'ee00a.c', 'ee00b.c',
          'ee06a.c', 'eect00.c', 'eform.c', 'eo06a.c', 'eors.c', 'epb.c',
          'epb2jd.c', 'epj.c', 'epj2jd.c', 'epv00.c', 'eqec06.c',
          'eqeq94.c', 'era00.c', 'fad03.c', 'fae03.c', 'faf03.c',
          'faju03.c', 'fal03.c', 'falp03.c', 'fama03.c', 'fame03.c',
          'fane03.c', 'faom03.c', 'fapa03.c', 'fasa03.c', 'faur03.c',
          'fave03.c', 'fk52h.c', 'fk5hip.c', 'fk5hz.c', 'fw2m.c',
          'fw2xy.c', 'g2icrs.c', 'gc2gd.c', 'gc2gde.c', 'gd2gc.c',
          'gd2gce.c', 'gmst00.c', 'gmst06.c', 'gmst82.c', 'gst00a.c',
          'gst00b.c', 'gst06.c', 'gst06a.c', 'gst94.c', 'h2fk5.c',
          'hfk5z.c', 'icrs2g.c', 'ir.c', 'jd2cal.c', 'jdcalf.c', 'ld.c',
          'ldn.c', 'ldsun.c', 'lteceq.c', 'ltecm.c', 'lteqec.c', 'ltp.c',
          'ltpb.c', 'ltpecl.c', 'ltpequ.c', 'num00a.c', 'num00b.c',
          'num06a.c', 'numat.c', 'nut00a.c', 'nut00b.c', 'nut06a.c',
          'nut80.c', 'nutm80.c', 'obl06.c', 'obl80.c', 'p06e.c', 'p2pv.c',
          'p2s.c', 'pap.c', 'pas.c', 'pb06.c', 'pdp.c', 'pfw06.c',
          'plan94.c', 'pm.c', 'pmat00.c', 'pmat06.c', 'pmat76.c', 'pmp.c',
          'pmpx.c', 'pmsafe.c', 'pn.c', 'pn00.c', 'pn00a.c', 'pn00b.c',
          'pn06.c', 'pn06a.c', 'pnm00a.c', 'pnm00b.c', 'pnm06a.c',
          'pnm80.c', 'pom00.c', 'ppp.c', 'ppsp.c', 'pr00.c', 'prec76.c',
          'pv2p.c', 'pv2s.c', 'pvdpv.c', 'pvm.c', 'pvmpv.c', 'pvppv.c',
          'pvstar.c', 'pvtob.c', 'pvu.c', 'pvup.c', 'pvxpv.c', 'pxp.c',
          'refco.c', 'rm2v.c', 'rv2m.c', 'rx.c', 'rxp.c', 'rxpv.c',
          'rxr.c', 'ry.c', 'rz.c', 's00.c', 's00a.c', 's00b.c', 's06.c',
          's06a.c', 's2c.c', 's2p.c', 's2pv.c', 's2xpv.c', 'sepp.c',
          'seps.c', 'sp00.c', 'starpm.c', 'starpv.c', 'sxp.c', 'sxpv.c',
          't_erfa_c.c', 'taitt.c', 'taiut1.c', 'taiutc.c', 'tcbtdb.c',
          'tcgtt.c', 'tdbtcb.c', 'tdbtt.c', 'tf2a.c', 'tf2d.c', 'tr.c',
          'trxp.c', 'trxpv.c', 'tttai.c', 'tttcg.c', 'tttdb.c', 'ttut1.c',
          'ut1tai.c', 'ut1tt.c', 'ut1utc.c', 'utctai.c', 'utcut1.c',
          'xy06.c', 'xys00a.c', 'xys00b.c', 'xys06a.c', 'zp.c', 'zpv.c',
          'zr.c')

#  List the C source files for unimplemeneted AST classes:
ast_c_extra = ('fitstable.c', 'intramap.c', 'plot3d.c', 'selectormap.c',
               'stc.c', 'stccatalogentrylocation.c', 'stcobsdatalocation.c',
               'stcresourceprofile.c', 'stcsearchlocation.c', 'switchmap.c',
               'table.c', 'xmlchan.c')

#  Initialise the list of sources files needed to build the starlink.Ast
#  module.
sources = [os.path.join('starlink', 'ast', 'Ast.c')]

#  Append all the .c and .h files needed to build the AST library locally.
for cfile in ast_c:
    sources.append(os.path.join('ast', cfile))
for cfile in cminpack_c:
    sources.append(os.path.join(os.path.join('ast', 'cminpack'), cfile))
for cfile in erfa_c:
    sources.append(os.path.join(os.path.join('ast', 'erfa'), cfile))
for cfile in ast_c_extra:
    sources.append(os.path.join('ast', cfile))

# Test the compiler
define_macros = []
compiler = ccompiler.new_compiler()
if compiler.has_function('strtok_r'):
    define_macros.append(('HAVE_STRTOK_R', '1'))

if compiler.has_function('strerror_r'):
    define_macros.append(('HAVE_STRERROR_R', '1'))

if compiler.has_function('isfinite'):
    define_macros.append(('HAVE_DECL_ISFINITE', '1'))

#  We need to tell AST what type a 64-bit int will have
#  Not really sure how to determine whether we have int64_t

define_macros.append(('SIZEOF_LONG', ctypes.sizeof(ctypes.c_long)))
define_macros.append(('SIZEOF_LONG_LONG', ctypes.sizeof(ctypes.c_longlong)))

# Assume we have isnan() available and assume we have a working sscanf
# configure would test for these but we no longer run configure
define_macros.append(('HAVE_DECL_ISNAN', '1'))

#  Create the description of the starlink.Ast module.
Ast = Extension('starlink.Ast',
                include_dirs=include_dirs,
                define_macros=define_macros,
                sources=sources)

# OSX needs to hide all the normal AST symbols to prevent
# name clashes when loaded alongside libast itself (eg from pyndf)
symbol_list = "public_symbols.txt"
if sys.platform.startswith("darwin"):
    symfile = open(symbol_list, "w")
    if sys.version_info[0] > 2:
        symname = "_PyInit_Ast"
    else:
        symname = "_initAst"
    print(symname, file=symfile)
    symfile.close()
    Ast.extra_link_args = ["-exported_symbols_list", symbol_list]


setup(name='starlink-pyast',
      version='3.12.1',
      description='A Python wrapper for the Starlink AST library',
      url='http://www.starlink.ac.uk/ast',
      author='David Berry',
      author_email='d.berry@eaobservatory.org',
      packages=['starlink'],
      package_data={'starlink': [os.path.join('include', 'star', 'pyast.h')]},
      ext_modules=[Ast],
      py_modules=['starlink.Grf', 'starlink.Atl'],
      classifiers=[
           'Intended Audience :: Developers',
           'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
           'Programming Language :: Python',
           'Programming Language :: C',
          'Topic :: Scientific/Engineering :: Astronomy'
      ])

if os.path.exists(symbol_list):
    os.unlink(symbol_list)


from __future__ import print_function

from distutils import ccompiler
from distutils.core import setup, Extension
import sys, os, subprocess, numpy, tarfile
from tools import make_exceptions, make_attributes

include_dirs = []

include_dirs.append(numpy.get_include())
include_dirs.append(os.path.join('.','starlink','include'))
include_dirs.append(os.path.join('.','ast'))

#  Create the support files needed for the build. These find the AST
#  source code using the environment variable AST_SOURCE, so set AST_SOURCE
#  to point to the AST source code directory distributed with PyAST.
os.environ[ "AST_SOURCE" ] = os.path.join( os.getcwd(), 'ast' )
make_exceptions.make_exceptions( os.path.join('starlink','ast') )
make_attributes.make_attributes( os.path.join('starlink','ast') )

#  Extract the AST documentation
if not os.path.exists( 'sun211.htx'):
   tar = tarfile.open('ast/sun211.htx_tar')
   tar.extractall()
   tar.close()

#  List the C source files for implemeneted AST classes:
ast_c = ( 'axis.c', 'box.c', 'channel.c', 'circle.c', 'cmpframe.c',
          'cmpmap.c', 'cmpregion.c', 'dsbspecframe.c', 'dssmap.c',
	  'ellipse.c', 'error.c', 'fitschan.c', 'fluxframe.c', 'frame.c',
	  'frameset.c', 'globals.c', 'grf3d.c', 'grf_2.0.c', 'grf_3.2.c',
	  'grf_5.6.c', 'grismmap.c', 'interval.c', 'keymap.c',
	  'levmar.c', 'loader.c', 'lutmap.c', 'mapping.c', 'mathmap.c',
	  'matrixmap.c', 'memory.c', 'normmap.c', 'nullregion.c',
	  'object.c', 'pal.c', 'pcdmap.c', 'permmap.c', 'plot.c',
	  'pointlist.c', 'pointset.c', 'polygon.c', 'polymap.c',
	  'prism.c', 'proj.c', 'ratemap.c', 'region.c', 'shiftmap.c',
	  'skyaxis.c', 'skyframe.c', 'specfluxframe.c', 'specframe.c',
	  'sphmap.c', 'stcschan.c', 'timeframe.c', 'timemap.c', 'tpn.c',
	  'tranmap.c', 'unit.c', 'unitmap.c', 'wcsmap.c', 'wcstrig.c',
	  'winmap.c', 'xml.c', 'zoommap.c')

#  List the C source files for unimplemeneted AST classes:
ast_c_extra = ( 'fitstable.c', 'intramap.c', 'plot3d.c', 'selectormap.c',
                'slamap.c', 'specmap.c', 'stccatalogentrylocation.c',
		'stc.c', 'stcobsdatalocation.c', 'stcresourceprofile.c',
		'stcsearchlocation.c', 'switchmap.c', 'table.c',
		'xmlchan.c')

#  Initialise the list of sources files needed to build the starlink.Ast
#  module.
sources = [os.path.join('starlink', 'ast', 'Ast.c')]

#  Append all the .c and .h files needed to build the AST library locally.
for cfile in ast_c:
   sources.append( os.path.join( 'ast', cfile ) )
for cfile in ast_c_extra:
   sources.append( os.path.join( 'ast', cfile ) )

# Test the compiler
define_macros = []
compiler=ccompiler.new_compiler()
if compiler.has_function('strtok_r'):
   define_macros.append(('HAVE_STRTOK_R','1'))

if compiler.has_function('strerror_r'):
   define_macros.append(('HAVE_STRERROR_R','1'))

if compiler.has_function('isfinite'):
   define_macros.append(('HAVE_DECL_ISFINITE','1'))

#  We need to tell AST what type a 64-bit int will have
#  Not really sure how to determine whether we have int64_t
import ctypes

define_macros.append(('SIZEOF_LONG', ctypes.sizeof(ctypes.c_long)))
define_macros.append(('SIZEOF_LONG_LONG', ctypes.sizeof(ctypes.c_longlong)))

# Assume we have isnan() available and assume we have a working sscanf
# configure would test for these but we no longer run configure
define_macros.append(('HAVE_DECL_ISNAN','1'))

#  Create the description of the starlink.Ast module.
Ast = Extension('starlink.Ast',
                include_dirs  = include_dirs,
                define_macros = define_macros,
                sources       = sources )

# OSX needs to hide all the normal AST symbols to prevent
# name clashes when loaded alongside libast itself (eg from pyndf)
symbol_list = "public_symbols.txt"
if sys.platform.startswith("darwin"):
   symfile = open( symbol_list, "w" )
   if sys.version_info[0] > 2:
      symname = "_PyInit_Ast"
   else:
      symname = "_initAst"
   print(symname,file=symfile)
   symfile.close()
   Ast.extra_link_args = [ "-exported_symbols_list", symbol_list]


setup (name = 'starlink-pyast',
       version = '2.1',
       description = 'A Python wrapper for the Starlink AST library',
       url = 'http://starlink.jach.hawaii.edu/starlink/AST',
       author = 'David Berry',
       author_email = 'd.berry@jach.hawaii.edu',
       packages =['starlink'],
       package_data = { 'starlink': [os.path.join('include','star','pyast.h')] },
       ext_modules=[Ast],
       py_modules=['starlink.Grf','starlink.Atl'],
       classifiers=[
          'Intended Audience :: Developers',
          'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
          'Programming Language :: Python',
          'Programming Language :: C',
          'Topic :: Scientific/Engineering :: Astronomy'
       ])

if os.path.exists(symbol_list):
   os.unlink(symbol_list)

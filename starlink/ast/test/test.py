import unittest
import starlink.Ast
import copy
import numpy
import math
import copy
import filecmp
import string
import sys
import os.path

#  A class that defines Channel source and sink functions that store text
#  in an internal list.
class TextStream():

   def __init__( self ):
      self.reset_sink()
      self.reset_source()

   def reset_sink( self ):
      self.text = []

   def reset_source( self ):
      self.index = 0;

   def sink( self, text ):
      self.text.append(text)

   def source( self ):
      if self.index < len( self.text ):
         result = self.text[ self.index ]
         self.index += 1
      else:
         result = None
      return result

   def get( self ):
      return copy.deepcopy(self.text)


#  A dummy object used to test channel error reporting.
class DummyStream():
   pass

#  A class that provides primitive graphics operatons for a Plot. All it
#  does is record the values supplied to it by the Plot class.
class DummyGrf():
   def __init__( self ):
      self.Reset()

   def Reset( self ):
      self.linex = []
      self.liney = []
      self.nline = 0
      self.markx = []
      self.marky = []
      self.markt = []
      self.nmark = 0
      self.attr = []
      self.value = []
      self.prim = []
      self.textt = []
      self.textx = []
      self.texty = []
      self.textj = []
      self.ntext = 0

   def Attr( self, attr, value, prim ):
      self.attr += [attr]
      self.value += [value]
      self.prim += [prim]
      return 1

   def BBuf( self ):
      return 0

   def Cap( self, cap, value ):
      if cap == starlink.Ast.grfSCALES:
         return 1
      elif cap == starlink.Ast.grfMJUST:
         return 0;
      elif cap == starlink.Ast.grfESC:
         return 0;
      else:
         return 0

   def EBuf( self ):
      return 0

   def Flush( self ):
      return 0

   def Line( self, n, x, y ):
      self.linex.extend(x)
      self.liney.extend(y)
      self.nline += n

   def Mark( self, n, x, y, type ):
      self.markx.extend(x)
      self.marky.extend(y)
      self.markt += (type,)
      self.nmark += n

   def Qch( self ):
      return (1.0,1.0)

   def Scales( self ):
      return (1.0,1.0)

   def Text( self, text, x, y, just, upx, upy ):
      self.textt += [text]
      self.textx += [x]
      self.texty += [y]
      self.textj += [just]
      self.ntext += 1

   def TxExt( self, text, x, y, just, upx, upy ):
      return (x-0.1,x+0.1,x+0.1,x-0.1,y-0.1,y-0.1,y+0.1,y+0.1)


#  Tester
class TestAst(unittest.TestCase):

   def test_Static(self):
      self.assertEqual( starlink.Ast.tunec("HRDel"), "%-%^50+%s70+h%+" )
      starlink.Ast.tunec( "HRDel", "fred" )
      self.assertEqual( starlink.Ast.tunec("HRDel"), "fred" )
      self.assertEqual( starlink.Ast.escapes(1), 0 )
      self.assertEqual( starlink.Ast.escapes(0), 1 )
      self.assertEqual( starlink.Ast.escapes(-1), 0 )
      self.assertEqual( starlink.Ast.tune("ObjectCaching", 1 ), 0 )
      self.assertEqual( starlink.Ast.tune("ObjectCaching", starlink.Ast.TUNULL ), 1 )
      self.assertEqual( starlink.Ast.tune("ObjectCaching", 0 ), 1 )
      self.assertGreaterEqual( starlink.Ast.version(), 5007002 )
      self.assertTrue( os.path.isfile( os.path.join( starlink.Ast.get_include(),'star','pyast.h' ) ) )

   def test_Object(self):
      with self.assertRaises(TypeError):
         obj = starlink.Ast.Object()

   def test_ZoomMap(self):
      zoommap = starlink.Ast.ZoomMap( 1, 1.2 )
      self.assertEqual( zoommap.Class, "ZoomMap" )
      self.assertIsInstance( zoommap, starlink.Ast.ZoomMap )
      self.assertEqual( zoommap.Nobject, 1 )
      with self.assertRaises(AttributeError):
         zoommap.fred = 1.0
      with self.assertRaises(TypeError):
         zoommap.ID = 1.0
      zoommap.ID = "Hello there"
      self.assertEqual( zoommap.ID, "Hello there" )
      self.assertEqual( zoommap.get("ID"), "Hello there" )
      self.assertTrue( zoommap.test("ID") )
      zoommap.clear("ID")
      self.assertFalse( zoommap.test("ID") )
      self.assertEqual( zoommap.ID, "" )
      zoommap.set("ID=fred")
      self.assertEqual( zoommap.ID, "fred" )
      self.assertTrue( zoommap.UseDefs )
      self.assertEqual( zoommap.Nin, 1 )
      self.assertEqual( zoommap.Nout, 1 )
      self.assertEqual( zoommap.Zoom, 1.2 )
      zoommap.Zoom = -1.3
      self.assertEqual( zoommap.Zoom, -1.3 )
      self.assertEqual( zoommap.get("Zoom"), "-1.3" )
      zm = copy.deepcopy(zoommap)
      self.assertEqual( zoommap.Nobject, 2 )
      self.assertIsInstance( zm, starlink.Ast.ZoomMap )
      self.assertEqual( zm.Zoom, -1.3 )
      self.assertEqual( zoommap.Zoom, -1.3 )
      zm.Zoom = 3.0
      self.assertEqual( zm.Zoom, 3.0 )
      self.assertEqual( zoommap.Zoom, -1.3 )
      zm2 = zoommap.copy()
      self.assertEqual( zoommap.Nobject, 3 )
      self.assertIsInstance( zm2, starlink.Ast.ZoomMap )
      self.assertEqual( zm2.Zoom, -1.3 )
      self.assertEqual( zoommap.Zoom, -1.3 )
      zm2.Zoom = 3.0
      self.assertEqual( zm2.Zoom, 3.0 )
      self.assertEqual( zoommap.Zoom, -1.3 )
      zm2 = None
      self.assertEqual( zoommap.Nobject, 2 )
      self.assertTrue( zoommap.same(zoommap) )
      self.assertFalse( zoommap.same(zm) )
      del zm
      self.assertEqual( zoommap.Nobject, 1 )
      self.assertTrue( zoommap.hasattribute("ID") )
      self.assertFalse( zoommap.hasattribute("FID") )
      self.assertTrue( zoommap.isaobject() )
      self.assertTrue( zoommap.isamapping() )
      self.assertTrue( zoommap.isazoommap() )

      self.assertEqual( zoommap.RefCount, 1 )
      with self.assertRaises(AttributeError):
         zoommap.Nin = 3
      with self.assertRaises(starlink.Ast.AstError):
         zoommap = starlink.Ast.ZoomMap( 1, 0 )
      with self.assertRaises(starlink.Ast.ZOOMI):
         zoommap = starlink.Ast.ZoomMap( 1, 0 )
      zoommap = starlink.Ast.ZoomMap( 1, 12.25, "Zoom=2.1" )
      self.assertEqual( zoommap.Zoom, 2.1 )
      with self.assertRaises(starlink.Ast.NOWRT):
         zoommap = starlink.Ast.ZoomMap( 1, 12.25, "Nin=3" )

      self.assertTrue( starlink.Ast.ZoomMap( 1, 1.0 ).simplify().isaunitmap() )


      zoommap.lock(1)

   def test_FrameSimple(self):
      frame = starlink.Ast.Frame( 2, "label(1)=a b,label(2)=c d" )
      self.assertIsInstance( frame, starlink.Ast.Frame )
      self.assertEqual( frame.Nin, 2 )
      self.assertEqual( frame.Nout, 2 )
      testtitle = "Test Frame"
      frame.Title = testtitle
      self.assertEqual( frame.Title, testtitle)
      self.assertEqual( frame.get("Title"), testtitle)
      self.assertEqual( frame.Label_1, "a b" )
      self.assertEqual( frame.Label_2, "c d" )
      frame.Label_2 = "A new label"
      self.assertEqual( frame.Label_2, "A new label" )
      frame.Label_2 = None
      self.assertEqual( frame.Label_2, "Axis 2" )

      # Some methods
   def test_FrameAngle(self):
      frame = starlink.Ast.Frame(2)
      angle = frame.angle( [4,3], [0,0], [4,0] )
      self.assertEqual( angle, math.atan2(3,4) )

   def test_FrameAxis(self):
      frame = starlink.Ast.Frame(2)
      angle = frame.axangle( [0,0], [4,3], 1 )
      self.assertEqual( angle, -math.atan2(3,4) )
      distance = frame.axdistance( 1, 0, 4 )
      self.assertEqual( distance, 4 )
      axoffset = frame.axoffset( 1, 1, 4 )
      self.assertEqual( axoffset, 5 )

   def test_FrameDistance(self):
      frame = starlink.Ast.Frame(2)
      distance = frame.distance( [0,0],[4,3] )
      self.assertEqual( distance, 5 )

   def test_FrameFormat(self):
      frame = starlink.Ast.Frame(2)
      format = frame.format( 1, 55.270)
      self.assertEqual( format, "55.27" )

   def test_FrameIntersect(self):
      frame = starlink.Ast.Frame(2)
      cross = frame.intersect( [-1,1],[1,1],[0,0],[2,2] )
      self.assertEqual( cross[0], 1.0 )
      self.assertEqual( cross[1], 1.0 )

   def test_FrameMatchAxes(self):
      frame = starlink.Ast.Frame(2)
      frame2 = starlink.Ast.Frame( 3 )
      axes = frame.matchaxes( frame2 )
      self.assertEqual( axes[0], 1 )
      self.assertEqual( axes[1], 2 )
      self.assertEqual( axes[2], 0 )

   def test_FrameNorm(self):
      frame = starlink.Ast.Frame(2)
      coords = [3,2]
      ncoords = frame.norm( coords )
      self.assertEqual( ncoords[0], coords[0] )

   def test_FrameOffset(self):
      frame = starlink.Ast.Frame(2)
      point = frame.offset( [0,0], [4,3], 10 )
      self.assertEqual( point[0], 8 )
      self.assertEqual( point[1], 6 )
      direction, point = frame.offset2( [0,0], math.atan2(4,3), 10 )
      self.assertAlmostEqual( point[0], 8 )
      self.assertAlmostEqual( point[1], 6 )

   def test_FramePerm(self):
      frame = starlink.Ast.Frame(2)
      frame.permaxes( [2,1] )
      nframe,mapping = frame.pickaxes( [ 2 ] )
      self.assertEqual(nframe.Nin, 1 )
      self.assertIsInstance( frame, starlink.Ast.Frame )
      self.assertIsInstance( mapping, starlink.Ast.PermMap )
      self.assertEqual( mapping.Nin, 2 )
      self.assertEqual( mapping.Nout, 1 )

   def test_FrameConvert(self):
      frame = starlink.Ast.Frame(2)
      nframe = starlink.Ast.Frame( 2 )
      fset = frame.convert( nframe )
      self.assertIsInstance( fset, starlink.Ast.FrameSet )
      self.assertEqual( fset.Nframe, 2 )
      fset2 = fset.findframe( nframe )
      self.assertIsInstance( fset, starlink.Ast.FrameSet )

   def test_FrameResolve(self):
      frame = starlink.Ast.Frame(2)
      point4,d1,d2 = frame.resolve( [0,0], [3,3], [0,4] )
      self.assertAlmostEqual( d2, 0.0 )
      self.assertAlmostEqual( d1, math.sqrt(18) )

   def test_FrameUnformat(self):
      frame = starlink.Ast.Frame(2)
      nchars,value = frame.unformat( 1, "56.4 #" )
      self.assertEqual( nchars, 5 )
      self.assertEqual( value, 56.4 )

   def test_FrameActiveUnit(self):
      frame = starlink.Ast.Frame(2)
      self.assertFalse( frame.ActiveUnit )
      frame.ActiveUnit = True
      self.assertTrue( frame.ActiveUnit )

   def test_FrameSet(self):
      frame = starlink.Ast.Frame( 2 )
      frameset = starlink.Ast.FrameSet( frame )
      self.assertIsInstance( frameset, starlink.Ast.FrameSet )
      self.assertEqual( frameset.Nframe, 1 )
      frameset.addframe( 1, starlink.Ast.UnitMap( 2 ), starlink.Ast.Frame(2) )
      self.assertEqual( frameset.Nframe, 2 )
      frame2 = frameset.getframe( 1 )
      self.assertTrue( frame2.same(frame) )
      mapping = frameset.getmapping( 1, 2 )
      self.assertIsInstance( mapping, starlink.Ast.UnitMap )
      frameset.remapframe( 1, starlink.Ast.UnitMap(2) )
      frameset.removeframe( 1 )
      self.assertEqual( frameset.Nframe, 1 )

   def test_Mapping(self):
      with self.assertRaises(TypeError):
         mapping = starlink.Ast.Mapping()
      zoommap = starlink.Ast.ZoomMap( 1, 1.2 )
      map1,map2,series,invert1,invert2 = zoommap.decompose()
      self.assertIsInstance( map1, starlink.Ast.ZoomMap )
      self.assertEqual( map1.Zoom, 1.2 )
      self.assertIsNone( map2 )
      self.assertFalse( invert1 )
      self.assertFalse( invert2 )
      self.assertTrue( series )
      self.assertFalse( zoommap.Invert )
      zoommap.Invert = True
      self.assertTrue( zoommap.Invert )
      zoommap.invert()
      self.assertFalse( zoommap.Invert )
      self.assertTrue( zoommap.IsLinear )
      self.assertFalse( zoommap.IsSimple )
      self.assertFalse( zoommap.Report )
      self.assertTrue( zoommap.TranForward )
      self.assertTrue( zoommap.TranInverse )

      xin = numpy.linspace( -1, 1, 10 )
      xout = zoommap.trann( xin, True )
      d = (1.2*xin - xout)**2
      self.assertEqual( d.sum(), 0.0 )

      xa = [ 0., 1., 2., -1., -2., -3., 1., 2., 4., 5. ]
      zoommap.trann( xa, True, xout )
      d = (1.2*numpy.array( xa ) - xout)**2
      self.assertEqual( d.sum(), 0.0 )

      zoommap = starlink.Ast.ZoomMap( 3, 2.0 )
      pin = numpy.array( [[1.,2.,3], [0.,1.,2], [2.,3.,4]] )
      pout = zoommap.trann( pin, False )
      d = (0.5*pin - pout)**2
      self.assertEqual( d.sum(), 0.0 )

      zoommap = starlink.Ast.ZoomMap( 2, 2.0 )
      pout = zoommap.trangrid( [1,0], [3,2], 0.001, 100, True )
      answer = numpy.array( [[ 2., 4., 6., 2., 4., 6., 2., 4., 6.],
                             [ 0., 0., 0., 2., 2., 2., 4., 4., 4.]] )
      d = (answer - pout)**2
      self.assertEqual( d.sum(), 0.0 )

      islin,fit = zoommap.linearapprox(  [1,0], [3,2], 0.001 )
      answer = numpy.array( [ 0., 0., 2., 0., 0., 2.] )
      d = (answer - fit)**2
      self.assertEqual( d.sum(), 0.0 )
      self.assertTrue( islin )

      lb,ub,xl,xu = zoommap.mapbox(  [1,0], [3,2], True, 2 )
      self.assertEqual( lb, 0 )
      self.assertEqual( ub, 4 )
      self.assertEqual( xl[1], 0 )
      self.assertEqual( xu[1], 2 )

      isquad,fit,rms = zoommap.quadapprox(  [1,0], [3,2], 3, 3 )

      self.assertTrue( isquad )
      self.assertEqual( rms, 0.0 )
      answer = numpy.array( [ 0.,2.,0.,0.,0.,0.,0.,0.,2.,0.,0.,0.] )
      d = (answer - fit)**2
      self.assertEqual( d.sum(), 0.0 )

      self.assertEqual( zoommap.rate( [1,1], 2, 2 ), 2.0 )
      self.assertEqual( zoommap.rate( [1,1], 1, 2 ), 0.0 )

      data_in = numpy.linspace( 1, 9, 9 )
      zoommap = starlink.Ast.ZoomMap( 2, 1.0 )
      out,outv = zoommap.rebin( 0.5, [1,0], [3,2], data_in, None,
                                starlink.Ast.NEAREST, None, starlink.Ast.USEBAD,
                                0.0, 100, starlink.Ast.BAD, [2,0], [4,2], [1,0],
                                [3,2] )

      answer = numpy.array( [ 2., 3., starlink.Ast.BAD,
                              5., 6., starlink.Ast.BAD,
                              8., 9., starlink.Ast.BAD] )
      d = (answer - out)**2
      self.assertEqual( d.sum(), 0.0 )
      self.assertIsNone( outv )

      data_in = numpy.array( [[1,2,3],[4,5,6],[7,8,9]], dtype=numpy.int32 )
      data_out = numpy.empty( (3,3), dtype=numpy.int32 )
      weights = numpy.zeros( (3,3), dtype=numpy.float64 )

      flags = starlink.Ast.USEBAD | starlink.Ast.REBININIT
      nused = 0
      nused = zoommap.rebinseq( 0.5, [1,0], [3,2], data_in, None,
                                starlink.Ast.LINEAR, None, flags,
                                0.0, 100, -999, [2,0], [4,2],
                                [1,0], [3,2], data_out, None, weights,
                                nused )
      flags = starlink.Ast.USEBAD | starlink.Ast.REBINEND
      nused = zoommap.rebinseq( 0.5, [1,0], [3,2], data_in, None,
                                starlink.Ast.LINEAR, None, flags,
                                0.0, 100, -999, [2,0], [4,2],
                                [1,0], [3,2], data_out, None, weights,
                                nused )


      answer = numpy.array( [[ 2., 3., -999],
                             [ 5., 6., -999],
                             [ 8., 9., -999]] )
      d = (answer - data_out)**2
      self.assertEqual( d.sum(), 0.0 )
      self.assertEqual( nused, 12 )

      data_in = numpy.linspace( 1, 9, 9 )
      zoommap = starlink.Ast.ZoomMap( 2, 1.0 )
      npix,out,outv = zoommap.resample( [1,0], [3,2], data_in, None,
                                   starlink.Ast.NEAREST, None, starlink.Ast.USEBAD,
                                   0.0, 100, starlink.Ast.BAD, [2,0], [4,2], [2,0],
                                   [4,2] )

      answer = numpy.array( [ 2., 3., starlink.Ast.BAD,
                              5., 6., starlink.Ast.BAD,
                              8., 9., starlink.Ast.BAD] )
      d = (answer - out)**2
      self.assertEqual( d.sum(), 0.0 )
      self.assertIsNone( outv )
      self.assertEqual( npix, 3 )



   def test_PermMap(self):
      permmap = starlink.Ast.PermMap( [2,1,3],[1,2] )
      self.assertIsInstance( permmap, starlink.Ast.PermMap )
      self.assertEqual( permmap.Nin, 3 )
      self.assertEqual( permmap.Nout, 2 )

   def test_ShiftMap(self):
      shiftmap = starlink.Ast.ShiftMap( [1,2] )
      self.assertIsInstance( shiftmap, starlink.Ast.ShiftMap )
      self.assertIsInstance( shiftmap, starlink.Ast.Mapping )
      self.assertEqual( shiftmap.Nout, 2 )

   def test_SphMap(self):
      sphmap = starlink.Ast.SphMap( "UnitRadius=1" )
      self.assertIsInstance( sphmap, starlink.Ast.SphMap )
      self.assertIsInstance( sphmap, starlink.Ast.Mapping )
      self.assertEqual( sphmap.Nout, 2 )
      self.assertTrue( sphmap.UnitRadius )

   def test_RateMap(self):
      ratemap = starlink.Ast.RateMap( starlink.Ast.UnitMap(2), 1, 2 )
      self.assertIsInstance( ratemap, starlink.Ast.RateMap )
      self.assertIsInstance( ratemap, starlink.Ast.Mapping )
      self.assertEqual( ratemap.Nout, 1 )

   def test_WcsMap(self):
      wcsmap = starlink.Ast.WcsMap( 2, starlink.Ast.TAN, 1, 2 )
      self.assertIsInstance( wcsmap, starlink.Ast.WcsMap )
      self.assertIsInstance( wcsmap, starlink.Ast.Mapping )
      self.assertEqual( wcsmap.Nout,2 )
      with self.assertRaises(starlink.Ast.WCSTY):
         wcsmap = starlink.Ast.WcsMap( 2, 4000, 1, 2 )
      wcsmap.set( "PV2_0=1.2" )
      self.assertEqual( wcsmap.ProjP_0, 1.2 )
      self.assertEqual( wcsmap.WcsType, starlink.Ast.TAN )

   def test_PcdMap(self):
      pcdmap = starlink.Ast.PcdMap( 0.2, [1,2] )
      self.assertIsInstance( pcdmap, starlink.Ast.PcdMap )
      self.assertIsInstance( pcdmap, starlink.Ast.Mapping )
      self.assertEqual( pcdmap.Nout, 2 )
      self.assertEqual( pcdmap.Disco, 0.2 )
      with self.assertRaises(ValueError):
         pcdmap = starlink.Ast.PcdMap( 0.5, [1,2,3])

   def test_LutMap(self):
      lutmap = starlink.Ast.LutMap( [1,2,3,4,5], 1, 0.5 )
      self.assertIsInstance( lutmap, starlink.Ast.LutMap )
      self.assertIsInstance( lutmap, starlink.Ast.Mapping )
      self.assertEqual( lutmap.Nout, 1 )

   def test_UnitMap(self):
      unitmap = starlink.Ast.UnitMap( 3 )
      self.assertIsInstance( unitmap, starlink.Ast.UnitMap )
      self.assertEqual( unitmap.Nin, 3 )

   def test_GrismMap(self):
      grismmap = starlink.Ast.GrismMap("GrismM=1")
      self.assertIsInstance( grismmap, starlink.Ast.GrismMap )
      self.assertIsInstance( grismmap, starlink.Ast.Mapping )
      self.assertEqual( grismmap.GrismM, 1 )

   def test_WinMap(self):
      with self.assertRaises(ValueError):
         winmap = starlink.Ast.WinMap( [1], [1,2], [3,4],[5,6])
      winmap = starlink.Ast.WinMap( [1,1],[2,2], [1,1],[3,3] )
      self.assertIsInstance( winmap, starlink.Ast.WinMap )
      self.assertEqual( winmap.Nin, 2 )

   def test_NormMap(self):
      with self.assertRaises(TypeError):
         normmap = starlink.Ast.NormMap( starlink.Ast.UnitMap( 1 ) )
      normmap = starlink.Ast.NormMap( starlink.Ast.Frame( 2 ) )
      self.assertIsInstance( normmap, starlink.Ast.NormMap )
      self.assertEqual( normmap.Nin, 2 )

   def test_TimeMap(self):
      with self.assertRaises(ValueError):
         timemap = starlink.Ast.TimeMap( 1 )
      timemap = starlink.Ast.TimeMap( 0 )
      self.assertIsInstance( timemap, starlink.Ast.TimeMap )
      self.assertIsInstance( timemap, starlink.Ast.Mapping )
      self.assertEqual( timemap.Nin, 1 )
      timemap.timeadd( "BEPTOMJD", [560,720] )
      with self.assertRaises(starlink.Ast.TIMIN):
         timemap.timeadd( "UNRECOGNIZED", [1] )

   def test_CmpFrame(self):
      cmpframe = starlink.Ast.CmpFrame( starlink.Ast.Frame(2), starlink.Ast.Frame(2) )
      self.assertIsInstance( cmpframe, starlink.Ast.CmpFrame )
      self.assertIsInstance( cmpframe, starlink.Ast.Frame )

   def test_CmpMap(self):
      cmpmap = starlink.Ast.CmpMap( starlink.Ast.UnitMap(2), starlink.Ast.UnitMap(2), True )
      self.assertIsInstance( cmpmap, starlink.Ast.CmpMap )
      self.assertIsInstance( cmpmap, starlink.Ast.Mapping )

   def test_TranMap(self):
      tranmap = starlink.Ast.TranMap( starlink.Ast.UnitMap(2), starlink.Ast.UnitMap(2) )
      self.assertIsInstance( tranmap, starlink.Ast.TranMap )
      self.assertIsInstance( tranmap, starlink.Ast.Mapping )
      with self.assertRaises(TypeError):
         tranmap = starlink.Ast.TranMap( starlink.Ast.UnitMap( 1 ), [1] )

   def test_SpecFrame(self):
      specframe = starlink.Ast.SpecFrame()
      self.assertIsInstance( specframe, starlink.Ast.Frame )
      self.assertIsInstance( specframe, starlink.Ast.SpecFrame )
      sky = starlink.Ast.SkyFrame()
      specframe.setrefpos( sky, 0, 1 )
      refpos = specframe.getrefpos(sky)
      self.assertAlmostEqual( refpos[0], 0, 5 )
      self.assertAlmostEqual( refpos[1], 1 )

   def test_DSBSpecFrame(self):
      dsbspecframe = starlink.Ast.DSBSpecFrame( "IF=4.0,AlignSideBand=1")
      self.assertIsInstance( dsbspecframe, starlink.Ast.Frame )
      self.assertIsInstance( dsbspecframe, starlink.Ast.SpecFrame )
      self.assertIsInstance( dsbspecframe, starlink.Ast.DSBSpecFrame )
      self.assertTrue( dsbspecframe.AlignSideBand )
      self.assertEqual( dsbspecframe.IF, 4.0 )

   def test_SkyFrame(self):
      skyframe = starlink.Ast.SkyFrame()
      self.assertIsInstance( skyframe, starlink.Ast.Frame )
      self.assertIsInstance( skyframe, starlink.Ast.SkyFrame )
      mapping = skyframe.skyoffsetmap()
      self.assertIsInstance( mapping, starlink.Ast.Mapping )

   def test_TimeFrame(self):
      tframe = starlink.Ast.TimeFrame( "TimeScale=TAI" )
      self.assertIsInstance( tframe, starlink.Ast.Frame )
      self.assertEqual( tframe.TimeScale, "TAI" )
      self.assertGreater( tframe.currenttime(), 55700.0)

   def test_FluxFrame(self):
      fframe = starlink.Ast.FluxFrame( 52.5, starlink.Ast.SpecFrame() )
      self.assertIsInstance( fframe, starlink.Ast.Frame )
      self.assertIsInstance( fframe, starlink.Ast.FluxFrame )

   def test_SpecFluxFrame(self):
      sfframe = starlink.Ast.SpecFluxFrame( starlink.Ast.SpecFrame(),
                                            starlink.Ast.FluxFrame(57, starlink.Ast.SpecFrame()))
      self.assertIsInstance( sfframe, starlink.Ast.Frame )
      self.assertIsInstance( sfframe, starlink.Ast.CmpFrame )
      self.assertIsInstance( sfframe, starlink.Ast.SpecFluxFrame )

   def test_Box(self):
      box = starlink.Ast.Box( starlink.Ast.Frame(2), 1,
                             [0,0],[3,4] )
      self.assertIsInstance( box, starlink.Ast.Box )
      self.assertIsInstance( box, starlink.Ast.Region )
      self.assertIsInstance( box, starlink.Ast.Frame )
      self.assertIsInstance( box.getregionframe(), starlink.Ast.Frame )
      lbnd,ubnd = box.getregionbounds()
      self.assertEqual( lbnd[0], 0 )
      self.assertEqual( ubnd[0], 3 )
      testbox = starlink.Ast.Box( starlink.Ast.Frame(2), 1,
                                 [1,1],[4,5])
      overlap = box.overlap( testbox )
      self.assertEqual( overlap, 4 )
      self.assertTrue(  box.removeregions().isaframe() )

   def test_Circle(self):
      circle = starlink.Ast.Circle( starlink.Ast.Frame(2), 0,
                             [0,0],[3,4] )
      self.assertIsInstance( circle, starlink.Ast.Circle )
      self.assertIsInstance( circle, starlink.Ast.Region )
      self.assertIsInstance( circle, starlink.Ast.Frame )
      self.assertIsInstance( circle.getregionframe(), starlink.Ast.Frame )
      testcircle = starlink.Ast.Circle( starlink.Ast.Frame(2), 1,
                                 [0,0],5)
      overlap = circle.overlap( testcircle )
      self.assertEqual( overlap, 5 )

   def test_Ellipse(self):
      ell = starlink.Ast.Ellipse( starlink.Ast.Frame(2), 0,
                             [0,0],[3,4], [2,3] )
      self.assertIsInstance( ell, starlink.Ast.Ellipse )
      self.assertIsInstance( ell, starlink.Ast.Region )
      self.assertIsInstance( ell, starlink.Ast.Frame )

   def test_NullRegion(self):
      null = starlink.Ast.NullRegion( starlink.Ast.Frame(2) )
      self.assertIsInstance( null, starlink.Ast.NullRegion )
      self.assertIsInstance( null, starlink.Ast.Region )
      self.assertFalse( null.Negated )

   def test_Interval(self):
      interval = starlink.Ast.Interval( starlink.Ast.Frame(2), [1,2],[5,10] )
      self.assertIsInstance( interval, starlink.Ast.Interval )
      self.assertIsInstance( interval, starlink.Ast.Region )

   def test_CmpRegion(self):
      circle = starlink.Ast.Circle( starlink.Ast.Frame(2), 0,
                             [0,0],[3,4] )
      box = starlink.Ast.Box( starlink.Ast.Frame(2), 1,
                              [0,0],[3,4] )
      cmp = starlink.Ast.CmpRegion( circle, box, starlink.Ast.AND )
      self.assertIsInstance( cmp, starlink.Ast.CmpRegion )
      self.assertIsInstance( cmp, starlink.Ast.Region )

   def test_Prism(self):
      circle = starlink.Ast.Circle( starlink.Ast.Frame(2), 0,
                             [0,0],[3,4] )
      interval = starlink.Ast.Interval( starlink.Ast.Frame(1), [6],[100] )
      prism = starlink.Ast.Prism( circle, interval )
      self.assertIsInstance( prism, starlink.Ast.Prism )
      self.assertIsInstance( prism, starlink.Ast.Region )

   def test_Channel(self):
      channel = starlink.Ast.Channel()
      self.assertIsInstance( channel, starlink.Ast.Object)
      self.assertIsInstance( channel, starlink.Ast.Channel )
      self.assertTrue(  channel.isachannel() )
      self.assertTrue(  channel.isaobject() )
      zoommap = starlink.Ast.ZoomMap( 2, 0.1, "ID=Hello there" )
      channel.SinkFile= "fred.txt"
      n = channel.write( zoommap )
      self.assertEqual( n, 1 )
      channel.SourceFile= "fred.txt"
      with self.assertRaises(starlink.Ast.RDERR):
         obj = channel.read( )
      channel.SinkFile = None
      channel.SourceFile= "fred.txt"
      obj = channel.read( )
      channel.SinkFile = "fred2.txt"
      channel.write( obj )
      channel.SinkFile = None
      channel.SourceFile = None
      self.assertTrue( filecmp.cmp("fred.txt", "fred2.txt", shallow=False)  )

   def test_MyChannel(self):
      ss = DummyStream()
      with self.assertRaises(TypeError):
         channel = starlink.Ast.Channel( ss, ss )
      ss = TextStream()
      channel = starlink.Ast.Channel( ss, ss )
      zoommap = starlink.Ast.ZoomMap( 2, 0.1, "ID=Hello there" )
      n = channel.write( zoommap )
      self.assertEqual( n, 1 )
      a = ss.get()
      obj = channel.read( )
      ss.reset_sink()
      ss.reset_source()
      n = channel.write( obj )
      self.assertEqual( n, 1 )
      b = ss.get()
      self.assertEqual( a, b )

   def test_FitsChan(self):
      mycards = ("NAXIS1  =                  200                                                  ",
                 "NAXIS2  =                  200                                                  ",
                 "CTYPE1  = 'RA--TAN '                                                            ",
                 "CTYPE2  = 'DEC-TAN '                                                            ",
                 "CRPIX1  =                  100                                                  ",
                 "CRPIX2  =                  100                                                  ",
                 "CDELT1  =                0.001                                                  ",
                 "CDELT2  =                0.001                                                  ",
                 "CRVAL1  =                    0                                                  ",
                 "CRVAL2  =                    0                                                  ")

      fc = starlink.Ast.FitsChan(mycards)
      self.assertIsInstance( fc, starlink.Ast.Object)
      self.assertEqual( fc[ "CRVAL1" ], 0.0 )

      fc = starlink.Ast.FitsChan()
      self.assertIsInstance( fc, starlink.Ast.Object)
      self.assertIsInstance( fc, starlink.Ast.Channel )
      self.assertIsInstance( fc, starlink.Ast.FitsChan )
      self.assertTrue( fc.isafitschan() )
      self.assertTrue( fc.isachannel() )
      self.assertTrue( fc.isaobject() )

      fc.setfitsI( "FRED", 99, "Hello there", True)
      there,value = fc.getfitsI( "FRED" )
      self.assertTrue( there )
      self.assertEqual( value, 99 )
      there,value = fc.getfitsS( "FRED" )
      self.assertTrue( there )
      self.assertEqual( value, "99" )
      fc.setfitsF( "FRED1", 99.9, "Hello there", True)
      there,value = fc.getfitsS( "FRED1" )
      self.assertTrue( there )
      self.assertEqual( value, "99.9" )
      fc.setfitsCF( "FRED2", (99.9,99.8), "Hello there", True)
      there,(real,imag) = fc.getfitsCF( "FRED2" )
      self.assertTrue( there )
      self.assertEqual( real, 99.9 )
      self.assertEqual( imag, 99.8 )
      fc.setfitsCI( "FRED3", (99,98), "Hello there", True)
      there,value = fc.getfitsS( "FRED3" )
      self.assertTrue( there )
      self.assertEqual( value, "99 98" )
      fc.setfitsS( "FRED4", "-12", "Hello there", True)
      there,value = fc.getfitsI( "FRED4" )
      self.assertTrue( there )
      self.assertEqual( value, -12 )
      fc.emptyfits()
      self.assertEqual( fc.Ncard, 0 )

      cards = "CRVAL1  = 0                                                                     CRVAL2  = 0                                                                     "
      fc.putcards( cards )
      self.assertEqual( fc.Card, 1 )
      fc.set( "Card=10" )
      self.assertEqual( fc.Card, 3 )
      fc.Card = None
      self.assertEqual( fc.Card, 1 )

      for card in mycards[0:8]:
         fc.putfits( card, False )

      self.assertEqual( fc.Ncard, 10 )
      self.assertEqual( fc.Card, 9 )
      self.assertEqual( fc.Encoding, "FITS-WCS" )
      there, card = fc.findfits( "%f", False )
      self.assertTrue( there )
      self.assertEqual( card, "CRVAL1  =                    0                                                  " )
      fc.delfits()
      self.assertEqual( fc.Ncard, 9 )
      self.assertEqual( fc.Card, 9 )
      there, card = fc.findfits( "%f", False )
      self.assertTrue( there )
      self.assertEqual( card, "CRVAL2  =                    0                                                  " )
      fc.putfits( "CRVAL1  = 0", False )
      self.assertEqual( fc.Ncard, 10 )
      self.assertEqual( fc.Card, 10 )
      there, card = fc.findfits( "%f", False )
      self.assertTrue( there )
      self.assertEqual( card, "CRVAL2  =                    0                                                  " )

      for cards in zip(fc,mycards):
         self.assertEqual( cards[0], cards[1] )

      fc.Card = None
      obj = fc.read()
      self.assertIsInstance( obj, starlink.Ast.FrameSet)
      self.assertTrue( obj.isaframeset() )

      with self.assertRaises(TypeError):
         fc = starlink.Ast.FitsChan("Encoding=FITS-WCS")

      fc = starlink.Ast.FitsChan( None, None, "Encoding=FITS-WCS")

      n = fc.write( obj )
      self.assertEqual( n, 1 )
      fc.Card = None

      there, card = fc.findfits( "%f", True )
      card = card.rstrip()
      self.assertTrue( there )
      self.assertEqual(card, "WCSAXES =                    2 / Number of WCS axes" )

      there, card = fc.findfits( "%f", True )
      card = card.rstrip()
      self.assertTrue( there )
      self.assertEqual(card, "CRPIX1  =                100.0 / Reference pixel on axis 1" )

      there, card = fc.findfits( "%f", True )
      card = card.rstrip()
      self.assertTrue( there )
      self.assertEqual(card, "CRPIX2  =                100.0 / Reference pixel on axis 2" )

      there, card = fc.findfits( "%f", True )
      card = card.rstrip()
      self.assertTrue( there )
      self.assertEqual(card, "CRVAL1  =                  0.0 / Value at ref. pixel on axis 1" )

      there, card = fc.findfits( "%f", True )
      card = card.rstrip()
      self.assertTrue( there )
      self.assertEqual(card, "CRVAL2  =                  0.0 / Value at ref. pixel on axis 2" )

      there, card = fc.findfits( "%f", True )
      card = card.rstrip()
      self.assertTrue( there )
      self.assertEqual(card, "CTYPE1  = 'RA---TAN'           / Type of co-ordinate on axis 1" )

      there, card = fc.findfits( "%f", True )
      card = card.rstrip()
      self.assertTrue( there )
      self.assertEqual(card, "CTYPE2  = 'DEC--TAN'           / Type of co-ordinate on axis 2" )

      there, card = fc.findfits( "%f", True )
      card = card.rstrip()
      self.assertTrue( there )
      self.assertEqual(card, "CDELT1  =                0.001 / Pixel size on axis 1" )

      there, card = fc.findfits( "%f", True )
      card = card.rstrip()
      self.assertTrue( there )
      self.assertEqual(card, "CDELT2  =                0.001 / Pixel size on axis 2" )

      there, card = fc.findfits( "%f", True )
      card = card.rstrip()
      self.assertTrue( there )
      self.assertEqual(card, "RADESYS = 'ICRS    '           / Reference frame for RA/DEC values" )

      ss = TextStream()
      fc = starlink.Ast.FitsChan( ss, ss, "Encoding=FITS-WCS")
      n = fc.write( obj )
      self.assertEqual( n, 1 )
      fc.writefits()
      a = ss.get()
      fc.readfits()
      obj = fc.read( )
      ss.reset_sink()
      ss.reset_source()
      n = fc.write( obj )
      self.assertEqual( n, 1 )
      fc = None
      b = ss.get()
      self.assertEqual( a, b )

   def test_FitsChan_AsMapping(self):
      fc = starlink.Ast.FitsChan()
      fc["NAXIS1"] = 200
      fc["NAXIS2"] = 200
      fc["CTYPE1"] = "RA--TAN "
      fc["CTYPE2"] = "DEC-TAN "
      fc["CRPIX1"] = 100
      fc["CRPIX2"] = 100
      fc["CDELT1"] = 0.001
      fc["CDELT2"] = 0.001
      fc["CRVAL1"] = 0
      fc["CRVAL2"] = 0

      self.assertTrue( "NAXIS1" in fc )

      self.assertEqual( len(fc), 10 )
      self.assertEqual( fc[2], "CTYPE1  = 'RA--TAN '                                                            ")
      fc.Card = None
      obj = fc.read( )
      self.assertEqual( len(fc), 2 )

      with self.assertRaises(KeyError):
         a = fc["CRVAL1"]

      self.assertEqual( fc["NAXIS1"], 200 )
      fc["NAXIS2"] = None
      self.assertEqual( len(fc), 1 )
      fc["NAXIS2"] = "This is a string"
      self.assertEqual( len(fc), 2 )
      self.assertEqual( fc["NAXIS2"], "This is a string")
      fc.setfitsI( "NAXIS2", -12, "Hello there", False )
      self.assertEqual( fc.Ncard, 3 )
      self.assertEqual( fc.Nkey, 2 )
      self.assertEqual( len(fc), 2 )
      self.assertEqual( fc["NAXIS2"][0], -12 )
      self.assertEqual( fc["NAXIS2"][1], "This is a string" )
      fc["NAXIS2"] = 1000
      self.assertEqual( fc["NAXIS2"], 1000 )
      self.assertEqual( fc.Ncard, 2 )
      self.assertEqual( fc.Nkey, 2 )
      self.assertEqual( len(fc), 2 )

      fc[100] = "CTYPE1  = 'RA--TAN '                                                            "
      self.assertEqual( fc[2], "CTYPE1  = 'RA--TAN '                                                            ")
      self.assertEqual( fc.Ncard, 3 )
      self.assertEqual( fc.Nkey, 3 )
      self.assertEqual( len(fc), 3 )

      fc[0] = "NEWKEY  = 123.456"
      self.assertEqual( fc[0], "NEWKEY  =              123.456                                                  ")
      self.assertEqual( fc.Ncard, 3 )
      self.assertEqual( fc.Nkey, 3 )
      self.assertEqual( len(fc), 3 )

   def test_StcsChan(self):
      ss = TextStream()
      ch = starlink.Ast.StcsChan( ss, ss, "ReportLevel=3" )
      self.assertIsInstance( ch, starlink.Ast.Object)
      self.assertIsInstance( ch, starlink.Ast.Channel )
      self.assertIsInstance( ch, starlink.Ast.StcsChan )
      self.assertTrue( ch.isastcschan() )
      self.assertTrue( ch.isachannel() )
      self.assertTrue( ch.isaobject() )
      ss.sink("StartTime 1900-01-01 Circle ICRS 148.9 69.1 2.0")
      ss.sink("SpeCtralInterval 4000 7000 unit Angstrom")
      obj = ch.read()
      self.assertIsInstance( obj, starlink.Ast.Prism )
      self.assertEqual( obj.Class, "Prism" )
      self.assertEqual( obj.Naxes, 4 )
      lbnd,ubnd = obj.getregionbounds()
      self.assertEqual( ubnd[0], sys.float_info.max )
      self.assertAlmostEqual( lbnd[1], 2.50080939227851 )
      self.assertAlmostEqual( ubnd[1], 2.6967811201606 )
      self.assertAlmostEqual( lbnd[2], 1.171115928088195 )
      self.assertAlmostEqual( ubnd[2], 1.24091013301998 )
      self.assertEqual( lbnd[3], 4000.0 )
      self.assertEqual( ubnd[3], 7000.0 )

   def test_KeyMap(self):

      with self.assertRaises(starlink.Ast.BADAT):
         km = starlink.Ast.KeyMap( "ReportLevel=3" )

      km = starlink.Ast.KeyMap( "KeyCase=0" )
      self.assertIsInstance( km, starlink.Ast.Object)
      self.assertIsInstance( km, starlink.Ast.KeyMap )
      self.assertTrue( km.isakeymap() )
      self.assertTrue( km.isaobject() )
      self.assertEqual( len(km), 0 )
      km['Hello'] = 'fred'
      self.assertEqual( len(km), 1 )
      self.assertEqual( km['hELlo'], 'fred' )
      km['Hello'] = None
      self.assertEqual( len(km), 0 )
      km['TTT'] = (1.2, 3, 4.5)
      self.assertEqual( len(km), 1 )
      self.assertEqual( km['ttt'], (1.2, 3, 4.5) )
      km['SS'] = 'hello'
      self.assertEqual( len(km), 2 )

      self.assertEqual( km[0], ('TTT', (1.2, 3.0, 4.5)) )
      self.assertEqual( km[1], ('SS', 'hello'))
      self.assertTrue( "SS" in km )

      i = 0
      for entry in km:
         self.assertEqual( entry, km[i] )
         i += 1

      km[0] = None
      self.assertEqual( len(km), 1 )
      self.assertEqual( km[0], ('SS', 'hello'))
      km[0] = 'Goofbye'
      self.assertEqual( km[0], ('SS', 'Goofbye'))
      with self.assertRaises(starlink.Ast.MPIND):
         km[1] = 'Nooooooo'

   def test_Plot(self):
      with self.assertRaises(TypeError):
         plot = starlink.Ast.Plot( None, [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0],
                                   TextStream() )
      mygrf = DummyGrf()
      plot = starlink.Ast.Plot( None, [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0],
                                mygrf )
      self.assertIsInstance( plot, starlink.Ast.Object)
      self.assertIsInstance( plot, starlink.Ast.Mapping )
      self.assertIsInstance( plot, starlink.Ast.Frame )
      self.assertIsInstance( plot, starlink.Ast.FrameSet )
      self.assertIsInstance( plot, starlink.Ast.Plot )
      self.assertTrue( plot.isaplot() )
      self.assertTrue( plot.isaframeset() )
      self.assertTrue( plot.isaframe() )
      self.assertTrue( plot.isamapping() )
      self.assertTrue( plot.isaobject() )

      self.assertEqual( plot.Tol, 0.01 )
      plot.Tol = 0.5
      self.assertEqual( plot.Tol, 0.5 )

      self.assertEqual( plot.Colour_Border, 1 )
      plot.Colour_Border = 2
      self.assertEqual( plot.Colour_Border, 2 )

      plot.border()

      self.assertAlmostEqual( mygrf.linex, [0.0, 0.071428574621677399, 0.1428571492433548, 0.2142857164144516, 0.28571429848670959, 0.3571428656578064, 0.4285714328289032, 0.5, 0.57142859697341919, 0.6428571343421936, 0.71428573131561279, 0.78571426868438721, 0.8571428656578064, 0.92857140302658081, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.92857140302658081, 0.8571428656578064, 0.78571426868438721, 0.71428573131561279, 0.6428571343421936, 0.57142859697341919, 0.5, 0.4285714328289032, 0.3571428656578064, 0.28571429848670959, 0.2142857164144516, 0.1428571492433548, 0.071428574621677399, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] )
      self.assertAlmostEqual( mygrf.liney, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.071428574621677399, 0.1428571492433548, 0.2142857164144516, 0.28571429848670959, 0.3571428656578064, 0.4285714328289032, 0.5, 0.57142859697341919, 0.6428571343421936, 0.71428573131561279, 0.78571426868438721, 0.8571428656578064, 0.92857140302658081, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.92857140302658081, 0.8571428656578064, 0.78571426868438721, 0.71428573131561279, 0.6428571343421936, 0.57142859697341919, 0.5, 0.4285714328289032, 0.3571428656578064, 0.28571429848670959, 0.2142857164144516, 0.1428571492433548, 0.071428574621677399, 0.0] )
      self.assertEqual( mygrf.nline, 57 )
      self.assertEqual( mygrf.attr, [4, 4] )
      self.assertAlmostEqual( mygrf.value, [2.0, 1.0] )
      self.assertEqual( mygrf.prim, [1, 1] )

      mygrf.Reset()
      plot.grid()
      self.assertEqual( mygrf.textt, ['0', '0.3', '0.6', '0.9', '0.4', '0.7', '1', 'Axis 1', 'Axis 2', '2-d coordinate system'])
      self.assertEqual( mygrf.textj,  ['TC', 'TC', 'TC', 'TC', 'CR', 'CR', 'CR', 'TC', 'BC', 'BC'])
      self.assertEqual( mygrf.ntext, 10)

      mygrf.Reset()
      pin = numpy.array( [[0.5,1.,0.], [0.5,1.,0.5]] )
      plot.polycurve( pin )
      self.assertEqual( mygrf.nline, 29 )

      lbnd,ubnd = plot.boundingbox()
      self.assertAlmostEqual( lbnd[0], 0 )
      self.assertAlmostEqual( lbnd[1], 0.5 )
      self.assertAlmostEqual( ubnd[0], 1. )
      self.assertAlmostEqual( ubnd[1], 1. )

      plot.Tol = 0.0000001
      plot.clip( starlink.Ast.CURRENT, [0.1,0.1], [0.9,0.9] )
      plot.polycurve( pin )
      lbnd,ubnd = plot.boundingbox()
      self.assertAlmostEqual( lbnd[0], 0.1 )
      self.assertAlmostEqual( lbnd[1], 0.5 )
      self.assertAlmostEqual( ubnd[0], 0.9 )
      self.assertAlmostEqual( ubnd[1], 0.9 )

      plot.curve( [0.0,0.0], [1.0,1.0] )
      lbnd,ubnd = plot.boundingbox()
      self.assertAlmostEqual( lbnd[0], 0.1 )
      self.assertAlmostEqual( lbnd[1], 0.1 )
      self.assertAlmostEqual( ubnd[0], 0.9 )
      self.assertAlmostEqual( ubnd[1], 0.9 )

      plot.clip( starlink.Ast.NOFRAME, [], [] )
      plot.curve( [0.0,0.0], [1.0,1.0] )
      lbnd,ubnd = plot.boundingbox()
      self.assertAlmostEqual( lbnd[0], 0.0 )
      self.assertAlmostEqual( lbnd[1], 0.0 )
      self.assertAlmostEqual( ubnd[0], 1.0 )
      self.assertAlmostEqual( ubnd[1], 1.0 )

      xlut = starlink.Ast.LutMap( [0,1], 0, 1.0 )
      ylut = starlink.Ast.LutMap( [0,0.5], 0, 1.0 )
      xylut = starlink.Ast.CmpMap( xlut, ylut, False )
      pm = starlink.Ast.PermMap( [1], [1,1] )
      map = starlink.Ast.CmpMap( pm, xylut, True )
      plot.gencurve( map )
      lbnd,ubnd = plot.boundingbox()
      self.assertAlmostEqual( lbnd[0], 0.0 )
      self.assertAlmostEqual( lbnd[1], 0.0 )
      self.assertAlmostEqual( ubnd[0], 1.0 )
      self.assertAlmostEqual( ubnd[1], 0.5 )

      plot.gridline( 1, [0.5,0.5], 0.5 )
      lbnd,ubnd = plot.boundingbox()
      self.assertAlmostEqual( lbnd[0], 0.5 )
      self.assertAlmostEqual( lbnd[1], 0.5 )
      self.assertAlmostEqual( ubnd[0], 1.0 )
      self.assertAlmostEqual( ubnd[1], 0.5 )

      mygrf.Reset()
      plot.mark( pin, 1 )
      lbnd,ubnd = plot.boundingbox()
      self.assertAlmostEqual( lbnd[0], 0.0 )
      self.assertAlmostEqual( lbnd[1], 0.5 )
      self.assertAlmostEqual( ubnd[0], 1.0 )
      self.assertAlmostEqual( ubnd[1], 1.0 )

      self.assertAlmostEqual( pin[0][0], mygrf.markx[0] )
      self.assertAlmostEqual( pin[0][1], mygrf.markx[1] )
      self.assertAlmostEqual( pin[0][2], mygrf.markx[2] )
      self.assertAlmostEqual( pin[1][0], mygrf.marky[0] )
      self.assertAlmostEqual( pin[1][1], mygrf.marky[1] )
      self.assertAlmostEqual( pin[1][2], mygrf.marky[2] )
      self.assertEqual( 1, mygrf.markt[0] )
      self.assertEqual( 3, mygrf.nmark )

      mygrf.Reset()
      plot.text( "Hello", [0.5,0.5], [0.0,1.0], "CC" )
      self.assertEqual( mygrf.textt, ['Hello'] )
      self.assertEqual( mygrf.textx, [0.5] )
      self.assertEqual( mygrf.texty, [0.5] )
      self.assertEqual( mygrf.textj, ['CC'] )
      self.assertEqual( mygrf.ntext, 1 )

   def test_MatrixMap(self):

      with self.assertRaises(ValueError):
         mm = starlink.Ast.MatrixMap( [ [ [1.0] ] ]  )

      mm = starlink.Ast.MatrixMap( [ -1.0, 2.0 ]  )
      self.assertIsInstance( mm, starlink.Ast.Object)
      self.assertIsInstance( mm, starlink.Ast.Mapping )
      self.assertIsInstance( mm, starlink.Ast.MatrixMap )
      self.assertTrue(  mm.isaobject() )
      self.assertTrue(  mm.isamapping() )
      self.assertTrue(  mm.isamatrixmap() )
      self.assertEqual( mm.Nin, 2 )
      self.assertEqual( mm.Nout, 2 )
      pin = numpy.array( [[1.,2.,3], [0.,1.,2]] )
      pout = mm.trann( pin, False )
      self.assertEqual( pout[0][0], -1.0 )
      self.assertEqual( pout[0][1], -2.0 )
      self.assertEqual( pout[0][2], -3.0 )
      self.assertEqual( pout[1][0], 0.0/2.0 )
      self.assertEqual( pout[1][1], 1.0/2.0 )
      self.assertEqual( pout[1][2], 2.0/2.0 )

      mm = starlink.Ast.MatrixMap( [[0.0,1.0], [2.0,3.0], [-1.0, -2.0]] )
      self.assertTrue( mm.TranForward )
      self.assertFalse( mm.TranInverse )
      self.assertEqual( mm.Nout, 3 )
      self.assertEqual( mm.Nin, 2 )
      pout = mm.trann( pin, True )
      self.assertEqual( pout[0][0], 0.0 )
      self.assertEqual( pout[0][1], 1.0 )
      self.assertEqual( pout[0][2], 2.0 )
      self.assertEqual( pout[1][0], 2.0 )
      self.assertEqual( pout[1][1], 7.0 )
      self.assertEqual( pout[1][2], 12.0 )
      self.assertEqual( pout[2][0], -1.0 )
      self.assertEqual( pout[2][1], -4.0 )
      self.assertEqual( pout[2][2], -7.0 )


   def test_PolyMap(self):
      pm = starlink.Ast.PolyMap( [[1.2,1.,2.,0.],[-0.5,1.,1.,1.],
                                  [1.0,2.,0.,1.]])
      self.assertIsInstance( pm, starlink.Ast.Object)
      self.assertIsInstance( pm, starlink.Ast.Mapping )
      self.assertIsInstance( pm, starlink.Ast.PolyMap )
      self.assertTrue(  pm.isaobject() )
      self.assertTrue(  pm.isamapping() )
      self.assertTrue(  pm.isapolymap() )
      self.assertEqual( pm.Nin, 2 )
      self.assertEqual( pm.Nout, 2 )
      pin = numpy.array( [[1.,2.,3], [0.,1.,2]] )
      pout = pm.trann( pin, True )
      for (xi,yi,xo,yo) in zip(pin[0],pin[1],pout[0],pout[1]):
         xn = 1.2*xi*xi - 0.5*yi*xi
         yn = yi
         self.assertAlmostEqual( xn, xo )
         self.assertAlmostEqual( yn, yo )

      pm = starlink.Ast.PolyMap( None, [[1.2,1.,2.,0.],[-0.5,1.,1.,1.],
                                        [1.0,2.,0.,1.]])
      self.assertEqual( pm.Nin, 2 )
      self.assertEqual( pm.Nout, 2 )
      pout = pm.trann( pin, False )
      for (xi,yi,xo,yo) in zip(pin[0],pin[1],pout[0],pout[1]):
         xn = 1.2*xi*xi - 0.5*yi*xi
         yn = yi
         self.assertAlmostEqual( xn, xo )
         self.assertAlmostEqual( yn, yo )

      pm = starlink.Ast.PolyMap( [[ 1., 1., 1., 0.],
                                  [ 1., 1., 0., 1.],
                                  [ 1., 2., 1., 0.],
                                  [-1., 2., 0., 1.]],
                                 [[ 0.5, 1., 1., 0.],
                                  [ 0.5, 1., 0., 1.],
                                  [ 0.5, 2., 1., 0.],
                                  [-0.5, 2., 0., 1.]] )

      self.assertEqual( pm.Nin, 2 )
      self.assertEqual( pm.Nout, 2 )
      pout = pm.trann( pin, True )
      pnew = pm.trann( pout, False )

      for (xi,yi,xn,yn) in zip(pin[0],pin[1],pnew[0],pnew[1]):
         self.assertAlmostEqual( xn, xi )
         self.assertAlmostEqual( yn, yi )

      self.assertEqual( pm.NiterInverse, 4 )
      self.assertEqual( pm.TolInverse, 1.0E-6 )


      new = pm.polytran( False, 0.01, 0.1, 3, [-1.0, 1.0], [-1.0, 1.0] )

#  The above call to astPolyTran currently does a really bad job so skip
#  the following test until the problem is fixed in polymap.c
#      pout = new.trann( pin, True )
#      pnew = new.trann( pout, False )
#      for (xi,yi,xn,yn) in zip(pin[0],pin[1],pnew[0],pnew[1]):
#         self.assertAlmostEqual( xn, xi )
#         self.assertAlmostEqual( yn, yi )



if __name__ == "__main__":
    #unittest.main()
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAst)
    unittest.TextTestRunner(verbosity=2).run(suite)

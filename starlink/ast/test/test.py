import unittest
import starlink.Ast
import copy
import numpy
import math

class TestAst(unittest.TestCase):

   def test_Static(self):
      self.assertEqual( starlink.Ast.escapes(1), 0 )
      self.assertEqual( starlink.Ast.escapes(0), 1 )
      self.assertEqual( starlink.Ast.escapes(-1), 0 )
      self.assertEqual( starlink.Ast.tune("ObjectCaching", 1 ), 0 )
      self.assertEqual( starlink.Ast.tune("ObjectCaching", starlink.Ast.TUNULL ), 1 )
      self.assertEqual( starlink.Ast.tune("ObjectCaching", 0 ), 1 )
      self.assertGreaterEqual( starlink.Ast.version(), 5007002 )

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

      zoommap.lock(1)

   def test_FrameSimple(self):
      frame = starlink.Ast.Frame( 2 )
      self.assertIsInstance( frame, starlink.Ast.Frame )
      self.assertEqual( frame.Nin, 2 )
      self.assertEqual( frame.Nout, 2 )
      testtitle = "Test Frame"
      frame.Title = testtitle
      self.assertEqual( frame.Title, testtitle)
      self.assertEqual( frame.get("Title"), testtitle)

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

   def test_LutMap(self):
      lutmap = starlink.Ast.LutMap( [1,2,3,4,5], 1, 0.5 )
      self.assertIsInstance( lutmap, starlink.Ast.LutMap )
      self.assertIsInstance( lutmap, starlink.Ast.Mapping )
      self.assertEqual( lutmap.Nout, 1 )

   def test_UnitMap(self):
      unitmap = starlink.Ast.UnitMap( 3 )
      self.assertIsInstance( unitmap, starlink.Ast.UnitMap )
      self.assertEqual( unitmap.Nin, 3 )

   def test_CmpFrame(self):
      cmpframe = starlink.Ast.CmpFrame( starlink.Ast.Frame(2), starlink.Ast.Frame(2) )
      self.assertIsInstance( cmpframe, starlink.Ast.CmpFrame )
      self.assertIsInstance( cmpframe, starlink.Ast.Frame )

   def test_CmpMap(self):
      cmpmap = starlink.Ast.CmpMap( starlink.Ast.UnitMap(2), starlink.Ast.UnitMap(2), True )
      self.assertIsInstance( cmpmap, starlink.Ast.CmpMap )
      self.assertIsInstance( cmpmap, starlink.Ast.Mapping )

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

if __name__ == "__main__":
    #unittest.main()
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAst)
    unittest.TextTestRunner(verbosity=2).run(suite)

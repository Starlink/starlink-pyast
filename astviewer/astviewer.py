#!/usr/bin/python

#  To run the app under the pdb debugger and break at a given line,
#  copy the following line to the place where the break point is
#  required, and uncomment.
# import pdb; pdb.set_trace()




import os
import sys
from PyQt4 import QtCore, QtGui, QtOpenGL
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from math import *
import starlink.Ast as Ast
import random
import subprocess
import math

#  Do we have astropy fits? If not do we have pyfits? If not, not FITS
#  support.
fits_supported = False
try:
   import astropy.io.fits as pyfits
   fits_supported = True
except ImportError:
   pass

if not fits_supported:
   try:
      import pyfits
      fits_supported = True
   except ImportError:
      pass

if fits_supported:
   import starlink.Atl as Atl

#  Ignore warnings (pyfits issues warnings when it tries to open a
#  non-FITS file)
   import warnings
   warnings.filterwarnings('ignore')

else:
   print("!! astropy/pyfits not found. No FITS support")

#  Do we have ATOOLS? If not, no NDF support (can't get pyndf to work).
ndf_supported = False
starlink = os.environ.get("STARLINK_DIR")
if starlink:
   astcopy = starlink+"/bin/atools/astcopy"
   if os.path.isfile( astcopy ):
      ndf_supported = True
if not ndf_supported:
   print("!! Starlink ATOOLS not found. No NDF support")





#  astviewer options

OPTIONS_FNAME = '.astviewerrc'
OPT_CHANGED = 'changed'
OPT_FCATTS = 'fcatts'

option_defs = {}
option_defs[ OPT_FCATTS ] = [ "FitsChan attributes to use when reading FITS-WCS", " ", True ]






#  Other constants

light_grey = QColor( 240, 240, 240 )
black = QColor( 0, 0, 0 )
pen1 = QPen( Qt.black, 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
pen2 = QPen( Qt.black, 1, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
pen3 = QPen( Qt.blue, 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
pen4 = QPen( Qt.red, 2, Qt.DotLine, Qt.RoundCap, Qt.RoundJoin)


#  Invoke a starlink command
def invoke(command):
   os.environ["ADAM_NOPROMPT"] = "1"
   os.environ["ADAM_EXIT"] = "1"
   os.environ["MSG_SZOUT"] = "0"
   outtxt = ""
   proc = subprocess.Popen(command,shell=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
   while True:

      line = proc.stdout.readline()
      while line is not None and len(line) > 0:
         if isinstance( line, bytes ):
            line = line.decode("ascii","ignore")
         line = line.rstrip()
         outtxt = "{0}\n{1}".format(outtxt,line)
         line = proc.stdout.readline()

      status = proc.poll()
      if status is not None:
         break

      time.sleep(1.0)

   if status != 0:
      if outtxt:
         msg = outtxt
         raise RuntimeError("\n\n{0}".format(msg))
      else:
         raise RuntimeError()

   return outtxt


#  Return a string describing a Frame.
def frameDesc( frame ):
   text = ""
   dom0 = frame.Domain
   title0 = frame.Title
   epoch0 = frame.Epoch
   if dom0:
      text = "{0}Domain: {1}\n".format(text,dom0)
   if title0:
      text = "{0}Title: {1}\n".format(text,title0)
   if epoch0:
      if float(epoch0) < 1984.0:
         text = "{0}Epoch: B{1}\n".format(text,epoch0)
      else:
         text = "{0}Epoch: J{1}\n".format(text,epoch0)

   spec_axis = 0
   dsb_axis = 0
   skylon_axis = 0
   skylat_axis = 0
   time_axis = 0
   flux_axis = 0

   for i in range( frame.Naxes ):
      text = "{0}\nAxis {1}:\n".format(text,i+1)
      classified = False

      try:
         junk = frame.get( "IsLatAxis({0})".format( i+1 ) )
         if junk == '1':
            skylat_axis = i + 1
         else:
            skylon_axis = i + 1
         classified = True
      except Ast.BADAT:
         pass

      try:
         junk = frame.get( "SideBand({0})".format( i+1 ) )
         dsb_axis = i + 1
         classified = True
      except Ast.BADAT:
         try:
            junk = frame.get( "StdOfRest({0})".format( i+1 ) )
            spec_axis = i + 1
            classified = True
         except Ast.BADAT:
            pass

      try:
         junk = frame.get( "TimeScale({0})".format( i+1 ) )
         time_axis = i + 1
         classified = True
      except Ast.BADAT:
         pass

      try:
         junk = frame.get( "SpecVal({0})".format( i+1 ) )
         flux_axis = i + 1
         classified = True
      except Ast.BADAT:
         pass

      item = frame.get( "Domain({0})".format( i+1 ) )
      if item and item != dom0 and not classified:
         text = "{0}   Domain: {1}\n".format(text,item)
      item = frame.get("Label({0})".format( i+1 ))
      if item:
         text = "{0}   Label: {1}\n".format(text,item)
      item = frame.get("InternalUnit({0})".format( i+1 ))
      if item:
         text = "{0}   Unit: {1}\n".format(text,item)
      item = frame.get("Format({0})".format( i+1 ))
      if item:
         text = "{0}   Format: {1}\n".format(text,item)

   sky_axis = max( [skylon_axis, skylat_axis ] )
   if sky_axis > 0:
      text = "{0}\nCelestial co-ordinates:\n".format(text)
      item = frame.get("System({0})".format( sky_axis ))
      if item:
         text = "{0}   System: {1}\n".format(text,item)
      item = frame.get("Equinox({0})".format( sky_axis ))
      if item:
         text = "{0}   Equinox: {1}\n".format(text,item)

      if skylon_axis > 0 and skylat_axis > 0:
         att1 = "SkyRef({0})".format( skylon_axis )
         att2 = "SkyRef({0})".format( skylat_axis )
         print( frame.test(att1), frame.test(att2) )
         if frame.test(att1) and frame.test(att2):
            fmt1 = frame.format( skylon_axis, float(frame.get(att1)) )
            fmt2 = frame.format( skylat_axis, float(frame.get(att2)) )
            text = "{0}   Ref. position: {1} {2}\n".format(text,fmt1,fmt2)
            item = frame.get("SkyRefIs({0})".format( sky_axis ))
            text = "{0}   Ref. position is {1}\n".format(text,item)


   if spec_axis == 0:
      spec_axis = dsb_axis

   if spec_axis > 0:
      text = "{0}\nSpectral co-ordinates:\n".format(text)
      item = frame.get("System({0})".format( spec_axis ))
      if item:
         text = "{0}   System: {1}\n".format(text,item)
      item = frame.get("RefRA({0})".format( spec_axis ))
      if item:
         text = "{0}   Ref. RA: {1}\n".format(text,item)
      item = frame.get("RefDec({0})".format( spec_axis ))
      if item:
         text = "{0}   Ref. Dec: {1}\n".format(text,item)
      item = frame.get("RestFreq({0})".format( spec_axis ))
      if item:
         text = "{0}   Rest freq: {1} GHz\n".format(text,item)
      item = frame.get("StdOfRest({0})".format( spec_axis ))
      if item:
         text = "{0}   Standard of rest: {1}\n".format(text,item)





   return text


#  ================================================================
class SettingsDialog(QDialog):

#  ----------------------------------------------------------------
   def __init__(self, parent, options, viewer ):
      super(SettingsDialog,self).__init__( parent )

      self.options = options
      self.viewer = viewer

      vlayout = QVBoxLayout(self)
      self.setLayout(vlayout)

      self.lineedits = {}
      grid = QGridLayout()
      irow = 0
      for key in options:
         if key != OPT_CHANGED:
            value = options[ key ]
            prompt = option_defs[ key ][ 0 ]
            self.lineedits[key] =  QLineEdit( str(value), self )
            grid.addWidget( QLabel( prompt+":" ), irow, 0 )
            grid.addWidget( self.lineedits[key], irow, 1 )
            irow += 1

      vlayout.addLayout( grid )

      buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)
      buttons.accepted.connect(self.accept)
      buttons.rejected.connect(self.reject)

      vlayout.addWidget(buttons)

   def accept(self):
      for key in self.options:
         if key != OPT_CHANGED:
            text = str(self.lineedits[ key ].text()).strip()
            if text != self.options[ key ]:
               self.options[ key ] = text
               if option_defs[ key ][ 2 ]:
                  self.viewer.redraw = True
               self.options[ OPT_CHANGED ] = True
      super(SettingsDialog,self).accept()


#  ================================================================
class DumpDialog(QDialog):

#  ----------------------------------------------------------------
   def __init__(self, parent, object ):
      super(DumpDialog,self).__init__( parent )

      self.object = object

      vlayout = QVBoxLayout(self)
      self.setLayout(vlayout)

      self.menubar = QMenuBar( self )
      vlayout.addWidget( self.menubar )

      fileMenu = self.menubar.addMenu('&File')

      saveAction = QAction('&Save', self)
      saveAction.setShortcut('Ctrl+S')
      saveAction.setStatusTip('Save object to text file')
      saveAction.triggered.connect(self.save)
      fileMenu.addAction(saveAction)

      vlayout.setContentsMargins(10,10,10,10)
      vlayout.setSpacing(5)

      self.tabs = QTabWidget()
      vlayout.addWidget( self.tabs )

      s = QScrollArea()
      text = QLabel( "{}".format(object) )
      text.setMargin(10)
      s.setWidget( text )
      self.tabs.addTab( s, "Raw AST data" )

      buttons = QDialogButtonBox(
            QDialogButtonBox.Ok,
            Qt.Horizontal, self)
      buttons.accepted.connect(self.accept)
      buttons.rejected.connect(self.reject)

      vlayout.addWidget(buttons)

   def accept(self):
      super(DumpDialog,self).accept()

   def save(self):
      fname = QtGui.QFileDialog.getSaveFileName(self, 'Save {0} to file:'.format(self.object.Class))
      if fname:
         Ast.Channel( None, None, "SinkFile={0}".format(fname) ).write( self.object )

#  ================================================================
class FrameDialog(DumpDialog):

#  ----------------------------------------------------------------
   def __init__(self, parent, object ):
      super(FrameDialog,self).__init__( parent, object )

      text = "Domain: {0}\n".format( object.Domain )


      s = QScrollArea()
      text = QLabel( frameDesc( self.object ) )
      text.setMargin(10)
      s.setWidget( text )
      itab = self.tabs.addTab( s, "Axis descriptions" )
      self.tabs.setCurrentIndex( itab )


#  ================================================================
class MappingDialog(QDialog):

#  ----------------------------------------------------------------
   def __init__(self, parent, object ):
      super(MappingDialog,self).__init__( parent )
      QApplication.setOverrideCursor(Qt.WaitCursor)

      self.object = object

      vlayout = QVBoxLayout(self)
      self.setLayout(vlayout)
      vlayout.setContentsMargins(0,0,0,0)
      vlayout.setSpacing(0)

      s = QScrollArea()
      vlayout.addWidget( s )
      text = QLabel( "{}".format(object) )
      s.setWidget( text )

      buttons = QDialogButtonBox(
            QDialogButtonBox.Ok,
            Qt.Horizontal, self)
      buttons.accepted.connect(self.accept)
      buttons.rejected.connect(self.reject)

      vlayout.addWidget(buttons)
      QApplication.restoreOverrideCursor()

   def accept(self):
      super(MappingDialog,self).accept()


#  ================================================================
class NodeIcon(QGraphicsEllipseItem):

#  ----------------------------------------------------------------
   def __init__(self, inode, scene, radius=7 ):
      super(NodeIcon,self).__init__( 0, 0, radius, radius )
      self.inode = inode
      self.scene = scene
      self.radius = radius
      self.setBrush( black )
      self.setPen( pen1 )
      self.setAcceptHoverEvents(True)
      self.childmaps = []
      self.parentmap = None

#  ----------------------------------------------------------------
#  Show help in the status bar when the mouse enters the FrameIcon.
   def hoverEnterEvent(self,event):
      self.scene.statusbar.showMessage( "Click and drag to move a node and all its children." )
      QApplication.setOverrideCursor(Qt.DragMoveCursor)

#  ----------------------------------------------------------------
#  Clear the status bar when the mouse leaves the FrameIcon.
   def hoverLeaveEvent(self,event):
      self.scene.statusbar.clearMessage()
      QApplication.restoreOverrideCursor()

#  ----------------------------------------------------------------
#  Return X at centre of icon.
   def centreX(self):
      return self.sceneBoundingRect().center().x()

#  ----------------------------------------------------------------
#  Return Y at centre of icon.
   def centreY(self):
      return self.sceneBoundingRect().center().y()

#  ----------------------------------------------------------------
#  Centre the icon on a given (X,Y) value
   def setCentre(self,x,y):
      dx = self.centreX() - x
      dy = self.centreY() - y
      self.setPos( self.x() - dx, self.y() - dy )

#  ----------------------------------------------------------------
#  Return the point on the circle that is closest to the supplied point.
   def connectPos(self, x, y):
      x0 = self.centreX()
      y0 = self.centreY()

      dx = x - x0
      dy = y - y0

      l = sqrt( dx*dx + dy*dy )
      if l > 0.0:
         x = x0 + 0.5*dx*self.radius/l
         y = y0 + 0.5*dy*self.radius/l

      return ( x, y )

#  ----------------------------------------------------------------
#  Record the current position of the node and all its children.
   def mark( self ):
      self.markx = self.x()
      self.marky = self.y()
      if self.inode in self.scene.tree:
         for inode in self.scene.tree[self.inode]:
            self.scene.nodes[ inode ].mark()

#  ----------------------------------------------------------------
#  Move the current position of the node and all its children.
   def move( self, dx, dy ):
      self.setPos( self.markx + dx, self.marky + dy )
      if self.inode in self.scene.tree:
         for inode in self.scene.tree[self.inode]:
            self.scene.nodes[ inode ].move( dx, dy )

      if self.parentmap:
         self.parentmap.setEnds()
      for map in self.childmaps:
         map.setEnds()

#  ----------------------------------------------------------------
#  When the left mouse button is pressed over the node, prepare to drag
#  the node and all its children.
   def mousePressEvent( self, event):
      if event.modifiers() != QtCore.Qt.ControlModifier:
         curpos = event.scenePos()
         self.drag_curx = curpos.x()
         self.drag_cury = curpos.y()
         self.mark()

#  ----------------------------------------------------------------
#  When the mouse is moved, move the node and all its children.
   def mouseMoveEvent( self, event):
      if event.modifiers() != QtCore.Qt.ControlModifier:
         curpos = event.scenePos()
         dx = curpos.x() - self.drag_curx
         dy = curpos.y() - self.drag_cury
         self.move( dx, dy )

#  ----------------------------------------------------------------
#  Add a mapping to a child node.
   def addChildMapping( self, mapping ):
      self.childmaps.append( mapping )

#  ----------------------------------------------------------------
#  Set the mapping from the parent node.
   def setParentMapping( self, mapping ):
      self.parentmap = mapping


#  ================================================================
class FrameIcon(QGraphicsRectItem):

#  ----------------------------------------------------------------
   def __init__(self, inode, iframe, frame, scene, label="" ):
      super(FrameIcon,self).__init__()

      self.inode = inode
      self.iframe = iframe
      self.frame = frame
      self.scene = scene
      self.rb = None
      self.origin = None
      self.childmaps = []
      self.parentmap = None
      self.dragged = False

      self.setAcceptHoverEvents(True)
      self.setEnabled(True)
      self.setActive(True)
      self.setFlag( QGraphicsItem.ItemIsSelectable )
#      self.setFlag( QGraphicsItem.ItemIsMovable )

      self.setBrush( light_grey )
      self.setPen( pen2 )

#  Create a SimpleTextItem to hold the text.
      self.text = QGraphicsSimpleTextItem()

#  Attach it to this RectItem so that the text apears inside the box.
      self.text.setParentItem(self)

#  Assign the default text.
      if label:
         self.setText(  "Frame: {} ({})\nDomain: {}".format( iframe, label, frame.Domain) )
      else:
         self.setText(  "Frame: {}\nDomain: {}".format( iframe, frame.Domain) )

#  ----------------------------------------------------------------
#  Show help in the status bar when the mouse enters the FrameIcon.
   def hoverEnterEvent(self,event):
      self.scene.statusbar.showMessage( "Click to see Frame properties. "
                                        "Click and drag to move a Frame and its children. "
                                        "Control-click and drag to another Frame "
                                        "to see Mapping between two Frames" )
      QApplication.setOverrideCursor(Qt.PointingHandCursor)

#  ----------------------------------------------------------------
#  Clear the status bar when the mouse leaves the FrameIcon.
   def hoverLeaveEvent(self,event):
      self.scene.statusbar.clearMessage()
      QApplication.restoreOverrideCursor()

#  ----------------------------------------------------------------
#  Record the current position of the frame and all its children.
   def mark( self ):
      self.markx = self.x()
      self.marky = self.y()
      if self.inode in self.scene.tree:
         for inode in self.scene.tree[self.inode]:
            self.scene.nodes[ inode ].mark()

#  ----------------------------------------------------------------
#  Move the current position of the frame and all its children.
   def move( self, dx, dy ):
      self.setPos( self.markx + dx, self.marky + dy )
      if self.inode in self.scene.tree:
         for inode in self.scene.tree[self.inode]:
            self.scene.nodes[ inode ].move( dx, dy )

      if self.parentmap:
         self.parentmap.setEnds()
      for map in self.childmaps:
         map.setEnds()

#  ----------------------------------------------------------------
#  When the left mouse button is pressed over the FrameIcon, prepare to drag
#  out a line to the pointer as it is moved.
   def mousePressEvent( self, event):
      self.dragged = False
      if event.modifiers() == QtCore.Qt.ControlModifier:
         self.origin = QPointF( self.centreX(), self.centreY() )
         if not self.rb:
            self.rb = ArrowItem( self.origin.x(), self.origin.y(),
                                 self.origin.x(), self.origin.y() )
            self.rb.setPen( pen4 )
            self.scene.addItem( self.rb )
      else:
         curpos = event.scenePos()
         self.drag_curx = curpos.x()
         self.drag_cury = curpos.y()
         self.mark()

#  ----------------------------------------------------------------
#  When the mouse is moved, drag out a line to the pointer.
   def mouseMoveEvent( self, event):
      self.dragged = True
      if event.modifiers() == QtCore.Qt.ControlModifier:
         here = event.scenePos()
         self.rb.setArrow( self.origin.x(), self.origin.y(), here.x(), here.y() )
      else:
         curpos = event.scenePos()
         dx = curpos.x() - self.drag_curx
         dy = curpos.y() - self.drag_cury
         self.move( dx, dy )

#  ----------------------------------------------------------------
#  When the left mouse button is released, display the Frame details if
#  the mouse is still over the original FrameIcon, or the Mapping joining
#  the two Frames if it over a differetn FrameIcon.
   def mouseReleaseEvent(self, event):

#  Remove the arrow first or else itemAt will return the arrow rather
#  than the TextItem.
      done = False
      if event.modifiers() == QtCore.Qt.ControlModifier:
         if self.rb:
            self.scene.removeItem( self.rb )

         it = self.scene.itemAt( event.scenePos() )
         if it and isinstance( it, QGraphicsSimpleTextItem ):
            it = it.parentItem()

         self.scene.clearSelection()
         self.setSelected(True)
         if it and isinstance( it, FrameIcon ) and it != self:
            it.setSelected(True)
            self.scene.addItem( self.rb )
            map = self.scene.frameset.getmapping( self.iframe, it.iframe )
            dialog = MappingDialog( None, map )
            dialog.exec_()
            self.scene.removeItem( self.rb )
            done = True

         self.rb = None

      if not done and not self.dragged:
         dialog = FrameDialog( None, self.frame )
         dialog.exec_()


      return QGraphicsRectItem.mouseReleaseEvent(self,event)

#  ----------------------------------------------------------------
#  Return X at centre of icon.
   def centreX(self):
      return self.sceneBoundingRect().center().x()

#  ----------------------------------------------------------------
#  Return Y at centre of icon.
   def centreY(self):
      return self.sceneBoundingRect().center().y()

#  ----------------------------------------------------------------
#  Centre the icon on a given (X,Y) value
   def setCentre(self,x,y):
      dx = self.centreX() - x
      dy = self.centreY() - y
      self.setPos( self.x() - dx, self.y() - dy )

#  ----------------------------------------------------------------
#  Return the point on the rectangle that is closest to the supplied point.
   def connectPos(self, x, y):
      rect = self.sceneBoundingRect()
      h = rect.height()
      w = rect.width()

      x0 = rect.center().x()
      y0 = rect.center().y()

      dx = x - x0
      dy = y - y0

      if abs(dx) > w/2 or abs(dy) > h/2:
         if dx > 0:
            if dy*w > h*dx:
               x = x0 + 0.5*h*dx/dy
               y = y0 + 0.5*h
            elif dy*w < -h*dx:
               x = x0 - 0.5*h*dx/dy
               y = y0 - 0.5*h
            else:
               x = x0 + 0.5*w
               y = y0 + 0.5*w*dy/dx

         elif dx < 0:
            if dy/(-dx) > h/w:
               x = x0 + 0.5*h*dx/dy
               y = y0 + 0.5*h
            elif dy/(-dx) < -h/w:
               x = x0 - 0.5*h*dx/dy
               y = y0 - 0.5*h
            else:
               x = x0 - 0.5*w
               y = y0 - 0.5*w*dy/dx

         else:
            if dy > 0.0:
               x = x0
               y = y0 + 0.5*h
            else:
               x = x0
               y = y0 - 0.5*h

      return ( x, y )

#  ----------------------------------------------------------------
   def setText(self, text ):

#  Store the text  in the SimpleTextItem.
      self.text.setText( text )

#  Set the size of the box to give a border around the text.
      self.setRect( -10, -10, self.text.boundingRect().width() + 20,
                              self.text.boundingRect().height() + 20 )


#  ----------------------------------------------------------------
#  Add a mapping to a child node.
   def addChildMapping( self, mapping ):
      self.childmaps.append( mapping )

#  ----------------------------------------------------------------
#  Set the mapping from the parent node.
   def setParentMapping( self, mapping ):
      self.parentmap = mapping





#  ================================================================
class ArrowItem(QGraphicsPolygonItem):

#  ----------------------------------------------------------------
   def __init__(self):
      super(ArrowItem,self).__init__()

   def __init__(self,x1,y1,x2,y2):
      super(ArrowItem,self).__init__()
      self.setArrow( x1, y1, x2, y2 )

#  Set the start and end of the arrow.
   def setArrow( self, x1, y1, x2, y2 ):

#  Arrow head dimensions (pixels)
      head_width = 4
      head_length = 10

#  Create an empty polygon.
      poly = QPolygonF()

#  Add points to the polygon so that the polygon forms an arrow from
#  (x1,y1) to (x2,y2)
      dx = x2 - x1
      dy = y2 - y1
      l = sqrt( dx*dx + dy*dy )
      if l > 0.0:

         bar_length = l - head_length
         vx = dx/l
         vy = dy/l

         x = x1
         y = y1
         poly += QPointF( x, y )

         x += vx*bar_length
         y += vy*bar_length
         poly += QPointF( x, y )

         x += vy*head_width
         y += -vx*head_width
         poly += QPointF( x, y )

         x += -vy*head_width + vx*head_length
         y += vx*head_width + vy*head_length
         poly += QPointF( x, y )

         x += -vx*head_length - vy*head_width
         y += -vy*head_length + vx*head_width
         poly += QPointF( x, y )

         x += vy*head_width
         y += -vx*head_width
         poly += QPointF( x, y )

#  Use the polygon.
         self.setPolygon(poly)


#  ================================================================
class MappingIcon(ArrowItem):

#  ----------------------------------------------------------------
   def __init__(self,node1,node2,mapping,scene):
      super(ArrowItem,self).__init__()
      self.setAcceptHoverEvents(True)
      self.setEnabled(True)
      self.setActive(True)
      self.setFlag( QGraphicsItem.ItemIsSelectable )
#      self.setFlag( QGraphicsItem.ItemIsMovable )

      self.mapping = mapping
      self.scene = scene
      self.node1 = node1
      self.node2 = node2
      self.setPen( pen1 )
      self.setBrush( black )
      node1.addChildMapping( self )
      node2.setParentMapping( self )
      self.setEnds()

#  -------------------------------------------------------------------
   def hoverEnterEvent(self,event):
      self.scene.statusbar.showMessage( "Click to see Mapping properties.")
      QApplication.setOverrideCursor(Qt.PointingHandCursor)

#  -------------------------------------------------------------------
   def hoverLeaveEvent(self,event):
      self.scene.statusbar.clearMessage()
      QApplication.restoreOverrideCursor()

#  -------------------------------------------------------------------
   def mouseReleaseEvent(self, event):
      dialog = MappingDialog( None, self.mapping )
      dialog.exec_()
      return QGraphicsPolygonItem.mouseReleaseEvent(self,event)

#  -------------------------------------------------------------------
#  Set he positions of hte two ends of the MappingIcon so that they match
#  the attached nodes.
   def setEnds( self ):

#  Get central (x,y) for each node.
      x1 = self.node1.centreX()
      y1 = self.node1.centreY()
      x2 = self.node2.centreX()
      y2 = self.node2.centreY()

      dx = x2 - x1
      dy = y2 - y1
      length = math.sqrt(dx*dx+dy*dy)
      cosa = dx/length
      sina = dy/length

#  Modify the position of the arrow end so that it is on the nearest
#  point of the box representing node2.
      (x2,y2) = self.node2.connectPos( x1, y1 )

#  Modify the position of the arrow start so that it is on the nearest
#  point of the box representing node1.
      (x1,y1) = self.node1.connectPos( x2, y2 )

#  Set the start and end of the arrow.
      self.setArrow( x1, y1, x2, y2 )



#  ================================================================
class AstScene(QGraphicsScene):

#  ----------------------------------------------------------------
   def __init__(self,frameset,view,parent):
        super(AstScene,self).__init__(parent)

        w = 800
        h = 500
        self.setSceneRect( 0, 0, w, h )

        self.view = view
        self.frameset = frameset
        self.nodes = []
        self.parents = []
        self.statusbar = parent.statusbar
        self.baseIcon = None
        self.currentIcon = None
        self.nodegrid = {}

#  Get the indices of the base and current Frames.
        ibase = frameset.Base
        icurrent = frameset.Current

#  Get the number of nodes in the FrameSet.
        ( ok, nnode, iframen, mapn, parent ) = frameset.getnode( -1 )

#  Loop round all nodes in the FrameSet.
        for inode in range( nnode ):

#  Get the details of the FrameSet node.
           ( ok, nnode, iframe, map, parent ) = frameset.getnode( inode )

#  If the node is associated with a Frame, create a FramceIcon to add to
#  the QGraphicsScene. Otherwise, create a NodeIcon. They all have the default
#  position (0,0) to begin with.
           if iframe != Ast.NOFRAME:
              if iframe == ibase:
                 label = "base"
                 if iframe == icurrent:
                    label += " and current"
              elif iframe == icurrent:
                 label = "current"
              else:
                 label = ""

              frame = frameset.getframe( iframe )
              item = FrameIcon( inode, iframe, frame, self, label )

              if iframe == ibase:
                 item.setPen( pen3 )
                 self.baseIcon = item

              if iframe == icurrent:
                 item.setPen( pen3 )
                 self.currentIcon = item

           else:
              item = NodeIcon( inode, self )

#  Add the item to the scene, and append the icon to the list of node
#  icons. Also record the node index of the parent node (the node that
#  feeds the current node).
           self.addItem( item )
           self.nodes.append( item )
           self.parents.append( parent )

#  Create a tree holding the node indices. Each node in the tree holds
#  nodes representing the child nodes. Each node is represented by its
#  index in the "self.nodes" array.
        self.tree = {}
        nnode  = len( self.nodes )
        for inode in range( nnode ):
           iparent = self.parents[ inode ]
           if iparent < 0:
              self.iroot = inode
           else:
              if not iparent in self.tree:
                 self.tree[iparent] = []
              self.tree[iparent].append(inode)

#  Assign a spatial position to each node icon. Make sure the mean position
#  is the center of the window.
        icon_spacing = 110
        self.layout( icon_spacing, w/2, h/2 )
        self.layout2( icon_spacing, w/2, h/2 )

#  Now that the nodes have spatial positions, we can join them together
#  using MappingIcons. Must do these down the tree.
#        self.MakeMappingIcons( self.iroot, frameset )
        for inode in range( nnode ):
           iparent =  self.parents[inode]
           if iparent >= 0:
              from_node = self.nodes[ iparent ]
              to_node = self.nodes[ inode ]
              ( ok, nnode, iframe, map, parent ) = frameset.getnode( inode )
              item = MappingIcon( from_node, to_node, map, self )
              self.addItem( item )


#  ----------------------------------------------------------------
#  Create MappingIcons that connect the specified node to each of its
#  children.
   def MakeMappingIcons( self, inode, frameset ):

#  Check the specified node has some children.
      if inode in self.tree:
         from_node = self.nodes[ inode ]

#  Loop round creating a MappingIcon that connect the specified node to
#  each child.
         children = self.tree[ inode ]
         nchild = len( children )
         result = 0
         for ichild in range(nchild):
           inode_child = children[ ichild ]
           to_node = self.nodes[ inode_child ]
           ( ok, nnode, iframe, map, parent ) = frameset.getnode( inode_child )
           item = MappingIcon( from_node, to_node, map, self )
           self.addItem( item )

#  Now create MappingIcons that connect the child to its own children.
           self.MakeMappingIcons( children[ ichild ], frameset )

#  ----------------------------------------------------------------
#  Ensure each node has an optimal position in the graph
   def layout(self, spacing, centrex, centrey ):

#  Find the weight for each node.
      nnode  = len( self.nodes )
      self.node_weight = []
      for inode in range( nnode ):
         self.node_weight.append( self.getNodeWeight( self.tree, inode ) )

#  Place the root node at its original position, then recursively place
#  each descendant on a set of concentric rings centred on the root node.
      x = self.nodes[ self.iroot ].centreX()
      y = self.nodes[ self.iroot ].centreY()
      (xmin, xmax, ymin, ymax) = self.placeNode( self.tree, self.iroot, x, y,
                                                 -0.5*pi, 1.5*pi, spacing, "" )

#  Adjust the above positions to put the centre of the bounding box at
#  the centre of the window, and assign the adjusted position to the icon.
      dx = 0.5*( xmax + xmin ) - centrex
      dy = 0.5*( ymax + ymin ) - centrey
      for inode in range( nnode ):
         x = self.nodes[ inode ].x()
         y = self.nodes[ inode ].y()
         self.nodes[ inode ].setPos( x - dx, y - dy )






#  ----------------------------------------------------------------
#  Ensure each node has an optimal position in the graph. This updtes the
#  positions created by "layout1" using a dynamical model in which all
#  nodes repel each other with a inverse square force, and each node is
#  connected to its parent with a spring of natural length "spacing".
   def layout2( self, spacing, centrex, centrey ):

      K = -100000
      A = 0.1
      E = 0.1

      nnode = len( self.nodes )
      iter = -1
      delta_maxl = 2E30
      delta_max = 1E30
      while delta_max < delta_maxl and iter < 200:
         iter += 1
         delta_maxl = delta_max
         delta_max = 0
         xmin = 1.0E30
         xmax = -1.0E30
         ymin = 1.0E30
         ymax = -1.0E30

         deltas = []
         newxs = []
         newys = []
         for inode in range(nnode):

            this_node = self.nodes[ inode ]
            this_x = this_node.centreX()
            this_y = this_node.centreY()

            if isinstance( this_node, FrameIcon):
               lab = "{0} {1}".format(inode,this_node.text.text())
            else:
               lab = "{0}".format(inode)

            fx = 0
            fy = 0
            for jnode in range(nnode):
               if inode != jnode:
                  node = self.nodes[ jnode ]
                  dx = node.centreX() - this_x
                  dy = node.centreY() - this_y
                  l2 = dx*dx + dy*dy
                  if l2 > 0.0:
                     l = math.sqrt( l2 )
                     f = K/l2
                     fx += f*dx/l
                     fy += f*dy/l

            iparent = self.parents[ inode ]
            if iparent >= 0:
               parent_node = self.nodes[ iparent ]
               par_x = parent_node.centreX()
               par_y = parent_node.centreY()
               dx = par_x - this_x
               dy = par_y - this_y
               l = math.sqrt( dx*dx + dy*dy )

               fx += E*(l-spacing)*dx/l
               fy += E*(l-spacing)*dy/l

               newx = this_x + A*fx
               newy = this_y + A*fy

               dx = newx - this_x
               dy = newy - this_y
               delta = math.sqrt( dx*dx + dy*dy )
               if delta > delta_max:
                  delta_max = delta

               dx = this_x - par_x
               dy = this_y - par_y
               l = math.sqrt( dx*dx + dy*dy )

               dx = newx - par_x
               dy = newy - par_y
               lnew = math.sqrt( dx*dx + dy*dy )

               if isinstance( parent_node, FrameIcon):
                  plab = "{0} {1}".format(iparent,parent_node.text.text())
               else:
                  plab = "{0}".format(iparent)

            else:
               newx = this_x
               newy = this_y
               delta = 0

            newxs.append( newx )
            newys.append( newy )
            deltas.append( delta )

            if newx > xmax:
               xmax = newx
            if newx < xmin:
               xmin = newx
            if newy > ymax:
               ymax = newy
            if newy < ymin:
               ymin = newy

         for inode in range(nnode):
            self.nodes[inode].setCentre( newxs[inode], newys[inode] )

#  Adjust the above positions to put the centre of the bounding box at
#  the centre of the window, and assign the adjusted position to the icon.
      dx = 0.5*( xmax + xmin ) - centrex
      dy = 0.5*( ymax + ymin ) - centrey
      for inode in range( nnode ):
         x = self.nodes[ inode ].x()
         y = self.nodes[ inode ].y()
         self.nodes[ inode ].setPos( x - dx, y - dy )





#  ----------------------------------------------------------------
#  Return the angular weight for a specified node.
   def getNodeWeight( self, tree, inode ):

#  If the tree has no node for the given index, it is a leaf node so give
#  it a weight of 1.0.
      if inode not in tree:
         return 1.0

#  Otherwise, summing up the angular weight of its children.
      else:
         children = tree[ inode ]
         nchild = len( children )
         result = 0
         for ichild in range(nchild):
            result += self.getNodeWeight( tree, children[ ichild ] )

#  Apply a factor so that smaller weight are given to nodes deeper in the tree.
         result = 1.0 + 0.5*result

      return result

#  ----------------------------------------------------------------
#  Place a given node at the specified position, and then place all
#  descendants on concentric rings centred on the supplied node, but
#  restricted to a specified angular section of each concentric ring.
   def placeNode( self, tree, inode, x, y, a1, a2, spacing, indent ):
      node = self.nodes[ inode ]
      if isinstance( node, FrameIcon ):
         text = "(Frame {0})".format(node.iframe)
      else:
         text = ""
#      print("{6}Drawing node {0} {1} at ({2},{3}) with children between {4} and {5} (weight {7})".
#            format( inode, text, x, y, 57.29578*a1, 57.29578*a2, indent, self.node_weight[inode] ))

#  Initialise the bounding box containing the supplied node and all its
#  decendants.
      xmax = -1.0E30
      xmin = 1.0E30
      ymax = -1.0E30
      ymin = 1.0E30

#  Find the offset from origin to centre of the node, and then set the
#  node's origin position so as to get the requested centre position.
      dx = node.centreX() - node.x()
      dy = node.centreY() - node.y()
      node.setPos( x - dx, y - dy )

#  Initialise the bounding box containing the supplied node and all its
#  decendants.
      xmax = x
      xmin = x
      ymax = y
      ymin = y

#  If this node has any children, draw them.
      if inode in tree:

#  Get the number of children.
         children = tree[ inode ]
         nchild = len( children )

#  Get the total weight of all children of the current node.
         wtot = 0.0
         for ichild in range(nchild):
            wtot += self.node_weight[ children[ ichild ] ]

#  Divide up the angular range available to this node so that each unit
#  weight gets the same angular width.
         delta = ( a2 - a1 )/wtot
         if delta > 0.5*pi:
            delta = 0.5*pi

#  INitialise the central angle for the first child

#  Loop round drawing each child.
         b2 = a1
         for ichild in range(nchild):
            inode_child = children[ ichild ]

#  Get the angular width for the current node, based on its weight. */
            awidth = delta*self.node_weight[ children[ ichild ] ]

#  Get the upper and lower angular limits for the node.
            b1 = b2
            b2 = b1 + awidth

#  Get the position for the child.
            if isinstance( self.nodes[inode_child], NodeIcon ) or \
               isinstance( self.nodes[inode], NodeIcon ):
               sp = spacing
            else:
               sp = spacing
            a0 = 0.5*( b1 + b2 )
            cx = x + sp*sin( a0 )
            cy = y + sp*cos( a0 )

#  Call this function recursively to draw the child and all its decendants.
            (cxmin, cxmax, cymin, cymax) = self.placeNode( tree, inode_child,
                                                      cx, cy, b1-0.2, b2+0.2, spacing, indent+"  " )

#  Update the bounding box.
            if cxmin < xmin:
               xmin = cxmin
            if cxmax > xmax:
               xmax = cxmax
            if cymin < ymin:
               ymin = cymin
            if cymax > ymax:
               ymax = cymax

#  Return the bounding box.
      return (xmin, xmax, ymin, ymax)


#  ================================================================
class AstView(QGraphicsView):
   def __init__(self, tab, parent = None):
      super(AstView, self).__init__(parent)
      self.tab = tab

#  ================================================================
class AstTab(QWidget):
   def __init__(self, label, parent = None):
      super(AstTab, self).__init__(parent)
      self.label = label
      self.view =  AstView(self)
      layout = QHBoxLayout()
      layout.addWidget( self.view )
      self.setLayout(layout)

   def setSize(self):
      self.view.setSceneRect( self.sceneRect() )

   def sceneRect(self):
      rect = self.view.scene().itemsBoundingRect()
      width = rect.width()*1.1
      height = rect.height()*1.1
      cent = rect.center()
      left = cent.x() - width/2
      top = cent.y() - height/2
      return QRectF( left, top, width, height )


#  ================================================================
class AstTabs(QTabWidget):
   def __init__(self, parent = None):
      super(AstTabs, self).__init__(parent)
      self.tabs = {}

   def addTab(self,label="",tabtext=""):
      self.s = QScrollArea()
      tab = AstTab( label )
      self.s.setWidget( tab )
      self.s.setWidgetResizable(True)
      itab = super(AstTabs, self).addTab( self.s, label )
      self.tabs[label] = tab
      self.setTabText( self.count()-1, tabtext)
      return itab

   def removeCurrentTab(self):
      itab = self.currentIndex()
      tab = self.widget(itab).widget()
      for (key,value) in self.tabs.items():
         if value == tab:
            del self.tabs[key]
      self.removeTab( itab )
      return self.count()

   def getCurrentView(self):
      result = self.currentWidget()
      if result is not None:
         result = result.widget()
      if result is not None:
         result = result.view
      return result

   def setSize(self):
      self.currentWidget().widget().setSize()

#  ================================================================
class AstViewer(QMainWindow):

#  ----------------------------------------------------------------
   def __init__(self, fname ):
      super(AstViewer,self).__init__()
      self.resize(QDesktopWidget().availableGeometry(self).size() * 0.6)
      self.loadOptions()

      self.exampleTab = -1
      self.file = None
      self.object = None
      self.tabs = AstTabs()
      self.setCentralWidget( self.tabs )
      self.statusbar = self.statusBar()
      self.titles = {}
      self.fname = None
      self.redraw = False

      self.tabs.currentChanged.connect(self.tabChanged)

      if fname:
         self.readFile( fname )
      else:
         self.showExample( )

      exitAction = QAction('&Exit', self)
      exitAction.setShortcut('Ctrl+Q')
      exitAction.setStatusTip('Exit application')
      exitAction.triggered.connect(self.close)

      openFileAction = QAction('Open', self)
      openFileAction.setShortcut('Ctrl+O')

      if ndf_supported:
         if fits_supported:
            text = 'Open new File - text, NDF or FITS'
         else:
            text = 'Open new File - text or NDF'
      elif fits_supported:
         text = 'Open new File - text or FITS'
      else:
         text = 'Open new File - text only'

      openFileAction.setStatusTip(text)
      openFileAction.triggered.connect(self.showOpenFileDialog)

      closeTabAction = QAction('&Close', self)
      closeTabAction.setStatusTip('Close tab')
      closeTabAction.triggered.connect(self.closeTab)

      settingsAction = QAction('Preferences', self)
      settingsAction.setStatusTip('Change global preferences')
      settingsAction.triggered.connect(self.showSettingsDialog)

      baseToCurrentAction = QAction('&Base->Current Mapping', self)
      baseToCurrentAction.setShortcut('Ctrl+M')
      baseToCurrentAction.setStatusTip('View the Mapping from the base to '
                                       'the current Frame')
      baseToCurrentAction.triggered.connect(self.btoc)

      currentAction = QAction('&Current Frame', self)
      currentAction.setShortcut('Ctrl+C')
      currentAction.setStatusTip('View the current Frame')
      currentAction.triggered.connect(self.cframe)

      baseAction = QAction('&Base Frame', self)
      baseAction.setShortcut('Ctrl+B')
      baseAction.setStatusTip('View the base Frame')
      baseAction.triggered.connect(self.bframe)

      menubar = self.menuBar()
      fileMenu = menubar.addMenu('&File')
      fileMenu.addAction(openFileAction)
      fileMenu.addAction(closeTabAction)
      fileMenu.addAction(settingsAction)
      fileMenu.addAction(exitAction)

      viewMenu = menubar.addMenu('&View')
      viewMenu.addAction(baseAction)
      viewMenu.addAction(currentAction)
      viewMenu.addAction(baseToCurrentAction)

#  ----------------------------------------------------------------
#  Display the base to current Mapping
   def btoc(self ):
      if self.scene and self.scene.frameset:
         self.scene.clearSelection()
         self.scene.baseIcon.setSelected(True)
         self.scene.currentIcon.setSelected(True)

         x1 = self.scene.baseIcon.centreX()
         y1 = self.scene.baseIcon.centreY()
         x2 = self.scene.currentIcon.centreX()
         y2 = self.scene.currentIcon.centreY()

         (x2,y2) = self.scene.currentIcon.connectPos( x1, y1 )
         (x1,y1) = self.scene.baseIcon.connectPos( x2, y2 )

         rb = ArrowItem( x1, y1, x2, y2 )
         rb.setPen( pen4 )
         self.scene.addItem( rb )
         map = self.scene.frameset.getmapping( Ast.BASE, Ast.CURRENT )
         dialog = MappingDialog( None, map )
         dialog.exec_()
         self.scene.removeItem( rb )

#  ----------------------------------------------------------------
#  Display the current Frame
   def cframe(self ):
      if self.scene and self.scene.frameset:
         self.scene.clearSelection()
         self.scene.currentIcon.setSelected(True)
         frame = self.scene.frameset.getframe( Ast.CURRENT )
         dialog = FrameDialog( None, frame )
         dialog.exec_()

#  ----------------------------------------------------------------
   def bframe(self ):
#  Display the base Frame
      if self.scene and self.scene.frameset:
         self.scene.clearSelection()
         self.scene.baseIcon.setSelected(True)
         frame = self.scene.frameset.getframe( Ast.BASE )
         dialog = FrameDialog( None, frame )
         dialog.exec_()

#  ----------------------------------------------------------------
#  Display a FrameSet in a tab
   def showObject(self, object, title, label ):
      if object:
         self.tabs.blockSignals(True)
         if self.exampleTab >= 0:
            self.tabs.removeTab( self.exampleTab )
            self.exampleTab = -1

         itab = self.tabs.addTab( label, label )
         self.tabs.setCurrentIndex( itab )

         view = self.tabs.getCurrentView()
         scene = AstScene( object, view, self )
         view.setScene( scene )
         self.titles[ scene ] = title;

         self.showTab()
         self.tabs.blockSignals(False)

#  ----------------------------------------------------------------
#  Invoked when the user clicks on a new tab.
   def tabChanged( self, i ):
      self.showTab()

#  ----------------------------------------------------------------
#  Invoked when the current tab changes.
   def showTab( self ):
      self.view = self.tabs.getCurrentView()
      if self.view:
         self.view.setMouseTracking(True)
         self.scene = self.view.scene()
         self.setWindowTitle( "astviewer: "+self.titles[ self.scene ] )
         self.object = object
         self.tabs.setSize( )

#  ----------------------------------------------------------------
#  Close a tab. Close the app if no other tabs left.
   def closeTab(self):
      if self.tabs.removeCurrentTab() == 0:
         self.close()

#  ----------------------------------------------------------------
   def readFile(self, fname ):
      self.fname = fname
      return self.drawFile()

#  ----------------------------------------------------------------
   def drawFile( self ):
      ok = False
      obj = None

      if self.fname:
         fname = str(self.fname)
         if not os. path. isfile(fname):
            QMessageBox.warning( self, "Message", "Cannot find file '{}'".format( fname ) )
            fname = None

#  See if it a binary file - usually works.
      if fname:
         textchars = bytearray({7,8,9,10,12,13,27} | set(range(0x20,0x100)) - {0x7f})
         is_binary_string = lambda bytes: bool(bytes.translate(None, textchars))
         binfile = is_binary_string(open(fname, 'rb').read(32768))

#  If binary, first try as an NDF. Convert to text using astcopy.
         if binfile and ndf_supported:
            try:
               dumpfile = "astviewer.tmp"
               invoke( "{} this={} result={}".format(astcopy,fname,dumpfile))
               binfile = False
            except:
               dumpfile = fname
         else:
            dumpfile = fname

#  Now try reading the file as a text file.
         if not binfile:
            try:
               obj = Ast.Channel( None, None, "SourceFile="+dumpfile ).read()
            except:
               obj = None

            if not obj:
               try:
                  fc = Ast.FitsChan( None, None, self.options[ OPT_FCATTS ] )
                  fc.SourceFile = dumpfile
                  obj = fc.read()
               except:
                  obj = None

         if dumpfile != fname:
            os.remove( dumpfile )

         if not obj and fits_supported and binfile:
            try:
               fits = pyfits.open( fname )
               obj = Ast.FitsChan( Atl.PyFITSAdapter(fits[ 0 ]), None,
                                   self.options[ OPT_FCATTS ] ).read()
            except:
               obj = None

#  If we have a suitable object, display it. Otherwise warn the user.
         if obj:
            if obj.isaframeset():
               self.showObject( obj, fname, os.path.basename(fname) )
               ok = True
            else:
               QMessageBox.warning( self, "Message", "Read an AST '{}' from  "
                                    "file '{}' - only FrameSets can be displayed"
                                    .format(obj.Class, fname) )
         else:
            QMessageBox.warning( self, "Message", "Failed to read an AST "
                                 "object from file '{}'".format( fname ) )
      if ok:
         self.redraw = False

      return ok


#  ----------------------------------------------------------------
   def showExample(self):
      f1 = Ast.Frame( 2, "Domain=D1" )
      f2 = Ast.Frame( 2, "Domain=D2" )
      f3 = Ast.Frame( 2, "Domain=D3" )
      f4 = Ast.Frame( 2, "Domain=D4" )
      f5 = Ast.Frame( 2, "Domain=D5" )
      f6 = Ast.Frame( 2, "Domain=D6" )
      f7 = Ast.Frame( 2, "Domain=D7" )
      f8 = Ast.Frame( 2, "Domain=D8" )
      f9 = Ast.Frame( 2, "Domain=D9" )
      f10 = Ast.Frame( 2, "Domain=D10" )

      m2 = Ast.ZoomMap( 2, 2.0 )
      m3 = Ast.UnitMap( 2 )
      m4 = Ast.UnitMap( 2 )
      m5 = Ast.UnitMap( 2 )
      m6 = Ast.UnitMap( 2 )
      m7 = Ast.UnitMap( 2 )
      m8 = Ast.UnitMap( 2 )

      fs = Ast.FrameSet( f1 )
      fs.addframe( 1, m2, f2 )
      fs.addframe( 1, m3, f3 )
      fs.addframe( 3, m4, f4 )
      fs.addframe( 3, m5, f5 )
      fs.addframe( 4, m5, f6 )
      fs.addframe( 4, m6, f7 )
      fs.addframe( 6, m5, f8 )
      fs.addframe( 6, m6, f9 )
      fs.addframe( 6, m6, f10 )

      fs.removeframe( 6 )
      fs.removeframe( 3 )

      self.showObject( fs, "An example FrameSet", "Example" )
      self.exampleTab = self.tabs.currentIndex()

#  ----------------------------------------------------------------
   def showOpenFileDialog(self):
      fname = QFileDialog.getOpenFileName(self, 'Open file', '.')
      self.readFile( fname )


#  ----------------------------------------------------------------
   def showSettingsDialog(self):
      dialog = SettingsDialog( self, self.options, self )
      dialog.exec_()
      if self.redraw:
         self.drawFile()

#  ----------------------------------------------------------------
   def optionsPath(self):
      home = os.environ.get("HOME")
      if home:
         return "{}/{}".format( home, OPTIONS_FNAME )
      else:
         return ""

#  ----------------------------------------------------------------
   def loadOptions(self):
      self.options = {}
      self.options[ OPT_CHANGED ] = False

      for key in option_defs:
         self.options[ key ] = option_defs[ key ][ 1 ]

      opath = self.optionsPath()
      if opath:
         try:
            with open(opath) as file:
                for line in file:
                   line = line.strip()
                   if line and not line.startswith('#'):
                      if ':' in line:
                         (key,value) = line.split( ':', 1 )
                         if key in self.options:
                            self.options[key] = value
                         else:
                            print("!! Ignoring unknown key '{}' in astviewer "
                                  "options file ({}).".format(key,opath) )
                      else:
                         print("!! Ignoring bad line '{}' in astviewer "
                               "options file ({}).".format(line,opath) )


         except IOError:
            pass

#  ----------------------------------------------------------------
   def saveOptions(self):
      if self.options[ OPT_CHANGED ]:
         path = self.optionsPath()
         if path:
            with open(path,'w') as file:
               file.write( "# astviewer options file\n" )
               for key in self.options:
                  if key != OPT_CHANGED:
                     file.write( "{}:{}\n".format( key, self.options[key] ))


#  ----------------------------------------------------------------
   def closeEvent(self, event):
      self.saveOptions()
      event.accept()

#  ================================================================
if __name__ == "__main__":
   app = QtGui.QApplication(sys.argv)

   if len(sys.argv) > 1:
      infile = sys.argv[1]
   else:
      infile = None

   astview = AstViewer( infile )
   astview.show()
   sys.exit(app.exec_())


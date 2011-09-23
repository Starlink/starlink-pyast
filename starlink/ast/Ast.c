/* Issues:

   - providing more base methods (equal, etc)
   - are there any memory leaks (either in AST or Python)?
   - implement more methods and classes
   - is starlink.Ast.BAD (== AST__BAD) implemented in the best way?
   - needs proper docs

*/

#include <Python.h>
#include <string.h>
#include "numpy/arrayobject.h"
#include "ast.h"

/* Define the name of the package and module, and initialise the current
   class and method name so that we have something to undef. */
#define MODULE "starlink.Ast"
#define CLASS
#define NAME

/* Prototypes for local functions (need to come here since they may be
   referred to inside pyast.h). */
static PyArrayObject *GetArray( PyObject *object, int type, int append, int ndim, int *dims, const char *arg, const char *fun );
static PyArrayObject *GetArray1D( PyObject *object, int *dim, const char *arg, const char *fun );
static PyArrayObject *GetArray1I( PyObject *object, int *dim, const char *arg, const char *fun );
static PyObject *PyAst_FromString( const char *string );
static char *DumpToString( AstObject *object, const char *options );
static char *GetString( void *mem, PyObject *value );
static char *PyAst_ToString( PyObject *self );
static const char *AttNorm( const char *att, char *buff );
static void Sinka( const char *text );

/* Macros used in this file */
#define PYAST_MODULE
#include "star/pyast.h"

/* Include code that intercepts error reports issued by AST and raises
   appropriate Python exceptions instead. */
#include "exceptions.c"

/* Object */
/* ====== */

/* Define a string holding the fully qualified Python class name. */
#undef CLASS
#define CLASS MODULE ".Object"

/* Define the class structure */
typedef struct {
   PyObject_HEAD
   AstObject *ast_object;
} Object;

/* Prototypes for class functions */
static PyObject *NewObject( AstObject *this );
static PyObject *Object_clear( Object *self, PyObject *args );
static PyObject *Object_copy( Object *self );
static PyObject *Object_deepcopy( Object *self, PyObject *args );
static PyObject *Object_get( Object *self, PyObject *args );
static PyObject *Object_hasattribute( Object *self, PyObject *args );
static PyObject *Object_lock( Object *self, PyObject *args );
static PyObject *Object_repr( PyObject *self );
static PyObject *Object_same( Object *self, PyObject *args );
static PyObject *Object_set( Object *self, PyObject *args );
static PyObject *Object_show( Object *self );
static PyObject *Object_str( PyObject *self );
static PyObject *Object_test( Object *self, PyObject *args );
static PyObject *Object_unlock( Object *self, PyObject *args );
static PyTypeObject *GetType( AstObject *this );
static int SetProxy( AstObject *this, Object *self );
static void Object_dealloc( Object *self );

/* Class membership functions (probably not needed, but just in case).
These are functions of the base Object class since it should be possible
to test an object of any sub-class for membership of any other sub-class. */
MAKE_ISA(Box)
MAKE_ISA(Channel)
MAKE_ISA(Circle)
MAKE_ISA(CmpFrame)
MAKE_ISA(CmpMap)
MAKE_ISA(CmpRegion)
MAKE_ISA(DSBSpecFrame)
MAKE_ISA(Ellipse)
MAKE_ISA(FitsChan)
MAKE_ISA(FluxFrame)
MAKE_ISA(Frame)
MAKE_ISA(FrameSet)
MAKE_ISA(GrismMap)
MAKE_ISA(Interval)
MAKE_ISA(LutMap)
MAKE_ISA(Mapping)
MAKE_ISA(NormMap)
MAKE_ISA(NullRegion)
MAKE_ISA(Object)
MAKE_ISA(PcdMap)
MAKE_ISA(PermMap)
MAKE_ISA(Prism)
MAKE_ISA(RateMap)
MAKE_ISA(Region)
MAKE_ISA(ShiftMap)
MAKE_ISA(SkyFrame)
MAKE_ISA(SpecFluxFrame)
MAKE_ISA(SpecFrame)
MAKE_ISA(SphMap)
MAKE_ISA(StcsChan)
MAKE_ISA(TimeFrame)
MAKE_ISA(TimeMap)
MAKE_ISA(TranMap)
MAKE_ISA(UnitMap)
MAKE_ISA(WcsMap)
MAKE_ISA(WinMap)
MAKE_ISA(ZoomMap)

/* Describe the methods of the class */
static PyMethodDef Object_methods[] = {
   DEF_ISA(Box,box),
   DEF_ISA(Channel,channel),
   DEF_ISA(Circle,circle),
   DEF_ISA(CmpFrame,cmpframe),
   DEF_ISA(CmpMap,cmpmap),
   DEF_ISA(CmpRegion,cmpregion),
   DEF_ISA(DSBSpecFrame,dsbspecframe),
   DEF_ISA(Ellipse,ellipse),
   DEF_ISA(FitsChan,fitschan),
   DEF_ISA(FluxFrame,fluxframe),
   DEF_ISA(Frame,frame),
   DEF_ISA(FrameSet,frameset),
   DEF_ISA(GrismMap,grismmap),
   DEF_ISA(Interval,interval),
   DEF_ISA(LutMap,lutmap),
   DEF_ISA(Mapping,mapping),
   DEF_ISA(NormMap,normmap),
   DEF_ISA(NullRegion,nullregion),
   DEF_ISA(Object,object),
   DEF_ISA(PcdMap,pcdmap),
   DEF_ISA(PermMap,permmap),
   DEF_ISA(Prism,prism),
   DEF_ISA(RateMap,ratemap),
   DEF_ISA(Region,region),
   DEF_ISA(ShiftMap,shiftmap),
   DEF_ISA(SkyFrame,skyframe),
   DEF_ISA(SpecFluxFrame,specfluxframe),
   DEF_ISA(SpecFrame,specframe),
   DEF_ISA(SphMap,sphmap),
   DEF_ISA(StcsChan,stcschan),
   DEF_ISA(TimeFrame,timeframe),
   DEF_ISA(TimeMap,timemap),
   DEF_ISA(TranMap,tranmap),
   DEF_ISA(UnitMap,unitmap),
   DEF_ISA(WcsMap,wcsmap),
   DEF_ISA(WinMap,winmap),
   DEF_ISA(ZoomMap,zoommap),
   {"__deepcopy__", (PyCFunction)Object_deepcopy, METH_VARARGS, "Create a deep copy of an Object - used by the copy module"},
   {"clear", (PyCFunction)Object_clear, METH_VARARGS, "Clear attribute values for an Object"},
   {"copy", (PyCFunction)Object_copy, METH_NOARGS, "Create a deep copy of an Object"},
   {"get", (PyCFunction)Object_get, METH_VARARGS, "Get an attribute value for an Object as a string"},
   {"hasattribute", (PyCFunction)Object_hasattribute, METH_VARARGS, "Test if an Object has a named attribute"},
   {"lock", (PyCFunction)Object_lock, METH_VARARGS, "Lock an Object for exclusive use by the calling thread"},
   {"set", (PyCFunction)Object_set, METH_VARARGS, "Set attribute values for an Object"},
   {"show", (PyCFunction)Object_show, METH_NOARGS, "Show the structure of the Object on standard output"},
   {"same", (PyCFunction)Object_same, METH_VARARGS, "Test if two references refer to the same Object"},
   {"test", (PyCFunction)Object_test, METH_VARARGS, "Test if an Object attribute value is set"},
   {"unlock", (PyCFunction)Object_unlock, METH_VARARGS, "Unlock an Object for use by other threads."},
   {NULL, NULL, 0, NULL}  /* Sentinel */
};


/* Define the AST attributes of the class */
MAKE_GETROC(Object,  Class)
MAKE_GETSETC(Object, ID)
MAKE_GETSETC(Object, Ident)
MAKE_GETROI(Object,  Nobject)
MAKE_GETROI(Object,  ObjSize)
MAKE_GETROI(Object,  RefCount)
MAKE_GETSETL(Object, UseDefs)
static PyGetSetDef Object_getseters[] = {
   DEFATT(Class,"Object AST class name"),
   DEFATT(ID,"Object identification string"),
   DEFATT(Ident,"Permanent Object identification string"),
   DEFATT(Nobject,"Number of Objects in class"),
   DEFATT(ObjSize,"The in-memory size of the AST Object in bytes"),
   DEFATT(RefCount,"Count of active Object pointers"),
   DEFATT(UseDefs,"Use default values for unspecified attributes?"),
   {NULL, NULL, NULL, NULL, NULL}  /* Sentinel */
};

/* Define the class Python type structure */
static PyTypeObject ObjectType = {
   PyVarObject_HEAD_INIT(NULL, 0)
   CLASS,                     /* tp_name */
   sizeof(Object),            /* tp_basicsize */
   0,                         /* tp_itemsize */
   (destructor)Object_dealloc,/* tp_dealloc */
   0,                         /* tp_print */
   0,                         /* tp_getattr */
   0,                         /* tp_setattr */
   0,                         /* tp_reserved */
   Object_repr,               /* tp_repr */
   0,                         /* tp_as_number */
   0,                         /* tp_as_sequence */
   0,                         /* tp_as_mapping */
   0,                         /* tp_hash  */
   0,                         /* tp_call */
   Object_str,                /* tp_str */
   0,                         /* tp_getattro */
   0,                         /* tp_setattro */
   0,                         /* tp_as_buffer */
   Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE, /* tp_flags */
   "AST Object",              /* tp_doc */
   0,		              /* tp_traverse */
   0,		              /* tp_clear */
   0,		              /* tp_richcompare */
   0,		              /* tp_weaklistoffset */
   0,		              /* tp_iter */
   0,		              /* tp_iternext */
   Object_methods,            /* tp_methods */
   0,                         /* tp_members */
   Object_getseters,          /* tp_getset */
};


/* Define the class methods */
#undef NAME
#define NAME CLASS ".clear"
static PyObject *Object_clear( Object *self, PyObject *args ) {
   PyObject *result = NULL;
   const char *attrib;

   if( PyErr_Occurred() ) return NULL;

   if( PyArg_ParseTuple( args, "s:" NAME, &attrib ) ) {
      astClear( THIS, attrib);
      if( astOK ) result = Py_None;
   }
   TIDY;
   return result;
}

static PyObject *Object_copy( Object *self ) {
   PyObject *result = NULL;
   AstObject *new = astCopy( THIS );

   if( PyErr_Occurred() ) return NULL;

   if( astOK ) {
      result = NewObject( new );
      new = astAnnul( new );
   }
   TIDY;
   return result;
}

static void Object_dealloc( Object *self ) {
   if( THIS ) {
      astSetProxy( THIS, NULL );
      THIS = astAnnul( THIS );
   }
   Py_TYPE(self)->tp_free((PyObject*)self);
   TIDY;
}

static PyObject *Object_repr( PyObject *self ) {
   char *p1 = NULL;
   char *p2 = NULL;
   int nc = 0;
   PyObject *result = NULL;

   if( PyErr_Occurred() ) return NULL;

   if( THIS ) {
      p1 = DumpToString( THIS, "Comment=0,Full=-1" );
      p2 = astAppendString( p2, &nc, "<" );
      p2 = astAppendString( p2, &nc, p1 );
      p2 = astAppendString( p2, &nc, ">" );
      result = Py_BuildValue( "s", p2 );
      p2 = astFree( p2 );
      p1 = astFree( p1 );
   }

   TIDY;
   return result;
}

static PyObject *Object_str( PyObject *self ) {
   char *p1 = NULL;
   PyObject *result = NULL;

   if( PyErr_Occurred() ) return NULL;

   if( THIS ) {
      p1 = DumpToString( THIS, "Comment=1,Full=0" );
      result = Py_BuildValue( "s", p1 );
      p1 = astFree( p1 );
   }

   TIDY;
   return result;
}

static PyObject *Object_deepcopy( Object *self, PyObject *args ) {
   if( PyErr_Occurred() ) return NULL;
   return Object_copy( self );
}

#undef NAME
#define NAME CLASS ".get"
static PyObject *Object_get( Object *self, PyObject *args ) {
   PyObject *result = NULL;
   const char *attrib;
   const char *value;
   if( PyErr_Occurred() ) return NULL;
   if( PyArg_ParseTuple( args, "s:" NAME, &attrib ) ) {
      value = astGetC( THIS, attrib );
      if( astOK ) result = Py_BuildValue( "s", value );
   }
   TIDY;
   return result;
}

#undef NAME
#define NAME CLASS ".hasattribute"
static PyObject *Object_hasattribute( Object *self, PyObject *args ) {
   PyObject *result = NULL;
   const char *attrib;
   int value;
   if( PyErr_Occurred() ) return NULL;
   if( PyArg_ParseTuple( args, "s:" NAME, &attrib ) ) {
      value = astHasAttribute( THIS, attrib);
      if( astOK ) result = Py_BuildValue( "O", (value ? Py_True : Py_False) );
   }
   TIDY;
   return result;
}

#undef NAME
#define NAME CLASS ".lock"
static PyObject *Object_lock( Object *self, PyObject *args ) {
   PyObject *result = NULL;
   int wait;
   if( PyErr_Occurred() ) return NULL;
   if( PyArg_ParseTuple( args, "i:" NAME, &wait ) ) {
      astLock( THIS, wait );
      if( astOK ) result = Py_None;
   }
   TIDY;
   return result;
}

#undef NAME
#define NAME CLASS ".same"
static PyObject *Object_same( Object *self, PyObject *args ) {
   PyObject *result = NULL;
   Object *other;
   int value;
   if( PyErr_Occurred() ) return NULL;
   if( PyArg_ParseTuple( args, "O!:" NAME, &ObjectType,
                         (PyObject **) &other ) ) {
      value = astSame( THIS, THAT );
      if( astOK ) result = Py_BuildValue( "O", (value ?  Py_True : Py_False) );
   }
   TIDY;
   return result;
}

#undef NAME
#define NAME CLASS ".set"
static PyObject *Object_set( Object *self, PyObject *args ) {
   PyObject *result = NULL;
   const char *settings;
   if( PyErr_Occurred() ) return NULL;
   if( PyArg_ParseTuple( args, "s:" NAME, &settings ) ) {
      astSet( THIS, settings );
      if( astOK ) result = Py_None;
   }
   TIDY;
   return result;
}

/* Define the AST methods of the class. */
static PyObject *Object_show( Object *self ) {
   PyObject *result = NULL;
   if( PyErr_Occurred() ) return NULL;
   astShow( THIS );
   if( astOK ) result = Py_None;
   TIDY;
   return result;
}

#undef NAME
#define NAME CLASS ".test"
static PyObject *Object_test( Object *self, PyObject *args ) {
   PyObject *result = NULL;
   const char *attrib;
   int value;
   if( PyErr_Occurred() ) return NULL;
   if( PyArg_ParseTuple( args, "s:" NAME, &attrib ) ) {
      value = astTest( THIS, attrib);
      if( astOK ) result = Py_BuildValue( "O", (value ?  Py_True : Py_False));
   }
   TIDY;
   return result;
}

#undef NAME
#define NAME CLASS ".unlock"
static PyObject *Object_unlock( Object *self, PyObject *args ) {
   PyObject *result = NULL;
   int report;
   if( PyErr_Occurred() ) return NULL;
   if( PyArg_ParseTuple( args, "i:" NAME, &report ) ) {
      astUnlock( THIS, report );
      if( astOK ) result = Py_None;
   }
   TIDY;
   return result;
}



/* Mapping */
/* ======= */

/* Define a string holding the fully qualified Python class name. */
#undef CLASS
#define CLASS MODULE ".Mapping"

/* Define the class structure */
typedef struct {
   Object parent;
} Mapping;

/* Prototypes for class functions */
static PyObject *Mapping_decompose( Mapping *self );
static PyObject *Mapping_invert( Mapping *self );
static PyObject *Mapping_linearapprox( Mapping *self, PyObject *args );
static PyObject *Mapping_mapbox( Mapping *self, PyObject *args );
static PyObject *Mapping_quadapprox( Mapping *self, PyObject *args );
static PyObject *Mapping_rate( Mapping *self, PyObject *args );
static PyObject *Mapping_rebin( Mapping *self, PyObject *args );
static PyObject *Mapping_rebinseq( Mapping *self, PyObject *args );
static PyObject *Mapping_resample( Mapping *self, PyObject *args );
static PyObject *Mapping_removeregions( Mapping *self );
static PyObject *Mapping_simplify( Mapping *self );
static PyObject *Mapping_trangrid( Mapping *self, PyObject *args );
static PyObject *Mapping_trann( Mapping *self, PyObject *args );

/* Describe the methods of the class */
static PyMethodDef Mapping_methods[] = {
   {"decompose", (PyCFunction)Mapping_decompose, METH_NOARGS, "Decompose a Mapping into two component Mappings"},
   {"invert", (PyCFunction)Mapping_invert, METH_NOARGS, "Invert a Mapping"},
   {"mapbox", (PyCFunction)Mapping_mapbox, METH_VARARGS, " Find a bounding box for a Mapping."},
   {"linearapprox", (PyCFunction)Mapping_linearapprox, METH_VARARGS, "Obtain a linear approximation to a Mapping, if appropriate."},
   {"quadapprox", (PyCFunction)Mapping_quadapprox, METH_VARARGS, "Obtain a quadratic approximation to a 2D Mapping"},
   {"rate", (PyCFunction)Mapping_rate, METH_VARARGS, "Calculate the rate of change of a Mapping output"},
   {"rebin", (PyCFunction)Mapping_rebin, METH_VARARGS, "Rebin a region of a data grid"},
   {"rebinseq", (PyCFunction)Mapping_rebinseq, METH_VARARGS, "Rebin a region of a sequence of data grids"},
   {"resample", (PyCFunction)Mapping_resample, METH_VARARGS, "Resample a region of a data grid"},
   {"removeregions", (PyCFunction)Mapping_removeregions, METH_NOARGS, "Remove any Regions from a Mapping"},
   {"simplify", (PyCFunction)Mapping_simplify, METH_NOARGS, "Simplify a Mapping"},
   {"trann", (PyCFunction)Mapping_trann, METH_VARARGS, "Transform N-dimensional coordinates"},
   {"trangrid", (PyCFunction)Mapping_trangrid, METH_VARARGS, "Transform a grid of positions"},
   {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* Define the AST attributes of the class */
MAKE_GETSETI(Mapping, Invert)
MAKE_GETROL(Mapping,  IsLinear)
MAKE_GETROL(Mapping,  IsSimple)
MAKE_GETROI(Mapping,  Nin)
MAKE_GETROI(Mapping,  Nout)
MAKE_GETSETI(Mapping, Report)
MAKE_GETROL(Mapping,  TranForward)
MAKE_GETROL(Mapping,  TranInverse)
static PyGetSetDef Mapping_getseters[] = {
   DEFATT(Invert,"Mapping inversion flag"),
   DEFATT(IsLinear,"Is the Mapping linear?"),
   DEFATT(IsSimple,"Has the Mapping been simplified?"),
   DEFATT(Nin,"Number of input coordinates for a Mapping"),
   DEFATT(Nout,"Number of output coordinates for a Mapping"),
   DEFATT(Report,"Report transformed coordinates?"),
   DEFATT(TranForward,"Forward transformation defined?"),
   DEFATT(TranInverse,"Inverse transformation defined?"),
   {NULL, NULL, NULL, NULL, NULL}  /* Sentinel */
};

/* Define the class Python type structure */
static PyTypeObject MappingType = {
   PyVarObject_HEAD_INIT(NULL, 0)
   CLASS,                     /* tp_name */
   sizeof(Mapping),           /* tp_basicsize */
   0,                         /* tp_itemsize */
   0,                         /* tp_dealloc */
   0,                         /* tp_print */
   0,                         /* tp_getattr */
   0,                         /* tp_setattr */
   0,                         /* tp_reserved */
   0,                         /* tp_repr */
   0,                         /* tp_as_number */
   0,                         /* tp_as_sequence */
   0,                         /* tp_as_mapping */
   0,                         /* tp_hash  */
   0,                         /* tp_call */
   0,                         /* tp_str */
   0,                         /* tp_getattro */
   0,                         /* tp_setattro */
   0,                         /* tp_as_buffer */
   Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE, /* tp_flags */
   "AST Mapping",             /* tp_doc */
   0,		              /* tp_traverse */
   0,		              /* tp_clear */
   0,		              /* tp_richcompare */
   0,		              /* tp_weaklistoffset */
   0,		              /* tp_iter */
   0,		              /* tp_iternext */
   Mapping_methods,           /* tp_methods */
   0,                         /* tp_members */
   Mapping_getseters,         /* tp_getset */
};


/* Define the class methods */

static PyObject *Mapping_decompose( Mapping *self ) {
   PyObject *result = NULL;
   PyObject *map1_object = NULL;
   PyObject *map2_object = NULL;
   AstMapping *map1;
   AstMapping *map2;
   int series;
   int invert1;
   int invert2;

   if( PyErr_Occurred() ) return NULL;
   astDecompose( THIS, &map1, &map2, &series, &invert1, &invert2  );
   if( astOK ) {
      map1_object = NewObject( (AstObject *) map1 );
      map2_object = NewObject( (AstObject *) map2 );
      if (map1_object && map2_object) {
        result = Py_BuildValue( "OOiii", map1_object, map2_object, series,
                                invert1, invert2 );
      }
      Py_XDECREF(map1_object);
      Py_XDECREF(map2_object);
   }
   if( map1 ) map1 = astAnnul( map1 );
   if( map2 ) map2 = astAnnul( map2 );

   TIDY;
   return result;
}

static PyObject *Mapping_invert( Mapping *self ) {
   PyObject *result = NULL;
   if( PyErr_Occurred() ) return NULL;
   astInvert( THIS );
   if( astOK ) result = Py_None;
   TIDY;
   return result;
}

#undef NAME
#define NAME CLASS ".linearapprox"
static PyObject *Mapping_linearapprox( Mapping *self, PyObject *args ) {
   PyObject *result = NULL;
   PyObject *islinear = NULL;
   PyArrayObject *fit = NULL;
   PyArrayObject *lbnd = NULL;
   PyArrayObject *ubnd = NULL;
   PyObject *lbnd_object = NULL;
   PyObject *ubnd_object = NULL;
   double tol;
   int ncoord_in;
   int ncoord_out;
   npy_intp dims[2];

   if( PyErr_Occurred() ) return NULL;
   ncoord_in = astGetI( THIS, "Nin" );
   ncoord_out = astGetI( THIS, "Nin" );
   if( PyArg_ParseTuple( args, "OOd:" NAME, &lbnd_object, &ubnd_object,
                         &tol ) && astOK ) {
      lbnd = GetArray1D( lbnd_object, &ncoord_in, "lbnd", NAME );
      ubnd = GetArray1D( ubnd_object, &ncoord_in, "ubnd", NAME );
      if( lbnd && ubnd ) {
         dims[ 0 ] = ( ncoord_in + 1 )*ncoord_out;
         fit = (PyArrayObject *) PyArray_SimpleNew( 1, dims, PyArray_DOUBLE );
         if( fit ) {
            islinear = astLinearApprox( THIS, (const double *)lbnd->data,
                                        (const double *)ubnd->data, tol,
                                        (double *)fit->data ) ? Py_True : Py_False;
            if( astOK ) result = Py_BuildValue( "OO", islinear, fit );
            Py_XDECREF( islinear );
            Py_DECREF( fit );
         }
      }
      Py_XDECREF( lbnd );
      Py_XDECREF( ubnd );
   }

   TIDY;
   return result;
}

#undef NAME
#define NAME CLASS ".mapbox"
static PyObject *Mapping_mapbox( Mapping *self, PyObject *args ) {
   PyArrayObject *lbnd_in = NULL;
   PyArrayObject *ubnd_in = NULL;
   PyArrayObject *xl = NULL;
   PyArrayObject *xu = NULL;
   PyObject *lbnd_in_object = NULL;
   PyObject *result = NULL;
   PyObject *ubnd_in_object = NULL;
   double lbnd_out;
   double ubnd_out;
   int coord_out;
   int forward;
   int ncoord_in;
   npy_intp dims[1];

   if( PyErr_Occurred() ) return NULL;
   ncoord_in = astGetI( THIS, "Nin" );
   if( PyArg_ParseTuple( args, "OOii:" NAME, &lbnd_in_object, &ubnd_in_object,
                         &forward, &coord_out ) && astOK ) {
      lbnd_in = GetArray1D( lbnd_in_object, &ncoord_in, "lbnd_in", NAME );
      ubnd_in = GetArray1D( ubnd_in_object, &ncoord_in, "ubnd_in", NAME );
      if( lbnd_in && ubnd_in ) {
         dims[ 0 ] = ncoord_in;
         xl = (PyArrayObject *) PyArray_SimpleNew( 1, dims, PyArray_DOUBLE );
         xu = (PyArrayObject *) PyArray_SimpleNew( 1, dims, PyArray_DOUBLE );
         if( xl && xu ) {
            astMapBox( THIS, (const double *)lbnd_in->data,
                       (const double *)ubnd_in->data, forward, coord_out,
                        &lbnd_out, &ubnd_out, (double *)xl->data,
                       (double *)xu->data );
            if( astOK ) result = Py_BuildValue( "ddOO", lbnd_out, ubnd_out,
                                                xl, xu );
         }
         Py_XDECREF( xl );
         Py_XDECREF( xu );
      }
      Py_XDECREF( lbnd_in );
      Py_XDECREF( ubnd_in );
   }

   TIDY;
   return result;
}

#undef NAME
#define NAME CLASS ".quadapprox"
static PyObject *Mapping_quadapprox( Mapping *self, PyObject *args ) {
   PyObject *result = NULL;
   PyObject *isquad;
   PyArrayObject *fit = NULL;
   PyArrayObject *lbnd = NULL;
   PyArrayObject *ubnd = NULL;
   PyObject *lbnd_object = NULL;
   PyObject *ubnd_object = NULL;
   double rms;
   int ncoord_in;
   int ncoord_out;
   npy_intp dims[2];
   int nx;
   int ny;

   if( PyErr_Occurred() ) return NULL;
   ncoord_in = astGetI( THIS, "Nin" );
   ncoord_out = astGetI( THIS, "Nin" );
   if( PyArg_ParseTuple( args, "OOii:" NAME, &lbnd_object, &ubnd_object, &nx,
                         &ny ) && astOK ) {
      lbnd = GetArray1D( lbnd_object, &ncoord_in, "lbnd", NAME );
      ubnd = GetArray1D( ubnd_object, &ncoord_in, "ubnd", NAME );
      if( lbnd && ubnd ) {
         dims[ 0 ] = 6*ncoord_out;
         fit = (PyArrayObject *) PyArray_SimpleNew( 1, dims, PyArray_DOUBLE );
         if( fit ) {
            isquad = astQuadApprox( THIS, (const double *)lbnd->data,
                                    (const double *)ubnd->data, nx, ny,
                                    (double *)fit->data, &rms ) ? Py_True : Py_False;
            if( astOK ) result = Py_BuildValue( "OOd", isquad, fit, rms );
            Py_XDECREF( isquad );
            Py_DECREF( fit );
         }
      }
      Py_XDECREF( lbnd );
      Py_XDECREF( ubnd );
   }

   TIDY;
   return result;
}

#undef NAME
#define NAME CLASS ".rate"
static PyObject *Mapping_rate( Mapping *self, PyObject *args ) {
   PyObject *result = NULL;
   PyArrayObject *at = NULL;
   PyObject *at_object = NULL;
   int ax1;
   int ax2;
   int ncoord_in;
   double value;

   if( PyErr_Occurred() ) return NULL;
   ncoord_in = astGetI( THIS, "Nin" );
   if( PyArg_ParseTuple( args, "Oii:" NAME, &at_object, &ax1, &ax2)
       && astOK ) {
      at = GetArray1D( at_object, &ncoord_in, "at", NAME );
      if( at ) {
         value = astRate( THIS, (double *)at->data, ax1, ax2 );
         if( astOK ) result = Py_BuildValue( "d", value );
      }
      Py_XDECREF( at );
   }

   TIDY;
   return result;
}

#undef NAME
#define NAME CLASS ".rebin"
static PyObject *Mapping_rebin( Mapping *self, PyObject *args ) {
   PyArrayObject *in = NULL;
   PyArrayObject *in_var = NULL;
   PyArrayObject *lbnd = NULL;
   PyArrayObject *lbnd_in = NULL;
   PyArrayObject *lbnd_out = NULL;
   PyArrayObject *out = NULL;
   PyArrayObject *out_var = NULL;
   PyArrayObject *params = NULL;
   PyArrayObject *ubnd = NULL;
   PyArrayObject *ubnd_in = NULL;
   PyArrayObject *ubnd_out = NULL;
   PyObject *in_object = NULL;
   PyObject *in_var_object = NULL;
   PyObject *lbnd_in_object = NULL;
   PyObject *lbnd_object = NULL;
   PyObject *lbnd_out_object = NULL;
   PyObject *params_object = NULL;
   PyObject *result = NULL;
   PyObject *ubnd_in_object = NULL;
   PyObject *ubnd_object = NULL;
   PyObject *ubnd_out_object = NULL;
   char buf[200];
   char format[] = "dOOOOiOididOOOO:" NAME;
   double badval_d;
   double tol;
   double wlim;
   float badval_f;
   int badval_i;
   int dims[ MXDIM ];
   int flags;
   int i;
   int maxpix;
   int ncoord_in;
   int ncoord_out;
   int ndim = 0;
   int nparam;
   int spread;
   int type = 0;
   npy_intp *pdims = NULL;
   void *pbadval = NULL;

   if( PyErr_Occurred() ) return NULL;

/* Get the number of inputs and outputs for the Mapping */
   ncoord_in = astGetI( THIS, "Nin" );
   ncoord_out = astGetI( THIS, "Nin" );

/* We do not know yet what format code to use for badval. We need to parse
   the arguments twice. The first time, we determine the data type from
   the "in" array. This allows us to choose the correct format code for
   badval, so we then parse the arguments a second time, using the
   correct code. */
   if( PyArg_ParseTuple( args, format, &wlim, &lbnd_in_object,
                         &ubnd_in_object, &in_object, &in_var_object,
                         &spread, &params_object, &flags, &tol, &maxpix,
                         &badval_d, &lbnd_out_object, &ubnd_out_object,
                         &lbnd_object, &ubnd_object ) && astOK ) {

      type = ((PyArrayObject*) in_object)->descr->type_num;
      if( type == PyArray_DOUBLE ) {
         format[ 10 ] = 'd';
         pbadval = &badval_d;
      } else if( type == PyArray_FLOAT ) {
         format[ 10 ] = 'f';
         pbadval = &badval_f;
      } else if( type == PyArray_INT ) {
         format[ 10 ] = 'i';
         pbadval = &badval_i;
      } else {
         PyErr_SetString( PyExc_ValueError, "The 'in' array supplied "
                          "to " NAME " has a data type that is not "
                          "supported by " NAME " (must be float64, "
                          "float32 or int32)." );
      }

/* Also record the number of axes and dimensions in the input array. */
      ndim = ((PyArrayObject*) in_object)->nd;
      pdims = ((PyArrayObject*) in_object)->dimensions;
      if( ndim > MXDIM ) {
         sprintf( buf, "The 'in' array supplied to " NAME " has too "
                  "many (%d) dimensions (must be no more than %d).",
                  ndim, MXDIM );
         PyErr_SetString( PyExc_ValueError, buf );
         pbadval = NULL;
      } else {
         for( i = 0; i < ndim; i++ ) {
            dims[ i ] = pdims[ i ];
         }
      }
   }

/* Parse the arguments again, this time with the correct code for
   badval. */
   if( PyArg_ParseTuple( args, format, &wlim, &lbnd_in_object,
                         &ubnd_in_object, &in_object, &in_var_object,
                         &spread, &params_object, &flags, &tol, &maxpix,
                         pbadval, &lbnd_out_object, &ubnd_out_object,
                         &lbnd_object, &ubnd_object ) && pbadval ) {

      lbnd_in = GetArray1I( lbnd_in_object, &ncoord_in, "lbnd_in", NAME );
      ubnd_in = GetArray1I( ubnd_in_object, &ncoord_in, "ubnd_in", NAME );

      in = GetArray( in_object, type, 1, ndim, dims, "in", NAME );
      if( in_var_object != Py_None ) {
         in_var = GetArray( in_var_object, type, 1, ndim, dims, "in_var", NAME );
      }

      if( params_object != Py_None ) {
         nparam = 0;
         params = GetArray1D( params_object, &nparam, "params", NAME );
      }

      lbnd_out = GetArray1I( lbnd_out_object, &ncoord_out, "lbnd_out", NAME );
      ubnd_out = GetArray1I( ubnd_out_object, &ncoord_out, "ubnd_out", NAME );

      lbnd = GetArray1I( lbnd_object, &ncoord_in, "lbnd", NAME );
      ubnd = GetArray1I( ubnd_object, &ncoord_in, "ubnd", NAME );

      if( lbnd_in && ubnd_in && lbnd_out && ubnd_out && lbnd && ubnd && in ) {

         out = (PyArrayObject *) PyArray_SimpleNew( ndim, pdims, type );
         if( in_var ) out_var = (PyArrayObject *) PyArray_SimpleNew( ndim,
                                                                 pdims, type );

         if( out && ( ( in_var && out_var ) || !in_var ) ) {

            if( type == PyArray_DOUBLE ) {
               astRebinD( THIS, wlim, ncoord_in, (const int *)lbnd_in->data,
                          (const int *)ubnd_in->data, (const double *)in->data,
                          (in_var ? (const double *)in_var->data : NULL),
                          spread, (params ? (const double *)params->data : NULL),
                          flags, tol, maxpix, badval_d, ncoord_out,
                          (const int *)lbnd_out->data, (const int *)ubnd_out->data,
                          (const int *)lbnd->data, (const int *)ubnd->data,
                          (double *)out->data,
                          (out_var ? (double *)out_var->data : NULL ) );
            } else if( type == PyArray_FLOAT ) {
               astRebinF( THIS, wlim, ncoord_in, (const int *)lbnd_in->data,
                          (const int *)ubnd_in->data, (const float *)in->data,
                          (in_var ? (const float *)in_var->data : NULL),
                          spread, (params ? (const double *)params->data : NULL),
                          flags, tol, maxpix, badval_f, ncoord_out,
                          (const int *)lbnd_out->data, (const int *)ubnd_out->data,
                          (const int *)lbnd->data, (const int *)ubnd->data,
                          (float *)out->data,
                          (out_var ? (float *)out_var->data : NULL ) );
            } else {
               astRebinI( THIS, wlim, ncoord_in, (const int *)lbnd_in->data,
                          (const int *)ubnd_in->data, (const int *)in->data,
                          (in_var ? (const int *)in_var->data : NULL),
                          spread, (params ? (const double *)params->data : NULL),
                          flags, tol, maxpix, badval_i, ncoord_out,
                          (const int *)lbnd_out->data, (const int *)ubnd_out->data,
                          (const int *)lbnd->data, (const int *)ubnd->data,
                          (int *)out->data,
                          (out_var ? (int *)out_var->data : NULL ) );
            }

            if( astOK ) {
               if( !out_var ) out_var = (PyArrayObject *) Py_None;
               result = Py_BuildValue( "OO", out, out_var );
            }
         }

         Py_XDECREF( out );
         Py_XDECREF( out_var );
      }

      Py_XDECREF( lbnd );
      Py_XDECREF( ubnd );
      Py_XDECREF( lbnd_in );
      Py_XDECREF( ubnd_in );
      Py_XDECREF( lbnd_out );
      Py_XDECREF( ubnd_out );
      Py_XDECREF( in );
      Py_XDECREF( in_var );
   }

   TIDY;
   return result;
}

#undef NAME
#define NAME CLASS ".rebinseq"
static PyObject *Mapping_rebinseq( Mapping *self, PyObject *args ) {
   PyArrayObject *in = NULL;
   PyArrayObject *in_var = NULL;
   PyArrayObject *lbnd = NULL;
   PyArrayObject *lbnd_in = NULL;
   PyArrayObject *lbnd_out = NULL;
   PyArrayObject *out = NULL;
   PyArrayObject *out_var = NULL;
   PyArrayObject *params = NULL;
   PyArrayObject *ubnd = NULL;
   PyArrayObject *ubnd_in = NULL;
   PyArrayObject *ubnd_out = NULL;
   PyArrayObject *weights = NULL;
   PyObject *in_object = NULL;
   PyObject *in_var_object = NULL;
   PyObject *lbnd_in_object = NULL;
   PyObject *lbnd_object = NULL;
   PyObject *lbnd_out_object = NULL;
   PyObject *out_object = NULL;
   PyObject *out_var_object = NULL;
   PyObject *params_object = NULL;
   PyObject *result = NULL;
   PyObject *ubnd_in_object = NULL;
   PyObject *ubnd_object = NULL;
   PyObject *ubnd_out_object = NULL;
   PyObject *weights_object = NULL;
   char buf[200];
   char format[] = "dOOOOiOididOOOOOOOi:" NAME;
   double badval_d;
   double tol;
   double wlim;
   float badval_f;
   int badval_i;
   int dims[ MXDIM ];
   int flags;
   int i;
   int maxpix;
   int ncoord_in;
   int ncoord_out;
   int ndim = 0;
   int nparam;
   int nused;
   int spread;
   int type = 0;
   int wdims[ MXDIM + 1 ];
   npy_intp *pdims = NULL;
   void *pbadval = NULL;

   if( PyErr_Occurred() ) return NULL;

/* Get the number of inputs and outputs for the Mapping */
   ncoord_in = astGetI( THIS, "Nin" );
   ncoord_out = astGetI( THIS, "Nin" );

/* We do not know yet what format code to use for badval. We need to parse
   the arguments twice. The first time, we determine the data type from
   the "in" array. This allows us to choose the correct format code for
   badval, so we then parse the arguments a second time, using the
   correct code. */
   if( PyArg_ParseTuple( args, format, &wlim, &lbnd_in_object,
                         &ubnd_in_object, &in_object, &in_var_object,
                         &spread, &params_object, &flags, &tol, &maxpix,
                         &badval_d, &lbnd_out_object, &ubnd_out_object,
                         &lbnd_object, &ubnd_object, &out_object,
                         &out_var_object, &weights_object, &nused ) && astOK ) {

      type = ((PyArrayObject*) in_object)->descr->type_num;
      if( type == PyArray_DOUBLE ) {
         format[ 10 ] = 'd';
         pbadval = &badval_d;
      } else if( type == PyArray_FLOAT ) {
         format[ 10 ] = 'f';
         pbadval = &badval_f;
      } else if( type == PyArray_INT ) {
         format[ 10 ] = 'i';
         pbadval = &badval_i;
      } else {
         PyErr_SetString( PyExc_ValueError, "The 'in' array supplied "
                          "to " NAME " has a data type that is not "
                          "supported by " NAME " (must be float64, "
                          "float32 or int32)." );
      }

/* Also record the number of axes and dimensions in the input array. */
      ndim = ((PyArrayObject*) in_object)->nd;
      pdims = ((PyArrayObject*) in_object)->dimensions;
      if( ndim > MXDIM ) {
         sprintf( buf, "The 'in' array supplied to " NAME " has too "
                  "many (%d) dimensions (must be no more than %d).",
                  ndim, MXDIM );
         PyErr_SetString( PyExc_ValueError, buf );
         pbadval = NULL;
      } else {
         for( i = 0; i < ndim; i++ ) {
            dims[ i ] = pdims[ i ];
         }
      }

/* Report an error if the weights array is not double. */
      if( ((PyArrayObject*) weights_object)->descr->type_num != PyArray_DOUBLE ) {
         PyErr_SetString( PyExc_ValueError, "The 'weights' array supplied to "
                          NAME " is not of type float64." );
         pbadval = NULL;
      }
   }

/* Parse the arguments again, this time with the correct code for
   badval. */
   if( PyArg_ParseTuple( args, format, &wlim, &lbnd_in_object,
                         &ubnd_in_object, &in_object, &in_var_object,
                         &spread, &params_object, &flags, &tol, &maxpix,
                         pbadval, &lbnd_out_object, &ubnd_out_object,
                         &lbnd_object, &ubnd_object, &out_object,
                         &out_var_object, &weights_object, &nused ) && pbadval ) {

      lbnd_in = GetArray1I( lbnd_in_object, &ncoord_in, "lbnd_in", NAME );
      ubnd_in = GetArray1I( ubnd_in_object, &ncoord_in, "ubnd_in", NAME );

      in = GetArray( in_object, type, 1, ndim, dims, "in", NAME );
      if( in_var_object != Py_None ) {
         in_var = GetArray( in_var_object, type, 1, ndim, dims, "in_var", NAME );
      }

      if( params_object != Py_None ) {
         nparam = 0;
         params = GetArray1D( params_object, &nparam, "params", NAME );
      }

      lbnd_out = GetArray1I( lbnd_out_object, &ncoord_out, "lbnd_out", NAME );
      ubnd_out = GetArray1I( ubnd_out_object, &ncoord_out, "ubnd_out", NAME );

      lbnd = GetArray1I( lbnd_object, &ncoord_in, "lbnd", NAME );
      ubnd = GetArray1I( ubnd_object, &ncoord_in, "ubnd", NAME );

      out = GetArray( out_object, type, 1, ndim, dims, "out", NAME );
      if( out_var_object != Py_None ) {
         out_var = GetArray( out_var_object, type, 1, ndim, dims, "out_var", NAME );
      }

      if( flags & AST__GENVAR ) {
         wdims[ 0 ] = 2;
         for( i = 0; i < ndim; i++ ) {
            wdims[ i + 1 ] = dims[ i ];
         }
         weights = GetArray( weights_object, PyArray_DOUBLE, 1, ndim + 1,
                             wdims, "weights", NAME );
      } else {
         weights = GetArray( weights_object, PyArray_DOUBLE, 1, ndim,
                             dims, "weights", NAME );
      }

      if( lbnd_in && ubnd_in && lbnd_out && ubnd_out && lbnd && ubnd &&
          in && out && weights ) {

         if( type == PyArray_DOUBLE ) {
            astRebinSeqD( THIS, wlim, ncoord_in, (const int *)lbnd_in->data,
                       (const int *)ubnd_in->data, (const double *)in->data,
                       (in_var ? (const double *)in_var->data : NULL),
                       spread, (params ? (const double *)params->data : NULL),
                       flags, tol, maxpix, badval_d, ncoord_out,
                       (const int *)lbnd_out->data, (const int *)ubnd_out->data,
                       (const int *)lbnd->data, (const int *)ubnd->data,
                       (double *)out->data,
                       (out_var ? (double *)out_var->data : NULL ),
                       (double *)weights->data, &nused );

         } else if( type == PyArray_FLOAT ) {
            astRebinSeqF( THIS, wlim, ncoord_in, (const int *)lbnd_in->data,
                       (const int *)ubnd_in->data, (const float *)in->data,
                       (in_var ? (const float *)in_var->data : NULL),
                       spread, (params ? (const double *)params->data : NULL),
                       flags, tol, maxpix, badval_f, ncoord_out,
                       (const int *)lbnd_out->data, (const int *)ubnd_out->data,
                       (const int *)lbnd->data, (const int *)ubnd->data,
                       (float *)out->data,
                       (out_var ? (float *)out_var->data : NULL ),
                       (double *)weights->data, &nused );
         } else {
            astRebinSeqI( THIS, wlim, ncoord_in, (const int *)lbnd_in->data,
                       (const int *)ubnd_in->data, (const int *)in->data,
                       (in_var ? (const int *)in_var->data : NULL),
                       spread, (params ? (const double *)params->data : NULL),
                       flags, tol, maxpix, badval_i, ncoord_out,
                       (const int *)lbnd_out->data, (const int *)ubnd_out->data,
                       (const int *)lbnd->data, (const int *)ubnd->data,
                       (int *)out->data,
                       (out_var ? (int *)out_var->data : NULL ),
                       (double *)weights->data, &nused );
         }

         if( astOK ) {
            if( !out_var ) out_var = (PyArrayObject *) Py_None;
            result = Py_BuildValue( "i", nused );
         }

      }

      Py_XDECREF( lbnd );
      Py_XDECREF( ubnd );
      Py_XDECREF( lbnd_in );
      Py_XDECREF( ubnd_in );
      Py_XDECREF( lbnd_out );
      Py_XDECREF( ubnd_out );
      Py_XDECREF( in );
      Py_XDECREF( in_var );
      Py_XDECREF( out );
      Py_XDECREF( out_var );
      Py_XDECREF( weights );
   }

   TIDY;
   return result;
}

#undef NAME
#define NAME CLASS ".resample"
static PyObject *Mapping_resample( Mapping *self, PyObject *args ) {
   PyArrayObject *in = NULL;
   PyArrayObject *in_var = NULL;
   PyArrayObject *lbnd = NULL;
   PyArrayObject *lbnd_in = NULL;
   PyArrayObject *lbnd_out = NULL;
   PyArrayObject *out = NULL;
   PyArrayObject *out_var = NULL;
   PyArrayObject *params = NULL;
   PyArrayObject *ubnd = NULL;
   PyArrayObject *ubnd_in = NULL;
   PyArrayObject *ubnd_out = NULL;
   PyObject *in_object = NULL;
   PyObject *in_var_object = NULL;
   PyObject *lbnd_in_object = NULL;
   PyObject *lbnd_object = NULL;
   PyObject *lbnd_out_object = NULL;
   PyObject *params_object = NULL;
   PyObject *result = NULL;
   PyObject *ubnd_in_object = NULL;
   PyObject *ubnd_object = NULL;
   PyObject *ubnd_out_object = NULL;
   char badval_b;
   char buf[200];
   char format[] = "OOOOiOididOOOO:" NAME;
   double badval_d;
   double tol;
   float badval_f;
   int badval_i;
   int dims[ MXDIM ];
   int flags;
   int i;
   int interp;
   int maxpix;
   int ncoord_in;
   int ncoord_out;
   int ndim = 0;
   int noutpix = 0;
   int nparam;
   int type = 0;
   npy_intp *pdims = NULL;
   short int badval_h;
   unsigned char badval_B;
   unsigned int badval_I;
   unsigned short int badval_H;
   void *pbadval = NULL;

   if( PyErr_Occurred() ) return NULL;

/* Get the number of inputs and outputs for the Mapping */
   ncoord_in = astGetI( THIS, "Nin" );
   ncoord_out = astGetI( THIS, "Nin" );

/* We do not know yet what format code to use for badval. We need to parse
   the arguments twice. The first time, we determine the data type from
   the "in" array. This allows us to choose the correct format code for
   badval, so we then parse the arguments a second time, using the
   correct code. */
   if( PyArg_ParseTuple( args, format, &lbnd_in_object,
                         &ubnd_in_object, &in_object, &in_var_object,
                         &interp, &params_object, &flags, &tol, &maxpix,
                         &badval_d, &lbnd_out_object, &ubnd_out_object,
                         &lbnd_object, &ubnd_object ) && astOK ) {

      type = ((PyArrayObject*) in_object)->descr->type_num;
      if( type == PyArray_DOUBLE ) {
         format[ 9 ] = 'd';
         pbadval = &badval_d;
      } else if( type == PyArray_FLOAT ) {
         format[ 9 ] = 'f';
         pbadval = &badval_f;
      } else if( type == PyArray_INT ) {
         format[ 9 ] = 'i';
         pbadval = &badval_i;
      } else if( type == PyArray_SHORT ) {
         format[ 9 ] = 'h';
         pbadval = &badval_h;
      } else if( type == PyArray_BYTE ) {
         format[ 9 ] = 'b';
         pbadval = &badval_b;
      } else if( type == PyArray_UINT ) {
         format[ 9 ] = 'I';
         pbadval = &badval_I;
      } else if( type == PyArray_USHORT ) {
         format[ 9 ] = 'H';
         pbadval = &badval_H;
      } else if( type == PyArray_UBYTE ) {
         format[ 9 ] = 'B';
         pbadval = &badval_B;
      } else {
         PyErr_SetString( PyExc_ValueError, "The 'in' array supplied "
                          "to " NAME " has a data type that is not "
                          "supported by " NAME "." );
      }

/* Also record the number of axes and dimensions in the input array. */
      ndim = ((PyArrayObject*) in_object)->nd;
      pdims = ((PyArrayObject*) in_object)->dimensions;
      if( ndim > MXDIM ) {
         sprintf( buf, "The 'in' array supplied to " NAME " has too "
                  "many (%d) dimensions (must be no more than %d).",
                  ndim, MXDIM );
         PyErr_SetString( PyExc_ValueError, buf );
         pbadval = NULL;
      } else {
         for( i = 0; i < ndim; i++ ) {
            dims[ i ] = pdims[ i ];
         }
      }
   }

/* Parse the arguments again, this time with the correct code for
   badval. */
   if( PyArg_ParseTuple( args, format, &lbnd_in_object,
                         &ubnd_in_object, &in_object, &in_var_object,
                         &interp, &params_object, &flags, &tol, &maxpix,
                         pbadval, &lbnd_out_object, &ubnd_out_object,
                         &lbnd_object, &ubnd_object ) && pbadval ) {

      lbnd_in = GetArray1I( lbnd_in_object, &ncoord_in, "lbnd_in", NAME );
      ubnd_in = GetArray1I( ubnd_in_object, &ncoord_in, "ubnd_in", NAME );

      in = GetArray( in_object, type, 1, ndim, dims, "in", NAME );
      if( in_var_object != Py_None ) {
         in_var = GetArray( in_var_object, type, 1, ndim, dims, "in_var", NAME );
      }

      if( params_object != Py_None ) {
         nparam = 0;
         params = GetArray1D( params_object, &nparam, "params", NAME );
      }

      lbnd_out = GetArray1I( lbnd_out_object, &ncoord_out, "lbnd_out", NAME );
      ubnd_out = GetArray1I( ubnd_out_object, &ncoord_out, "ubnd_out", NAME );

      lbnd = GetArray1I( lbnd_object, &ncoord_in, "lbnd", NAME );
      ubnd = GetArray1I( ubnd_object, &ncoord_in, "ubnd", NAME );

      if( lbnd_in && ubnd_in && lbnd_out && ubnd_out && lbnd && ubnd && in ) {

         out = (PyArrayObject *) PyArray_SimpleNew( ndim, pdims, type );
         if( in_var ) out_var = (PyArrayObject *) PyArray_SimpleNew( ndim,
                                                                 pdims, type );

         if( out && ( ( in_var && out_var ) || !in_var ) ) {

            if( type == PyArray_DOUBLE ) {
               noutpix = astResampleD( THIS, ncoord_in, (const int *)lbnd_in->data,
                          (const int *)ubnd_in->data, (const double *)in->data,
                          (in_var ? (const double *)in_var->data : NULL),
                          interp, NULL, (params ? (const double *)params->data : NULL),
                          flags, tol, maxpix, badval_d, ncoord_out,
                          (const int *)lbnd_out->data, (const int *)ubnd_out->data,
                          (const int *)lbnd->data, (const int *)ubnd->data,
                          (double *)out->data,
                          (out_var ? (double *)out_var->data : NULL ) );
            } else if( type == PyArray_FLOAT ) {
               noutpix = astResampleF( THIS, ncoord_in, (const int *)lbnd_in->data,
                          (const int *)ubnd_in->data, (const float *)in->data,
                          (in_var ? (const float *)in_var->data : NULL),
                          interp, NULL, (params ? (const double *)params->data : NULL),
                          flags, tol, maxpix, badval_f, ncoord_out,
                          (const int *)lbnd_out->data, (const int *)ubnd_out->data,
                          (const int *)lbnd->data, (const int *)ubnd->data,
                          (float *)out->data,
                          (out_var ? (float *)out_var->data : NULL ) );
            } else if( type == PyArray_INT ) {
               noutpix = astResampleI( THIS, ncoord_in, (const int *)lbnd_in->data,
                          (const int *)ubnd_in->data, (const int *)in->data,
                          (in_var ? (const int *)in_var->data : NULL),
                          interp, NULL, (params ? (const double *)params->data : NULL),
                          flags, tol, maxpix, badval_i, ncoord_out,
                          (const int *)lbnd_out->data, (const int *)ubnd_out->data,
                          (const int *)lbnd->data, (const int *)ubnd->data,
                          (int *)out->data,
                          (out_var ? (int *)out_var->data : NULL ) );
            } else if( type == PyArray_SHORT ) {
               noutpix = astResampleS( THIS, ncoord_in, (const int *)lbnd_in->data,
                          (const int *)ubnd_in->data, (const short int *)in->data,
                          (in_var ? (const short int *)in_var->data : NULL),
                          interp, NULL, (params ? (const double *)params->data : NULL),
                          flags, tol, maxpix, badval_f, ncoord_out,
                          (const int *)lbnd_out->data, (const int *)ubnd_out->data,
                          (const int *)lbnd->data, (const int *)ubnd->data,
                          (short int *)out->data,
                          (out_var ? (short int *)out_var->data : NULL ) );
            } else if( type == PyArray_BYTE ) {
               noutpix = astResampleB( THIS, ncoord_in, (const int *)lbnd_in->data,
                          (const int *)ubnd_in->data, (const signed char *)in->data,
                          (in_var ? (const signed char *)in_var->data : NULL),
                          interp, NULL, (params ? (const double *)params->data : NULL),
                          flags, tol, maxpix, badval_f, ncoord_out,
                          (const int *)lbnd_out->data, (const int *)ubnd_out->data,
                          (const int *)lbnd->data, (const int *)ubnd->data,
                          (signed char *)out->data,
                          (out_var ? (signed char *)out_var->data : NULL ) );
            } else if( type == PyArray_UINT ) {
               noutpix = astResampleUI( THIS, ncoord_in, (const int *)lbnd_in->data,
                          (const int *)ubnd_in->data, (const unsigned int *)in->data,
                          (in_var ? (const unsigned int *)in_var->data : NULL),
                          interp, NULL, (params ? (const double *)params->data : NULL),
                          flags, tol, maxpix, badval_i, ncoord_out,
                          (const int *)lbnd_out->data, (const int *)ubnd_out->data,
                          (const int *)lbnd->data, (const int *)ubnd->data,
                          (unsigned int *)out->data,
                          (out_var ? (unsigned int *)out_var->data : NULL ) );
            } else if( type == PyArray_USHORT ) {
               noutpix = astResampleUS( THIS, ncoord_in, (const int *)lbnd_in->data,
                          (const int *)ubnd_in->data, (const unsigned short int *)in->data,
                          (in_var ? (const unsigned short int *)in_var->data : NULL),
                          interp, NULL, (params ? (const double *)params->data : NULL),
                          flags, tol, maxpix, badval_f, ncoord_out,
                          (const int *)lbnd_out->data, (const int *)ubnd_out->data,
                          (const int *)lbnd->data, (const int *)ubnd->data,
                          (unsigned short int *)out->data,
                          (out_var ? (unsigned short int *)out_var->data : NULL ) );
            } else if( type == PyArray_UBYTE ) {
               noutpix = astResampleUB( THIS, ncoord_in, (const int *)lbnd_in->data,
                          (const int *)ubnd_in->data, (const unsigned char *)in->data,
                          (in_var ? (const unsigned char *)in_var->data : NULL),
                          interp, NULL, (params ? (const double *)params->data : NULL),
                          flags, tol, maxpix, badval_f, ncoord_out,
                          (const int *)lbnd_out->data, (const int *)ubnd_out->data,
                          (const int *)lbnd->data, (const int *)ubnd->data,
                          (unsigned char *)out->data,
                          (out_var ? (unsigned char *)out_var->data : NULL ) );
            } else {
               PyErr_SetString( PyExc_ValueError, "The 'in' array supplied "
                                "to " NAME " has a data type that is not "
                                "supported by " NAME "." );
            }

            if( astOK ) {
               if( !out_var ) out_var = (PyArrayObject *) Py_None;
               result = Py_BuildValue( "iOO", noutpix, out, out_var );
            }
         }

         Py_XDECREF( out );
         Py_XDECREF( out_var );
      }

      Py_XDECREF( lbnd );
      Py_XDECREF( ubnd );
      Py_XDECREF( lbnd_in );
      Py_XDECREF( ubnd_in );
      Py_XDECREF( lbnd_out );
      Py_XDECREF( ubnd_out );
      Py_XDECREF( in );
      Py_XDECREF( in_var );
   }

   TIDY;
   return result;
}

static PyObject *Mapping_removeregions( Mapping *self ) {
   PyObject *result = NULL;
   PyObject *map_object = NULL;
   AstMapping *map;

   if( PyErr_Occurred() ) return NULL;

   map = astRemoveRegions( THIS );
   if( astOK ) {
      map_object = NewObject( (AstObject *) map );
      if( map_object ) {
         result = Py_BuildValue( "O", map_object );
      }
      Py_XDECREF(map_object);
   }
   if( map ) map = astAnnul( map );

   TIDY;
   return result;
}

static PyObject *Mapping_simplify( Mapping *self ) {
   PyObject *result = NULL;
   PyObject *map_object = NULL;
   AstMapping *map;

   if( PyErr_Occurred() ) return NULL;

   map = astSimplify( THIS );
   if( astOK ) {
      map_object = NewObject( (AstObject *) map );
      if( map_object ) {
         result = Py_BuildValue( "O", map_object );
      }
      Py_XDECREF(map_object);
   }
   if( map ) map = astAnnul( map );

   TIDY;
   return result;
}

#undef NAME
#define NAME CLASS ".trangrid"
static PyObject *Mapping_trangrid( Mapping *self, PyObject *args ) {
   PyArrayObject *lbnd = NULL;
   PyArrayObject *pout = NULL;
   PyArrayObject *ubnd = NULL;
   PyObject *lbnd_object = NULL;
   PyObject *result = NULL;
   PyObject *ubnd_object = NULL;
   const int *lb;
   const int *ub;
   double tol;
   int forward;
   int i;
   int maxpix;
   int ncoord_in;
   int ncoord_out;
   int outdim;
   npy_intp dims[2];

   if( PyErr_Occurred() ) return NULL;

   ncoord_in = astGetI( THIS, "Nin" );
   ncoord_out = astGetI( THIS, "Nout" );
   if( PyArg_ParseTuple( args, "OOdii:" NAME, &lbnd_object, &ubnd_object,
                         &tol, &maxpix, &forward ) && astOK ) {
      lbnd = GetArray1I( lbnd_object, &ncoord_in, "lbnd", NAME );
      ubnd = GetArray1I( ubnd_object, &ncoord_in, "ubnd", NAME );
      if( lbnd && ubnd ) {

         lb = (const int *) lbnd->data;
         ub = (const int *) ubnd->data;
         outdim = 1;
         for( i = 0; i < ncoord_in; i++ ) {
            outdim *= *(ub++) - *(lb++) + 1;
         }

         dims[ 0 ] = ncoord_out;
         dims[ 1 ] = outdim;

         pout = (PyArrayObject *) PyArray_SimpleNew( 2, dims, PyArray_DOUBLE );
         if( pout ) {

            astTranGrid( THIS, ncoord_in, (const int *)lbnd->data,
                         (const int *)ubnd->data, tol, maxpix, forward,
                         ncoord_out, outdim, (double *) pout->data );
            if( astOK ) {
               result = (PyObject *) pout;
            } else {
               Py_DECREF( pout );
            }
         }
      }
      Py_XDECREF( lbnd );
      Py_XDECREF( ubnd );
   }
   TIDY;
   return result;
}

#undef NAME
#define NAME CLASS ".trann"
static PyObject *Mapping_trann( Mapping *self, PyObject *args ) {
   PyArrayObject *in = NULL;
   PyArrayObject *out = NULL;
   PyObject *result = NULL;
   PyObject *in_object = NULL;
   PyObject *out_object = NULL;
   int forward;
   int npoint;
   int ncoord_in;
   int ncoord_out;
   npy_intp pdims[2];
   int dims[ 2 ];
   int ndim;

   if( PyErr_Occurred() ) return NULL;

   ncoord_in = astGetI( THIS, "Nin" );
   ncoord_out = astGetI( THIS, "Nout" );
   if( PyArg_ParseTuple( args, "Oi|O:" NAME, &in_object, &forward,
                         &out_object ) && astOK ) {
      dims[ 0 ] = ncoord_in;
      dims[ 1 ] = 0;
      in = GetArray( in_object, PyArray_DOUBLE, 0, 2, dims, "in", NAME );

      if( in ) {
         dims[ 0 ] = ncoord_out;
         if( out_object ) {
            out = GetArray( out_object, PyArray_DOUBLE, 0, 2, dims, "out", NAME );
         } else {
            if( in->nd == 1 ){
               ndim = 1;
               pdims[ 0 ] = dims[ 1 ];
            } else {
               ndim = 2;
               pdims[ 0 ] = dims[ 0 ];
               pdims[ 1 ] = dims[ 1 ];
            }
            out = (PyArrayObject *) PyArray_SimpleNew( ndim, pdims,
                                                       PyArray_DOUBLE );
         }
      }

      if( out ) {
         npoint = dims[ 1 ];
         astTranN( THIS, npoint, ncoord_in, npoint, (const double *) in->data,
                   forward, ncoord_out, npoint, (double *) out->data );
         if( astOK ) result = (PyObject *) out;
      }

      Py_XDECREF( in );
   }

   TIDY;
   return result;
}


/* ZoomMap */
/* ======= */

/* Define a string holding the fully qualified Python class name. */
#undef CLASS
#define CLASS MODULE ".ZoomMap"

/* Define the class structure */
typedef struct {
   Mapping parent;
} ZoomMap;

/* Prototypes for class functions */
static int ZoomMap_init( ZoomMap *self, PyObject *args, PyObject *kwds );

/* Define the AST attributes of the class */
MAKE_GETSETD(ZoomMap,Zoom)
static PyGetSetDef ZoomMap_getseters[] = {
   DEFATT(Zoom," ZoomMap scale factor"),
   {NULL, NULL, NULL, NULL, NULL}  /* Sentinel */
};

/* Define the class Python type structure */
static PyTypeObject ZoomMapType = {
   PyVarObject_HEAD_INIT(NULL, 0)
   CLASS,                     /* tp_name */
   sizeof(ZoomMap),           /* tp_basicsize */
   0,                         /* tp_itemsize */
   0,                         /* tp_dealloc */
   0,                         /* tp_print */
   0,                         /* tp_getattr */
   0,                         /* tp_setattr */
   0,                         /* tp_reserved */
   0,                         /* tp_repr */
   0,                         /* tp_as_number */
   0,                         /* tp_as_sequence */
   0,                         /* tp_as_mapping */
   0,                         /* tp_hash  */
   0,                         /* tp_call */
   0,                         /* tp_str */
   0,                         /* tp_getattro */
   0,                         /* tp_setattro */
   0,                         /* tp_as_buffer */
   Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE, /* tp_flags */
   "AST ZoomMap",             /* tp_doc */
   0,		              /* tp_traverse */
   0,		              /* tp_clear */
   0,		              /* tp_richcompare */
   0,		              /* tp_weaklistoffset */
   0,		              /* tp_iter */
   0,		              /* tp_iternext */
   0,                         /* tp_methods */
   0,                         /* tp_members */
   ZoomMap_getseters,         /* tp_getset */
   0,                         /* tp_base */
   0,                         /* tp_dict */
   0,                         /* tp_descr_get */
   0,                         /* tp_descr_set */
   0,                         /* tp_dictoffset */
   (initproc)ZoomMap_init,    /* tp_init */
   0,                         /* tp_alloc */
   0,                         /* tp_new */
};


/* Define the class methods */
static int ZoomMap_init( ZoomMap *self, PyObject *args, PyObject *kwds ){
   const char *options = " ";
   double zoom;
   int ncoord;
   int result = -1;

   if( PyArg_ParseTuple(args, "id|s:" CLASS, &ncoord, &zoom, &options ) ) {
      AstZoomMap *this = astZoomMap( ncoord, zoom, options );
      result = SetProxy( (AstObject *) this, (Object *) self );
      this = astAnnul( this );
   }

   TIDY;
   return result;
}

/* SphMap */
/* ====== */

/* Define a string holding the fully qualified Python class name. */
#undef CLASS
#define CLASS MODULE ".SphMap"

/* Define the class structure */
typedef struct {
   Mapping parent;
} SphMap;

/* Prototypes for class functions */
static int SphMap_init( SphMap *self, PyObject *args, PyObject *kwds );

/* Define the AST attributes of the class */
MAKE_GETSETL(SphMap,UnitRadius)
MAKE_GETSETD(SphMap,PolarLong)
static PyGetSetDef SphMap_getseters[] = {
   DEFATT(UnitRadius,"SphMap input vectors lie on a unit sphere?"),
   DEFATT(PolarLong,"The longitude value to assign to either pole"),
   {NULL, NULL, NULL, NULL, NULL}  /* Sentinel */
};

/* Define the class Python type structure */
static PyTypeObject SphMapType = {
   PyVarObject_HEAD_INIT(NULL, 0)
   CLASS,                     /* tp_name */
   sizeof(SphMap),            /* tp_basicsize */
   0,                         /* tp_itemsize */
   0,                         /* tp_dealloc */
   0,                         /* tp_print */
   0,                         /* tp_getattr */
   0,                         /* tp_setattr */
   0,                         /* tp_reserved */
   0,                         /* tp_repr */
   0,                         /* tp_as_number */
   0,                         /* tp_as_sequence */
   0,                         /* tp_as_mapping */
   0,                         /* tp_hash  */
   0,                         /* tp_call */
   0,                         /* tp_str */
   0,                         /* tp_getattro */
   0,                         /* tp_setattro */
   0,                         /* tp_as_buffer */
   Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE, /* tp_flags */
   "AST SphMap",              /* tp_doc */
   0,		              /* tp_traverse */
   0,		              /* tp_clear */
   0,		              /* tp_richcompare */
   0,		              /* tp_weaklistoffset */
   0,		              /* tp_iter */
   0,		              /* tp_iternext */
   0,                         /* tp_methods */
   0,                         /* tp_members */
   SphMap_getseters,          /* tp_getset */
   0,                         /* tp_base */
   0,                         /* tp_dict */
   0,                         /* tp_descr_get */
   0,                         /* tp_descr_set */
   0,                         /* tp_dictoffset */
   (initproc)SphMap_init,     /* tp_init */
   0,                         /* tp_alloc */
   0,                         /* tp_new */
};


/* Define the class methods */
static int SphMap_init( SphMap *self, PyObject *args, PyObject *kwds ){
   const char *options = " ";
   int result = -1;

   if( PyArg_ParseTuple(args, "|s:" CLASS, &options ) ) {
      AstSphMap *this = astSphMap( options );
      result = SetProxy( (AstObject *) this, (Object *) self );
      this = astAnnul( this );
   }

   TIDY;
   return result;
}

/* GrismMap */
/* ======== */

/* Define a string holding the fully qualified Python class name. */
#undef CLASS
#define CLASS MODULE ".GrismMap"

/* Define the class structure */
typedef struct {
   Mapping parent;
} GrismMap;

/* Prototypes for class functions */
static int GrismMap_init( GrismMap *self, PyObject *args, PyObject *kwds );

/* Define the AST attributes of the class */
MAKE_GETSETD(GrismMap,GrismNR)
MAKE_GETSETD(GrismMap,GrismNRP)
MAKE_GETSETD(GrismMap,GrismWaveR)
MAKE_GETSETD(GrismMap,GrismAlpha)
MAKE_GETSETD(GrismMap,GrismG)
MAKE_GETSETI(GrismMap,GrismM)
MAKE_GETSETD(GrismMap,GrismEps)
MAKE_GETSETD(GrismMap,GrismTheta)

static PyGetSetDef GrismMap_getseters[] = {
   DEFATT(GrismNR,"The refractive index at the reference wavelength"),
   DEFATT(GrismNRP,"Rate of change of refractive index with wavelength"),
   DEFATT(GrismWaveR,"The reference wavelength"),
   DEFATT(GrismAlpha,"The angle of incidence of the incoming light"),
   DEFATT(GrismG,"The grating ruling density"),
   DEFATT(GrismM,"The interference order"),
   DEFATT(GrismEps,"The angle between the normal and the dispersion plane"),
   DEFATT(GrismTheta,"Angle between normal to detector plane and reference ray"),
   {NULL, NULL, NULL, NULL, NULL}  /* Sentinel */
};

/* Define the class Python type structure */
static PyTypeObject GrismMapType = {
   PyVarObject_HEAD_INIT(NULL, 0)
   CLASS,                     /* tp_name */
   sizeof(GrismMap),          /* tp_basicsize */
   0,                         /* tp_itemsize */
   0,                         /* tp_dealloc */
   0,                         /* tp_print */
   0,                         /* tp_getattr */
   0,                         /* tp_setattr */
   0,                         /* tp_reserved */
   0,                         /* tp_repr */
   0,                         /* tp_as_number */
   0,                         /* tp_as_sequence */
   0,                         /* tp_as_mapping */
   0,                         /* tp_hash  */
   0,                         /* tp_call */
   0,                         /* tp_str */
   0,                         /* tp_getattro */
   0,                         /* tp_setattro */
   0,                         /* tp_as_buffer */
   Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE, /* tp_flags */
   "AST GrismMap",            /* tp_doc */
   0,		              /* tp_traverse */
   0,		              /* tp_clear */
   0,		              /* tp_richcompare */
   0,		              /* tp_weaklistoffset */
   0,		              /* tp_iter */
   0,		              /* tp_iternext */
   0,                         /* tp_methods */
   0,                         /* tp_members */
   GrismMap_getseters,        /* tp_getset */
   0,                         /* tp_base */
   0,                         /* tp_dict */
   0,                         /* tp_descr_get */
   0,                         /* tp_descr_set */
   0,                         /* tp_dictoffset */
   (initproc)GrismMap_init,   /* tp_init */
   0,                         /* tp_alloc */
   0,                         /* tp_new */
};


/* Define the class methods */
static int GrismMap_init( GrismMap *self, PyObject *args, PyObject *kwds ){
   const char *options = " ";
   int result = -1;

   if( PyArg_ParseTuple(args, "|s:" CLASS, &options ) ) {
      AstGrismMap *this = astGrismMap( options );
      result = SetProxy( (AstObject *) this, (Object *) self );
      this = astAnnul( this );
   }

   TIDY;
   return result;
}

/* PcdMap */
/* ====== */

/* Define a string holding the fully qualified Python class name. */
#undef CLASS
#define CLASS MODULE ".PcdMap"

/* Define the class structure */
typedef struct {
   Mapping parent;
} PcdMap;

/* Prototypes for class functions */
static int PcdMap_init( PcdMap *self, PyObject *args, PyObject *kwds );

/* Define the AST attributes of the class */
#include "PcdCen_def.c"
MAKE_GETSETD(PcdMap,Disco)

static PyGetSetDef PcdMap_getseters[] = {
   #include "PcdCen_desc.c"
   DEFATT(Disco,"PcdMap pincushion/barrel distortion coefficient"),
   {NULL, NULL, NULL, NULL, NULL}  /* Sentinel */
};

/* Define the class Python type structure */
static PyTypeObject PcdMapType = {
   PyVarObject_HEAD_INIT(NULL, 0)
   CLASS,                     /* tp_name */
   sizeof(PcdMap),            /* tp_basicsize */
   0,                         /* tp_itemsize */
   0,                         /* tp_dealloc */
   0,                         /* tp_print */
   0,                         /* tp_getattr */
   0,                         /* tp_setattr */
   0,                         /* tp_reserved */
   0,                         /* tp_repr */
   0,                         /* tp_as_number */
   0,                         /* tp_as_sequence */
   0,                         /* tp_as_mapping */
   0,                         /* tp_hash  */
   0,                         /* tp_call */
   0,                         /* tp_str */
   0,                         /* tp_getattro */
   0,                         /* tp_setattro */
   0,                         /* tp_as_buffer */
   Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE, /* tp_flags */
   "AST PcdMap",              /* tp_doc */
   0,		              /* tp_traverse */
   0,		              /* tp_clear */
   0,		              /* tp_richcompare */
   0,		              /* tp_weaklistoffset */
   0,		              /* tp_iter */
   0,		              /* tp_iternext */
   0,                         /* tp_methods */
   0,                         /* tp_members */
   PcdMap_getseters,          /* tp_getset */
   0,                         /* tp_base */
   0,                         /* tp_dict */
   0,                         /* tp_descr_get */
   0,                         /* tp_descr_set */
   0,                         /* tp_dictoffset */
   (initproc)PcdMap_init,     /* tp_init */
   0,                         /* tp_alloc */
   0,                         /* tp_new */
};


/* Define the class methods */
static int PcdMap_init( PcdMap *self, PyObject *args, PyObject *kwds ){
   const char *options = " ";
   int result = -1;
   double disco;
   PyArrayObject * pcdcen = NULL;
   PyObject * pcdcen_object = NULL;

   if( PyArg_ParseTuple(args, "dO|s:" CLASS, &disco, &pcdcen_object, &options ) ) {
      AstPcdMap * this = NULL;
      int ncoord = 2;
      pcdcen = GetArray1D( pcdcen_object, &ncoord, "pcdcen", NAME );
      if (pcdcen) {
	this = astPcdMap( disco, (const double *)pcdcen->data, options );
	result = SetProxy( (AstObject *) this, (Object *) self );
	this = astAnnul( this );
      }
      Py_XDECREF( pcdcen );
   }

   TIDY;
   return result;
}

/* WcsMap */
/* ====== */

/* Define a string holding the fully qualified Python class name. */
#undef CLASS
#define CLASS MODULE ".WcsMap"

/* Define the class structure */
typedef struct {
   Mapping parent;
} WcsMap;

/* Prototypes for class functions */
static int WcsMap_init( WcsMap *self, PyObject *args, PyObject *kwds );

/* Define the AST attributes of the class */
#include "ProjP_def.c"
#include "PVMax_def.c"
#include "WcsAxis_def.c"
// PVi_m - would required 10000 properties so not done.
MAKE_GETROI(WcsMap,WcsType)
MAKE_GETROD(WcsMap,NatLon)
MAKE_GETROD(WcsMap,NatLat)

static PyGetSetDef WcsMap_getseters[] = {
#include "ProjP_desc.c"
   #include "PVMax_desc.c"
   #include "WcsAxis_desc.c"
   DEFATT(NatLat,"Native latitude of the reference point of a FITS-WCS projection"),
   DEFATT(NatLon,"Native longitude of the reference point of a FITS-WCS projection"),
   DEFATT(WcsType,"FITS-WCS projection type"),
   {NULL, NULL, NULL, NULL, NULL}  /* Sentinel */
};

/* Define the class Python type structure */
static PyTypeObject WcsMapType = {
   PyVarObject_HEAD_INIT(NULL, 0)
   CLASS,                     /* tp_name */
   sizeof(WcsMap),            /* tp_basicsize */
   0,                         /* tp_itemsize */
   0,                         /* tp_dealloc */
   0,                         /* tp_print */
   0,                         /* tp_getattr */
   0,                         /* tp_setattr */
   0,                         /* tp_reserved */
   0,                         /* tp_repr */
   0,                         /* tp_as_number */
   0,                         /* tp_as_sequence */
   0,                         /* tp_as_mapping */
   0,                         /* tp_hash  */
   0,                         /* tp_call */
   0,                         /* tp_str */
   0,                         /* tp_getattro */
   0,                         /* tp_setattro */
   0,                         /* tp_as_buffer */
   Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE, /* tp_flags */
   "AST WcsMap",              /* tp_doc */
   0,		              /* tp_traverse */
   0,		              /* tp_clear */
   0,		              /* tp_richcompare */
   0,		              /* tp_weaklistoffset */
   0,		              /* tp_iter */
   0,		              /* tp_iternext */
   0,                         /* tp_methods */
   0,                         /* tp_members */
   WcsMap_getseters,          /* tp_getset */
   0,                         /* tp_base */
   0,                         /* tp_dict */
   0,                         /* tp_descr_get */
   0,                         /* tp_descr_set */
   0,                         /* tp_dictoffset */
   (initproc)WcsMap_init,     /* tp_init */
   0,                         /* tp_alloc */
   0,                         /* tp_new */
};


/* Define the class methods */
static int WcsMap_init( WcsMap *self, PyObject *args, PyObject *kwds ){
   const char *options = " ";
   int result = -1;
   int ncoord;
   int type;
   int lonax;
   int latax;

   if( PyArg_ParseTuple(args, "iiii|s:" CLASS, &ncoord,
                        &type, &lonax, &latax, &options ) ) {
      AstWcsMap * this = NULL;
      this = astWcsMap( ncoord, type, lonax, latax, options );
      result = SetProxy( (AstObject *) this, (Object *) self );
      this = astAnnul( this );
   }

   TIDY;
   return result;
}

/* UnitMap */
/* ======= */

/* Define a string holding the fully qualified Python class name. */
#undef CLASS
#define CLASS MODULE ".UnitMap"

/* Define the class structure */
typedef struct {
   Mapping parent;
} UnitMap;

/* Prototypes for class functions */
static int UnitMap_init( UnitMap *self, PyObject *args, PyObject *kwds );

/* Define the class Python type structure */
static PyTypeObject UnitMapType = {
   PyVarObject_HEAD_INIT(NULL, 0)
   CLASS,                     /* tp_name */
   sizeof(UnitMap),           /* tp_basicsize */
   0,                         /* tp_itemsize */
   0,                         /* tp_dealloc */
   0,                         /* tp_print */
   0,                         /* tp_getattr */
   0,                         /* tp_setattr */
   0,                         /* tp_reserved */
   0,                         /* tp_repr */
   0,                         /* tp_as_number */
   0,                         /* tp_as_sequence */
   0,                         /* tp_as_mapping */
   0,                         /* tp_hash  */
   0,                         /* tp_call */
   0,                         /* tp_str */
   0,                         /* tp_getattro */
   0,                         /* tp_setattro */
   0,                         /* tp_as_buffer */
   Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE, /* tp_flags */
   "AST UnitMap",             /* tp_doc */
   0,		              /* tp_traverse */
   0,		              /* tp_clear */
   0,		              /* tp_richcompare */
   0,		              /* tp_weaklistoffset */
   0,		              /* tp_iter */
   0,		              /* tp_iternext */
   0,                         /* tp_methods */
   0,                         /* tp_members */
   0,                         /* tp_getset */
   0,                         /* tp_base */
   0,                         /* tp_dict */
   0,                         /* tp_descr_get */
   0,                         /* tp_descr_set */
   0,                         /* tp_dictoffset */
   (initproc)UnitMap_init,    /* tp_init */
   0,                         /* tp_alloc */
   0,                         /* tp_new */
};


/* Define the class methods */
static int UnitMap_init( UnitMap *self, PyObject *args, PyObject *kwds ){
   const char *options = " ";
   int ncoord;
   int result = -1;

   if( PyArg_ParseTuple(args, "i|s:" CLASS, &ncoord, &options ) ) {
      AstUnitMap *this = astUnitMap( ncoord, options );
      result = SetProxy( (AstObject *) this, (Object *) self );
      this = astAnnul( this );
   }

   TIDY;
   return result;
}

/* TimeMap */
/* ======= */

/* Define a string holding the fully qualified Python class name. */
#undef CLASS
#define CLASS MODULE ".TimeMap"

/* Define the class structure */
typedef struct {
   Mapping parent;
} TimeMap;

/* Prototypes for class functions */
static int TimeMap_init( TimeMap *self, PyObject *args, PyObject *kwds );
static PyObject *TimeMap_timeadd( TimeMap *self, PyObject *args );

/* Describe the methods of the class */
static PyMethodDef TimeMap_methods[] = {
   {"timeadd", (PyCFunction)TimeMap_timeadd, METH_VARARGS, "Add a time coordinate conversion to a TimeMap"},
   {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* Define the class Python type structure */
static PyTypeObject TimeMapType = {
   PyVarObject_HEAD_INIT(NULL, 0)
   CLASS,                     /* tp_name */
   sizeof(TimeMap),           /* tp_basicsize */
   0,                         /* tp_itemsize */
   0,                         /* tp_dealloc */
   0,                         /* tp_print */
   0,                         /* tp_getattr */
   0,                         /* tp_setattr */
   0,                         /* tp_reserved */
   0,                         /* tp_repr */
   0,                         /* tp_as_number */
   0,                         /* tp_as_sequence */
   0,                         /* tp_as_mapping */
   0,                         /* tp_hash  */
   0,                         /* tp_call */
   0,                         /* tp_str */
   0,                         /* tp_getattro */
   0,                         /* tp_setattro */
   0,                         /* tp_as_buffer */
   Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE, /* tp_flags */
   "AST TimeMap",             /* tp_doc */
   0,		              /* tp_traverse */
   0,		              /* tp_clear */
   0,		              /* tp_richcompare */
   0,		              /* tp_weaklistoffset */
   0,		              /* tp_iter */
   0,		              /* tp_iternext */
   TimeMap_methods,           /* tp_methods */
   0,                         /* tp_members */
   0,                         /* tp_getset */
   0,                         /* tp_base */
   0,                         /* tp_dict */
   0,                         /* tp_descr_get */
   0,                         /* tp_descr_set */
   0,                         /* tp_dictoffset */
   (initproc)TimeMap_init,    /* tp_init */
   0,                         /* tp_alloc */
   0,                         /* tp_new */
};


/* Define the class methods */
static int TimeMap_init( TimeMap *self, PyObject *args, PyObject *kwds ){
   const char *options = " ";
   int flags;
   int result = -1;

   if( PyArg_ParseTuple(args, "i|s:" CLASS, &flags, &options ) ) {
      if (flags != 0) {
         PyErr_SetString( PyExc_ValueError, "The TimeMap flags argument must currently always be zero");
      } else {
         AstTimeMap *this = astTimeMap( flags, options );
         result = SetProxy( (AstObject *) this, (Object *) self );
         this = astAnnul( this );
      }
   }

   TIDY;
   return result;
}

#undef NAME
#define NAME CLASS ".timeadd"
static PyObject *TimeMap_timeadd( TimeMap *self, PyObject *args ) {
  PyObject *result = NULL;
  PyArrayObject *astargs = NULL;
  PyObject *astargs_object = NULL;
  const char * cvt = "";

  if( PyErr_Occurred() ) return NULL;

  /* Currently every cvt option takes an argument so we do not mark
     args as optional */
  if ( PyArg_ParseTuple( args, "sO:" NAME, &cvt,
                         &astargs_object ) && astOK ) {
    /* Ideally we would like to determine how many elements we
       have in "args" to make sure it is correct. Putting the code
       here and in AST seems silly though. */
    astargs = (PyArrayObject *) PyArray_ContiguousFromAny( astargs_object,
                                                           PyArray_DOUBLE, 0, 100);
    if (astargs) {
      astTimeAdd( THIS, cvt, (const double *)astargs->data );
      if (astOK) result = Py_None;
    }
    Py_XDECREF( astargs );
  }

  TIDY;
  return result;
}

/* RateMap */
/* ======= */

/* Define a string holding the fully qualified Python class name. */
#undef CLASS
#define CLASS MODULE ".RateMap"

/* Define the class structure */
typedef struct {
   Mapping parent;
} RateMap;

/* Prototypes for class functions */
static int RateMap_init( RateMap *self, PyObject *args, PyObject *kwds );

/* Define the class Python type structure */
static PyTypeObject RateMapType = {
   PyVarObject_HEAD_INIT(NULL, 0)
   CLASS,                     /* tp_name */
   sizeof(RateMap),           /* tp_basicsize */
   0,                         /* tp_itemsize */
   0,                         /* tp_dealloc */
   0,                         /* tp_print */
   0,                         /* tp_getattr */
   0,                         /* tp_setattr */
   0,                         /* tp_reserved */
   0,                         /* tp_repr */
   0,                         /* tp_as_number */
   0,                         /* tp_as_sequence */
   0,                         /* tp_as_mapping */
   0,                         /* tp_hash  */
   0,                         /* tp_call */
   0,                         /* tp_str */
   0,                         /* tp_getattro */
   0,                         /* tp_setattro */
   0,                         /* tp_as_buffer */
   Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE, /* tp_flags */
   "AST RateMap",             /* tp_doc */
   0,		              /* tp_traverse */
   0,		              /* tp_clear */
   0,		              /* tp_richcompare */
   0,		              /* tp_weaklistoffset */
   0,		              /* tp_iter */
   0,		              /* tp_iternext */
   0,                         /* tp_methods */
   0,                         /* tp_members */
   0,                         /* tp_getset */
   0,                         /* tp_base */
   0,                         /* tp_dict */
   0,                         /* tp_descr_get */
   0,                         /* tp_descr_set */
   0,                         /* tp_dictoffset */
   (initproc)RateMap_init,    /* tp_init */
   0,                         /* tp_alloc */
   0,                         /* tp_new */
};


/* Define the class methods */
static int RateMap_init( RateMap *self, PyObject *args, PyObject *kwds ){
   const char *options = " ";
   Mapping *other;
   int ax1;
   int ax2;
   int result = -1;

   if( PyArg_ParseTuple(args, "O!ii|s:" CLASS, &MappingType, (PyObject**)&other,
			&ax1, &ax2, &options ) ) {
      AstRateMap *this = astRateMap( THAT, ax1, ax2, options );
      result = SetProxy( (AstObject *) this, (Object *) self );
      this = astAnnul( this );
   }

   TIDY;
   return result;
}

/* CmpMap */
/* ====== */

/* Define a string holding the fully qualified Python class name. */
#undef CLASS
#define CLASS MODULE ".CmpMap"

/* Define the class structure */
typedef struct {
   Mapping parent;
} CmpMap;

/* Prototypes for class functions */
static int CmpMap_init( CmpMap *self, PyObject *args, PyObject *kwds );

/* Define the class Python type structure */
static PyTypeObject CmpMapType = {
   PyVarObject_HEAD_INIT(NULL, 0)
   CLASS,                     /* tp_name */
   sizeof(CmpMap),            /* tp_basicsize */
   0,                         /* tp_itemsize */
   0,                         /* tp_dealloc */
   0,                         /* tp_print */
   0,                         /* tp_getattr */
   0,                         /* tp_setattr */
   0,                         /* tp_reserved */
   0,                         /* tp_repr */
   0,                         /* tp_as_number */
   0,                         /* tp_as_sequence */
   0,                         /* tp_as_mapping */
   0,                         /* tp_hash  */
   0,                         /* tp_call */
   0,                         /* tp_str */
   0,                         /* tp_getattro */
   0,                         /* tp_setattro */
   0,                         /* tp_as_buffer */
   Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE, /* tp_flags */
   "AST CmpMap",              /* tp_doc */
   0,		              /* tp_traverse */
   0,		              /* tp_clear */
   0,		              /* tp_richcompare */
   0,		              /* tp_weaklistoffset */
   0,		              /* tp_iter */
   0,		              /* tp_iternext */
   0,                         /* tp_methods */
   0,                         /* tp_members */
   0,                         /* tp_getset */
   0,                         /* tp_base */
   0,                         /* tp_dict */
   0,                         /* tp_descr_get */
   0,                         /* tp_descr_set */
   0,                         /* tp_dictoffset */
   (initproc)CmpMap_init,     /* tp_init */
   0,                         /* tp_alloc */
   0,                         /* tp_new */
};


/* Define the class methods */
static int CmpMap_init( CmpMap *self, PyObject *args, PyObject *kwds ){
   const char *options = " ";
   Mapping *other;
   Mapping *another;
   int series;
   int result = -1;

   if( PyArg_ParseTuple(args, "O!O!i|s:" CLASS, &MappingType, (PyObject**)&other,
                        &MappingType, (PyObject**)&another, &series, &options ) ) {
      AstCmpMap *this = astCmpMap( THAT, ANOTHER, series, options );
      result = SetProxy( (AstObject *) this, (Object *) self );
      this = astAnnul( this );
   }

   TIDY;
   return result;
}

/* TranMap */
/* ======= */

/* Define a string holding the fully qualified Python class name. */
#undef CLASS
#define CLASS MODULE ".TranMap"

/* Define the class structure */
typedef struct {
   Mapping parent;
} TranMap;

/* Prototypes for class functions */
static int TranMap_init( TranMap *self, PyObject *args, PyObject *kwds );

/* Define the class Python type structure */
static PyTypeObject TranMapType = {
   PyVarObject_HEAD_INIT(NULL, 0)
   CLASS,                     /* tp_name */
   sizeof(TranMap),           /* tp_basicsize */
   0,                         /* tp_itemsize */
   0,                         /* tp_dealloc */
   0,                         /* tp_print */
   0,                         /* tp_getattr */
   0,                         /* tp_setattr */
   0,                         /* tp_reserved */
   0,                         /* tp_repr */
   0,                         /* tp_as_number */
   0,                         /* tp_as_sequence */
   0,                         /* tp_as_mapping */
   0,                         /* tp_hash  */
   0,                         /* tp_call */
   0,                         /* tp_str */
   0,                         /* tp_getattro */
   0,                         /* tp_setattro */
   0,                         /* tp_as_buffer */
   Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE, /* tp_flags */
   "AST TranMap",             /* tp_doc */
   0,		              /* tp_traverse */
   0,		              /* tp_clear */
   0,		              /* tp_richcompare */
   0,		              /* tp_weaklistoffset */
   0,		              /* tp_iter */
   0,		              /* tp_iternext */
   0,                         /* tp_methods */
   0,                         /* tp_members */
   0,                         /* tp_getset */
   0,                         /* tp_base */
   0,                         /* tp_dict */
   0,                         /* tp_descr_get */
   0,                         /* tp_descr_set */
   0,                         /* tp_dictoffset */
   (initproc)TranMap_init,    /* tp_init */
   0,                         /* tp_alloc */
   0,                         /* tp_new */
};


/* Define the class methods */
static int TranMap_init( TranMap *self, PyObject *args, PyObject *kwds ){
   const char *options = " ";
   Mapping *other;
   Mapping *another;
   int result = -1;

   if( PyArg_ParseTuple(args, "O!O!|s:" CLASS, &MappingType, (PyObject**)&other,
                        &MappingType, (PyObject**)&another, &options ) ) {
      AstTranMap *this = astTranMap( THAT, ANOTHER, options );
      result = SetProxy( (AstObject *) this, (Object *) self );
      this = astAnnul( this );
   }

   TIDY;
   return result;
}

/* PermMap */
/* ======= */

/* Define a string holding the fully qualified Python class name. */
#undef CLASS
#define CLASS MODULE ".PermMap"

/* Define the class structure */
typedef struct {
   Mapping parent;
} PermMap;

/* Prototypes for class functions */
static int PermMap_init( PermMap *self, PyObject *args, PyObject *kwds );

/* Define the class Python type structure */
static PyTypeObject PermMapType = {
   PyVarObject_HEAD_INIT(NULL, 0)
   CLASS,                     /* tp_name */
   sizeof(PermMap),           /* tp_basicsize */
   0,                         /* tp_itemsize */
   0,                         /* tp_dealloc */
   0,                         /* tp_print */
   0,                         /* tp_getattr */
   0,                         /* tp_setattr */
   0,                         /* tp_reserved */
   0,                         /* tp_repr */
   0,                         /* tp_as_number */
   0,                         /* tp_as_sequence */
   0,                         /* tp_as_mapping */
   0,                         /* tp_hash  */
   0,                         /* tp_call */
   0,                         /* tp_str */
   0,                         /* tp_getattro */
   0,                         /* tp_setattro */
   0,                         /* tp_as_buffer */
   Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE, /* tp_flags */
   "AST PermMap",             /* tp_doc */
   0,		              /* tp_traverse */
   0,		              /* tp_clear */
   0,		              /* tp_richcompare */
   0,		              /* tp_weaklistoffset */
   0,		              /* tp_iter */
   0,		              /* tp_iternext */
   0,                         /* tp_methods */
   0,                         /* tp_members */
   0,                         /* tp_getset */
   0,                         /* tp_base */
   0,                         /* tp_dict */
   0,                         /* tp_descr_get */
   0,                         /* tp_descr_set */
   0,                         /* tp_dictoffset */
   (initproc)PermMap_init,    /* tp_init */
   0,                         /* tp_alloc */
   0,                         /* tp_new */
};


/* Define the class methods */
static int PermMap_init( PermMap *self, PyObject *args, PyObject *kwds ){
   const char *options = " ";
   PyArrayObject * inperm = NULL;
   PyArrayObject * outperm = NULL;
   PyArrayObject * constant = NULL;
   PyObject * inperm_object = NULL;
   PyObject * outperm_object = NULL;
   PyObject * constant_object = NULL;

   int result = -1;

   // We get nin and nou from the arrays themselves
   if( PyArg_ParseTuple(args, "OO|Os:" CLASS, &inperm_object,
                        &outperm_object, &constant_object, &options ) ) {
      inperm = (PyArrayObject *) PyArray_ContiguousFromAny( inperm_object,
                                                            PyArray_INT, 0, 100);
      outperm = (PyArrayObject *) PyArray_ContiguousFromAny( outperm_object,
                                                             PyArray_INT, 0, 100);
      if (constant_object) {
        constant = (PyArrayObject *) PyArray_ContiguousFromAny( constant_object,
                                                                PyArray_DOUBLE, 0, 100);
      }
      if (inperm && outperm) {
         AstPermMap * this = NULL;

         /* May want to sanity check the "constant" array since we know how
            big it is and we can search through inperm and outperm for negative
            values */
         this = astPermMap( PyArray_Size( (PyObject*)inperm),
                            (const int *)inperm->data,
                            PyArray_Size( (PyObject*)outperm),
                            (const int *)outperm->data,
                            (constant ? (const double*)constant->data : NULL),
                            options);
         result = SetProxy( (AstObject *) this, (Object *) self );
         this = astAnnul( this );
      }
      Py_XDECREF( inperm );
      Py_XDECREF( outperm );
      Py_XDECREF( constant );
   }

   TIDY;
   return result;
}

/* ShiftMap */
/* ======== */

/* Define a string holding the fully qualified Python class name. */
#undef CLASS
#define CLASS MODULE ".ShiftMap"

/* Define the class structure */
typedef struct {
   Mapping parent;
} ShiftMap;

/* Prototypes for class functions */
static int ShiftMap_init( ShiftMap *self, PyObject *args, PyObject *kwds );

/* Define the class Python type structure */
static PyTypeObject ShiftMapType = {
   PyVarObject_HEAD_INIT(NULL, 0)
   CLASS,                     /* tp_name */
   sizeof(ShiftMap),          /* tp_basicsize */
   0,                         /* tp_itemsize */
   0,                         /* tp_dealloc */
   0,                         /* tp_print */
   0,                         /* tp_getattr */
   0,                         /* tp_setattr */
   0,                         /* tp_reserved */
   0,                         /* tp_repr */
   0,                         /* tp_as_number */
   0,                         /* tp_as_sequence */
   0,                         /* tp_as_mapping */
   0,                         /* tp_hash  */
   0,                         /* tp_call */
   0,                         /* tp_str */
   0,                         /* tp_getattro */
   0,                         /* tp_setattro */
   0,                         /* tp_as_buffer */
   Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE, /* tp_flags */
   "AST ShiftMap",            /* tp_doc */
   0,		              /* tp_traverse */
   0,		              /* tp_clear */
   0,		              /* tp_richcompare */
   0,		              /* tp_weaklistoffset */
   0,		              /* tp_iter */
   0,		              /* tp_iternext */
   0,                         /* tp_methods */
   0,                         /* tp_members */
   0,                         /* tp_getset */
   0,                         /* tp_base */
   0,                         /* tp_dict */
   0,                         /* tp_descr_get */
   0,                         /* tp_descr_set */
   0,                         /* tp_dictoffset */
   (initproc)ShiftMap_init,   /* tp_init */
   0,                         /* tp_alloc */
   0,                         /* tp_new */
};


/* Define the class methods */
static int ShiftMap_init( ShiftMap *self, PyObject *args, PyObject *kwds ){
   const char *options = " ";
   PyArrayObject * shift = NULL;
   PyObject * shift_object = NULL;

   int result = -1;

   // We get nin and nou from the arrays themselves
   if( PyArg_ParseTuple(args, "O|s:" CLASS, &shift_object,
                        &options ) ) {
      shift = (PyArrayObject *) PyArray_ContiguousFromAny( shift_object,
                                                            PyArray_DOUBLE, 0, 100);
      if (shift) {
         AstShiftMap * this = NULL;
         this = astShiftMap( PyArray_Size( (PyObject*)shift),
                            (const double *)shift->data,
                            options);
         result = SetProxy( (AstObject *) this, (Object *) self );
         this = astAnnul( this );
      }
      Py_XDECREF( shift );
   }

   TIDY;
   return result;
}

/* LutMap */
/* ====== */

/* Define a string holding the fully qualified Python class name. */
#undef CLASS
#define CLASS MODULE ".LutMap"

/* Define the class structure */
typedef struct {
   Mapping parent;
} LutMap;

/* Prototypes for class functions */
static int LutMap_init( LutMap *self, PyObject *args, PyObject *kwds );

/* Define the class Python type structure */
static PyTypeObject LutMapType = {
   PyVarObject_HEAD_INIT(NULL, 0)
   CLASS,                     /* tp_name */
   sizeof(LutMap),            /* tp_basicsize */
   0,                         /* tp_itemsize */
   0,                         /* tp_dealloc */
   0,                         /* tp_print */
   0,                         /* tp_getattr */
   0,                         /* tp_setattr */
   0,                         /* tp_reserved */
   0,                         /* tp_repr */
   0,                         /* tp_as_number */
   0,                         /* tp_as_sequence */
   0,                         /* tp_as_mapping */
   0,                         /* tp_hash  */
   0,                         /* tp_call */
   0,                         /* tp_str */
   0,                         /* tp_getattro */
   0,                         /* tp_setattro */
   0,                         /* tp_as_buffer */
   Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE, /* tp_flags */
   "AST LutMap",              /* tp_doc */
   0,		              /* tp_traverse */
   0,		              /* tp_clear */
   0,		              /* tp_richcompare */
   0,		              /* tp_weaklistoffset */
   0,		              /* tp_iter */
   0,		              /* tp_iternext */
   0,                         /* tp_methods */
   0,                         /* tp_members */
   0,                         /* tp_getset */
   0,                         /* tp_base */
   0,                         /* tp_dict */
   0,                         /* tp_descr_get */
   0,                         /* tp_descr_set */
   0,                         /* tp_dictoffset */
   (initproc)LutMap_init,     /* tp_init */
   0,                         /* tp_alloc */
   0,                         /* tp_new */
};


/* Define the class methods */
static int LutMap_init( LutMap *self, PyObject *args, PyObject *kwds ){
   const char *options = " ";
   PyArrayObject * lut = NULL;
   PyObject * lut_object = NULL;
   double start;
   double inc;

   int result = -1;

   // We get nin and nout from the arrays themselves
   if( PyArg_ParseTuple(args, "Odd|s:" CLASS, &lut_object,
                        &start, &inc, &options ) ) {
      lut = (PyArrayObject *) PyArray_ContiguousFromAny( lut_object,
                                                         PyArray_DOUBLE, 0, 100);
      if (lut) {
         AstLutMap * this = NULL;
         this = astLutMap( PyArray_Size( (PyObject*)lut),
                           (const double *)lut->data,
                           start, inc, options);
         result = SetProxy( (AstObject *) this, (Object *) self );
         this = astAnnul( this );
      }
      Py_XDECREF( lut );
   }

   TIDY;
   return result;
}

/* WinMap */
/* ====== */

/* Define a string holding the fully qualified Python class name. */
#undef CLASS
#define CLASS MODULE ".WinMap"

/* Define the class structure */
typedef struct {
   Mapping parent;
} WinMap;

/* Prototypes for class functions */
static int WinMap_init( WinMap *self, PyObject *args, PyObject *kwds );

/* Define the class Python type structure */
static PyTypeObject WinMapType = {
   PyVarObject_HEAD_INIT(NULL, 0)
   CLASS,                     /* tp_name */
   sizeof(WinMap),            /* tp_basicsize */
   0,                         /* tp_itemsize */
   0,                         /* tp_dealloc */
   0,                         /* tp_print */
   0,                         /* tp_getattr */
   0,                         /* tp_setattr */
   0,                         /* tp_reserved */
   0,                         /* tp_repr */
   0,                         /* tp_as_number */
   0,                         /* tp_as_sequence */
   0,                         /* tp_as_mapping */
   0,                         /* tp_hash  */
   0,                         /* tp_call */
   0,                         /* tp_str */
   0,                         /* tp_getattro */
   0,                         /* tp_setattro */
   0,                         /* tp_as_buffer */
   Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE, /* tp_flags */
   "AST WinMap",              /* tp_doc */
   0,		              /* tp_traverse */
   0,		              /* tp_clear */
   0,		              /* tp_richcompare */
   0,		              /* tp_weaklistoffset */
   0,		              /* tp_iter */
   0,		              /* tp_iternext */
   0,                         /* tp_methods */
   0,                         /* tp_members */
   0,                         /* tp_getset */
   0,                         /* tp_base */
   0,                         /* tp_dict */
   0,                         /* tp_descr_get */
   0,                         /* tp_descr_set */
   0,                         /* tp_dictoffset */
   (initproc)WinMap_init,     /* tp_init */
   0,                         /* tp_alloc */
   0,                         /* tp_new */
};


/* Define the class methods */
static int WinMap_init( WinMap *self, PyObject *args, PyObject *kwds ){
   const char *options = " ";
   PyArrayObject * ina = NULL;
   PyArrayObject * inb = NULL;
   PyArrayObject * outa= NULL;
   PyArrayObject * outb= NULL;
   PyObject * ina_object = NULL;
   PyObject * inb_object = NULL;
   PyObject * outa_object = NULL;
   PyObject * outb_object = NULL;

   int result = -1;

   // We get nin and nou from the arrays themselves
   if( PyArg_ParseTuple(args, "OOOO|s:" CLASS, &ina_object,
                        &inb_object, &outa_object, &outb_object, &options ) ) {
      ina = (PyArrayObject *) PyArray_ContiguousFromAny( ina_object,
                                                         PyArray_DOUBLE, 0, 100);
      inb = (PyArrayObject *) PyArray_ContiguousFromAny( inb_object,
                                                         PyArray_DOUBLE, 0, 100);
      outa = (PyArrayObject *) PyArray_ContiguousFromAny( outa_object,
                                                         PyArray_DOUBLE, 0, 100);
      outb = (PyArrayObject *) PyArray_ContiguousFromAny( outb_object,
                                                         PyArray_DOUBLE, 0, 100);
      if (ina && inb && outa && outb ) {
         AstWinMap * this = NULL;
         // Sanity check size
         size_t ncoord = PyArray_Size( (PyObject*)ina );
         if ( ncoord == PyArray_Size( (PyObject*)inb ) &&
              ncoord == PyArray_Size( (PyObject*)outa) &&
              ncoord == PyArray_Size( (PyObject*)outb) ) {
           this = astWinMap( ncoord,
                             (const double *)ina->data,
                             (const double *)inb->data,
                             (const double *)outa->data,
                             (const double *)outb->data,
                             options);
           result = SetProxy( (AstObject *) this, (Object *) self );
           this = astAnnul( this );
         } else {
           PyErr_SetString( PyExc_ValueError,
                            "All input arrays must have the same number of elements for WinMap");
         }
      }
      Py_XDECREF( ina );
      Py_XDECREF( inb );
      Py_XDECREF( outa );
      Py_XDECREF( outb );
   }

   TIDY;
   return result;
}

/* Frame */
/* ===== */

/* Define a string holding the fully qualified Python class name. */
#undef CLASS
#define CLASS MODULE ".Frame"

/* Define the class structure */
typedef struct {
   Mapping parent;
} Frame;

/* Prototypes for class functions */
static int Frame_init( Frame *self, PyObject *args, PyObject *kwds );
static PyObject *Frame_angle( Frame *self, PyObject *args );
static PyObject *Frame_axangle( Frame *self, PyObject *args );
static PyObject *Frame_axdistance( Frame *self, PyObject *args );
static PyObject *Frame_axoffset( Frame *self, PyObject *args );
static PyObject *Frame_convert( Frame *self, PyObject *args );
static PyObject *Frame_distance( Frame *self, PyObject *args );
static PyObject *Frame_findframe( Frame *self, PyObject *args );
static PyObject *Frame_format( Frame *self, PyObject *args );
static PyObject *Frame_intersect( Frame *self, PyObject *args );
static PyObject *Frame_matchaxes( Frame *self, PyObject *args );
static PyObject *Frame_norm( Frame *self, PyObject *args );
static PyObject *Frame_offset( Frame *self, PyObject *args );
static PyObject *Frame_offset2( Frame *self, PyObject *args );
static PyObject *Frame_permaxes( Frame *self, PyObject *args );
static PyObject *Frame_pickaxes( Frame *self, PyObject *args );
static PyObject *Frame_resolve( Frame *self, PyObject *args );
static PyObject *Frame_unformat( Frame *self, PyObject *args );

/* Describe the methods of the class */
static PyMethodDef Frame_methods[] = {
  {"angle", (PyCFunction)Frame_angle, METH_VARARGS, "Calculate the angle subtended by two points at a this point"},
  {"axangle", (PyCFunction)Frame_axangle, METH_VARARGS, "Returns the angle from an axis, to a line through two points"},
  {"axdistance", (PyCFunction)Frame_axdistance, METH_VARARGS, "Find the distance between two axis values"},
  {"axoffset", (PyCFunction)Frame_axoffset, METH_VARARGS, "Add an increment onto a supplied axis value"},
  {"convert", (PyCFunction)Frame_convert, METH_VARARGS, "Determine how to convert between two coordinate systems"},
  {"distance", (PyCFunction)Frame_distance, METH_VARARGS, "Calculate the distance between two points in a Frame"},
  {"findframe", (PyCFunction)Frame_findframe, METH_VARARGS, "Find a coordinate system with specified characteristics"},
  {"format", (PyCFunction)Frame_format, METH_VARARGS, "Format a coordinate value for a Frame axis"},
  // astGetActiveUnit is implemented as an attribute
  {"intersect", (PyCFunction)Frame_intersect, METH_VARARGS, "Find the point of intersection between two geodesic curves"},
  {"matchaxes", (PyCFunction)Frame_matchaxes, METH_VARARGS, "Find any corresponding axes in two Frames"},
  {"norm", (PyCFunction)Frame_norm, METH_VARARGS, "Normalise a set of Frame coordinates"},
  {"offset", (PyCFunction)Frame_offset, METH_VARARGS, "Calculate an offset along a geodesic curve"},
  {"offset2", (PyCFunction)Frame_offset2, METH_VARARGS, "Calculate an offset along a geodesic curve in a 2D Frame"},
  {"permaxes", (PyCFunction)Frame_permaxes, METH_VARARGS, "Permute the axis order in a Frame"},
  {"pickaxes", (PyCFunction)Frame_pickaxes, METH_VARARGS, "Crate a new Frame by picking axes from an existing one"},
  {"resolve", (PyCFunction)Frame_resolve, METH_VARARGS, "Resolve a vector into two orthogonal components"},
  {"unformat", (PyCFunction)Frame_unformat, METH_VARARGS, "Read a formatted coordinate value for a Frame"},
  {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* Define the AST attributes of the class */
#include "Bottom_def.c"
#include "Digits_def.c"
#include "Direction_def.c"
#include "Format_def.c"
#include "Label_def.c"
#include "NormUnit_def.c"
#include "Symbol_def.c"
#include "Top_def.c"
#include "Unit_def.c"
MAKE_GETROI(Frame,Naxes)
MAKE_GETSETC(Frame,AlignSystem)
MAKE_GETSETC(Frame,Domain)
MAKE_GETSETC(Frame,System)
MAKE_GETSETC(Frame,Title)
MAKE_GETSETD(Frame,Dut1)
MAKE_GETSETD(Frame,Epoch)
MAKE_GETSETD(Frame,ObsAlt)
MAKE_GETSETD(Frame,ObsLon)
MAKE_GETSETD(Frame,ObsLat)
MAKE_GETSETI(Frame,MaxAxes)
MAKE_GETSETI(Frame,MinAxes)
MAKE_GETSETL(Frame,MatchEnd)
MAKE_GETSETL(Frame,Permute)
MAKE_GETSETL(Frame,PreserveAxes)

// Have to write our own GETSET routines for ActiveUnit
MAKE_GET( Frame, ActiveUnit,
          astGetActiveUnit( THIS ) ? Py_True : Py_False );
MAKE_SET( Frame, ActiveUnit, Bool, boolean, astSetActiveUnit( THIS, ( value == Py_True ) ); if (astOK) result = 0; );


static PyGetSetDef Frame_getseters[] = {
   #include "Bottom_desc.c"
   #include "Digits_desc.c"
   #include "Direction_desc.c"
   #include "Format_desc.c"
   #include "Label_desc.c"
   #include "NormUnit_desc.c"
   #include "Symbol_desc.c"
   #include "Top_desc.c"
   #include "Unit_desc.c"
   DEFATT(AlignSystem,"Coordinate system used to align Frames"),
   DEFATT(Domain, "Coordinate system domain"),
   DEFATT(Dut1, "Difference between the UT1 and UTC timescale"),
   DEFATT(Epoch, "Epoch of observation"),
   DEFATT(MatchEnd, "Match trailing axes?"),
   DEFATT(MaxAxes, "Maximum number of Frame axes to match"),
   DEFATT(MinAxes, "Minimum number of Frame axes to match"),
   DEFATT(Naxes, "Number of Frame axes"),
   DEFATT(ObsAlt, "Geodetic altitude of observer"),
   DEFATT(ObsLat, "Geodetic latitude of observer"),
   DEFATT(ObsLon, "Geodetic longitude of observer"),
   DEFATT(Permute, "Permute axis order?"),
   DEFATT(PreserveAxes, "Preserve axes?"),
   DEFATT(System, "Coordinate system used to describe the domain"),
   DEFATT(Title, "Frame title"),
   DEFATT(ActiveUnit, "Control how the frame behaves when it is used to match anothe Frame"),
   {NULL, NULL, NULL, NULL, NULL}  /* Sentinel */
};

/* Define the class Python type structure */
static PyTypeObject FrameType = {
   PyVarObject_HEAD_INIT(NULL, 0)
   CLASS,                     /* tp_name */
   sizeof(Frame),             /* tp_basicsize */
   0,                         /* tp_itemsize */
   0,                         /* tp_dealloc */
   0,                         /* tp_print */
   0,                         /* tp_getattr */
   0,                         /* tp_setattr */
   0,                         /* tp_reserved */
   0,                         /* tp_repr */
   0,                         /* tp_as_number */
   0,                         /* tp_as_sequence */
   0,                         /* tp_as_mapping */
   0,                         /* tp_hash  */
   0,                         /* tp_call */
   0,                         /* tp_str */
   0,                         /* tp_getattro */
   0,                         /* tp_setattro */
   0,                         /* tp_as_buffer */
   Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE, /* tp_flags */
   "AST Frame",               /* tp_doc */
   0,		              /* tp_traverse */
   0,		              /* tp_clear */
   0,		              /* tp_richcompare */
   0,		              /* tp_weaklistoffset */
   0,		              /* tp_iter */
   0,		              /* tp_iternext */
   Frame_methods,             /* tp_methods */
   0,                         /* tp_members */
   Frame_getseters,           /* tp_getset */
   0,                         /* tp_base */
   0,                         /* tp_dict */
   0,                         /* tp_descr_get */
   0,                         /* tp_descr_set */
   0,                         /* tp_dictoffset */
   (initproc)Frame_init,      /* tp_init */
   0,                         /* tp_alloc */
   0,                         /* tp_new */
};


/* Define the class methods */
static int Frame_init( Frame *self, PyObject *args, PyObject *kwds ){
   const char *options = " ";
   int result = -1;
   int naxes;

   if( PyArg_ParseTuple(args, "i|s:" CLASS, &naxes, &options ) ) {
      AstFrame *this = astFrame( naxes, options );
      result = SetProxy( (AstObject *) this, (Object *) self );
      this = astAnnul( this );
   }

   TIDY;
   return result;
}

#undef NAME
#define NAME CLASS ".angle"
static PyObject *Frame_angle( Frame *self, PyObject *args ) {
  PyObject *result = NULL;
  PyArrayObject *a = NULL;
  PyArrayObject *b = NULL;
  PyArrayObject *c = NULL;
  PyObject *a_object = NULL;
  PyObject *b_object = NULL;
  PyObject *c_object = NULL;
  int naxes;

  if( PyErr_Occurred() ) return NULL;

  naxes = astGetI( THIS, "Naxes" );
  if ( PyArg_ParseTuple( args, "OOO:" NAME, &a_object, &b_object,
                         &c_object ) && astOK ) {
    a = GetArray1D( a_object, &naxes, "a", NAME );
    b = GetArray1D( b_object, &naxes, "b", NAME );
    c = GetArray1D( c_object, &naxes, "c", NAME );
    if (a && b && c ) {
      double angle = astAngle( THIS, (const double *)a->data,
                               (const double *)b->data,
                               (const double *)c->data);
      if (astOK) result = Py_BuildValue( "d", angle );
    }
    Py_XDECREF( a );
    Py_XDECREF( b );
    Py_XDECREF( c );
  }

  TIDY;
  return result;
}

#undef NAME
#define NAME CLASS ".axangle"
static PyObject *Frame_axangle( Frame *self, PyObject *args ) {
  PyObject *result = NULL;
  PyArrayObject *a = NULL;
  PyArrayObject *b = NULL;
  PyObject *a_object = NULL;
  PyObject *b_object = NULL;
  int axis;
  int naxes;

  if( PyErr_Occurred() ) return NULL;

  naxes = astGetI( THIS, "Naxes" );
  if ( PyArg_ParseTuple( args, "OOi:" NAME, &a_object, &b_object,
                         &axis ) && astOK ) {
    a = GetArray1D( a_object, &naxes, "a", NAME );
    b = GetArray1D( b_object, &naxes, "b", NAME );
    if (a && b ) {
      double axangle = astAxAngle( THIS, (const double *)a->data,
                                   (const double *)b->data, axis );
      if (astOK) result = Py_BuildValue( "d", axangle );
    }
    Py_XDECREF( a );
    Py_XDECREF( b );
  }

  TIDY;
  return result;
}

#undef NAME
#define NAME CLASS ".axdistance"
static PyObject *Frame_axdistance( Frame *self, PyObject *args ) {
  PyObject *result = NULL;
  int axis;
  double v1;
  double v2;

  if( PyErr_Occurred() ) return NULL;

  if ( PyArg_ParseTuple( args, "idd:" NAME, &axis, &v1, &v2 ) && astOK ) {
    double axdistance = astAxDistance( THIS, axis, v1, v2 );
    if (astOK) result = Py_BuildValue( "d", axdistance );
  }

  TIDY;
  return result;
}

#undef NAME
#define NAME CLASS ".axoffset"
static PyObject *Frame_axoffset( Frame *self, PyObject *args ) {
  PyObject *result = NULL;
  int axis;
  double v1;
  double dist;

  if ( PyArg_ParseTuple( args, "idd:" NAME, &axis, &v1, &dist ) && astOK ) {
    double axoffset = astAxOffset( THIS, axis, v1, dist );
    if (astOK) result = Py_BuildValue( "d", axoffset );
  }

  TIDY;
  return result;
}

#undef NAME
#define NAME CLASS ".convert"
static PyObject *Frame_convert( Frame *self, PyObject *args ) {
  Object *other = NULL;
  PyObject *result = NULL;
  const char *domainlist = NULL;

  if( PyErr_Occurred() ) return NULL;

  if( PyArg_ParseTuple(args, "O!|s:" NAME, &FrameType,
                       (PyObject**)&other, &domainlist ) && astOK ) {
      AstFrameSet *conversion = astConvert( THIS, THAT,
                                            (domainlist ? domainlist : "" ) );
      if (astOK) {
        PyObject *conversion_object = NewObject( (AstObject *)conversion );
        if (conversion_object) {
          result = Py_BuildValue( "O", conversion_object );
        }
        Py_XDECREF( conversion_object );
      }
      if (conversion) conversion = astAnnul( conversion );
   }

   TIDY;
   return result;
}

#undef NAME
#define NAME CLASS ".distance"
static PyObject *Frame_distance( Frame *self, PyObject *args ) {
  PyObject *result = NULL;
  PyArrayObject *point1 = NULL;
  PyArrayObject *point2 = NULL;
  PyObject *point1_object = NULL;
  PyObject *point2_object = NULL;
  int naxes;

  if( PyErr_Occurred() ) return NULL;

  naxes = astGetI( THIS, "Naxes" );
  if ( PyArg_ParseTuple( args, "OO:" NAME, &point1_object,
                          &point2_object ) && astOK ) {
    point1 = GetArray1D( point1_object, &naxes, "point1", NAME );
    point2 = GetArray1D( point2_object, &naxes, "point2", NAME );
    if (point1 && point2 ) {
      double distance = astDistance( THIS, (const double *)point1->data,
                                   (const double *)point2->data );
      if (astOK) result = Py_BuildValue( "d", distance );
    }
    Py_XDECREF( point1 );
    Py_XDECREF( point2 );
  }

  TIDY;
  return result;
}

#undef NAME
#define NAME CLASS ".findframe"
static PyObject *Frame_findframe( Frame *self, PyObject *args ) {
  Object *other = NULL;
  PyObject *result = NULL;
  const char *domainlist = NULL;

  if( PyErr_Occurred() ) return NULL;

  if( PyArg_ParseTuple(args, "O!|s:" NAME, &FrameType,
                       (PyObject**)&other, &domainlist ) && astOK ) {
      AstFrameSet *found = astFindFrame( THIS, THAT,
                                         (domainlist ? domainlist : "" ) );
      if (astOK) {
        PyObject *found_object = NewObject( (AstObject *)found );
        if (found_object) {
          result = Py_BuildValue( "O", found_object );
        }
        Py_XDECREF( found_object );
      }
      if (found) found = astAnnul( found );
   }

   TIDY;
   return result;
}

#undef NAME
#define NAME CLASS ".format"
static PyObject *Frame_format( Frame *self, PyObject *args ) {
  PyObject *result = NULL;
  int axis;
  double value;

  if( PyErr_Occurred() ) return NULL;

  if ( PyArg_ParseTuple( args, "id:" NAME, &axis, &value ) && astOK ) {
    const char * format = astFormat( THIS, axis, value );
    if (astOK) result = Py_BuildValue( "s", format );
  }

  TIDY;
  return result;
}

#undef NAME
#define NAME CLASS ".intersect"
static PyObject *Frame_intersect( Frame *self, PyObject *args ) {
  PyObject *result = NULL;
  PyArrayObject *a1 = NULL;
  PyArrayObject *a2 = NULL;
  PyArrayObject *b1 = NULL;
  PyArrayObject *b2 = NULL;
  PyArrayObject *out = NULL;
  PyObject *a1_object = NULL;
  PyObject *a2_object = NULL;
  PyObject *b1_object = NULL;
  PyObject *b2_object = NULL;
  int naxes;
  npy_intp dims[1];

  if( PyErr_Occurred() ) return NULL;

  naxes = astGetI( THIS, "Naxes" );
  if ( PyArg_ParseTuple( args, "OOOO:" NAME, &a1_object,
                         &a2_object, &b1_object, &b2_object ) && astOK ) {
    a1 = GetArray1D( a1_object, &naxes, "a1", NAME );
    a2 = GetArray1D( a2_object, &naxes, "a2", NAME );
    b1 = GetArray1D( b1_object, &naxes, "b1", NAME );
    b2 = GetArray1D( b2_object, &naxes, "b2", NAME );
    dims[0] = naxes;
    out = (PyArrayObject *) PyArray_SimpleNew( 1, dims, PyArray_DOUBLE );
    if (a1 && a2 && b1 && b2 && out ) {
      astIntersect( THIS, (const double *)a1->data,
                    (const double *)a2->data,
                    (const double *)b1->data,
                    (const double *)b2->data, (double *)out->data );
      if (astOK) result = Py_BuildValue("O", PyArray_Return(out));
    }
    Py_XDECREF( a1 );
    Py_XDECREF( a2 );
    Py_XDECREF( b1 );
    Py_XDECREF( b2 );
    Py_XDECREF( out );
  }

  TIDY;
  return result;
}

#undef NAME
#define NAME CLASS ".matchaxes"
static PyObject *Frame_matchaxes( Frame *self, PyObject *args ) {
   PyObject *result = NULL;
   Frame *other;
   npy_intp dims[1];
   PyArrayObject * axes = NULL;

   if( PyErr_Occurred() ) return NULL;

   if( PyArg_ParseTuple( args, "O!:" NAME, &FrameType,
                         (PyObject **) &other ) && astOK ) {
     dims[0] = astGetI( THAT, "Naxes" );
     axes = (PyArrayObject *) PyArray_SimpleNew( 1, dims, PyArray_INT );
     if (axes) {
       astMatchAxes( THIS, THAT, (int *)axes->data );
       if( astOK ) result = Py_BuildValue("O", PyArray_Return(axes));
     }
     Py_XDECREF( axes );
   }
   TIDY;
   return result;
}

#undef NAME
#define NAME CLASS ".norm"
static PyObject *Frame_norm( Frame *self, PyObject *args ) {
  PyObject *result = NULL;
  PyArrayObject *value = NULL;
  PyArrayObject *axes = NULL;
  PyObject *value_object = NULL;
  int naxes;
  npy_intp dims[1];

  if( PyErr_Occurred() ) return NULL;

  naxes = astGetI( THIS, "Naxes" );
  if ( PyArg_ParseTuple( args, "O:" NAME,
                          &value_object ) && astOK ) {
    value = GetArray1D( value_object, &naxes, "value", NAME );
    dims[0] = naxes;
    axes = (PyArrayObject *) PyArray_SimpleNew( 1, dims, PyArray_DOUBLE );
    if ( value && axes ) {
      memcpy( axes->data, value->data, sizeof(double)*naxes);
      astNorm( THIS, (double *)axes->data );
      if (astOK) result = Py_BuildValue( "O", PyArray_Return(axes) );
    }
    Py_XDECREF( value );
    Py_XDECREF( axes );
  }

  TIDY;
  return result;
}

#undef NAME
#define NAME CLASS ".offset"
static PyObject *Frame_offset( Frame *self, PyObject *args ) {
  PyObject *result = NULL;
  PyArrayObject *point1 = NULL;
  PyArrayObject *point2 = NULL;
  PyArrayObject *point3 = NULL;
  PyObject *point1_object = NULL;
  PyObject *point2_object = NULL;
  int naxes;
  npy_intp dims[1];
  double offset;

  if( PyErr_Occurred() ) return NULL;

  naxes = astGetI( THIS, "Naxes" );
  if ( PyArg_ParseTuple( args, "OOd:" NAME, &point1_object,
                         &point2_object, &offset ) && astOK ) {
    point1 = GetArray1D( point1_object, &naxes, "point1", NAME );
    point2 = GetArray1D( point2_object, &naxes, "point2", NAME );
    dims[0] = naxes;
    point3 = (PyArrayObject *) PyArray_SimpleNew( 1, dims, PyArray_DOUBLE );
    if (point1 && point2 && point3 ) {
      astOffset( THIS, (const double *)point1->data,
                 (const double *)point2->data, offset,
                 (double *)point3->data );
      if (astOK) result = Py_BuildValue("O", PyArray_Return(point3));
    }
    Py_XDECREF( point1 );
    Py_XDECREF( point2 );
    Py_XDECREF( point3);
  }

  TIDY;
  return result;
}

#undef NAME
#define NAME CLASS ".offset2"
static PyObject *Frame_offset2( Frame *self, PyObject *args ) {
  PyObject *result = NULL;
  PyArrayObject *point1 = NULL;
  PyArrayObject *point2 = NULL;
  PyObject *point1_object = NULL;
  int naxes;
  npy_intp dims[1];
  double offset;
  double angle;

  if( PyErr_Occurred() ) return NULL;

  naxes = astGetI( THIS, "Naxes" );
  if ( PyArg_ParseTuple( args, "Odd:" NAME, &point1_object,
                         &angle, &offset ) && astOK ) {
    point1 = GetArray1D( point1_object, &naxes, "point1", NAME );
    dims[0] = naxes;
    point2 = (PyArrayObject *) PyArray_SimpleNew( 1, dims, PyArray_DOUBLE );
    if (point1 && point2 ) {
      double direction = astOffset2( THIS, (const double *)point1->data,
                                    angle, offset,
                                    (double *)point2->data );
      if (astOK) result = Py_BuildValue("dO", direction, PyArray_Return(point2));
    }
    Py_XDECREF( point1 );
    Py_XDECREF( point2 );
  }

  TIDY;
  return result;
}

#undef NAME
#define NAME CLASS ".permaxes"
static PyObject *Frame_permaxes( Frame *self, PyObject *args ) {
  PyObject *result = NULL;
  PyArrayObject *perm = NULL;
  PyObject *perm_object = NULL;
  int naxes;

  if( PyErr_Occurred() ) return NULL;

  naxes = astGetI( THIS, "Naxes" );
  if ( PyArg_ParseTuple( args, "O:" NAME, &perm_object ) && astOK ) {
    perm = GetArray1I( perm_object, &naxes, "perm", NAME );
    if (perm) {
      astPermAxes( THIS, (const int *)perm->data );
      if (astOK) result = Py_None;
    }
    Py_XDECREF( perm );
  }

  TIDY;
  return result;
}

#undef NAME
#define NAME CLASS ".pickaxes"
static PyObject *Frame_pickaxes( Frame *self, PyObject *args ) {
  PyObject *result = NULL;
  PyArrayObject *axes = NULL;
  PyObject *axes_object = NULL;

  if( PyErr_Occurred() ) return NULL;

  // We get naxes from the axes argument
  if ( PyArg_ParseTuple( args, "O:" NAME, &axes_object ) && astOK ) {
    axes = (PyArrayObject *) PyArray_ContiguousFromAny( axes_object,
                                                        PyArray_INT, 0, 100);
    if (axes) {
      AstMapping *map = NULL;
      AstFrame * frame = NULL;
      int naxes;

      naxes = PyArray_Size( (PyObject*)axes );
      frame = astPickAxes( THIS, naxes,
                           (const int *)axes->data,
                           &map);
      if (astOK) {
        PyObject *map_object = NULL;
        PyObject *frame_object = NULL;
        frame_object = NewObject( (AstObject *)frame );
        map_object = NewObject( (AstObject *)map );
        if (frame_object && map_object ) {
          result = Py_BuildValue( "OO", frame_object, map_object );
        }
        Py_XDECREF( map_object );
        Py_XDECREF( frame_object );
      }
      if ( map ) map = astAnnul( map );
      if (frame) frame=astAnnul( frame );
    }
    Py_XDECREF( axes );
  }

  TIDY;
  return result;
}

#undef NAME
#define NAME CLASS ".resolve"
static PyObject *Frame_resolve( Frame *self, PyObject *args ) {
  PyObject *result = NULL;
  PyArrayObject *point1 = NULL;
  PyArrayObject *point2 = NULL;
  PyArrayObject *point3 = NULL;
  PyArrayObject *point4 = NULL;
  PyObject *point1_object = NULL;
  PyObject *point2_object = NULL;
  PyObject *point3_object = NULL;
  int naxes;
  npy_intp dims[1];

  if( PyErr_Occurred() ) return NULL;

  naxes = astGetI( THIS, "Naxes" );
  if ( PyArg_ParseTuple( args, "OOO:" NAME, &point1_object,
                         &point2_object, &point3_object ) && astOK ) {
    point1 = GetArray1D( point1_object, &naxes, "point1", NAME );
    point2 = GetArray1D( point2_object, &naxes, "point2", NAME );
    point3 = GetArray1D( point2_object, &naxes, "point3", NAME );
    dims[0] = naxes;
    point4 = (PyArrayObject *) PyArray_SimpleNew( 1, dims, PyArray_DOUBLE );
    if (point1 && point2 && point3 && point4) {
      double d1;
      double d2;
      astResolve( THIS, (const double *)point1->data,
                  (const double *)point2->data,
                  (const double *)point3->data,
                  (double *)point4->data, &d1, &d2);
      if (astOK) result = Py_BuildValue("Odd", PyArray_Return(point4), d1, d2);
    }
    Py_XDECREF( point1 );
    Py_XDECREF( point2 );
    Py_XDECREF( point3 );
    Py_XDECREF( point4 );
  }

  TIDY;
  return result;
}

#undef NAME
#define NAME CLASS ".unformat"
static PyObject *Frame_unformat( Frame *self, PyObject *args ) {
  PyObject *result = NULL;
  int axis;
  const char * string = NULL;

  if( PyErr_Occurred() ) return NULL;

  if ( PyArg_ParseTuple( args, "is:" NAME, &axis, &string ) && astOK ) {
    double value;
    int nchars;
    nchars = astUnformat( THIS, axis, string, &value );
    if (astOK) result = Py_BuildValue( "id", nchars, value );
  }

  TIDY;
  return result;
}

/* NormMap */
/* ======= */

/* Define a string holding the fully qualified Python class name. */
#undef CLASS
#define CLASS MODULE ".NormMap"

/* Define the class structure */
typedef struct {
   Mapping parent;
} NormMap;

/* Prototypes for class functions */
static int NormMap_init( NormMap *self, PyObject *args, PyObject *kwds );

/* Define the class Python type structure */
static PyTypeObject NormMapType = {
   PyVarObject_HEAD_INIT(NULL, 0)
   CLASS,                     /* tp_name */
   sizeof(NormMap),           /* tp_basicsize */
   0,                         /* tp_itemsize */
   0,                         /* tp_dealloc */
   0,                         /* tp_print */
   0,                         /* tp_getattr */
   0,                         /* tp_setattr */
   0,                         /* tp_reserved */
   0,                         /* tp_repr */
   0,                         /* tp_as_number */
   0,                         /* tp_as_sequence */
   0,                         /* tp_as_mapping */
   0,                         /* tp_hash  */
   0,                         /* tp_call */
   0,                         /* tp_str */
   0,                         /* tp_getattro */
   0,                         /* tp_setattro */
   0,                         /* tp_as_buffer */
   Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE, /* tp_flags */
   "AST NormMap",             /* tp_doc */
   0,		              /* tp_traverse */
   0,		              /* tp_clear */
   0,		              /* tp_richcompare */
   0,		              /* tp_weaklistoffset */
   0,		              /* tp_iter */
   0,		              /* tp_iternext */
   0,                         /* tp_methods */
   0,                         /* tp_members */
   0,                         /* tp_getset */
   0,                         /* tp_base */
   0,                         /* tp_dict */
   0,                         /* tp_descr_get */
   0,                         /* tp_descr_set */
   0,                         /* tp_dictoffset */
   (initproc)NormMap_init,    /* tp_init */
   0,                         /* tp_alloc */
   0,                         /* tp_new */
};


/* Define the class methods */
static int NormMap_init( NormMap *self, PyObject *args, PyObject *kwds ){
   const char *options = " ";
   Mapping *other;
   int result = -1;

   if( PyArg_ParseTuple(args, "O!|s:" CLASS, &FrameType, (PyObject**)&other,
                        &options ) ) {
      AstNormMap *this = astNormMap( THAT, options );
      result = SetProxy( (AstObject *) this, (Object *) self );
      this = astAnnul( this );
   }

   TIDY;
   return result;
}


/* FrameSet */
/* ======== */

/* Define a string holding the fully qualified Python class name. */
#undef CLASS
#define CLASS MODULE ".FrameSet"

/* Define the class structure */
typedef struct {
   Frame parent;
} FrameSet;

/* Prototypes for class functions */
static int FrameSet_init( FrameSet *self, PyObject *args, PyObject *kwds );
static PyObject *FrameSet_addframe( FrameSet *self, PyObject *args );
static PyObject *FrameSet_getframe( FrameSet *self, PyObject *args );
static PyObject *FrameSet_getmapping( FrameSet *self, PyObject *args );
static PyObject *FrameSet_remapframe( FrameSet *self, PyObject *args );
static PyObject *FrameSet_removeframe( FrameSet *self, PyObject *args );

/* Describe the methods of the class */
static PyMethodDef FrameSet_methods[] = {
  {"addframe", (PyCFunction)FrameSet_addframe, METH_VARARGS, "Add a Frame to a FrameSet to define a new coordinate system"},
  {"getframe", (PyCFunction)FrameSet_getframe, METH_VARARGS, "Obtain an reference to a specified Frame in a FrameSet"},
  {"getmapping", (PyCFunction)FrameSet_getmapping, METH_VARARGS, "Obtain a Mapping that converts between two Frames in a FrameSet"},
  {"remapframe", (PyCFunction)FrameSet_remapframe, METH_VARARGS, "Modify a Frame's relationship to other Frames in a FrameSet"},
  {"removeframe", (PyCFunction)FrameSet_removeframe, METH_VARARGS, "Remove a Frame from a FrameSet"},
   {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* Define the AST attributes of the class */
MAKE_GETSETI(FrameSet,Base)
MAKE_GETSETI(FrameSet,Current)
MAKE_GETSETI(FrameSet,Nframe)

static PyGetSetDef FrameSet_getseters[] = {
   DEFATT(Base,"FrameSet base Frame index"),
   DEFATT(Current,"FrameSet current Frame index"),
   DEFATT(Nframe,"Number of Frames in a FrameSet"),
   {NULL, NULL, NULL, NULL, NULL}  /* Sentinel */
};

/* Define the class Python type structure */
static PyTypeObject FrameSetType = {
   PyVarObject_HEAD_INIT(NULL, 0)
   CLASS,                     /* tp_name */
   sizeof(FrameSet),          /* tp_basicsize */
   0,                         /* tp_itemsize */
   0,                         /* tp_dealloc */
   0,                         /* tp_print */
   0,                         /* tp_getattr */
   0,                         /* tp_setattr */
   0,                         /* tp_reserved */
   0,                         /* tp_repr */
   0,                         /* tp_as_number */
   0,                         /* tp_as_sequence */
   0,                         /* tp_as_mapping */
   0,                         /* tp_hash  */
   0,                         /* tp_call */
   0,                         /* tp_str */
   0,                         /* tp_getattro */
   0,                         /* tp_setattro */
   0,                         /* tp_as_buffer */
   Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE, /* tp_flags */
   "AST FrameSet",            /* tp_doc */
   0,		              /* tp_traverse */
   0,		              /* tp_clear */
   0,		              /* tp_richcompare */
   0,		              /* tp_weaklistoffset */
   0,		              /* tp_iter */
   0,		              /* tp_iternext */
   FrameSet_methods,          /* tp_methods */
   0,                         /* tp_members */
   FrameSet_getseters,        /* tp_getset */
   0,                         /* tp_base */
   0,                         /* tp_dict */
   0,                         /* tp_descr_get */
   0,                         /* tp_descr_set */
   0,                         /* tp_dictoffset */
   (initproc)FrameSet_init,   /* tp_init */
   0,                         /* tp_alloc */
   0,                         /* tp_new */
};


/* Define the class methods */
static int FrameSet_init( FrameSet *self, PyObject *args, PyObject *kwds ){
   const char *options = " ";
   FrameSet *other;
   int result = -1;

   if( PyArg_ParseTuple(args, "O!|s:" CLASS, &FrameType, (PyObject**)&other, &options ) ) {
      AstFrameSet *this = astFrameSet( THAT, options );
      result = SetProxy( (AstObject *) this, (Object *) self );
      this = astAnnul( this );
   }

   TIDY;
   return result;
}

#undef NAME
#define NAME CLASS ".addframe"
static PyObject *FrameSet_addframe( FrameSet *self, PyObject *args ) {
  Object *other = NULL;
  Object *another = NULL;
  PyObject *result = NULL;
  int iframe;

  if( PyErr_Occurred() ) return NULL;

  if( PyArg_ParseTuple(args, "iO!O!:" NAME, &iframe,
                       &MappingType, (PyObject**)&other,
                       &FrameType, (PyObject**)&another ) && astOK ) {
      astAddFrame( THIS, iframe, THAT, ANOTHER );
      if (astOK) result = Py_None;
   }

   TIDY;
   return result;
}

#undef NAME
#define NAME CLASS ".getframe"
static PyObject *FrameSet_getframe( FrameSet *self, PyObject *args ) {
  PyObject *result = NULL;
  int iframe;

  if( PyErr_Occurred() ) return NULL;

  if( PyArg_ParseTuple(args, "i:" NAME, &iframe ) && astOK ) {
      AstFrame * frame = astGetFrame( THIS, iframe );
      if (astOK) {
        PyObject *frame_object = NULL;
        frame_object = NewObject( (AstObject *)frame );
        if (frame_object) {
          result = Py_BuildValue( "O", frame_object );
        }
        Py_XDECREF( frame_object );
        if (frame) frame = astAnnul(frame);
      }
   }

   TIDY;
   return result;
}

#undef NAME
#define NAME CLASS ".getmapping"
static PyObject *FrameSet_getmapping( FrameSet *self, PyObject *args ) {
  PyObject *result = NULL;
  int iframe1;
  int iframe2;

  if( PyErr_Occurred() ) return NULL;

  if( PyArg_ParseTuple(args, "ii:" NAME, &iframe1, &iframe2 ) && astOK ) {
      AstMapping * mapping = astGetMapping( THIS, iframe1, iframe2 );
      if (astOK) {
        PyObject *mapping_object = NULL;
        mapping_object = NewObject( (AstObject *)mapping );
        if (mapping_object) {
          result = Py_BuildValue( "O", mapping_object );
        }
        Py_XDECREF( mapping_object );
        if (mapping) mapping = astAnnul(mapping);
      }
   }

   TIDY;
   return result;
}

#undef NAME
#define NAME CLASS ".remapframe"
static PyObject *FrameSet_remapframe( FrameSet *self, PyObject *args ) {
  Object *other = NULL;
  PyObject *result = NULL;
  int iframe;

  if( PyErr_Occurred() ) return NULL;

  if( PyArg_ParseTuple(args, "iO!:" NAME, &iframe,
                       &MappingType, (PyObject**)&other ) && astOK ) {
      astRemapFrame( THIS, iframe, THAT );
      if (astOK) result = Py_None;
   }

   TIDY;
   return result;
}

#undef NAME
#define NAME CLASS ".removeframe"
static PyObject *FrameSet_removeframe( FrameSet *self, PyObject *args ) {
  PyObject *result = NULL;
  int iframe;

  if( PyErr_Occurred() ) return NULL;

  if( PyArg_ParseTuple(args, "i:" NAME, &iframe ) && astOK ) {
      astRemoveFrame( THIS, iframe );
      if (astOK) result = Py_None;
   }

   TIDY;
   return result;
}

/* CmpFrame */
/* ======== */

/* Define a string holding the fully qualified Python class name. */
#undef CLASS
#define CLASS MODULE ".CmpFrame"

/* Define the class structure */
typedef struct {
   Frame parent;
} CmpFrame;

/* Prototypes for class functions */
static int CmpFrame_init( CmpFrame *self, PyObject *args, PyObject *kwds );

/* Define the class Python type structure */
static PyTypeObject CmpFrameType = {
   PyVarObject_HEAD_INIT(NULL, 0)
   CLASS,                     /* tp_name */
   sizeof(CmpFrame),          /* tp_basicsize */
   0,                         /* tp_itemsize */
   0,                         /* tp_dealloc */
   0,                         /* tp_print */
   0,                         /* tp_getattr */
   0,                         /* tp_setattr */
   0,                         /* tp_reserved */
   0,                         /* tp_repr */
   0,                         /* tp_as_number */
   0,                         /* tp_as_sequence */
   0,                         /* tp_as_mapping */
   0,                         /* tp_hash  */
   0,                         /* tp_call */
   0,                         /* tp_str */
   0,                         /* tp_getattro */
   0,                         /* tp_setattro */
   0,                         /* tp_as_buffer */
   Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE, /* tp_flags */
   "AST CmpFrame",            /* tp_doc */
   0,		              /* tp_traverse */
   0,		              /* tp_clear */
   0,		              /* tp_richcompare */
   0,		              /* tp_weaklistoffset */
   0,		              /* tp_iter */
   0,		              /* tp_iternext */
   0,                         /* tp_methods */
   0,                         /* tp_members */
   0,                         /* tp_getset */
   0,                         /* tp_base */
   0,                         /* tp_dict */
   0,                         /* tp_descr_get */
   0,                         /* tp_descr_set */
   0,                         /* tp_dictoffset */
   (initproc)CmpFrame_init,   /* tp_init */
   0,                         /* tp_alloc */
   0,                         /* tp_new */
};


/* Define the class methods */
static int CmpFrame_init( CmpFrame *self, PyObject *args, PyObject *kwds ){
   const char *options = " ";
   FrameSet *other;
   FrameSet *another;
   int result = -1;

   if( PyArg_ParseTuple(args, "O!O!|s:" CLASS, &FrameType, (PyObject**)&other,
                        &FrameType, (PyObject**)&another, &options ) ) {
      AstCmpFrame *this = astCmpFrame( THAT, ANOTHER, options );
      result = SetProxy( (AstObject *) this, (Object *) self );
      this = astAnnul( this );
   }

   TIDY;
   return result;
}

/* SkyFrame */
/* ======== */

/* Define a string holding the fully qualified Python class name. */
#undef CLASS
#define CLASS MODULE ".SkyFrame"

/* Define the class structure */
typedef struct {
   Frame parent;
} SkyFrame;

/* Prototypes for class functions */
static int SkyFrame_init( SkyFrame *self, PyObject *args, PyObject *kwds );
static PyObject *SkyFrame_skyoffsetmap( SkyFrame *self, PyObject *args );

/* Describe the methods of the class */
static PyMethodDef SkyFrame_methods[] = {
  {"skyoffsetmap", (PyCFunction)SkyFrame_skyoffsetmap, METH_NOARGS,"Returns a Mapping which goes from absolute coordinates to offset coordinates"},
  {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* Define the AST attributes of the class */
#include "AsTime_def.c"
#include "IsLatAxis_def.c"
#include "IsLonAxis_def.c"
#include "SkyRefP_def.c"
#include "SkyRef_def.c"
MAKE_GETSETC(SkyFrame,Projection)
MAKE_GETSETC(SkyFrame,SkyRefIs)
MAKE_GETSETD(SkyFrame,Equinox)
MAKE_GETSETI(SkyFrame,LatAxis)
MAKE_GETSETI(SkyFrame,LonAxis)
MAKE_GETSETL(SkyFrame,AlignOffset)
MAKE_GETSETL(SkyFrame,NegLon)

static PyGetSetDef SkyFrame_getseters[] = {
   #include "AsTime_desc.c"
   #include "IsLatAxis_desc.c"
   #include "IsLonAxis_desc.c"
   #include "SkyRef_desc.c"
   #include "SkyRefP_desc.c"
   DEFATT(AlignOffset,"Align SkyFrames using the offset coordinate system?"),
   DEFATT(Equinox,"Epoch of the mean equinox"),
   DEFATT(LatAxis,"Index of the latitude axis"),
   DEFATT(LonAxis,"Index of the longitude axis"),
   DEFATT(NegLon,"Display longitude values in the range [-pi,pi]?"),
   DEFATT(Projection,"Sky projection description"),
   DEFATT(SkyRefIs,"Selects the nature of the offset coordinate system"),
   {NULL, NULL, NULL, NULL, NULL}  /* Sentinel */
};

/* Define the class Python type structure */
static PyTypeObject SkyFrameType = {
   PyVarObject_HEAD_INIT(NULL, 0)
   CLASS,                     /* tp_name */
   sizeof(SkyFrame),          /* tp_basicsize */
   0,                         /* tp_itemsize */
   0,                         /* tp_dealloc */
   0,                         /* tp_print */
   0,                         /* tp_getattr */
   0,                         /* tp_setattr */
   0,                         /* tp_reserved */
   0,                         /* tp_repr */
   0,                         /* tp_as_number */
   0,                         /* tp_as_sequence */
   0,                         /* tp_as_mapping */
   0,                         /* tp_hash  */
   0,                         /* tp_call */
   0,                         /* tp_str */
   0,                         /* tp_getattro */
   0,                         /* tp_setattro */
   0,                         /* tp_as_buffer */
   Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE, /* tp_flags */
   "AST SkyFrame",            /* tp_doc */
   0,		              /* tp_traverse */
   0,		              /* tp_clear */
   0,		              /* tp_richcompare */
   0,		              /* tp_weaklistoffset */
   0,		              /* tp_iter */
   0,		              /* tp_iternext */
   SkyFrame_methods,          /* tp_methods */
   0,                         /* tp_members */
   SkyFrame_getseters,        /* tp_getset */
   0,                         /* tp_base */
   0,                         /* tp_dict */
   0,                         /* tp_descr_get */
   0,                         /* tp_descr_set */
   0,                         /* tp_dictoffset */
   (initproc)SkyFrame_init,   /* tp_init */
   0,                         /* tp_alloc */
   0,                         /* tp_new */
};


/* Define the class methods */
static int SkyFrame_init( SkyFrame *self, PyObject *args, PyObject *kwds ){
   const char *options = " ";
   int result = -1;

   if( PyArg_ParseTuple(args, "|s:" CLASS, &options ) ) {
      AstSkyFrame *this = astSkyFrame( options );
      result = SetProxy( (AstObject *) this, (Object *) self );
      this = astAnnul( this );
   }

   TIDY;
   return result;
}

#undef NAME
#define NAME CLASS ".skyoffsetmap"
static PyObject *SkyFrame_skyoffsetmap( SkyFrame *self, PyObject *args ) {
   PyObject *result = NULL;
   AstMapping * mapping = NULL;

   if( PyErr_Occurred() ) return NULL;

   mapping = astSkyOffsetMap( THIS  );
   if (astOK) {
      PyObject *mapping_object = NULL;
      mapping_object = NewObject( (AstObject *)mapping );
      if (mapping_object) {
         result = Py_BuildValue( "O", mapping_object );
      }
      Py_XDECREF( mapping_object );
   }
   if (mapping) mapping = astAnnul(mapping);

   TIDY;
   return result;
}

/* SpecFrame */
/* ========= */

/* Define a string holding the fully qualified Python class name. */
#undef CLASS
#define CLASS MODULE ".SpecFrame"

/* Define the class structure */
typedef struct {
   Frame parent;
} SpecFrame;

/* Prototypes for class functions */
static int SpecFrame_init( SpecFrame *self, PyObject *args, PyObject *kwds );
static PyObject *SpecFrame_setrefpos( SpecFrame *self, PyObject *args );
static PyObject *SpecFrame_getrefpos( SpecFrame *self, PyObject *args );

/* Describe the methods of the class */
static PyMethodDef SpecFrame_methods[] = {
  {"setrefpos", (PyCFunction)SpecFrame_setrefpos, METH_VARARGS,"Set the reference position in a specified celestial coordinate system"},
  {"getrefpos", (PyCFunction)SpecFrame_getrefpos, METH_VARARGS, "Return the reference position in a specified celestial coordinate system"},
  {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* Define the AST attributes of the class */
MAKE_GETSETL(SpecFrame,AlignSpecOffset)
MAKE_GETSETC(SpecFrame,AlignStdOfRest)
MAKE_GETSETC(SpecFrame,RefDec)
MAKE_GETSETC(SpecFrame,RefRA)
MAKE_GETSETD(SpecFrame,RestFreq)
MAKE_GETSETC(SpecFrame,SourceSys)
MAKE_GETSETD(SpecFrame,SourceVel)
MAKE_GETSETC(SpecFrame,SourceVRF)
MAKE_GETSETD(SpecFrame,SpecOrigin)
MAKE_GETSETC(SpecFrame,StdOfRest)

static PyGetSetDef SpecFrame_getseters[] = {
   DEFATT(AlignSpecOffset,"Align SpecFrames using the offset coordinate system?"),
   DEFATT(AlignStdOfRest,"Standard of rest in which to align SpecFrames"),
   DEFATT(RefDec,"Declination of the source (FK5 J2000)"),
   DEFATT(RefRA,"Right ascension of the source (FK5 J2000)"),
   DEFATT(RestFreq,"Rest frequency"),
   DEFATT(SourceSys,"Source velocity spectral system"),
   DEFATT(SourceVel,"Source velocity"),
   DEFATT(SourceVRF,"Source velocity rest frame"),
   DEFATT(SpecOrigin,"The zero point for SpecFrame axis values"),
   DEFATT(StdOfRest,"Standard of rest"),
   {NULL, NULL, NULL, NULL, NULL}  /* Sentinel */
};

/* Define the class Python type structure */
static PyTypeObject SpecFrameType = {
   PyVarObject_HEAD_INIT(NULL, 0)
   CLASS,                     /* tp_name */
   sizeof(SpecFrame),         /* tp_basicsize */
   0,                         /* tp_itemsize */
   0,                         /* tp_dealloc */
   0,                         /* tp_print */
   0,                         /* tp_getattr */
   0,                         /* tp_setattr */
   0,                         /* tp_reserved */
   0,                         /* tp_repr */
   0,                         /* tp_as_number */
   0,                         /* tp_as_sequence */
   0,                         /* tp_as_mapping */
   0,                         /* tp_hash  */
   0,                         /* tp_call */
   0,                         /* tp_str */
   0,                         /* tp_getattro */
   0,                         /* tp_setattro */
   0,                         /* tp_as_buffer */
   Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE, /* tp_flags */
   "AST SpecFrame",           /* tp_doc */
   0,		              /* tp_traverse */
   0,		              /* tp_clear */
   0,		              /* tp_richcompare */
   0,		              /* tp_weaklistoffset */
   0,		              /* tp_iter */
   0,		              /* tp_iternext */
   SpecFrame_methods,         /* tp_methods */
   0,                         /* tp_members */
   SpecFrame_getseters,       /* tp_getset */
   0,                         /* tp_base */
   0,                         /* tp_dict */
   0,                         /* tp_descr_get */
   0,                         /* tp_descr_set */
   0,                         /* tp_dictoffset */
   (initproc)SpecFrame_init,  /* tp_init */
   0,                         /* tp_alloc */
   0,                         /* tp_new */
};


/* Define the class methods */
static int SpecFrame_init( SpecFrame *self, PyObject *args, PyObject *kwds ){
   const char *options = " ";
   int result = -1;

   if( PyArg_ParseTuple(args, "|s:" CLASS, &options ) ) {
      AstSpecFrame *this = astSpecFrame( options );
      result = SetProxy( (AstObject *) this, (Object *) self );
      this = astAnnul( this );
   }

   TIDY;
   return result;
}

#undef NAME
#define NAME CLASS ".setrefpos"
static PyObject *SpecFrame_setrefpos( SpecFrame *self, PyObject *args ) {
  Object *other = NULL;
  PyObject *result = NULL;
  double lon;
  double lat;

  if( PyErr_Occurred() ) return NULL;

  if( PyArg_ParseTuple(args, "O!dd:" NAME,
                       &SkyFrameType, (PyObject**)&other,
                       &lon, &lat ) && astOK ) {
      astSetRefPos( THIS, THAT, lon, lat );
      if (astOK) result = Py_None;
   }

   TIDY;
   return result;
}

#undef NAME
#define NAME CLASS ".getrefpos"
static PyObject *SpecFrame_getrefpos( SpecFrame *self, PyObject *args ) {
  Object *other = NULL;
  PyObject *result = NULL;
  double lon;
  double lat;

  if( PyErr_Occurred() ) return NULL;

  if( PyArg_ParseTuple(args, "O!:" NAME,
                       &SkyFrameType, (PyObject**)&other ) && astOK ) {
      astGetRefPos( THIS, THAT, &lon, &lat );
      if (astOK) result = Py_BuildValue( "dd", lon, lat );
   }

   TIDY;
   return result;
}

/* DSBSpecFrame */
/* ============ */

/* Define a string holding the fully qualified Python class name. */
#undef CLASS
#define CLASS MODULE ".DSBSpecFrame"

/* Define the class structure */
typedef struct {
   SpecFrame parent;
} DSBSpecFrame;

/* Prototypes for class functions */
static int DSBSpecFrame_init( DSBSpecFrame *self, PyObject *args, PyObject *kwds );

/* Define the AST attributes of the class */
MAKE_GETSETL(DSBSpecFrame,AlignSideBand)
MAKE_GETSETD(DSBSpecFrame,DSBCentre)
MAKE_GETSETD(DSBSpecFrame,IF)
MAKE_GETSETD(DSBSpecFrame,ImagFreq)
MAKE_GETSETC(DSBSpecFrame,SideBand)

static PyGetSetDef DSBSpecFrame_getseters[] = {
   DEFATT(AlignSideBand,"Should alignment occur between sidebands?"),
   DEFATT(DSBCentre,"The central position of interest"),
   DEFATT(IF,"The intermediate frequency used to define the LO frequency"),
   DEFATT(ImagFreq,"The image sideband equivalent of the rest frequency"),
   DEFATT(SideBand,"Indicates which sideband the DSBSpecFrame represents"),
   {NULL, NULL, NULL, NULL, NULL}  /* Sentinel */
};

/* Define the class Python type structure */
static PyTypeObject DSBSpecFrameType = {
   PyVarObject_HEAD_INIT(NULL, 0)
   CLASS,                     /* tp_name */
   sizeof(DSBSpecFrame),      /* tp_basicsize */
   0,                         /* tp_itemsize */
   0,                         /* tp_dealloc */
   0,                         /* tp_print */
   0,                         /* tp_getattr */
   0,                         /* tp_setattr */
   0,                         /* tp_reserved */
   0,                         /* tp_repr */
   0,                         /* tp_as_number */
   0,                         /* tp_as_sequence */
   0,                         /* tp_as_mapping */
   0,                         /* tp_hash  */
   0,                         /* tp_call */
   0,                         /* tp_str */
   0,                         /* tp_getattro */
   0,                         /* tp_setattro */
   0,                         /* tp_as_buffer */
   Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE, /* tp_flags */
   "AST DSBSpecFrame",        /* tp_doc */
   0,		              /* tp_traverse */
   0,		              /* tp_clear */
   0,		              /* tp_richcompare */
   0,		              /* tp_weaklistoffset */
   0,		              /* tp_iter */
   0,		              /* tp_iternext */
   0,                         /* tp_methods */
   0,                         /* tp_members */
   DSBSpecFrame_getseters,    /* tp_getset */
   0,                         /* tp_base */
   0,                         /* tp_dict */
   0,                         /* tp_descr_get */
   0,                         /* tp_descr_set */
   0,                         /* tp_dictoffset */
   (initproc)DSBSpecFrame_init,/* tp_init */
   0,                         /* tp_alloc */
   0,                         /* tp_new */
};


/* Define the class methods */
static int DSBSpecFrame_init( DSBSpecFrame *self, PyObject *args, PyObject *kwds ){
   const char *options = " ";
   int result = -1;

   if( PyArg_ParseTuple(args, "|s:" CLASS, &options ) ) {
      AstDSBSpecFrame *this = astDSBSpecFrame( options );
      result = SetProxy( (AstObject *) this, (Object *) self );
      this = astAnnul( this );
   }

   TIDY;
   return result;
}

/* TimeFrame */
/* ========= */

/* Define a string holding the fully qualified Python class name. */
#undef CLASS
#define CLASS MODULE ".TimeFrame"

/* Define the class structure */
typedef struct {
   Frame parent;
} TimeFrame;

/* Prototypes for class functions */
static int TimeFrame_init( TimeFrame *self, PyObject *args, PyObject *kwds );
static PyObject *TimeFrame_currenttime( TimeFrame *self );

/* Describe the methods of the class */
static PyMethodDef TimeFrame_methods[] = {
  {"currenttime", (PyCFunction)TimeFrame_currenttime, METH_NOARGS,"Return the current system time"},
  {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* Define the AST attributes of the class */
MAKE_GETSETC(TimeFrame,AlignTimeScale)
MAKE_GETSETD(TimeFrame,LTOffset)
MAKE_GETSETD(TimeFrame,TimeOrigin)
MAKE_GETSETC(TimeFrame,TimeScale)

static PyGetSetDef TimeFrame_getseters[] = {
   DEFATT(AlignTimeScale,"Time scale in which to align TimeFrames"),
   DEFATT(LTOffset,"The offset of Local Time from UTC in hours"),
   DEFATT(TimeOrigin,"The zero point for TimeFrame axis values"),
   DEFATT(TimeScale,"The timescale used by the TimeFrame"),
   {NULL, NULL, NULL, NULL, NULL}  /* Sentinel */
};

/* Define the class Python type structure */
static PyTypeObject TimeFrameType = {
   PyVarObject_HEAD_INIT(NULL, 0)
   CLASS,                     /* tp_name */
   sizeof(TimeFrame),          /* tp_basicsize */
   0,                         /* tp_itemsize */
   0,                         /* tp_dealloc */
   0,                         /* tp_print */
   0,                         /* tp_getattr */
   0,                         /* tp_setattr */
   0,                         /* tp_reserved */
   0,                         /* tp_repr */
   0,                         /* tp_as_number */
   0,                         /* tp_as_sequence */
   0,                         /* tp_as_mapping */
   0,                         /* tp_hash  */
   0,                         /* tp_call */
   0,                         /* tp_str */
   0,                         /* tp_getattro */
   0,                         /* tp_setattro */
   0,                         /* tp_as_buffer */
   Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE, /* tp_flags */
   "AST TimeFrame",            /* tp_doc */
   0,		              /* tp_traverse */
   0,		              /* tp_clear */
   0,		              /* tp_richcompare */
   0,		              /* tp_weaklistoffset */
   0,		              /* tp_iter */
   0,		              /* tp_iternext */
   TimeFrame_methods,         /* tp_methods */
   0,                         /* tp_members */
   TimeFrame_getseters,       /* tp_getset */
   0,                         /* tp_base */
   0,                         /* tp_dict */
   0,                         /* tp_descr_get */
   0,                         /* tp_descr_set */
   0,                         /* tp_dictoffset */
   (initproc)TimeFrame_init,   /* tp_init */
   0,                         /* tp_alloc */
   0,                         /* tp_new */
};


/* Define the class methods */
static int TimeFrame_init( TimeFrame *self, PyObject *args, PyObject *kwds ){
   const char *options = " ";
   int result = -1;

   if( PyArg_ParseTuple(args, "|s:" CLASS, &options ) ) {
      AstTimeFrame *this = astTimeFrame( options );
      result = SetProxy( (AstObject *) this, (Object *) self );
      this = astAnnul( this );
   }

   TIDY;
   return result;
}

#undef NAME
#define NAME CLASS ".currenttime"
static PyObject *TimeFrame_currenttime( TimeFrame *self ) {
   PyObject *result = NULL;

   if( PyErr_Occurred() ) return NULL;

   if ( astOK ) {
      double currtime = astCurrentTime( THIS );
      if (astOK) result = Py_BuildValue( "d", currtime );
   }

   TIDY;
   return result;
}

/* FluxFrame */
/* ========= */

/* Define a string holding the fully qualified Python class name. */
#undef CLASS
#define CLASS MODULE ".FluxFrame"

/* Define the class structure */
typedef struct {
   Frame parent;
} FluxFrame;

/* Prototypes for class functions */
static int FluxFrame_init( FluxFrame *self, PyObject *args, PyObject *kwds );

/* Define the AST attributes of the class */
MAKE_GETSETD(FluxFrame,SpecVal)

static PyGetSetDef FluxFrame_getseters[] = {
   DEFATT(SpecVal,"The spectral position at which the flux values are measured"),
   {NULL, NULL, NULL, NULL, NULL}  /* Sentinel */
};

/* Define the class Python type structure */
static PyTypeObject FluxFrameType = {
   PyVarObject_HEAD_INIT(NULL, 0)
   CLASS,                     /* tp_name */
   sizeof(FluxFrame),         /* tp_basicsize */
   0,                         /* tp_itemsize */
   0,                         /* tp_dealloc */
   0,                         /* tp_print */
   0,                         /* tp_getattr */
   0,                         /* tp_setattr */
   0,                         /* tp_reserved */
   0,                         /* tp_repr */
   0,                         /* tp_as_number */
   0,                         /* tp_as_sequence */
   0,                         /* tp_as_mapping */
   0,                         /* tp_hash  */
   0,                         /* tp_call */
   0,                         /* tp_str */
   0,                         /* tp_getattro */
   0,                         /* tp_setattro */
   0,                         /* tp_as_buffer */
   Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE, /* tp_flags */
   "AST FluxFrame",           /* tp_doc */
   0,		              /* tp_traverse */
   0,		              /* tp_clear */
   0,		              /* tp_richcompare */
   0,		              /* tp_weaklistoffset */
   0,		              /* tp_iter */
   0,		              /* tp_iternext */
   0,                         /* tp_methods */
   0,                         /* tp_members */
   FluxFrame_getseters,       /* tp_getset */
   0,                         /* tp_base */
   0,                         /* tp_dict */
   0,                         /* tp_descr_get */
   0,                         /* tp_descr_set */
   0,                         /* tp_dictoffset */
   (initproc)FluxFrame_init,  /* tp_init */
   0,                         /* tp_alloc */
   0,                         /* tp_new */
};


/* Define the class methods */
static int FluxFrame_init( FluxFrame *self, PyObject *args, PyObject *kwds ){
   const char *options = " ";
   int result = -1;
   double specval;
   Object *other;

   if( PyArg_ParseTuple(args, "dO!|s:" CLASS, &specval,
			&SpecFrameType, (PyObject**)&other, &options ) ) {
      AstFluxFrame *this = astFluxFrame( specval, THAT, options );
      result = SetProxy( (AstObject *) this, (Object *) self );
      this = astAnnul( this );
   }

   TIDY;
   return result;
}

/* SpecFluxFrame */
/* ============= */

/* Define a string holding the fully qualified Python class name. */
#undef CLASS
#define CLASS MODULE ".SpecFluxFrame"

/* Define the class structure */
typedef struct {
   CmpFrame parent;
} SpecFluxFrame;

/* Prototypes for class functions */
static int SpecFluxFrame_init( SpecFluxFrame *self, PyObject *args, PyObject *kwds );

/* Define the class Python type structure */
static PyTypeObject SpecFluxFrameType = {
   PyVarObject_HEAD_INIT(NULL, 0)
   CLASS,                     /* tp_name */
   sizeof(SpecFluxFrame),     /* tp_basicsize */
   0,                         /* tp_itemsize */
   0,                         /* tp_dealloc */
   0,                         /* tp_print */
   0,                         /* tp_getattr */
   0,                         /* tp_setattr */
   0,                         /* tp_reserved */
   0,                         /* tp_repr */
   0,                         /* tp_as_number */
   0,                         /* tp_as_sequence */
   0,                         /* tp_as_mapping */
   0,                         /* tp_hash  */
   0,                         /* tp_call */
   0,                         /* tp_str */
   0,                         /* tp_getattro */
   0,                         /* tp_setattro */
   0,                         /* tp_as_buffer */
   Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE, /* tp_flags */
   "AST SpecFluxFrame",       /* tp_doc */
   0,		              /* tp_traverse */
   0,		              /* tp_clear */
   0,		              /* tp_richcompare */
   0,		              /* tp_weaklistoffset */
   0,		              /* tp_iter */
   0,		              /* tp_iternext */
   0,                         /* tp_methods */
   0,                         /* tp_members */
   0,                         /* tp_getset */
   0,                         /* tp_base */
   0,                         /* tp_dict */
   0,                         /* tp_descr_get */
   0,                         /* tp_descr_set */
   0,                         /* tp_dictoffset */
   (initproc)SpecFluxFrame_init,/* tp_init */
   0,                         /* tp_alloc */
   0,                         /* tp_new */
};


/* Define the class methods */
static int SpecFluxFrame_init( SpecFluxFrame *self, PyObject *args, PyObject *kwds ){
   const char *options = " ";
   int result = -1;
   Object *other;
   Object *another;

   if( PyArg_ParseTuple(args, "O!O!|s:" CLASS,
			&SpecFrameType, (PyObject**)&other,
			&FluxFrameType, (PyObject**)&another, &options ) ) {
      AstSpecFluxFrame *this = astSpecFluxFrame( THAT, ANOTHER, options );
      result = SetProxy( (AstObject *) this, (Object *) self );
      this = astAnnul( this );
   }

   TIDY;
   return result;
}

/* Region */
/* ====== */

/* Define a string holding the fully qualified Python class name. */
#undef CLASS
#define CLASS MODULE ".Region"

/* Define the class structure */
typedef struct {
   Frame parent;
} Region;

/* Prototypes for class functions */
static PyObject *Region_getregionframe( Region *self );
static PyObject *Region_getregionbounds( Region *self );
static PyObject *Region_overlap( Region *self, PyObject * args );

/* Define the AST attributes of the class */
MAKE_GETSETL(Region,Adaptive)
MAKE_GETSETL(Region,Negated)
MAKE_GETSETL(Region,Closed)
MAKE_GETSETI(Region,MeshSize)
MAKE_GETSETD(Region,FillFactor)
MAKE_GETROL(Region,Bounded)

static PyGetSetDef Region_getseters[] = {
  DEFATT(Adaptive,"Should the area adapt to changes in the coordinate system?"),
  DEFATT(Negated,"Has the original region been negated?"),
  DEFATT(Closed,"Should the boundary be considered to be inside the region?"),
  DEFATT(MeshSize,"Number of points used to create a mesh covering the Region"),
  DEFATT(FillFactor,"Fraction of the Region which is of interest"),
  DEFATT(Bounded,"Is the Region bounded?"),
   {NULL, NULL, NULL, NULL, NULL}  /* Sentinel */
};

/* Describe the methods of the class */
static PyMethodDef Region_methods[] = {
  {"getregionframe", (PyCFunction)Region_getregionframe, METH_NOARGS, "Obtain an object of the encapsulated Frame within a Region"},
  {"getregionbounds", (PyCFunction)Region_getregionbounds, METH_NOARGS, "Returns the bounding box of Region"},
  {"overlap", (PyCFunction)Region_overlap, METH_VARARGS, "Test if two Regions overlap each other"},
  {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* Define the class Python type structure */
static PyTypeObject RegionType = {
   PyVarObject_HEAD_INIT(NULL, 0)
   CLASS,                     /* tp_name */
   sizeof(Region),            /* tp_basicsize */
   0,                         /* tp_itemsize */
   0,                         /* tp_dealloc */
   0,                         /* tp_print */
   0,                         /* tp_getattr */
   0,                         /* tp_setattr */
   0,                         /* tp_reserved */
   0,                         /* tp_repr */
   0,                         /* tp_as_number */
   0,                         /* tp_as_sequence */
   0,                         /* tp_as_mapping */
   0,                         /* tp_hash  */
   0,                         /* tp_call */
   0,                         /* tp_str */
   0,                         /* tp_getattro */
   0,                         /* tp_setattro */
   0,                         /* tp_as_buffer */
   Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE, /* tp_flags */
   "AST Region",              /* tp_doc */
   0,		              /* tp_traverse */
   0,		              /* tp_clear */
   0,		              /* tp_richcompare */
   0,		              /* tp_weaklistoffset */
   0,		              /* tp_iter */
   0,		              /* tp_iternext */
   Region_methods,            /* tp_methods */
   0,                         /* tp_members */
   Region_getseters,          /* tp_getset */
};


/* Define the class methods */
#undef NAME
#define NAME CLASS ".getregionbounds"
static PyObject *Region_getregionbounds( Region *self ) {
  PyObject *result = NULL;
  int naxes;
  npy_intp dims[1];
  PyArrayObject * lbnd = NULL;
  PyArrayObject * ubnd = NULL;

  if( PyErr_Occurred() ) return NULL;

  naxes = astGetI( THIS, "Naxes" );
  dims[0] = naxes;
  lbnd = (PyArrayObject *) PyArray_SimpleNew( 1, dims, PyArray_DOUBLE );
  ubnd = (PyArrayObject *) PyArray_SimpleNew( 1, dims, PyArray_DOUBLE );
  if( lbnd && ubnd ) {
     astGetRegionBounds( THIS, (double *)lbnd->data, (double*)ubnd->data );
     if (astOK) result = Py_BuildValue("OO", PyArray_Return(lbnd),
                                        PyArray_Return(ubnd));
  }
  Py_XDECREF(lbnd);
  Py_XDECREF(ubnd);

  TIDY;
  return result;
}

#undef NAME
#define NAME CLASS ".getregionframe"
static PyObject *Region_getregionframe( Region *self ) {
  PyObject *result = NULL;
  AstFrame * frame = NULL;

  if( PyErr_Occurred() ) return NULL;

  frame = astGetRegionFrame( THIS );
  if (astOK) {
     PyObject *frame_object = NULL;
     frame_object = NewObject( (AstObject *)frame );
     if (frame_object) {
       result = Py_BuildValue( "O", frame_object );
     }
     Py_XDECREF( frame_object );
     if (frame) frame = astAnnul(frame);
  }

  TIDY;
  return result;
}

#undef NAME
#define NAME CLASS ".overlap"
static PyObject *Region_overlap( Region *self, PyObject * args ) {
  PyObject *result = NULL;
  Region *other = NULL;

   if( PyErr_Occurred() ) return NULL;

   if( PyArg_ParseTuple( args, "O!:" NAME, &RegionType,
                         (PyObject **) &other ) && astOK ) {
      int overlap = astOverlap( THIS, THAT );
      if( astOK ) result = Py_BuildValue("i", overlap);
   }

  TIDY;
  return result;
}

/* Box */
/* === */

/* Define a string holding the fully qualified Python class name. */
#undef CLASS
#define CLASS MODULE ".Box"

/* Define the class structure */
typedef struct {
   Region parent;
} Box;

/* Prototypes for class functions */
static int Box_init( Box *self, PyObject *args, PyObject *kwds );

/* Define the class Python type structure */
static PyTypeObject BoxType = {
   PyVarObject_HEAD_INIT(NULL, 0)
   CLASS,                     /* tp_name */
   sizeof(Box),               /* tp_basicsize */
   0,                         /* tp_itemsize */
   0,                         /* tp_dealloc */
   0,                         /* tp_print */
   0,                         /* tp_getattr */
   0,                         /* tp_setattr */
   0,                         /* tp_reserved */
   0,                         /* tp_repr */
   0,                         /* tp_as_number */
   0,                         /* tp_as_sequence */
   0,                         /* tp_as_mapping */
   0,                         /* tp_hash  */
   0,                         /* tp_call */
   0,                         /* tp_str */
   0,                         /* tp_getattro */
   0,                         /* tp_setattro */
   0,                         /* tp_as_buffer */
   Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE, /* tp_flags */
   "AST box",                 /* tp_doc */
   0,		              /* tp_traverse */
   0,		              /* tp_clear */
   0,		              /* tp_richcompare */
   0,		              /* tp_weaklistoffset */
   0,		              /* tp_iter */
   0,		              /* tp_iternext */
   0,                         /* tp_methods */
   0,                         /* tp_members */
   0,                         /* tp_getset */
   0,                         /* tp_base */
   0,                         /* tp_dict */
   0,                         /* tp_descr_get */
   0,                         /* tp_descr_set */
   0,                         /* tp_dictoffset */
   (initproc)Box_init,        /* tp_init */
   0,                         /* tp_alloc */
   0,                         /* tp_new */
};


/* Define the class methods */
static int Box_init( Box *self, PyObject *args, PyObject *kwds ){
   const char *options = " ";
   Frame *other;
   Region *another = NULL;
   int form; /* boolean */
   PyArrayObject * point1 = NULL;
   PyArrayObject * point2 = NULL;
   PyObject * point1_object = NULL;
   PyObject * point2_object = NULL;
   int result = -1;

   if( PyArg_ParseTuple(args, "O!iOO|O!s:" CLASS,
			&FrameType, (PyObject**)&other,
                        &form, &point1_object, &point2_object,
			&RegionType, (PyObject**)&another, &options ) ) {
      int naxes;
      AstBox * this = NULL;
      AstRegion * unc = NULL;
      if (another) unc = (AstRegion *) ANOTHER;
      naxes = astGetI( THAT, "Naxes" );
      point1 = GetArray1D( point1_object, &naxes, "point1", NAME );
      point2 = GetArray1D( point2_object, &naxes, "point2", NAME );
      this = astBox( THAT, form, (const double*)point1->data,
                     (const double*)point2->data, unc, options );
      result = SetProxy( (AstObject *) this, (Object *) self );
      this = astAnnul( this );
   }

   TIDY;
   return result;
}

/* Circle */
/* ====== */

/* Define a string holding the fully qualified Python class name. */
#undef CLASS
#define CLASS MODULE ".Circle"

/* Define the class structure */
typedef struct {
   Region parent;
} Circle;

/* Prototypes for class functions */
static int Circle_init( Circle *self, PyObject *args, PyObject *kwds );

/* Define the class Python type structure */
static PyTypeObject CircleType = {
   PyVarObject_HEAD_INIT(NULL, 0)
   CLASS,                     /* tp_name */
   sizeof(Circle),            /* tp_basicsize */
   0,                         /* tp_itemsize */
   0,                         /* tp_dealloc */
   0,                         /* tp_print */
   0,                         /* tp_getattr */
   0,                         /* tp_setattr */
   0,                         /* tp_reserved */
   0,                         /* tp_repr */
   0,                         /* tp_as_number */
   0,                         /* tp_as_sequence */
   0,                         /* tp_as_mapping */
   0,                         /* tp_hash  */
   0,                         /* tp_call */
   0,                         /* tp_str */
   0,                         /* tp_getattro */
   0,                         /* tp_setattro */
   0,                         /* tp_as_buffer */
   Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE, /* tp_flags */
   "AST box",                 /* tp_doc */
   0,		              /* tp_traverse */
   0,		              /* tp_clear */
   0,		              /* tp_richcompare */
   0,		              /* tp_weaklistoffset */
   0,		              /* tp_iter */
   0,		              /* tp_iternext */
   0,                         /* tp_methods */
   0,                         /* tp_members */
   0,                         /* tp_getset */
   0,                         /* tp_base */
   0,                         /* tp_dict */
   0,                         /* tp_descr_get */
   0,                         /* tp_descr_set */
   0,                         /* tp_dictoffset */
   (initproc)Circle_init,     /* tp_init */
   0,                         /* tp_alloc */
   0,                         /* tp_new */
};


/* Define the class methods */
static int Circle_init( Circle *self, PyObject *args, PyObject *kwds ){
   const char *options = " ";
   Frame *other;
   Region *another = NULL;
   int form; /* boolean */
   PyArrayObject * centre = NULL;
   PyArrayObject * point = NULL;
   PyObject * centre_object = NULL;
   PyObject * point_object = NULL;
   int result = -1;

   if( PyArg_ParseTuple(args, "O!iOO|O!s:" CLASS,
			&FrameType, (PyObject**)&other,
                        &form, &centre_object, &point_object,
			&RegionType, (PyObject**)&another, &options ) ) {
      int naxes;
      AstCircle * this = NULL;
      AstRegion * unc = NULL;
      if (another) unc = (AstRegion *) ANOTHER;
      naxes = astGetI( THAT, "Naxes" );
      centre = GetArray1D( centre_object, &naxes, "centre", NAME );
      if (form == 1) naxes = 1;
      point = GetArray1D( point_object, &naxes, "point", NAME );
      if (centre && point) {
        this = astCircle( THAT, form, (const double*)centre->data,
                          (const double*)point->data, unc, options );
        result = SetProxy( (AstObject *) this, (Object *) self );
        this = astAnnul( this );
      }
   }

   TIDY;
   return result;
}

/* Ellipse */
/* ======= */

/* Define a string holding the fully qualified Python class name. */
#undef CLASS
#define CLASS MODULE ".Ellipse"

/* Define the class structure */
typedef struct {
   Region parent;
} Ellipse;

/* Prototypes for class functions */
static int Ellipse_init( Ellipse *self, PyObject *args, PyObject *kwds );

/* Define the class Python type structure */
static PyTypeObject EllipseType = {
   PyVarObject_HEAD_INIT(NULL, 0)
   CLASS,                     /* tp_name */
   sizeof(Ellipse),           /* tp_basicsize */
   0,                         /* tp_itemsize */
   0,                         /* tp_dealloc */
   0,                         /* tp_print */
   0,                         /* tp_getattr */
   0,                         /* tp_setattr */
   0,                         /* tp_reserved */
   0,                         /* tp_repr */
   0,                         /* tp_as_number */
   0,                         /* tp_as_sequence */
   0,                         /* tp_as_mapping */
   0,                         /* tp_hash  */
   0,                         /* tp_call */
   0,                         /* tp_str */
   0,                         /* tp_getattro */
   0,                         /* tp_setattro */
   0,                         /* tp_as_buffer */
   Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE, /* tp_flags */
   "AST box",                 /* tp_doc */
   0,		              /* tp_traverse */
   0,		              /* tp_clear */
   0,		              /* tp_richcompare */
   0,		              /* tp_weaklistoffset */
   0,		              /* tp_iter */
   0,		              /* tp_iternext */
   0,                         /* tp_methods */
   0,                         /* tp_members */
   0,                         /* tp_getset */
   0,                         /* tp_base */
   0,                         /* tp_dict */
   0,                         /* tp_descr_get */
   0,                         /* tp_descr_set */
   0,                         /* tp_dictoffset */
   (initproc)Ellipse_init,    /* tp_init */
   0,                         /* tp_alloc */
   0,                         /* tp_new */
};


/* Define the class methods */
static int Ellipse_init( Ellipse *self, PyObject *args, PyObject *kwds ){
   const char *options = " ";
   Frame *other;
   Region *another = NULL;
   int form; /* boolean */
   PyArrayObject * centre = NULL;
   PyArrayObject * point1 = NULL;
   PyArrayObject * point2 = NULL;
   PyObject * centre_object = NULL;
   PyObject * point1_object = NULL;
   PyObject * point2_object = NULL;
   int result = -1;

   if( PyArg_ParseTuple(args, "O!iOOO|O!s:" CLASS,
			&FrameType, (PyObject**)&other,
                        &form, &centre_object, &point1_object, &point2_object,
			&RegionType, (PyObject**)&another, &options ) ) {
      int naxes;
      AstEllipse * this = NULL;
      AstRegion * unc = NULL;
      if (another) unc = (AstRegion *) ANOTHER;
      naxes = 2;
      centre = GetArray1D( centre_object, &naxes, "centre", NAME );
      point1 = GetArray1D( point1_object, &naxes, "point1", NAME );
      point2 = GetArray1D( point2_object, &naxes, "point2", NAME );
      if (centre && point1 && point2 ) {
        this = astEllipse( THAT, form, (const double*)centre->data,
			   (const double*)point1->data,
			   (const double*)point2->data,
			   unc, options );
        result = SetProxy( (AstObject *) this, (Object *) self );
        this = astAnnul( this );
      }
   }

   TIDY;
   return result;
}

/* Interval */
/* ======== */

/* Define a string holding the fully qualified Python class name. */
#undef CLASS
#define CLASS MODULE ".Interval"

/* Define the class structure */
typedef struct {
   Region parent;
} Interval;

/* Prototypes for class functions */
static int Interval_init( Interval *self, PyObject *args, PyObject *kwds );

/* Define the class Python type structure */
static PyTypeObject IntervalType = {
   PyVarObject_HEAD_INIT(NULL, 0)
   CLASS,                     /* tp_name */
   sizeof(Interval),          /* tp_basicsize */
   0,                         /* tp_itemsize */
   0,                         /* tp_dealloc */
   0,                         /* tp_print */
   0,                         /* tp_getattr */
   0,                         /* tp_setattr */
   0,                         /* tp_reserved */
   0,                         /* tp_repr */
   0,                         /* tp_as_number */
   0,                         /* tp_as_sequence */
   0,                         /* tp_as_mapping */
   0,                         /* tp_hash  */
   0,                         /* tp_call */
   0,                         /* tp_str */
   0,                         /* tp_getattro */
   0,                         /* tp_setattro */
   0,                         /* tp_as_buffer */
   Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE, /* tp_flags */
   "AST box",                 /* tp_doc */
   0,		              /* tp_traverse */
   0,		              /* tp_clear */
   0,		              /* tp_richcompare */
   0,		              /* tp_weaklistoffset */
   0,		              /* tp_iter */
   0,		              /* tp_iternext */
   0,                         /* tp_methods */
   0,                         /* tp_members */
   0,                         /* tp_getset */
   0,                         /* tp_base */
   0,                         /* tp_dict */
   0,                         /* tp_descr_get */
   0,                         /* tp_descr_set */
   0,                         /* tp_dictoffset */
   (initproc)Interval_init,   /* tp_init */
   0,                         /* tp_alloc */
   0,                         /* tp_new */
};


/* Define the class methods */
static int Interval_init( Interval *self, PyObject *args, PyObject *kwds ){
   const char *options = " ";
   Frame *other;
   Region *another = NULL;
   PyArrayObject * ubnd = NULL;
   PyArrayObject * lbnd = NULL;
   PyObject * ubnd_object = NULL;
   PyObject * lbnd_object = NULL;
   int result = -1;

   if( PyArg_ParseTuple(args, "O!OO|O!s:" CLASS,
			&FrameType, (PyObject**)&other,
                        &lbnd_object, &ubnd_object,
			&RegionType, (PyObject**)&another, &options ) ) {
      int naxes;
      AstInterval * this = NULL;
      AstRegion * unc = NULL;
      if (another) unc = (AstRegion *) ANOTHER;
      naxes = astGetI( THAT, "Naxes" );
      lbnd = GetArray1D( lbnd_object, &naxes, "lbnd", NAME );
      ubnd = GetArray1D( ubnd_object, &naxes, "ubnd", NAME );
      if (lbnd && ubnd) {
        this = astInterval( THAT, (const double*)lbnd->data,
                          (const double*)ubnd->data, unc, options );
        result = SetProxy( (AstObject *) this, (Object *) self );
        this = astAnnul( this );
      }
   }

   TIDY;
   return result;
}

/* NullRegion */
/* ========== */

/* Define a string holding the fully qualified Python class name. */
#undef CLASS
#define CLASS MODULE ".NullRegion"

/* Define the class structure */
typedef struct {
   Region parent;
} NullRegion;

/* Prototypes for class functions */
static int NullRegion_init( NullRegion *self, PyObject *args, PyObject *kwds );

/* Define the class Python type structure */
static PyTypeObject NullRegionType = {
   PyVarObject_HEAD_INIT(NULL, 0)
   CLASS,                     /* tp_name */
   sizeof(NullRegion),        /* tp_basicsize */
   0,                         /* tp_itemsize */
   0,                         /* tp_dealloc */
   0,                         /* tp_print */
   0,                         /* tp_getattr */
   0,                         /* tp_setattr */
   0,                         /* tp_reserved */
   0,                         /* tp_repr */
   0,                         /* tp_as_number */
   0,                         /* tp_as_sequence */
   0,                         /* tp_as_mapping */
   0,                         /* tp_hash  */
   0,                         /* tp_call */
   0,                         /* tp_str */
   0,                         /* tp_getattro */
   0,                         /* tp_setattro */
   0,                         /* tp_as_buffer */
   Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE, /* tp_flags */
   "AST box",                 /* tp_doc */
   0,		              /* tp_traverse */
   0,		              /* tp_clear */
   0,		              /* tp_richcompare */
   0,		              /* tp_weaklistoffset */
   0,		              /* tp_iter */
   0,		              /* tp_iternext */
   0,                         /* tp_methods */
   0,                         /* tp_members */
   0,                         /* tp_getset */
   0,                         /* tp_base */
   0,                         /* tp_dict */
   0,                         /* tp_descr_get */
   0,                         /* tp_descr_set */
   0,                         /* tp_dictoffset */
   (initproc)NullRegion_init, /* tp_init */
   0,                         /* tp_alloc */
   0,                         /* tp_new */
};


/* Define the class methods */
static int NullRegion_init( NullRegion *self, PyObject *args, PyObject *kwds ){
   const char *options = " ";
   Frame *other;
   Region *another = NULL;
   int result = -1;

   if( PyArg_ParseTuple(args, "O!|O!s:" CLASS,
			&FrameType, (PyObject**)&other,
			&RegionType, (PyObject**)&another, &options ) ) {
      AstNullRegion * this = NULL;
      AstRegion * unc = NULL;
      if (another) unc = (AstRegion *) ANOTHER;
      this = astNullRegion( THAT, unc, options );
      result = SetProxy( (AstObject *) this, (Object *) self );
      this = astAnnul( this );
   }

   TIDY;
   return result;
}

/* CmpRegion */
/* ========= */

/* Define a string holding the fully qualified Python class name. */
#undef CLASS
#define CLASS MODULE ".CmpRegion"

/* Define the class structure */
typedef struct {
   Region parent;
} CmpRegion;

/* Prototypes for class functions */
static int CmpRegion_init( CmpRegion *self, PyObject *args, PyObject *kwds );

/* Define the class Python type structure */
static PyTypeObject CmpRegionType = {
   PyVarObject_HEAD_INIT(NULL, 0)
   CLASS,                     /* tp_name */
   sizeof(CmpRegion),         /* tp_basicsize */
   0,                         /* tp_itemsize */
   0,                         /* tp_dealloc */
   0,                         /* tp_print */
   0,                         /* tp_getattr */
   0,                         /* tp_setattr */
   0,                         /* tp_reserved */
   0,                         /* tp_repr */
   0,                         /* tp_as_number */
   0,                         /* tp_as_sequence */
   0,                         /* tp_as_mapping */
   0,                         /* tp_hash  */
   0,                         /* tp_call */
   0,                         /* tp_str */
   0,                         /* tp_getattro */
   0,                         /* tp_setattro */
   0,                         /* tp_as_buffer */
   Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE, /* tp_flags */
   "AST box",                 /* tp_doc */
   0,		              /* tp_traverse */
   0,		              /* tp_clear */
   0,		              /* tp_richcompare */
   0,		              /* tp_weaklistoffset */
   0,		              /* tp_iter */
   0,		              /* tp_iternext */
   0,                         /* tp_methods */
   0,                         /* tp_members */
   0,                         /* tp_getset */
   0,                         /* tp_base */
   0,                         /* tp_dict */
   0,                         /* tp_descr_get */
   0,                         /* tp_descr_set */
   0,                         /* tp_dictoffset */
   (initproc)CmpRegion_init,  /* tp_init */
   0,                         /* tp_alloc */
   0,                         /* tp_new */
};


/* Define the class methods */
static int CmpRegion_init( CmpRegion *self, PyObject *args, PyObject *kwds ){
   const char *options = " ";
   Region *other;
   Region *another;
   int result = -1;
   int oper;

   if( PyArg_ParseTuple(args, "O!O!i|s:" CLASS, &RegionType, (PyObject**)&other,
                        &RegionType, (PyObject**)&another, &oper, &options ) ) {
      AstCmpRegion *this = astCmpRegion( THAT, ANOTHER, oper, options );
      result = SetProxy( (AstObject *) this, (Object *) self );
      this = astAnnul( this );
   }

   TIDY;
   return result;
}

/* Prism */
/* ===== */

/* Define a string holding the fully qualified Python class name. */
#undef CLASS
#define CLASS MODULE ".Prism"

/* Define the class structure */
typedef struct {
   Region parent;
} Prism;

/* Prototypes for class functions */
static int Prism_init( Prism *self, PyObject *args, PyObject *kwds );

/* Define the class Python type structure */
static PyTypeObject PrismType = {
   PyVarObject_HEAD_INIT(NULL, 0)
   CLASS,                     /* tp_name */
   sizeof(Prism),             /* tp_basicsize */
   0,                         /* tp_itemsize */
   0,                         /* tp_dealloc */
   0,                         /* tp_print */
   0,                         /* tp_getattr */
   0,                         /* tp_setattr */
   0,                         /* tp_reserved */
   0,                         /* tp_repr */
   0,                         /* tp_as_number */
   0,                         /* tp_as_sequence */
   0,                         /* tp_as_mapping */
   0,                         /* tp_hash  */
   0,                         /* tp_call */
   0,                         /* tp_str */
   0,                         /* tp_getattro */
   0,                         /* tp_setattro */
   0,                         /* tp_as_buffer */
   Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE, /* tp_flags */
   "AST box",                 /* tp_doc */
   0,		              /* tp_traverse */
   0,		              /* tp_clear */
   0,		              /* tp_richcompare */
   0,		              /* tp_weaklistoffset */
   0,		              /* tp_iter */
   0,		              /* tp_iternext */
   0,                         /* tp_methods */
   0,                         /* tp_members */
   0,                         /* tp_getset */
   0,                         /* tp_base */
   0,                         /* tp_dict */
   0,                         /* tp_descr_get */
   0,                         /* tp_descr_set */
   0,                         /* tp_dictoffset */
   (initproc)Prism_init,      /* tp_init */
   0,                         /* tp_alloc */
   0,                         /* tp_new */
};


/* Define the class methods */
static int Prism_init( Prism *self, PyObject *args, PyObject *kwds ){
   const char *options = " ";
   Region *other;
   Region *another;
   int result = -1;

   if( PyArg_ParseTuple(args, "O!O!|s:" CLASS, &RegionType, (PyObject**)&other,
                        &RegionType, (PyObject**)&another, &options ) ) {
      AstPrism *this = astPrism( THAT, ANOTHER, options );
      result = SetProxy( (AstObject *) this, (Object *) self );
      this = astAnnul( this );
   }

   TIDY;
   return result;
}




/* Channel */
/* ======= */

/* Define a string holding the fully qualified Python class name. */
#undef CLASS
#define CLASS MODULE ".Channel"

/* Define the class structure */
typedef struct {
   Object parent;
   PyObject *source;
   PyObject *sink;
   char *source_line;
} Channel;

/* Prototypes for class functions */
/* static PyObject *Channel_warnings( Channel *self ); */
static PyObject *Channel_read( Channel *self );
static PyObject *Channel_write( Channel *self, PyObject *args );
static int Channel_init( Channel *self, PyObject *args, PyObject *kwds );
const char *source_wrapper( void );
void sink_wrapper( const char *text );
static int ChannelFuncs( Channel *self,  PyObject *source, PyObject *sink,
                         const char *(** source_wrap)( void ),
                         void (** sink_wrap)( const char * ) );


/* Describe the methods of the class */
static PyMethodDef Channel_methods[] = {
/*   {"warnings", (PyCFunction)Channel_warnings, METH_NOARGS, "Returns any warnings issued by the previous read or write operation"},*/
   {"read", (PyCFunction)Channel_read, METH_NOARGS, "Read an Object from a Channel."},
   {"write", (PyCFunction)Channel_write, METH_VARARGS, "Write an Object to a Channel."},
   {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* Define the AST attributes of the class */
MAKE_GETSETC(Channel,SourceFile)
MAKE_GETSETC(Channel,SinkFile)
MAKE_GETSETL(Channel,Comment)
MAKE_GETSETI(Channel,Full)
MAKE_GETSETI(Channel,Indent)
MAKE_GETSETI(Channel,ReportLevel)
MAKE_GETSETL(Channel,Skip)
MAKE_GETSETL(Channel,Strict)
static PyGetSetDef Channel_getseters[] = {
   DEFATT(SourceFile,"Input file from which to read data"),
   DEFATT(SinkFile,"Output file to which to data should be written"),
   DEFATT(Comment,"Include textual comments in output?"),
   DEFATT(Full,"Set level of output detail"),
   DEFATT(Indent,"Specifies the indentation to use in text produced by a Channel"),
   DEFATT(ReportLevel,"Determines which read/write conditions are reported"),
   DEFATT(Skip,"Skip irrelevant data?"),
   DEFATT(Strict,"Report an error if any unexpeted data items are found?"),
   {NULL, NULL, NULL, NULL, NULL}  /* Sentinel */
};

/* Define the class Python type structure */
static PyTypeObject ChannelType = {
   PyVarObject_HEAD_INIT(NULL, 0)
   CLASS,                     /* tp_name */
   sizeof(Channel),           /* tp_basicsize */
   0,                         /* tp_itemsize */
   0,                         /* tp_dealloc */
   0,                         /* tp_print */
   0,                         /* tp_getattr */
   0,                         /* tp_setattr */
   0,                         /* tp_reserved */
   0,                         /* tp_repr */
   0,                         /* tp_as_number */
   0,                         /* tp_as_sequence */
   0,                         /* tp_as_mapping */
   0,                         /* tp_hash  */
   0,                         /* tp_call */
   0,                         /* tp_str */
   0,                         /* tp_getattro */
   0,                         /* tp_setattro */
   0,                         /* tp_as_buffer */
   Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE, /* tp_flags */
   "AST Channel",             /* tp_doc */
   0,		              /* tp_traverse */
   0,		              /* tp_clear */
   0,		              /* tp_richcompare */
   0,		              /* tp_weaklistoffset */
   0,		              /* tp_iter */
   0,		              /* tp_iternext */
   Channel_methods,           /* tp_methods */
   0,                         /* tp_members */
   Channel_getseters,         /* tp_getset */
   0,                         /* tp_base */
   0,                         /* tp_dict */
   0,                         /* tp_descr_get */
   0,                         /* tp_descr_set */
   0,                         /* tp_dictoffset */
   (initproc)Channel_init,    /* tp_init */
   0,                         /* tp_alloc */
   0,                         /* tp_new */
};


/* Define the class methods */
static int Channel_init( Channel *self, PyObject *args, PyObject *kwds ){
   PyObject *source = NULL;
   PyObject *sink = NULL;
   const char *(* source_wrap)( void );
   void (* sink_wrap)( const char * );
   const char *options = " ";
   int result = -1;
   if( PyArg_ParseTuple(args, "|OOs:" CLASS, &source, &sink, &options ) ) {

/* Choose the source and sink wrapper functions and store info required
   by the source and sink functions in the Channel structure. */
      result = ChannelFuncs( self, source, sink, &source_wrap, &sink_wrap );

/* Create the channel using the above selected wrapper functions. */
      if( result == 0 ) {
         AstChannel *this = astChannel( source_wrap, sink_wrap, options );

/* Store a pointer to the PyObject Channel in the AST Channel so that the
   source and sink wrapper functions can get at it. */
         astPutChannelData( this, self );

/* Store self as the Python proxy for the AST Channel. */
         result = SetProxy( (AstObject *) this, (Object *) self );
         this = astAnnul( this );
      }
   }

   TIDY;
   return result;
}

/*
static PyObject *Channel_warnings( Channel *self ) {
   AstKeyMap *km;
   PyObject *result = NULL;
   PyObject *str;
   const char *text;
   int nkey;
   int ikey;

   if( PyErr_Occurred() ) return NULL;

   km = astWarnings( THIS );
   nkey = astMapSize( km );

   if( astOK ) {
      result = PyTuple_New( nkey );
      for( ikey = 0; ikey < nkey; ikey++ ) {
         astMapGet0C( THIS, astMapKey( THIS, ikey ), &text );
         str = Py_BuildValue( "s", text );
         PyTuple_SetItem( result, ikey, str );
      }
   }

   km = astAnnul( km );

   TIDY;
   return result;
}
*/
static PyObject *Channel_read( Channel *self ){
   PyObject *result = NULL;
   PyObject *object = NULL;
   AstObject *obj;

   if( PyErr_Occurred() ) return NULL;

   obj = astRead( THIS );
   self->source_line = astFree( self->source_line );
   if( astOK ) {
      if( obj ) {
         object = NewObject( (AstObject *) obj );
         if( object ) result = Py_BuildValue( "O", object );
         Py_XDECREF(object);
      } else {
         result = Py_None;
      }
   }
   if( obj ) obj = astAnnul( obj );
   TIDY;
   return result;
}


#undef NAME
#define NAME CLASS ".write"
static PyObject *Channel_write( Channel *self, PyObject *args ){
   Object *other = NULL;
   PyObject *result = NULL;
   int nwrite;

   if( PyErr_Occurred() ) return NULL;

   if( PyArg_ParseTuple(args, "O!:" NAME, &ObjectType, (PyObject**) &other ) ) {
      nwrite = astWrite( THIS, THAT );
      if (astOK) result = Py_BuildValue( "i", nwrite );
   }
   TIDY;
   return result;
}


static int ChannelFuncs( Channel *self, PyObject *source, PyObject *sink,
                         const char *(** source_wrap)( void ),
                         void (** sink_wrap)( const char * ) ) {
/*
*  Name:
*     ChannelFuncs

*  Purpose:
*     Choose the source and sink wrapper functions for a Channel, and
*     store information required by the source and sink functions in the
*     Channel structure.

*/

/* Initialise. */
   int result = 0;
   *source_wrap = NULL;
   *sink_wrap = NULL;

/* If a source object was supplied, we use the local "source_wrapper" function
   as a C-callable wrapper for the object's "source" method. Otherwise, we use
   a NULL wrapper. Also store a pointer to the source object in the Channel
   structure. */
   if( source ) {
      if( PyObject_HasAttrString( source, "source" ) ) {
         *source_wrap = source_wrapper;
         self->source = source;
      } else if( source != Py_None ){
         result = -1;
         PyErr_SetString( PyExc_TypeError, "The supplied 'source' "
                          "object does not have a 'source' method" );
      }
   }

/* Do the same for the sink object. */
   if( sink ) {
      if( PyObject_HasAttrString( sink, "sink" ) ) {
         *sink_wrap = sink_wrapper;
         self->sink = sink;
      } else if( sink != Py_None ) {
         result = -1;
         PyErr_SetString( PyExc_TypeError, "The supplied 'sink' "
                          "object does not have a 'sink' method" );
      }
   }

/* Initialise the pointer to the dynamically allocated string holding the
   line of text read most recently by the Channel's source function. */
   self->source_line = NULL;

/* Return the success flag */
   return result;
}

/* Source and sink functions which are called by the AST Channel C code.
   These invoke the source and sink methods on the Python Object
   associated with the Channel. Note, these cannot be static as they are
   called from within AST. */

const char *source_wrapper( void ){
   Channel *channel = astChannelData;
   PyObject *pytext = PyObject_CallMethod( channel->source, "source", NULL );
   channel->source_line = GetString( channel->source_line, pytext );
   Py_XDECREF(pytext);
   return channel->source_line;
}

void sink_wrapper( const char *text ){
   Channel *channel = astChannelData;
   PyObject *result = PyObject_CallMethod( channel->sink, "sink", "s", text );
   Py_XDECREF(result);
}


/* FitsChan */
/* ======== */

/* Define a string holding the fully qualified Python class name. */
#undef CLASS
#define CLASS MODULE ".FitsChan"

/* Define the class structure */
typedef struct {
   Channel parent;
} FitsChan;

/* Prototypes for class functions */
static PyObject *FitsChan_delfits( FitsChan *self );
static PyObject *FitsChan_emptyfits( FitsChan *self );
static PyObject *FitsChan_findfits( FitsChan *self, PyObject *args );
static PyObject *FitsChan_getfitsCF( FitsChan *self, PyObject *args );
static PyObject *FitsChan_getfitsCI( FitsChan *self, PyObject *args );
static PyObject *FitsChan_getfitsCN( FitsChan *self, PyObject *args );
static PyObject *FitsChan_getfitsF( FitsChan *self, PyObject *args );
static PyObject *FitsChan_getfitsI( FitsChan *self, PyObject *args );
static PyObject *FitsChan_getfitsL( FitsChan *self, PyObject *args );
static PyObject *FitsChan_getfitsS( FitsChan *self, PyObject *args );
static PyObject *FitsChan_getitem( PyObject *self, PyObject *keyword );
static PyObject *FitsChan_getiter( PyObject *self );
static PyObject *FitsChan_next( PyObject *self );
static PyObject *FitsChan_readfits( FitsChan *self );
static PyObject *FitsChan_writefits( FitsChan *self );
static Py_ssize_t FitsChan_length( PyObject *self );
static int FitsChan_init( FitsChan *self, PyObject *args, PyObject *kwds );
static int FitsChan_setitem( PyObject *self, PyObject *keyword, PyObject *value );
/* TBD static PyObject *FitsChan_gettables( FitsChan *self ); */
static PyObject *FitsChan_purgewcs( FitsChan *self );
static PyObject *FitsChan_putcards( FitsChan *self, PyObject *args );
static PyObject *FitsChan_putfits( FitsChan *self, PyObject *args );
/* TBD static PyObject *FitsChan_puttable( FitsChan *self, PyObject *args ); */
/* TBD static PyObject *FitsChan_puttables( FitsChan *self, PyObject *args ); */
/* TBD static PyObject *FitsChan_removetables( FitsChan *self, PyObject *args ); */
static PyObject *FitsChan_retainfits( FitsChan *self );
static PyObject *FitsChan_setfitsCF( FitsChan *self, PyObject *args );
static PyObject *FitsChan_setfitsCI( FitsChan *self, PyObject *args );
static PyObject *FitsChan_setfitsF( FitsChan *self, PyObject *args );
static PyObject *FitsChan_setfitsI( FitsChan *self, PyObject *args );
static PyObject *FitsChan_setfitsL( FitsChan *self, PyObject *args );
static PyObject *FitsChan_setfitsS( FitsChan *self, PyObject *args );
static PyObject *FitsChan_setfitsCN( FitsChan *self, PyObject *args );
static PyObject *FitsChan_testfits( FitsChan *self, PyObject *args );

/* Describe the methods of the class */
static PyMethodDef FitsChan_methods[] = {
   {"delfits", (PyCFunction)FitsChan_delfits, METH_NOARGS, "Delete the current FITS card in a FitsChan."},
   {"emptyfits", (PyCFunction)FitsChan_emptyfits, METH_NOARGS, "Delete all cards in a FitsChan."},
   {"findfits", (PyCFunction)FitsChan_findfits, METH_VARARGS, "Find a FITS card in a FitsChan by keyword."},
   {"getfitsCF", (PyCFunction)FitsChan_getfitsCF, METH_VARARGS, "Get a complex floating point keyword value from a FitsChan."},
   {"getfitsCI", (PyCFunction)FitsChan_getfitsCI, METH_VARARGS, "Get a complex integer keyword value from a FitsChan."},
   {"getfitsF", (PyCFunction)FitsChan_getfitsF, METH_VARARGS, "Get a floating point keyword value from a FitsChan."},
   {"getfitsI", (PyCFunction)FitsChan_getfitsI, METH_VARARGS, "Get an integer value from a FitsChan."},
   {"getfitsL", (PyCFunction)FitsChan_getfitsL, METH_VARARGS, "Get an integer value from a FitsChan."},
   {"getfitsS", (PyCFunction)FitsChan_getfitsS, METH_VARARGS, "Get a string keyword value from a FitsChan."},
   {"getfitsCN", (PyCFunction)FitsChan_getfitsCN, METH_VARARGS, "Get a string keyword value from a FitsChan."},
   {"purgewcs", (PyCFunction)FitsChan_purgewcs, METH_NOARGS, "Delete all WCS-related cards in a FitsChan."},
   {"putcards", (PyCFunction)FitsChan_putcards, METH_VARARGS, "Stores a set of FITS header card in a FitsChan."},
   {"putfits", (PyCFunction)FitsChan_putfits, METH_VARARGS, "Store a FITS header card in a FitsChan."},
   {"readfits", (PyCFunction)FitsChan_readfits, METH_NOARGS, "Read cards from the external source of a FitsChan."},
   {"retainfits", (PyCFunction)FitsChan_retainfits, METH_NOARGS, "Ensure current card is retained in a FitsChan."},
   {"setfitsCF", (PyCFunction)FitsChan_setfitsCF, METH_VARARGS, "Store a new complex floating point keyword value in a FitsChan."},
   {"setfitsCI", (PyCFunction)FitsChan_setfitsCI, METH_VARARGS, "Store a new complex integer keyword value in a FitsChan."},
   {"setfitsF", (PyCFunction)FitsChan_setfitsF, METH_VARARGS, "Store a new floating point keyword value in a FitsChan."},
   {"setfitsI", (PyCFunction)FitsChan_setfitsI, METH_VARARGS, "Store a new integer value in a FitsChan."},
   {"setfitsL", (PyCFunction)FitsChan_setfitsL, METH_VARARGS, "Store a new integer value in a FitsChan."},
   {"setfitsS", (PyCFunction)FitsChan_setfitsS, METH_VARARGS, "Store a new string keyword value in a FitsChan."},
   {"setfitsCN", (PyCFunction)FitsChan_setfitsCN, METH_VARARGS, "Store a new string keyword value in a FitsChan."},
   {"testfits", (PyCFunction)FitsChan_testfits, METH_VARARGS, "Test if a keyword has a defined value in a FitsChan."},
   {"writefits", (PyCFunction)FitsChan_writefits, METH_NOARGS, "Write out all cards to the external sink of a FitsChan."},
   {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* Define the AST attributes of the class */
MAKE_GETROC(FitsChan,AllWarnings)
MAKE_GETSETI(FitsChan,Card)
MAKE_GETSETL(FitsChan,CarLin)
MAKE_GETSETL(FitsChan,CDMatrix)
MAKE_GETSETL(FitsChan,Clean)
MAKE_GETSETL(FitsChan,DefB1950)
MAKE_GETSETC(FitsChan,Encoding)
MAKE_GETSETI(FitsChan,FitsDigits)
MAKE_GETSETL(FitsChan,Iwc)
MAKE_GETROI(FitsChan,Ncard)
MAKE_GETROI(FitsChan,Nkey)
/* TBD MAKE_GETSETL(FitsChan,TabOK)*/
MAKE_GETSETI(FitsChan,PolyTan)
MAKE_GETSETC(FitsChan,Warnings)

static PyGetSetDef FitsChan_getseters[] = {
   DEFATT(AllWarnings,"A list of the available conditions"),
   DEFATT(Card,"Index of current FITS card in a FitsChan"),
   DEFATT(CarLin,"Ignore spherical rotations on CAR projections?"),
   DEFATT(CDMatrix,"Use a CD matrix instead of a PC matrix?"),
   DEFATT(Clean,"Remove cards used whilst reading even if an error occurs?"),
   DEFATT(DefB1950,"Use FK4 B1950 as default equatorial coordinates?"),
   DEFATT(Encoding,"System for encoding Objects as FITS headers"),
   DEFATT(FitsDigits,"Digits of precision for floating-point FITS values"),
   DEFATT(Iwc,"Add a Frame describing Intermediate World Coords?"),
   DEFATT(Ncard,"Number of FITS header cards in a FitsChan"),
   DEFATT(Nkey,"Number of unique FITS keywords in a FitsChan"),
   DEFATT(PolyTan,"Use PVi_m keywords to define distorted TAN projection?"),
   DEFATT(Warnings,"Produces warnings about selected conditions"),
   {NULL, NULL, NULL, NULL, NULL}  /* Sentinel */
};

/* Define the methods needed to make a FitsChan behave as a mapping. */
static PyMappingMethods FitsChanAsMapping = {
   FitsChan_length,
   FitsChan_getitem,
   FitsChan_setitem,
};

/* Define the class Python type structure */
static PyTypeObject FitsChanType = {
   PyVarObject_HEAD_INIT(NULL, 0)
   CLASS,                     /* tp_name */
   sizeof(FitsChan),          /* tp_basicsize */
   0,                         /* tp_itemsize */
   0,                         /* tp_dealloc */
   0,                         /* tp_print */
   0,                         /* tp_getattr */
   0,                         /* tp_setattr */
   0,                         /* tp_reserved */
   0,                         /* tp_repr */
   0,                         /* tp_as_number */
   0,                         /* tp_as_sequence */
   &FitsChanAsMapping,        /* tp_as_mapping */
   0,                         /* tp_hash  */
   0,                         /* tp_call */
   0,                         /* tp_str */
   0,                         /* tp_getattro */
   0,                         /* tp_setattro */
   0,                         /* tp_as_buffer */
   Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE, /* tp_flags */
   "AST FitsChan",            /* tp_doc */
   0,		              /* tp_traverse */
   0,		              /* tp_clear */
   0,		              /* tp_richcompare */
   0,		              /* tp_weaklistoffset */
   FitsChan_getiter,	      /* tp_iter */
   FitsChan_next,	      /* tp_iternext */
   FitsChan_methods,          /* tp_methods */
   0,                         /* tp_members */
   FitsChan_getseters,        /* tp_getset */
   0,                         /* tp_base */
   0,                         /* tp_dict */
   0,                         /* tp_descr_get */
   0,                         /* tp_descr_set */
   0,                         /* tp_dictoffset */
   (initproc)FitsChan_init,   /* tp_init */
   0,                         /* tp_alloc */
   0,                         /* tp_new */
};


static int FitsChan_init( FitsChan *self, PyObject *args, PyObject *kwds ){
   PyObject *source = NULL;
   PyObject *sink = NULL;
   const char *(* source_wrap)( void );
   void (* sink_wrap)( const char * );
   const char *options = " ";
   int result = -1;
   if( PyArg_ParseTuple(args, "|OOs:" CLASS, &source, &sink, &options ) ) {

/* Choose the source and sink wrapper functions and store info required
   by the source and sink functions in the Channel structure. */
      result = ChannelFuncs( (Channel *) self, source, sink, &source_wrap,
                             &sink_wrap );

/* Create the FitsChan using the above selected wrapper functions. */
      if( result == 0 ) {
         AstFitsChan *this = astFitsChan( source_wrap, sink_wrap, options );

/* Store a pointer to the PyObject FitsChan in the AST FitsChan so that the
   source and sink wrapper functions can get at it. */
         astPutChannelData( this, self );

/* Store self as the Python proxy for the AST FitsChan. */
         result = SetProxy( (AstObject *) this, (Object *) self );
         this = astAnnul( this );
      }
   }

   TIDY;
   return result;
}


/* A function that returns a Python iterator for a FitsChan. In this
   case, the iterator is just the FitsChan itself, but the Card attribute
   is cleared by this function so that the first card is returned first.
   Note, this means that each FitsChan can only have one iterator associated
   with it at any one time. That is, calling this function will reset any
   prevoous iterators created by this function. Not ideal, but probably
   good enough for now. */
static PyObject *FitsChan_getiter( PyObject *self ) {
   PyObject *result = NULL;

   if( PyErr_Occurred() ) return NULL;

   astClear( THIS, "Card" );
   if( astOK ) {
      Py_INCREF( self );
      result = self;
   }
   TIDY;
   return result;
}

/* Return the next value from the iteration of a FitsChan. */
static PyObject *FitsChan_next( PyObject *self ) {
   PyObject *result = NULL;
   char card[ 81 ];
   if( PyErr_Occurred() ) return result;
   if( astFindFits( THIS, "%f", card, 1 ) ) {
      result = Py_BuildValue( "s", card );
   } else {
      PyErr_SetString( PyExc_StopIteration, "No more header cards in FitsChan" );
   }
   TIDY;
   return result;
}


/* Methods needed to make a FitsChan behave as a python mapping */

/* Return the number of unique keywords in the FitsChan. */
static Py_ssize_t FitsChan_length( PyObject *self ) {
   Py_ssize_t result = (Py_ssize_t) astGetI( THIS, "Nkey" );
   if( PyErr_Occurred() ) return -1;
   if( !astOK ) result = -1;
   TIDY;
   return result;
}

/* Return the value(s) of a given keyword, or the whole card with a given
   index. */
static PyObject *FitsChan_getitem( PyObject *self, PyObject *index ){
   PyObject *result = NULL;
   PyObject **vals;
   char *keyw = NULL;
   int icard;
   int ival;
   int nval;
   int type;

   if( PyErr_Occurred() ) return result;

/* Save the current card index. */
   icard = astGetI( THIS, "Card" );

/* If the index is actually an integer, treat it as the Card index. Set
   the Card attribute in the FitsChan, and then get the current card.
   Change from python zero-based index to ATS one-based index. */
   if( PyLong_Check( index ) ) {
      char card[ 81 ];
      long int lval = PyLong_AsLong( index );
      int val = (int) lval;
      if( (long int) val != lval ) {
         val = INT_MAX;
      } else {
         val++;
      }
      astSetI( THIS, "Card", val );
      if( astFindFits( THIS, "%f", card, 0 ) ) {
         result = Py_BuildValue( "s", card );
      }

/* Otherwise, get the keyword to be searched for. */
   } else {
      keyw = GetString( NULL, index );

/* Save the current card index, and then rewind the FitsChan. */
      icard = astGetI( THIS, "Card" );
      astClear( THIS, "Card" );

/* Search forward to the next occurrence of the requested keyword. It
   becomes the current card. */
      vals = NULL;
      nval = 0;
      while( astFindFits( THIS, keyw, NULL, 0 ) && astOK ) {

/* If a match was found, get its card index. */
         icard = astGetI( THIS, "Card" );

/* Get the data type of the card. */
         type = astGetI( THIS, "CardType" );

/* Use the appropriate astGetFits<X> function to get the value and
   build an appropriate PyObject. Note, astGetFITS<X> starts searching
   with the card *following* the current card, so decrement the current card
   so that astGetFits<X> will find the correct card. */
         astSetI( THIS, "Card", icard - 1 );

         if( type == AST__INT ) {
            int val;
            astGetFitsI( THIS, keyw, &val );
            vals = astGrow( vals, nval + 1, sizeof( *vals ) );
            if( astOK ) vals[ nval++ ] = Py_BuildValue( "i", val );

         } else if( type == AST__FLOAT ) {
            double val;
            astGetFitsF( THIS, keyw, &val );
            vals = astGrow( vals, nval + 1, sizeof( *vals ) );
            if( astOK ) vals[ nval++ ] = Py_BuildValue( "d", val );

         } else if( type == AST__LOGICAL ) {
            int val;
            astGetFitsL( THIS, keyw, &val );
            vals = astGrow( vals, nval + 1, sizeof( *vals ) );
            if( astOK ) vals[ nval++ ] = Py_BuildValue( "O", (val ? Py_True : Py_False) );

         } else {
            char *val;
            astGetFitsS( THIS, keyw, &val );
            vals = astGrow( vals, nval + 1, sizeof( *vals ) );
            if( astOK ) vals[ nval++ ] = Py_BuildValue( "s", val );
         }

/* Increment the current card so that astFindFits will not just find the
   same card again. */
         astSetI( THIS, "Card", icard + 1 );
      }

/* If there is more than one value to return, construct a tuple. */
      if( astOK ) {
         if( nval > 1 ) {
            result = PyTuple_New( nval );
            for( ival = 0; ival < nval; ival++ ) {
               PyTuple_SetItem( result, ival, vals[ ival ] );
            }

         } else if( nval == 1 ) {
            result = vals[ 0 ];

         } else {
            char buff[ 200 ];
            sprintf( buff, "FITS keyword %s not found in FitsChan.", keyw );
            PyErr_SetString( PyExc_KeyError, buff );
         }

      } else if( nval > 0 ) {
         for( ival = 0; ival < nval; ival++ ) {
            Py_XDECREF( vals[ ival ] );
         }
      }

      vals = astFree( vals );
      keyw = astFree( keyw );
   }

   astSetI( THIS, "Card", icard );
   TIDY;
   return result;
}

/* Set the value of a given keyword, replacing any old value(s). */
static int FitsChan_setitem( PyObject *self, PyObject *index, PyObject *value ){
   char *keyw;
   int icard;
   int result = -1;
   if( PyErr_Occurred() ) return result;

/* If the supplied index is an integer, overwrite the card with the
   corresponding index. */
   if( PyLong_Check( index ) ) {
      long int lval = PyLong_AsLong( index );
      int val = (int) lval;
      if( (long int) val != lval ) {
         val = INT_MAX;
      } else {
         val++;
      }
      astSetI( THIS, "Card", val );

      if( value && value != Py_None ) {
         PyObject *str = PyObject_Str( value );
         char *card = GetString( NULL, str );
         astPutFits( THIS, card, 1 );
         card = astFree( card );
         Py_DECREF(str);
      } else {
         astDelFits( THIS );
      }

/* Otherwise...get the keyword to be searched for. */
   } else {
      keyw = GetString( NULL, index );

/* If the keyword name is blank, just insert the supplied value (as a
   string) before the current card, with no keyword (i.e. as a comment card). */
      if( !keyw || astChrLen( keyw ) == 0 ) {
         if( value ) {
            PyObject *str = PyObject_Str( value );
            char *val = GetString( NULL, str );
            astSetFitsCM( THIS, val, 0 );
            val = astFree( val );
            Py_DECREF(str);
         }

/* Otherwise replace the named keyword with the supplied value */
      } else {

/* Record the initial current card, and then rewind the FitsChan. */
         icard = astGetI( THIS, "Card" );
         astClear( THIS, "Card" );

/* Find the first occurrence (if any) of the specified keyword in the
   FitsChan, and make it the current card. If not found, the FitsChan is
   left at "end-of-file". */
         astFindFits( THIS, keyw, NULL, 0 );

/* Store the supplied keyword value, overwriting the current card
   found above. */
         if( !value || value == Py_None ) {
            /* Do nothing if no value supplied - the current card will be
               deleted later */

         } else if( PyLong_Check( value ) ) {
            long int lval = PyLong_AsLong( value );
            int val = (int) lval;
            if( (long int) val != lval ) {
               char buff[ 200 ];
               sprintf( buff, "Cannot assign value %ld to FITS keyword %s - "
                        "integer overflow.", lval, keyw );
               PyErr_SetString( PyExc_OverflowError, buff );
               result = 0;
            }  else {
               astSetFitsI( THIS, keyw, val, NULL, 1 );
            }

         } else if( PyFloat_Check( value ) ) {
            double val = PyFloat_AsDouble( value );
            astSetFitsF( THIS, keyw, val, NULL, 1 );

         } else if( PyBool_Check( value ) ) {
            int val = ( value == Py_True );
            astSetFitsL( THIS, keyw, val, NULL, 1 );

         } else {
            PyObject *str = PyObject_Str( value );
            char *val = GetString( NULL, str );
            astSetFitsS( THIS, keyw, val, NULL, 1 );
            val = astFree( val );
            Py_DECREF(str);
         }

/* Search for any later occurrences of the same keyword, and delete them.
   Modify the original current card index if the original curent card is
   later in the FitsChan. */
         while( astFindFits( THIS, keyw, NULL, 0 ) && astOK ) {
            if( astGetI( THIS, "Card" ) < icard ) icard--;
            astDelFits( THIS );
         }

/* Re-instate the original current card. */
         astSetI( THIS, "Card", icard );
      }

      keyw = astFree( keyw );
   }
   if( astOK ) result = 0;
   TIDY;
   return result;
}



/* Define the AST methods of the class. */
static PyObject *FitsChan_delfits( FitsChan *self ) {
   PyObject *result = NULL;
   if( PyErr_Occurred() ) return NULL;
   astDelFits( THIS );
   if( astOK ) result = Py_None;
   TIDY;
   return result;
}

static PyObject *FitsChan_writefits( FitsChan *self ) {
   PyObject *result = NULL;
   if( PyErr_Occurred() ) return NULL;
   astWriteFits( THIS );
   if( astOK ) result = Py_None;
   TIDY;
   return result;
}

static PyObject *FitsChan_readfits( FitsChan *self ) {
   PyObject *result = NULL;
   if( PyErr_Occurred() ) return NULL;
   astReadFits( THIS );
   if( astOK ) result = Py_None;
   TIDY;
   return result;
}

static PyObject *FitsChan_emptyfits( FitsChan *self ) {
   PyObject *result = NULL;
   if( PyErr_Occurred() ) return NULL;
   astEmptyFits( THIS );
   if( astOK ) result = Py_None;
   TIDY;
   return result;
}


#undef NAME
#define NAME CLASS ".findfits"
static PyObject *FitsChan_findfits( FitsChan *self, PyObject *args ) {
   PyObject *result = NULL;
   int inc;
   const char *name = NULL;

   if( PyErr_Occurred() ) return NULL;

   if ( PyArg_ParseTuple( args, "si:" NAME, &name, &inc ) && astOK ) {
      char card[ 81 ];
      int found = astFindFits( THIS, name, card, inc );
      if( astOK ) {
         result = Py_BuildValue( "Os", (found ? Py_True : Py_False), card );
      }
   }

   TIDY;
   return result;
}

#define MAKE_GETFITS(typecode,type,fmt) \
\
static PyObject *FitsChan_getfits##typecode( FitsChan *self, PyObject *args ) { \
   PyObject *result = NULL; \
   const char *name = NULL; \
   if( PyErr_Occurred() ) return NULL; \
   if ( PyArg_ParseTuple( args, "s:" NAME ".getfits" #typecode, &name ) && astOK ) { \
      type value[2];\
      int there = astGetFits##typecode( THIS, name, value ); \
      if( astOK ) { \
         result = Py_BuildValue( "O" #fmt, (there ? Py_True : Py_False), \
                                  value[0], value[1] ); \
      } \
   } \
   TIDY; \
   return result; \
}

MAKE_GETFITS(CF,double,(dd))
MAKE_GETFITS(CI,int,(ii))

#undef MAKE_GETFITS

#define MAKE_GETFITS(typecode,type,fmt,valexp) \
\
static PyObject *FitsChan_getfits##typecode( FitsChan *self, PyObject *args ) { \
   PyObject *result = NULL; \
   const char *name = NULL; \
   if( PyErr_Occurred() ) return NULL; \
   if ( PyArg_ParseTuple( args, "s:" NAME ".getfits" #typecode, &name ) && astOK ) { \
      type value;\
      int there = astGetFits##typecode( THIS, name, &value ); \
      if( astOK ) { \
         result = Py_BuildValue( "O" #fmt, (there ? Py_True : Py_False), \
                                  valexp ); \
      } \
   } \
   TIDY; \
   return result; \
}

MAKE_GETFITS(F,double,d,value)
MAKE_GETFITS(I,int,i,value)
MAKE_GETFITS(L,int,O,(value?Py_True:Py_False))
MAKE_GETFITS(S,char *,s,value)
MAKE_GETFITS(CN,char *,s,value)

#undef MAKE_GETFITS

#undef NAME
#define NAME CLASS ".putcards"
static PyObject *FitsChan_putcards( FitsChan *self, PyObject *args ) {
   PyObject *result = NULL;
   const char *cards = NULL;
   if( PyErr_Occurred() ) return NULL;
   if ( PyArg_ParseTuple( args, "s:" NAME, &cards ) && astOK ) {
      astPutCards( THIS, cards );
      if( astOK ) result = Py_None;
   }
   TIDY;
   return result;
}

#undef NAME
#define NAME CLASS ".putfits"
static PyObject *FitsChan_putfits( FitsChan *self, PyObject *args ) {
   PyObject *result = NULL;
   int overwrite;
   const char *card = NULL;

   if( PyErr_Occurred() ) return NULL;

   if ( PyArg_ParseTuple( args, "si:" NAME, &card, &overwrite ) && astOK ) {
      astPutFits( THIS, card, overwrite );
      if( astOK ) result = Py_None;
   }
   TIDY;
   return result;
}

#define MAKE_SETFITS(typecode,type,fmt) \
\
static PyObject *FitsChan_setfits##typecode( FitsChan *self, PyObject *args ) { \
   PyObject *result = NULL; \
   const char *name = NULL; \
   const char *comment = NULL; \
   int overwrite; \
   type value[2]; \
   if( PyErr_Occurred() ) return NULL; \
   if ( PyArg_ParseTuple( args, "s" #fmt "si:" NAME ".setfits" #typecode, \
                          &name, value, value + 1, &comment, &overwrite) && astOK ) { \
      astSetFits##typecode( THIS, name, value, comment, overwrite ); \
      if( astOK ) result = Py_None; \
   } \
   TIDY; \
   return result; \
}

MAKE_SETFITS(CF,double,(dd))
MAKE_SETFITS(CI,int,(ii))

#undef MAKE_SETFITS

#define MAKE_SETFITS(typecode,type,fmt,valexp) \
\
static PyObject *FitsChan_setfits##typecode( FitsChan *self, PyObject *args ) { \
   PyObject *result = NULL; \
   const char *name = NULL; \
   const char *comment = NULL; \
   int overwrite; \
   type value; \
   if( PyErr_Occurred() ) return NULL; \
   if ( PyArg_ParseTuple( args, "s" #fmt "si:" NAME ".setfits" #typecode, \
                          &name, &value, &comment, &overwrite) && astOK ) { \
      astSetFits##typecode( THIS, name, valexp, comment, overwrite ); \
      if( astOK ) result = Py_None; \
   } \
   TIDY; \
   return result; \
}

MAKE_SETFITS(F,double,d,value)
MAKE_SETFITS(I,int,i,value)
MAKE_SETFITS(L,PyObject *,O,(value==Py_True))
MAKE_SETFITS(S,const char *,s,value)
MAKE_SETFITS(CN,const char *,s,value)

#undef MAKE_SETFITS


#undef NAME
#define NAME CLASS ".testfits"
static PyObject *FitsChan_testfits( FitsChan *self, PyObject *args ) {
   PyObject *result = NULL;
   const char *name;
   if( PyErr_Occurred() ) return NULL;
   if ( PyArg_ParseTuple( args, "s:" NAME, &name ) && astOK ) {
      int there;
      int ok = astTestFits( THIS, name, &there );
      if( astOK ) {
         result = Py_BuildValue( "OO", (ok ? Py_True : Py_False),
                                  (there ? Py_True : Py_False) );
      }
   }
   TIDY;
   return result;
}

static PyObject *FitsChan_retainfits( FitsChan *self ) {
   PyObject *result = NULL;
   if( PyErr_Occurred() ) return NULL;
   astRetainFits( THIS );
   if( astOK ) result = Py_None;
   TIDY;
   return result;
}

static PyObject *FitsChan_purgewcs( FitsChan *self ) {
   PyObject *result = NULL;
   if( PyErr_Occurred() ) return NULL;
   astPurgeWCS( THIS );
   if( astOK ) result = Py_None;
   TIDY;
   return result;
}


/* StcsChan */
/* ======== */

/* Define a string holding the fully qualified Python class name. */
#undef CLASS
#define CLASS MODULE ".StcsChan"

/* Define the class structure */
typedef struct {
   Channel parent;
} StcsChan;

/* Prototypes for class functions */
static int StcsChan_init( StcsChan *self, PyObject *args, PyObject *kwds );

/* Define the AST attributes of the class */
MAKE_GETSETL(StcsChan,StcsArea)
MAKE_GETSETL(StcsChan,StcsCoords)
MAKE_GETSETL(StcsChan,StcsProps)
MAKE_GETSETI(StcsChan,StcsLength)

static PyGetSetDef StcsChan_getseters[] = {
   DEFATT(StcsArea,"Return the CoordinateArea component when reading an STC-S document?"),
   DEFATT(StcsCoords,"Return the Coordinates component when reading an STC-S document?"),
   DEFATT(StcsProps,"Return all properties when reading an STC-S document?"),
   DEFATT(StcsLength,"Controls output line length."),
   {NULL, NULL, NULL, NULL, NULL}  /* Sentinel */
};

/* Define the class Python type structure */
static PyTypeObject StcsChanType = {
   PyVarObject_HEAD_INIT(NULL, 0)
   CLASS,                     /* tp_name */
   sizeof(StcsChan),          /* tp_basicsize */
   0,                         /* tp_itemsize */
   0,                         /* tp_dealloc */
   0,                         /* tp_print */
   0,                         /* tp_getattr */
   0,                         /* tp_setattr */
   0,                         /* tp_reserved */
   0,                         /* tp_repr */
   0,                         /* tp_as_number */
   0,                         /* tp_as_sequence */
   0,                         /* tp_as_mapping */
   0,                         /* tp_hash  */
   0,                         /* tp_call */
   0,                         /* tp_str */
   0,                         /* tp_getattro */
   0,                         /* tp_setattro */
   0,                         /* tp_as_buffer */
   Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE, /* tp_flags */
   "AST StcsChan",            /* tp_doc */
   0,		              /* tp_traverse */
   0,		              /* tp_clear */
   0,		              /* tp_richcompare */
   0,		              /* tp_weaklistoffset */
   0,		              /* tp_iter */
   0,		              /* tp_iternext */
   0,                         /* tp_methods */
   0,                         /* tp_members */
   StcsChan_getseters,        /* tp_getset */
   0,                         /* tp_base */
   0,                         /* tp_dict */
   0,                         /* tp_descr_get */
   0,                         /* tp_descr_set */
   0,                         /* tp_dictoffset */
   (initproc)StcsChan_init,   /* tp_init */
   0,                         /* tp_alloc */
   0,                         /* tp_new */
};


static int StcsChan_init( StcsChan *self, PyObject *args, PyObject *kwds ){
   PyObject *source = NULL;
   PyObject *sink = NULL;
   const char *(* source_wrap)( void ) = NULL;
   void (* sink_wrap)( const char * ) = NULL;
   const char *options = " ";
   int result = -1;
   if( PyArg_ParseTuple(args, "|OOs:" CLASS, &source, &sink, &options ) ) {

/* Choose the source and sink wrapper functions and store info required
   by the source and sink functions in the Channel structure. */
      result = ChannelFuncs( (Channel *) self, source, sink, &source_wrap,
                             &sink_wrap );

/* Create the StcsChan using the above selected wrapper functions. */
      if( result == 0 ) {
         AstStcsChan *this = astStcsChan( source_wrap, sink_wrap, options );

/* Store a pointer to the PyObject StcsChan in the AST StcsChan so that the
   source and sink wrapper functions can get at it. */
         astPutChannelData( this, self );

/* Store self as the Python proxy for the AST StcsChan. */
         result = SetProxy( (AstObject *) this, (Object *) self );
         this = astAnnul( this );
      }
   }

   TIDY;
   return result;
}



/* Now describe the whole AST module */
/* ================================= */

/* Static method prototypes */
static PyObject *PyAst_escapes( PyObject *self, PyObject *args );
static PyObject *PyAst_tune( PyObject *self, PyObject *args );
static PyObject *PyAst_version( PyObject *self );
static PyObject *PyAst_get_include( PyObject *self );

/* Static method implementations */

#undef NAME
#define NAME MODULE ".escapes"
static PyObject *PyAst_escapes( PyObject *self, PyObject *args ) {
   PyObject *result = NULL;
   int newval;
   int value;
   if( PyErr_Occurred() ) return NULL;
   if( PyArg_ParseTuple(args, "i:" NAME, &newval ) ) {
      value = astEscapes( newval );
      if( astOK ) result = Py_BuildValue( "i", value );
   }
   TIDY;
   return result;
}

#undef NAME
#define NAME MODULE ".tune"
static PyObject *PyAst_tune( PyObject *self, PyObject *args ) {
   PyObject *result = NULL;
   int value;
   int oldval;
   const char *name;
   if( PyErr_Occurred() ) return NULL;
   if( PyArg_ParseTuple(args, "si:" NAME, &name, &value ) ) {
      oldval = astTune( name, value );
      if( astOK ) result = Py_BuildValue( "i", oldval );
   }
   TIDY;
   return result;
}

static PyObject *PyAst_version( PyObject *self ) {
   PyObject *result = NULL;
   int version;
   if( PyErr_Occurred() ) return NULL;
   version = astVersion;
   if( astOK ) result = Py_BuildValue( "i", version );
   TIDY;
   return result;
}


/* Return the path to the directory holding "star/pyast.h".  */
static PyObject *PyAst_get_include( PyObject *self ) {
   PyObject *result = NULL;
   PyObject *str;
   char *buff;
   char *c;
   int nc;

/* Check no error has occurred already. */
   if( PyErr_Occurred() ) return result;

/* Get a string holding the full path to the pyast sharable library. */
   str = PyObject_GetAttrString( self, "__file__" );
   buff = GetString( NULL, str );
   Py_XDECREF( str );

/* Find the last directory separator ("/" or "\"). Removing the file
   basename that follows, and replace it with "include". */
   if( buff ) {
      c = buff + strlen ( buff ) - 1;
      while( *c != '/' && *c != '\\' && c > buff ) c--;

      if( *c == '/' || *c == '\\' ) {
         nc = c - buff + 1;
         astAppendString( buff, &nc, "include" );
         if( astOK ) result = Py_BuildValue( "s", buff );

      } else {
         PyErr_SetString( INTER_err, "Cannot determine the path to the "
                          "pyast header file" );
      }

      buff = astFree( buff );
   }
   return result;
}



/* Describe the static methods of the class */
static PyMethodDef PyAst_methods[] = {
   {"escapes", (PyCFunction)PyAst_escapes, METH_VARARGS, "Control whether graphical escape sequences are included in strings"},
   {"tune", (PyCFunction)PyAst_tune, METH_VARARGS,  "Set or get an AST global tuning parameter"},
   {"version", (PyCFunction)PyAst_version, METH_NOARGS,  "Return the version of the AST library being used"},
   {"get_include", (PyCFunction)PyAst_get_include, METH_NOARGS,  "Return the path to the directory containing pyast header files"},
   {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* Describe the properties of the module. */
static struct PyModuleDef astmodule = {
   PyModuleDef_HEAD_INIT,
   "Ast",
   "AST Python interface.",
   -1,
   PyAst_methods,
   NULL, NULL, NULL, NULL
};


/* Tell the python interpreter about this module. This includes telling
   the interpreter about each of the types defined by this module. */
PyMODINIT_FUNC PyInit_Ast(void) {
   static void *PyAst_API[ PyAst_API_pointers ];
   PyObject *c_api_object, *m;

   m = PyModule_Create(&astmodule);
   if( m == NULL ) return NULL;

/* Create singleton instances of the AST Exception classes. The
   RegisterErrors function is defined within file exceptions.c (generated
   automatically by the make_exceptions.py script on the basis of the ast_err.msg
   file). */
   if( !RegisterErrors( m ) ) return NULL;

/* Pointers to functions for use by other extension modules. */
   PyAst_API[PyAst_ToString_NUM] = (void *)PyAst_ToString;
   PyAst_API[PyAst_FromString_NUM] = (void *)PyAst_FromString;

/* Create a Capsule containing the API pointer array's address */
   c_api_object = PyCapsule_New( (void *) PyAst_API, MODULE "._C_API", NULL );
   if( c_api_object ) PyModule_AddObject( m, "_C_API", c_api_object );

/* The types provided by this module. */
   if( PyType_Ready(&ObjectType) < 0) return NULL;
   Py_INCREF(&ObjectType);
   PyModule_AddObject( m, "Object", (PyObject *)&ObjectType);

   MappingType.tp_base = &ObjectType;
   if( PyType_Ready(&MappingType) < 0) return NULL;
   Py_INCREF(&MappingType);
   PyModule_AddObject( m, "Mapping", (PyObject *)&MappingType);

   ZoomMapType.tp_new = PyType_GenericNew;
   ZoomMapType.tp_base = &MappingType;
   if( PyType_Ready(&ZoomMapType) < 0) return NULL;
   Py_INCREF(&ZoomMapType);
   PyModule_AddObject( m, "ZoomMap", (PyObject *)&ZoomMapType);

   SphMapType.tp_new = PyType_GenericNew;
   SphMapType.tp_base = &MappingType;
   if( PyType_Ready(&SphMapType) < 0) return NULL;
   Py_INCREF(&SphMapType);
   PyModule_AddObject( m, "SphMap", (PyObject *)&SphMapType);

   GrismMapType.tp_new = PyType_GenericNew;
   GrismMapType.tp_base = &MappingType;
   if( PyType_Ready(&GrismMapType) < 0) return NULL;
   Py_INCREF(&GrismMapType);
   PyModule_AddObject( m, "GrismMap", (PyObject *)&GrismMapType);

   PcdMapType.tp_new = PyType_GenericNew;
   PcdMapType.tp_base = &MappingType;
   if( PyType_Ready(&PcdMapType) < 0) return NULL;
   Py_INCREF(&PcdMapType);
   PyModule_AddObject( m, "PcdMap", (PyObject *)&PcdMapType);

   WcsMapType.tp_new = PyType_GenericNew;
   WcsMapType.tp_base = &MappingType;
   if( PyType_Ready(&WcsMapType) < 0) return NULL;
   Py_INCREF(&WcsMapType);
   PyModule_AddObject( m, "WcsMap", (PyObject *)&WcsMapType);

   UnitMapType.tp_new = PyType_GenericNew;
   UnitMapType.tp_base = &MappingType;
   if( PyType_Ready(&UnitMapType) < 0) return NULL;
   Py_INCREF(&UnitMapType);
   PyModule_AddObject( m, "UnitMap", (PyObject *)&UnitMapType);

   TimeMapType.tp_new = PyType_GenericNew;
   TimeMapType.tp_base = &MappingType;
   if( PyType_Ready(&TimeMapType) < 0) return NULL;
   Py_INCREF(&TimeMapType);
   PyModule_AddObject( m, "TimeMap", (PyObject *)&TimeMapType);

   RateMapType.tp_new = PyType_GenericNew;
   RateMapType.tp_base = &MappingType;
   if( PyType_Ready(&RateMapType) < 0) return NULL;
   Py_INCREF(&RateMapType);
   PyModule_AddObject( m, "RateMap", (PyObject *)&RateMapType);

   CmpMapType.tp_new = PyType_GenericNew;
   CmpMapType.tp_base = &MappingType;
   if( PyType_Ready(&CmpMapType) < 0) return NULL;
   Py_INCREF(&CmpMapType);
   PyModule_AddObject( m, "CmpMap", (PyObject *)&CmpMapType);

   TranMapType.tp_new = PyType_GenericNew;
   TranMapType.tp_base = &MappingType;
   if( PyType_Ready(&TranMapType) < 0) return NULL;
   Py_INCREF(&TranMapType);
   PyModule_AddObject( m, "TranMap", (PyObject *)&TranMapType);

   NormMapType.tp_new = PyType_GenericNew;
   NormMapType.tp_base = &MappingType;
   if( PyType_Ready(&NormMapType) < 0) return NULL;
   Py_INCREF(&NormMapType);
   PyModule_AddObject( m, "NormMap", (PyObject *)&NormMapType);

   PermMapType.tp_new = PyType_GenericNew;
   PermMapType.tp_base = &MappingType;
   if( PyType_Ready(&PermMapType) < 0) return NULL;
   Py_INCREF(&PermMapType);
   PyModule_AddObject( m, "PermMap", (PyObject *)&PermMapType);

   ShiftMapType.tp_new = PyType_GenericNew;
   ShiftMapType.tp_base = &MappingType;
   if( PyType_Ready(&ShiftMapType) < 0) return NULL;
   Py_INCREF(&ShiftMapType);
   PyModule_AddObject( m, "ShiftMap", (PyObject *)&ShiftMapType);

   LutMapType.tp_new = PyType_GenericNew;
   LutMapType.tp_base = &MappingType;
   if( PyType_Ready(&LutMapType) < 0) return NULL;
   Py_INCREF(&LutMapType);
   PyModule_AddObject( m, "LutMap", (PyObject *)&LutMapType);

   WinMapType.tp_new = PyType_GenericNew;
   WinMapType.tp_base = &MappingType;
   if( PyType_Ready(&WinMapType) < 0) return NULL;
   Py_INCREF(&WinMapType);
   PyModule_AddObject( m, "WinMap", (PyObject *)&WinMapType);

   FrameType.tp_new = PyType_GenericNew;
   FrameType.tp_base = &MappingType;
   if( PyType_Ready(&FrameType) < 0) return NULL;
   Py_INCREF(&FrameType);
   PyModule_AddObject( m, "Frame", (PyObject *)&FrameType);

   FrameSetType.tp_new = PyType_GenericNew;
   FrameSetType.tp_base = &FrameType;
   if( PyType_Ready(&FrameSetType) < 0) return NULL;
   Py_INCREF(&FrameSetType);
   PyModule_AddObject( m, "FrameSet", (PyObject *)&FrameSetType);

   CmpFrameType.tp_new = PyType_GenericNew;
   CmpFrameType.tp_base = &FrameType;
   if( PyType_Ready(&CmpFrameType) < 0) return NULL;
   Py_INCREF(&CmpFrameType);
   PyModule_AddObject( m, "CmpFrame", (PyObject *)&CmpFrameType);

   SpecFrameType.tp_new = PyType_GenericNew;
   SpecFrameType.tp_base = &FrameType;
   if( PyType_Ready(&SpecFrameType) < 0) return NULL;
   Py_INCREF(&SpecFrameType);
   PyModule_AddObject( m, "SpecFrame", (PyObject *)&SpecFrameType);

   DSBSpecFrameType.tp_new = PyType_GenericNew;
   DSBSpecFrameType.tp_base = &SpecFrameType;
   if( PyType_Ready(&DSBSpecFrameType) < 0) return NULL;
   Py_INCREF(&DSBSpecFrameType);
   PyModule_AddObject( m, "DSBSpecFrame", (PyObject *)&DSBSpecFrameType);

   SkyFrameType.tp_new = PyType_GenericNew;
   SkyFrameType.tp_base = &FrameType;
   if( PyType_Ready(&SkyFrameType) < 0) return NULL;
   Py_INCREF(&SkyFrameType);
   PyModule_AddObject( m, "SkyFrame", (PyObject *)&SkyFrameType);

   TimeFrameType.tp_new = PyType_GenericNew;
   TimeFrameType.tp_base = &FrameType;
   if( PyType_Ready(&TimeFrameType) < 0) return NULL;
   Py_INCREF(&TimeFrameType);
   PyModule_AddObject( m, "TimeFrame", (PyObject *)&TimeFrameType);

   FluxFrameType.tp_new = PyType_GenericNew;
   FluxFrameType.tp_base = &FrameType;
   if( PyType_Ready(&FluxFrameType) < 0) return NULL;
   Py_INCREF(&FluxFrameType);
   PyModule_AddObject( m, "FluxFrame", (PyObject *)&FluxFrameType);

   SpecFluxFrameType.tp_new = PyType_GenericNew;
   SpecFluxFrameType.tp_base = &CmpFrameType;
   if( PyType_Ready(&SpecFluxFrameType) < 0) return NULL;
   Py_INCREF(&SpecFluxFrameType);
   PyModule_AddObject( m, "SpecFluxFrame", (PyObject *)&SpecFluxFrameType);

   RegionType.tp_new = PyType_GenericNew;
   RegionType.tp_base = &FrameType;
   if( PyType_Ready(&RegionType) < 0) return NULL;
   Py_INCREF(&RegionType);
   PyModule_AddObject( m, "Region", (PyObject *)&RegionType);

   BoxType.tp_new = PyType_GenericNew;
   BoxType.tp_base = &RegionType;
   if( PyType_Ready(&BoxType) < 0) return NULL;
   Py_INCREF(&BoxType);
   PyModule_AddObject( m, "Box", (PyObject *)&BoxType);

   CircleType.tp_new = PyType_GenericNew;
   CircleType.tp_base = &RegionType;
   if( PyType_Ready(&CircleType) < 0) return NULL;
   Py_INCREF(&CircleType);
   PyModule_AddObject( m, "Circle", (PyObject *)&CircleType);

   EllipseType.tp_new = PyType_GenericNew;
   EllipseType.tp_base = &RegionType;
   if( PyType_Ready(&EllipseType) < 0) return NULL;
   Py_INCREF(&EllipseType);
   PyModule_AddObject( m, "Ellipse", (PyObject *)&EllipseType);

   IntervalType.tp_new = PyType_GenericNew;
   IntervalType.tp_base = &RegionType;
   if( PyType_Ready(&IntervalType) < 0) return NULL;
   Py_INCREF(&IntervalType);
   PyModule_AddObject( m, "Interval", (PyObject *)&IntervalType);

   NullRegionType.tp_new = PyType_GenericNew;
   NullRegionType.tp_base = &RegionType;
   if( PyType_Ready(&NullRegionType) < 0) return NULL;
   Py_INCREF(&NullRegionType);
   PyModule_AddObject( m, "NullRegion", (PyObject *)&NullRegionType);

   CmpRegionType.tp_new = PyType_GenericNew;
   CmpRegionType.tp_base = &RegionType;
   if( PyType_Ready(&CmpRegionType) < 0) return NULL;
   Py_INCREF(&CmpRegionType);
   PyModule_AddObject( m, "CmpRegion", (PyObject *)&CmpRegionType);

   PrismType.tp_new = PyType_GenericNew;
   PrismType.tp_base = &RegionType;
   if( PyType_Ready(&PrismType) < 0) return NULL;
   Py_INCREF(&PrismType);
   PyModule_AddObject( m, "Prism", (PyObject *)&PrismType);

   ChannelType.tp_new = PyType_GenericNew;
   ChannelType.tp_base = &ObjectType;
   if( PyType_Ready(&ChannelType) < 0) return NULL;
   Py_INCREF(&ChannelType);
   PyModule_AddObject( m, "Channel", (PyObject *)&ChannelType);

   FitsChanType.tp_new = PyType_GenericNew;
   FitsChanType.tp_base = &ChannelType;
   if( PyType_Ready(&FitsChanType) < 0) return NULL;
   Py_INCREF(&FitsChanType);
   PyModule_AddObject( m, "FitsChan", (PyObject *)&FitsChanType);

   StcsChanType.tp_new = PyType_GenericNew;
   StcsChanType.tp_base = &ChannelType;
   if( PyType_Ready(&StcsChanType) < 0) return NULL;
   Py_INCREF(&StcsChanType);
   PyModule_AddObject( m, "StcsChan", (PyObject *)&StcsChanType);

/* The constants provided by this module. */
#define ICONST(Name) \
   PyModule_AddIntConstant( m, #Name, AST__##Name )

#define DCONST(Name) \
   PyModule_AddObject( m, #Name, PyFloat_FromDouble(AST__##Name) )

   ICONST(TUNULL);

   ICONST(USEVAR);
   ICONST(USEBAD);
   ICONST(CONSERVEFLUX);
   ICONST(REBININIT);
   ICONST(REBINEND);
   ICONST(GENVAR);
   ICONST(VARWGT);
   ICONST(NOBAD);
   ICONST(DISVAR);
   ICONST(NEAREST);
   ICONST(LINEAR);
   ICONST(SINC);
   ICONST(SINCSINC);
   ICONST(SINCCOS);
   ICONST(SINCGAUSS);
   ICONST(BLOCKAVE);
   ICONST(GAUSS);
   ICONST(SOMB);
   ICONST(SOMBCOS);

   ICONST(AZP);
   ICONST(SZP);
   ICONST(TAN);
   ICONST(STG);
   ICONST(SIN);
   ICONST(ARC);
   ICONST(ZPN);
   ICONST(ZEA);
   ICONST(AIR);
   ICONST(CYP);
   ICONST(CEA);
   ICONST(CAR);
   ICONST(MER);
   ICONST(SFL);
   ICONST(PAR);
   ICONST(MOL);
   ICONST(AIT);
   ICONST(COP);
   ICONST(COE);
   ICONST(COD);
   ICONST(COO);
   ICONST(BON);
   ICONST(PCO);
   ICONST(TSC);
   ICONST(CSC);
   ICONST(QSC);
   ICONST(NCP);
   ICONST(GLS);
   ICONST(TPN);
   ICONST(HPX);
   ICONST(WCSBAD);

   ICONST(AND);
   ICONST(OR);
   ICONST(XOR);

   DCONST(BAD);

#undef ICONST
#undef DCONST

/* Initialise the numpi module */
   import_array();

   return m;
}


/* Functions mainly for use by other C modules */
/* =========================================== */

static PyObject *PyAst_FromString( const char *string ) {
/*
*  Name:
*     PyAst_FromString

*  Purpose:
*     Re-create a pyast Object from a string.

*  Arguments:
*     string
*        Pointer to a string created previously by PyAst_ToString.

*  Returned Value:
*     A pointer to a new pyast Object.

*/

/* Check no python error has occurred, and that a string was supplied. */
   if( PyErr_Occurred() || !string ) return NULL;

/* Crate an AST Object from the string. */
   AstObject *this = astFromString( string );

/* Report an error if unsuccesfull. */
   if( !this && !PyErr_Occurred() ) {
      char mess[255];
      sprintf( mess, "PyAst_FromString: Could not create an AST Object "
               "from supplied string (%.40s).", string );
      PyErr_SetString( PyExc_ValueError, mess );
      return NULL;
   }

/* Create a new Python Object to encapsulate the AST Object and return it. */
   PyObject *result = NewObject( this );
   TIDY;
   return result;
}

static char *PyAst_ToString( PyObject *self ) {
/*
*  Name:
*     PyAst_ToString

*  Purpose:
*     Convert a pyast Object to a string.

*  Arguments:
*     self
*        Pointer to a PyObject that is a pyast Object issued by this module.

*  Returned Value:
*     A dynamically allocated string holding a minimal serialisation of the
*     AST Object. It should be freed using astFree when no longer needed.

*/
   char *result;

/* Check no python error has occurred, and that an object was supplied. */
   if( PyErr_Occurred() || !self ) return NULL;

/* Report an error if supplied PyObject is not an AST Object */
   if( !PyObject_IsInstance( self, (PyObject *) &ObjectType ) ) {
      char mess[255];
      if( self->ob_type && self->ob_type->tp_doc ) {
         sprintf( mess, "PyAst_ToString: Expected an AST Object but a %.*s "
                  "was supplied.", (int)( sizeof(mess) - 60 ),
                  self->ob_type->tp_doc );
      } else {
         sprintf( mess, "PyAst_ToString: Expected an AST Object." );
      }
      PyErr_SetString( PyExc_TypeError, mess );
      return NULL;
   }

/* Invoke the astToString method on the AST Object and return the pointer. */
   result = astToString( THIS );
   TIDY;
   return result;
}


/* Utility functions */
/* ================= */

static char *GetString( void *mem, PyObject *value ) {
/*
*  Name:
*     GetString

*  Purpose:
*     Get a pointer to a null terminated string from a PyObject.

*  Arguments:
*     mem
*        Pointer to memory previously allocated by AST in which the
*        returned string should be stored. This memory will be extended
*        if required. New memory is allocated if NULL is supplied.
*     value
*        The PyObject containing the string to copy.

*  Returned Value:
*     A dynamically allocated copy of the string. It should be freed
*     using astFree when no longer needed. This may be a copy of "mem".

*/
   char *result = NULL;
   if( value && value != Py_None ) {
      PyObject *bytes = PyUnicode_AsASCIIString(value);
      if( bytes ) {
         const char *bytestr =  PyBytes_AS_STRING(bytes);
         result = astStore( mem, bytestr, PyBytes_Size( bytes ) + 1 );
         Py_DECREF(bytes);
      }
   }
   return result;
}

static PyObject *NewObject( AstObject *this ) {
/*
*  Name:
*     NewObject

*  Purpose:
*     Obtain a starlink.Ast.Object object to represent a given AST Object.
*     If there is already an existing starlink.Ast.Object object acting as
*     a proxy for the supplied AST Object, then a pointer to it is
*     returned (the reference count for the existing starlink.Ast.Object
*     is incremented). If the AST Object has no existing proxy, then a new
*     starlink.Ast.Object is created and recorded as the proxy for the AST
*     Object.

*/

/* Local Variables: */
   PyObject *result = NULL;
   Object *self;

/* Check inherited status */
   if( !astOK ) return result;

/* If a NULL pointer is supplied, return Py_None. */
   if( ! this ) {
      result = Py_None;

/* If the supplied AST object has an associated proxy object, return it. */
   } else {
      self = astGetProxy( this );
      if( self ) {
         result = (PyObject *) self;
         Py_INCREF( result );

/* If the supplied AST object does not have an associated proxy object,
   create a new starlink.Ast.object (of the same type as the AST object),
   store the AST Object pointer in it and record it as the proxy for the
   AST Object. Delete the starlink.Ast.object if anything goes wrong. */
      } else {
         PyTypeObject * type = GetType( this );
         if( type ) {
            self = (Object *) _PyObject_New( type );
            if( self ) {
              if( SetProxy( this, self ) == 0 ) {
                result = (PyObject *) self;
              } else {
                Object_dealloc( self );
              }
            }
         }
      }
   }

/* Return the result */
   return result;
}

static int SetProxy( AstObject *this, Object *self ) {
/*
*  Name:
*     SetProxy

*  Purpose:
*     Store a clone of the supplied AST Object pointer in the supplied
*     starlink.Ast.Object, and register the starlink.Ast.Object as the
*     Python proxy for the AST Object.

*/
   if( !astOK ) return -1;
   THIS = astClone( this );
   astSetProxy( this, self );
   return astOK ? 0 : -1;
}



static PyTypeObject *GetType( AstObject *this ) {
/*
*  Name:
*     GetType

*  Purpose:
*     Returns the Python object class corresponding to the supplied AST
*     object class.

*/
   PyTypeObject *result = NULL;
   if( !astOK ) return NULL;

   const char *class = astGetC( this, "Class" );
   if( class ) {
      if( !strcmp( class, "ZoomMap" ) ) {
         result = (PyTypeObject *) &ZoomMapType;
      } else if( !strcmp( class, "UnitMap" ) ) {
        result = (PyTypeObject *) &UnitMapType;
      } else if( !strcmp( class, "TimeMap" ) ) {
        result = (PyTypeObject *) &TimeMapType;
      } else if( !strcmp( class, "SphMap" ) ) {
        result = (PyTypeObject *) &SphMapType;
      } else if( !strcmp( class, "GrismMap" ) ) {
        result = (PyTypeObject *) &GrismMapType;
      } else if( !strcmp( class, "RateMap" ) ) {
        result = (PyTypeObject *) &RateMapType;
      } else if( !strcmp( class, "PcdMap" ) ) {
        result = (PyTypeObject *) &PcdMapType;
      } else if( !strcmp( class, "WcsMap" ) ) {
        result = (PyTypeObject *) &WcsMapType;
      } else if( !strcmp( class, "CmpMap" ) ) {
        result = (PyTypeObject *) &CmpMapType;
      } else if( !strcmp( class, "TranMap" ) ) {
        result = (PyTypeObject *) &TranMapType;
      } else if( !strcmp( class, "NormMap" ) ) {
        result = (PyTypeObject *) &NormMapType;
      } else if( !strcmp( class, "PermMap" ) ) {
        result = (PyTypeObject *) &PermMapType;
      } else if( !strcmp( class, "ShiftMap" ) ) {
        result = (PyTypeObject *) &ShiftMapType;
      } else if( !strcmp( class, "LutMap" ) ) {
        result = (PyTypeObject *) &LutMapType;
      } else if( !strcmp( class, "WinMap" ) ) {
        result = (PyTypeObject *) &WinMapType;
      } else if( !strcmp( class, "Frame" ) ) {
         result = (PyTypeObject *) &FrameType;
      } else if( !strcmp( class, "FrameSet" ) ) {
         result = (PyTypeObject *) &FrameSetType;
      } else if( !strcmp( class, "CmpFrame" ) ) {
         result = (PyTypeObject *) &CmpFrameType;
      } else if( !strcmp( class, "SpecFrame" ) ) {
         result = (PyTypeObject *) &SpecFrameType;
      } else if( !strcmp( class, "DSBSpecFrame" ) ) {
         result = (PyTypeObject *) &DSBSpecFrameType;
      } else if( !strcmp( class, "SkyFrame" ) ) {
         result = (PyTypeObject *) &SkyFrameType;
      } else if( !strcmp( class, "TimeFrame" ) ) {
         result = (PyTypeObject *) &TimeFrameType;
      } else if( !strcmp( class, "FluxFrame" ) ) {
         result = (PyTypeObject *) &FluxFrameType;
      } else if( !strcmp( class, "SpecFluxFrame" ) ) {
         result = (PyTypeObject *) &SpecFluxFrameType;
      } else if( !strcmp( class, "Region" ) ) {
         result = (PyTypeObject *) &RegionType;
      } else if( !strcmp( class, "Box" ) ) {
         result = (PyTypeObject *) &BoxType;
      } else if( !strcmp( class, "Circle" ) ) {
         result = (PyTypeObject *) &CircleType;
      } else if( !strcmp( class, "Ellipse" ) ) {
         result = (PyTypeObject *) &EllipseType;
      } else if( !strcmp( class, "Interval" ) ) {
         result = (PyTypeObject *) &IntervalType;
      } else if( !strcmp( class, "NullRegion" ) ) {
         result = (PyTypeObject *) &NullRegionType;
      } else if( !strcmp( class, "CmpRegion" ) ) {
         result = (PyTypeObject *) &CmpRegionType;
      } else if( !strcmp( class, "Prism" ) ) {
         result = (PyTypeObject *) &PrismType;
      } else if( !strcmp( class, "Channel" ) ) {
         result = (PyTypeObject *) &ChannelType;
      } else if( !strcmp( class, "FitsChan" ) ) {
         result = (PyTypeObject *) &FitsChanType;
      } else if( !strcmp( class, "StcsChan" ) ) {
         result = (PyTypeObject *) &StcsChanType;
      } else {
         char buff[ 200 ];
         sprintf( buff, "Python AST function GetType does not yet "
                  "support to the %s class", class );
         PyErr_SetString( INTER_err, buff );
      }
   }

   return result;
}

static PyArrayObject *GetArray( PyObject *object, int type, int append,
                                int ndim, int *dims, const char *arg,
                                const char *fun ){
/*
*  Name:
*     GetArrayObject

*  Purpose:
*     A wrapper for PyArray_ContiguousFromAny that issues better
*     error messages, and checks the ArrayObject has specified dimensions.

*/
   char buf[400];
   PyArrayObject *result = NULL;
   int error = 0;
   int i;
   int j;

/* Check a PyObject was supplied. */
   if( object ) {

/* Get a PyArrayObject from the PyObject, using the specified data type,
   but allowing any number of dimensions (so that we can produce a more
   helpful error message). */
      result = (PyArrayObject *) PyArray_ContiguousFromAny( object, type, 0,
                                                            100 );

/* Check the array was created succesfully. */
      if( result ) {

/* If the ArrayObject has more axes than requested, check that the first
   ndim axes have the correct length, and that all the extra trailing
   axes are degenerate (i.e. have a length of one). */
         if( result->nd > ndim ) {

            for( i = 0; i < ndim && !error; i++ ) {
               if( dims[ i ] > 0 && result->dimensions[ i ] != dims[ i ] ) {
                  sprintf( buf, "The '%s' array supplied to %s has a length "
                           "of %d for dimension %d (one-based) - should "
                           "be %d.", arg, fun, (int) result->dimensions[ i ],
                           i+1, dims[ i ] );
                  error = 1;
               }
               dims[ i ] = result->dimensions[ i ];
            }

            for( ; i < result->nd && !error; i++ ) {
               if( result->dimensions[ i ] > 1 ) {
                  sprintf( buf, "The '%s' array supplied to %s has too many "
                          "significant %s, but no more than %d %s allowed.",
                          arg, fun, (ndim==1?"dimension":"dimensions"),
                          ndim, (ndim==1?"is":"are") );
                  error = 1;
               }
            }

/* If the ArrayObject has exactly the right number of axes, check that
   they have the correct lengths. */
         } else if( result->nd == ndim ) {
            for( i = 0; i < ndim && !error; i++ ) {
               if( dims[ i ] > 0 && result->dimensions[ i ] != dims[ i ] ) {
                  sprintf( buf, "The '%s' array supplied to %s has a length "
                           "of %d for dimension %d (one-based) - should "
                           "be %d.", arg, fun, (int) result->dimensions[ i ],
                           i+1, dims[ i ] );
                  error = 1;
               }
               dims[ i ] = result->dimensions[ i ];
            }

/* If the ArrayObject has too few axes, and we are using the available
   ArrayObject axes as the leading axes (and therefore padding with
   trailing degenerate axes), check the available axes. */
         } else if( append ){

            for( i = 0; i < result->nd && !error; i++ ) {
               if( dims[ i ] > 0 && result->dimensions[ i ] != dims[ i ] ) {
                  sprintf( buf, "The '%s' array supplied to %s has a length "
                           "of %d for dimension %d (one-based) - should "
                           "be %d.", arg, fun, (int) result->dimensions[ i ],
                           i+1, dims[ i ] );
                  error = 1;
               }
               dims[ i ] = result->dimensions[ i ];
            }

            for( ; i < ndim && !error; i++ ) {
               if( dims[ i ] > 1 ) {
                  sprintf( buf, "The '%s' array supplied to %s has %d "
                          "%s, but %d %s required.", arg, fun, result->nd,
                          (ndim==1?"dimension":"dimensions"), ndim,
                          (ndim==1?"is":"are") );
                  error = 1;
               }
               dims[ i ] = 1;
            }

/* If the ArrayObject has too few axes, and we are using the available
   ArrayObject axes as the trailing axes (and therefore padding with
   leading degenerate axes), check the available axes. */
         } else {

            for( i = 0; i < ndim - result->nd && !error; i++ ) {
               if( dims[ i ] > 1 ) {
                  sprintf( buf, "The '%s' array supplied to %s has %d "
                          "%s, but %d %s required.", arg, fun, result->nd,
                          (ndim==1?"dimension":"dimensions"), ndim,
                          (ndim==1?"is":"are") );
                  error = 1;
               }
               dims[ i ] = 1;
            }

            for( j = 0; i < ndim && !error; i++,j++ ) {
               if( dims[ i ] > 0 && result->dimensions[ j ] != dims[ i ] ) {
                  sprintf( buf, "The '%s' array supplied to %s has a length "
                           "of %d for dimension %d (one-based) - should "
                           "be %d.", arg, fun, (int) result->dimensions[ j ],
                           j+1, dims[ i ] );
                  error = 1;
               }
               dims[ i ] = result->dimensions[ j ];
            }
         }
      }
   }

/* If an error was flagged, raise a ValueError exception, release the
   refererence to the ArrayObject, and nullify the returned pointer. */
   if( result && error ) {
      PyErr_SetString( PyExc_ValueError, buf );
      Py_DECREF(result);
      result = NULL;
   }

   return result;
}

static PyArrayObject *GetArray1D( PyObject *object, int *dim, const char *arg,
                                  const char *fun ){
/*
*  Name:
*     GetArray1D

*  Purpose:
*     A wrapper for PyArray_ContiguousFromAny that issues better
*     error messages, and checks the ArrayObject is 1-D with double
*     precision values.

*/
   return GetArray( object, PyArray_DOUBLE, 1, 1, dim, arg, fun );
}

static PyArrayObject *GetArray1I( PyObject *object, int *dim, const char *arg,
                                  const char *fun ){
/*
*  Name:
*     GetArray1I

*  Purpose:
*     A wrapper for PyArray_ContiguousFromAny that issues better
*     error messages, and checks the ArrayObject is 1-D with integer
*     values.

*/
   return GetArray( object, PyArray_INT, 1, 1, dim, arg, fun );
}

static char *DumpToString( AstObject *this, const char *options ){
/*
*  Name:
*     DumpToString

*  Purpose:
*     Returns a pointer to a dynamically allocated string containing a
*     dump of the supplied Object.

*/
   AstChannel *ch = NULL;
   char *result = NULL;

   if( !astOK ) return result;

   ch = astChannel( NULL, Sinka, options );
   astPutChannelData( ch, &result );
   astWrite( ch, this );
   ch = astAnnul( ch );

   return result;
}

static void Sinka( const char *text ){
/*
*  Name:
*     Sinka

*  Purpose:
*     Appends a supplied line of text to an expanding string associated
*     with a Channel.

*/
   if( text ) {
      char **store = astChannelData;
      int nc = astChrLen( *store );
      if( nc ) *store = astAppendString( *store, &nc, "\n" );
      *store = astAppendString( *store, &nc, text );
   }
}

static const char *AttNorm( const char *att, char *buff ){
/*
*  Name:
*     AttNorm

*  Purpose:
*     Normalise an attribute name. Multi-valued attribute names of the
*     form "<a>_<b>" are changed to "<a>(<b>)"

*/
   const char *result = att;
   if( att && buff ) {
      const char *us = strchr( att, '_' );
      if( us ) {
         sprintf( buff, "%.*s(%s)", (int)( us - att ), att, us + 1 );
         result = buff;
      }
   }
   return result;
}


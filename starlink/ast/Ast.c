/* Issues:

   - providing more base methods (repr, str, equal, etc)
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
   referred to inside Ast.h). */
static char *GetString( PyObject *value );
static PyArrayObject *GetArray( PyObject *object, int type, int append, int ndim,
       int *dims, const char *arg, const char *fun );
static PyArrayObject *GetArray1I( PyObject *object, int *dim, const char *arg,
       const char *fun );
static PyArrayObject *GetArray1D( PyObject *object, int *dim, const char *arg,
       const char *fun );

/* Macros used in this file */
#include "pyast.h"

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
static PyObject *Object_unlock( Object *self, PyObject *args );
static PyObject *Object_set( Object *self, PyObject *args );
static PyObject *Object_show( Object *self );
static PyObject *Object_same( Object *self, PyObject *args );
static PyObject *Object_test( Object *self, PyObject *args );
static PyTypeObject *GetType( AstObject *this );
static int SetProxy( AstObject *this, Object *self );
static void Object_dealloc( Object *self );

/* Standard AST class functons */
MAKE_ISA(Object)

/* Describe the methods of the class */
static PyMethodDef Object_methods[] = {
   DEF_ISA(Object,object),
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
   {NULL}  /* Sentinel */
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
   {NULL}  /* Sentinel */
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

static PyObject *Object_deepcopy( Object *self, PyObject *args ) {
   return Object_copy( self );
}

#undef NAME
#define NAME CLASS ".get"
static PyObject *Object_get( Object *self, PyObject *args ) {
   PyObject *result = NULL;
   const char *attrib;
   const char *value;
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
   if( PyArg_ParseTuple( args, "s:" NAME, &attrib ) ) {
      value = astHasAttribute( THIS, attrib);
      if( astOK ) result = value ? Py_True : Py_False;
   }
   TIDY;
   return result;
}

#undef NAME
#define NAME CLASS ".lock"
static PyObject *Object_lock( Object *self, PyObject *args ) {
   PyObject *result = NULL;
   int wait;
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
   if( PyArg_ParseTuple( args, "O!:" NAME, &ObjectType,
                         (PyObject **) &other ) ) {
      value = astSame( THIS, THAT );
      if( astOK ) result = value ?  Py_True : Py_False;
   }
   TIDY;
   return result;
}

#undef NAME
#define NAME CLASS ".set"
static PyObject *Object_set( Object *self, PyObject *args ) {
   PyObject *result = NULL;
   const char *settings;
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
   if( PyArg_ParseTuple( args, "s:" NAME, &attrib ) ) {
      value = astTest( THIS, attrib);
      if( astOK ) result = value ?  Py_True : Py_False;
   }
   TIDY;
   return result;
}

#undef NAME
#define NAME CLASS ".unlock"
static PyObject *Object_unlock( Object *self, PyObject *args ) {
   PyObject *result = NULL;
   int report;
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
static PyObject *Mapping_trann( Mapping *self, PyObject *args );
static PyObject *Mapping_trangrid( Mapping *self, PyObject *args );

/* Standard AST class functons */
MAKE_ISA(Mapping)

/* Describe the methods of the class */
static PyMethodDef Mapping_methods[] = {
   DEF_ISA(Mapping,mapping),
   {"decompose", (PyCFunction)Mapping_decompose, METH_NOARGS, "Decompose a Mapping into two component Mappings"},
   {"invert", (PyCFunction)Mapping_invert, METH_NOARGS, "Invert a Mapping"},
   {"mapbox", (PyCFunction)Mapping_mapbox, METH_VARARGS, " Find a bounding box for a Mapping."},
   {"linearapprox", (PyCFunction)Mapping_linearapprox, METH_VARARGS, "Obtain a linear approximation to a Mapping, if appropriate."},
   {"quadapprox", (PyCFunction)Mapping_quadapprox, METH_VARARGS, "Obtain a quadratic approximation to a 2D Mapping"},
   {"rate", (PyCFunction)Mapping_rate, METH_VARARGS, "Calculate the rate of change of a Mapping output"},
   {"rebin", (PyCFunction)Mapping_rebin, METH_VARARGS, "Rebin a region of a data grid"},
   {"rebinseq", (PyCFunction)Mapping_rebinseq, METH_VARARGS, "Rebin a region of a sequence of data grids"},
   {"trann", (PyCFunction)Mapping_trann, METH_VARARGS, "Transform N-dimensional coordinates"},
   {"trangrid", (PyCFunction)Mapping_trangrid, METH_VARARGS, "Transform a grid of positions"},
   {NULL}  /* Sentinel */
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
   {NULL}  /* Sentinel */
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

/* Standard AST class functons */
MAKE_ISA(ZoomMap)

/* Describe the methods of the class */
static PyMethodDef ZoomMap_methods[] = {
   DEF_ISA(ZoomMap,zoommap),
   {NULL}  /* Sentinel */
};

/* Define the AST attributes of the class */
MAKE_GETSETD(ZoomMap,Zoom)
static PyGetSetDef ZoomMap_getseters[] = {
   DEFATT(Zoom," ZoomMap scale factor"),
   {NULL}  /* Sentinel */
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
   ZoomMap_methods,           /* tp_methods */
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

/* Standard AST class functons */
MAKE_ISA(UnitMap)

/* Describe the methods of the class */
static PyMethodDef UnitMap_methods[] = {
   DEF_ISA(UnitMap,unitmap),
   {NULL}  /* Sentinel */
};

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
   UnitMap_methods,           /* tp_methods */
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

/* Standard AST class functons */
MAKE_ISA(PermMap)

/* Describe the methods of the class */
static PyMethodDef PermMap_methods[] = {
   DEF_ISA(PermMap,permmap),
   {NULL}  /* Sentinel */
};

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
   PermMap_methods,           /* tp_methods */
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

/* Frame */
/* ======= */

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

/* Standard AST class functons */
MAKE_ISA(Frame)

/* Describe the methods of the class */
static PyMethodDef Frame_methods[] = {
  DEF_ISA(Frame,frame),
  {"angle", (PyCFunction)Frame_angle, METH_VARARGS, "Calculate the angle subtended by two points at a this point"},
  {"axangle", (PyCFunction)Frame_axangle, METH_VARARGS, "Returns the angle from an axis, to a line through two points"},
  {"axdistance", (PyCFunction)Frame_axdistance, METH_VARARGS, "Find the distance between two axis values"},
  {"axoffset", (PyCFunction)Frame_axoffset, METH_VARARGS, "Add an increment onto a supplied axis value"},
  {"convert", (PyCFunction)Frame_convert, METH_VARARGS, "Determine how to convert between two coordinate systems"},
  {"distance", (PyCFunction)Frame_distance, METH_VARARGS, "Calculate the distance between two points in a Frame"},
  {"findframe", (PyCFunction)Frame_findframe, METH_VARARGS, "Find a coordinate system with specified characteristics"},
  {"format", (PyCFunction)Frame_format, METH_VARARGS, "Format a coordinate value for a Frame axis"},
  // astGetActiveUnit should be implemented as an attribute getter
  {"intersect", (PyCFunction)Frame_intersect, METH_VARARGS, "Find the point of intersection between two geodesic curves"},
  {"matchaxes", (PyCFunction)Frame_matchaxes, METH_VARARGS, "Find any corresponding axes in two Frames"},
  {"norm", (PyCFunction)Frame_norm, METH_VARARGS, "Normalise a set of Frame coordinates"},
  {"offset", (PyCFunction)Frame_offset, METH_VARARGS, "Calculate an offset along a geodesic curve"},
  {"offset2", (PyCFunction)Frame_offset2, METH_VARARGS, "Calculate an offset along a geodesic curve in a 2D Frame"},
  {"permaxes", (PyCFunction)Frame_permaxes, METH_VARARGS, "Permute the axis order in a Frame"},
  {"pickaxes", (PyCFunction)Frame_pickaxes, METH_VARARGS, "Crate a new Frame by picking axes from an existing one"},
  {"resolve", (PyCFunction)Frame_resolve, METH_VARARGS, "Resolve a vector into two orthogonal components"},
  {"unformat", (PyCFunction)Frame_unformat, METH_VARARGS, "Read a formatted coordinate value for a Frame"},
   {NULL}  /* Sentinel */
};

/* Define the AST attributes of the class */
MAKE_GETSETC(Frame,AlignSystem)
// Bottom(axis)
// Digits(axis)
// Direction(axis)
MAKE_GETSETC(Frame,Domain)
MAKE_GETSETD(Frame,Dut1)
MAKE_GETSETD(Frame,Epoch)
// Format(axis)
// Label(axis)
MAKE_GETSETL(Frame,MatchEnd)
MAKE_GETSETI(Frame,MaxAxes)
MAKE_GETSETI(Frame,MinAxes)
MAKE_GETROI(Frame,Naxes)
// NormUnits(axis)
MAKE_GETSETD(Frame,ObsAlt)
MAKE_GETSETD(Frame,ObsLat)
MAKE_GETSETD(Frame,ObsLon)
MAKE_GETSETL(Frame,Permute)
MAKE_GETSETL(Frame,PreserveAxes)
// Symbol(axis)
MAKE_GETSETC(Frame,System)
MAKE_GETSETC(Frame,Title)
// Top(axis)
// Unit(axis)

static PyGetSetDef Frame_getseters[] = {
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
   {NULL}  /* Sentinel */
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
   "AST Frame",             /* tp_doc */
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
   (initproc)Frame_init,    /* tp_init */
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

  if( PyArg_ParseTuple(args, "O!|s:" NAME, &FrameType, (PyObject**)&other, &domainlist ) ) {
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

  if( PyArg_ParseTuple(args, "O!|s:" NAME, &FrameType, (PyObject**)&other, &domainlist ) ) {
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
    Py_XDECREF( b2 );
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

   if( PyArg_ParseTuple( args, "O!:" NAME, &FrameType,
                         (PyObject **) &other ) ) {
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
  double offset;

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

  if ( PyArg_ParseTuple( args, "is:" NAME, &axis, &string ) && astOK ) {
    double value;
    int nchars;
    nchars = astUnformat( THIS, axis, string, &value );
    if (astOK) result = Py_BuildValue( "id", nchars, value );
  }

  TIDY;
  return result;
}

/* FrameSet */
/* ======= */

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

/* Standard AST class functons */
MAKE_ISA(FrameSet)

/* Describe the methods of the class */
static PyMethodDef FrameSet_methods[] = {
  DEF_ISA(FrameSet,frameset),
  {"addframe", (PyCFunction)FrameSet_addframe, METH_VARARGS, "Add a Frame to a FrameSet to define a new coordinate system"},
   {NULL}  /* Sentinel */
};

/* Define the AST attributes of the class */
MAKE_GETSETI(FrameSet,Base)
MAKE_GETSETI(FrameSet,Current)
MAKE_GETSETI(FrameSet,Nframe)

static PyGetSetDef FrameSet_getseters[] = {
   DEFATT(Base,"FrameSet base Frame index"),
   DEFATT(Current,"FrameSet current Frame index"),
   DEFATT(Nframe,"Number of Frames in a FrameSet"),
   {NULL}  /* Sentinel */
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
   "AST FrameSet",             /* tp_doc */
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
   (initproc)FrameSet_init,    /* tp_init */
   0,                         /* tp_alloc */
   0,                         /* tp_new */
};


/* Define the class methods */
static int FrameSet_init( FrameSet *self, PyObject *args, PyObject *kwds ){
   const char *options = " ";
   FrameSet *other;
   int result = -1;
   int naxes;

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

  if( PyArg_ParseTuple(args, "iO!O!:" NAME, &iframe,
                       &MappingType, (PyObject**)&other,
                       &FrameType, (PyObject**)&another ) && astOK ) {
      astAddFrame( THIS, iframe, THAT, ANOTHER );
      if (astOK) result = Py_None;
   }

   TIDY;
   return result;
}

/* Now describe the whole AST module */
/* ================================= */

/* Static method prototypes */
static PyObject *Static_escapes( PyObject *self, PyObject *args );
static PyObject *Static_tune( PyObject *self, PyObject *args );
static PyObject *Static_version( PyObject *self );

/* Static method implementations */

#undef NAME
#define NAME MODULE ".escapes"
static PyObject *Static_escapes( PyObject *self, PyObject *args ) {
   PyObject *result = NULL;
   int newval;
   int value;
   if( PyArg_ParseTuple(args, "i:" NAME, &newval ) ) {
      value = astEscapes( newval );
      if( astOK ) result = Py_BuildValue( "i", value );
   }
   TIDY;
   return result;
}

#undef NAME
#define NAME MODULE ".tune"
static PyObject *Static_tune( PyObject *self, PyObject *args ) {
   PyObject *result = NULL;
   int value;
   int oldval;
   const char *name;
   if( PyArg_ParseTuple(args, "si:" NAME, &name, &value ) ) {
      oldval = astTune( name, value );
      if( astOK ) result = Py_BuildValue( "i", oldval );
   }
   TIDY;
   return result;
}

static PyObject *Static_version( PyObject *self ) {
   PyObject *result = NULL;
   int version;
   version = astVersion;
   if( astOK ) result = Py_BuildValue( "i", version );
   TIDY;
   return result;
}

/* Describe the static methods of the class */
static PyMethodDef static_methods[] = {
   {"escapes", (PyCFunction)Static_escapes, METH_VARARGS, "Control whether graphical escape sequences are included in strings"},
   {"tune", (PyCFunction)Static_tune, METH_VARARGS,  "Set or get an AST global tuning parameter"},
   {"version", (PyCFunction)Static_version, METH_NOARGS,  "Return the version of the AST library being used"},
   {NULL}  /* Sentinel */
};

/* Describe the properties of the module. */
static struct PyModuleDef astmodule = {
   PyModuleDef_HEAD_INIT,
   "Ast",
   "AST Python interface.",
   -1,
   static_methods,
   NULL, NULL, NULL, NULL
};


/* Tell the python interpreter about this module. This includes telling
   the interpreter about each of the types defined by this module. */
PyMODINIT_FUNC PyInit_Ast(void) {
   PyObject *m;
   m = PyModule_Create(&astmodule);
   if( m == NULL ) return NULL;

/* Create singleton instances of the AST Exception classes. The
   RegisterErrors function is defined within file exceptions.c (generated
   automatically by the make_exceptions.py script on the basis of the ast_err.msg
   file). */
   if( !RegisterErrors( m ) ) return NULL;

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

   UnitMapType.tp_new = PyType_GenericNew;
   UnitMapType.tp_base = &MappingType;
   if( PyType_Ready(&UnitMapType) < 0) return NULL;
   Py_INCREF(&UnitMapType);
   PyModule_AddObject( m, "UnitMap", (PyObject *)&UnitMapType);

   PermMapType.tp_new = PyType_GenericNew;
   PermMapType.tp_base = &MappingType;
   if( PyType_Ready(&PermMapType) < 0) return NULL;
   Py_INCREF(&PermMapType);
   PyModule_AddObject( m, "PermMap", (PyObject *)&PermMapType);

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

/* The constants provided by this module. */
#define ICONST(Name) \
   PyModule_AddIntConstant( m, #Name, AST__##Name )

#define DCONST(Name) \
   PyModule_AddObject( m, #Name, PyFloat_FromDouble(AST__##Name) )

   ICONST(TUNULL);

   ICONST(URESAMP1);
   ICONST(URESAMP2);
   ICONST(URESAMP3);
   ICONST(URESAMP4);
   ICONST(USEVAR);
   ICONST(USEBAD);
   ICONST(CONSERVEFLUX);
   ICONST(REBININIT);
   ICONST(REBINEND);
   ICONST(GENVAR);
   ICONST(VARWGT);
   ICONST(NOBAD);
   ICONST(DISVAR);
   ICONST(UKERN1);
   ICONST(UINTERP);
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

   DCONST(BAD);

#undef ICONST
#undef DCONST

/* Initialise the numpi module */
   import_array();

   return m;
}


/* Utility functions */
/* ================= */

static char *GetString( PyObject *value ) {
/*
*  Name:
*     GetString

*  Purpose:
*     Read a null terminated string from a PyObject and store in
      dynamic memory (free the returned pointer using astFree).

*/
   char *result = NULL;
   if( value ) {
      PyObject *bytes = PyUnicode_AsASCIIString(value);
      if( bytes ) {
         const char *cval =  PyBytes_AS_STRING(bytes);
         if( cval ) {
            result = astStore( NULL, cval, strlen(cval ) + 1 );
         }
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
      } else if( !strcmp( class, "PermMap" ) ) {
        result = (PyTypeObject *) &PermMapType;
      } else if( !strcmp( class, "Frame" ) ) {
         result = (PyTypeObject *) &FrameType;
      } else if( !strcmp( class, "FrameSet" ) ) {
         result = (PyTypeObject *) &FrameSetType;
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


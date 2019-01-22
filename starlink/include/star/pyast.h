#ifndef PyAst_included
#define PyAst_included

#ifdef __cplusplus
extern "C" {
#endif

/* Prototypes for public functions */
#define PyAst_ToString_NUM 0
#define PyAst_ToString_RETURN char *
#define PyAst_ToString_PROTO ( PyObject *self )

#define PyAst_FromString_NUM 1
#define PyAst_FromString_RETURN PyObject *
#define PyAst_FromString_PROTO ( const char *string )

/* Total number of C API pointers */
#define PyAst_API_pointers 2

/* Following section is used only when compiling the pyast module. */
/* --------------------------------------------------------------- */
#ifdef PYAST_MODULE

/* Various simple short-hands */
#define LAST(pyobj) ((Object*)pyobj)->ast_object
#define LTHIS LAST(self)

#define AST(pyobj) (pyobj&&((PyObject*)pyobj!=Py_None)?LAST(pyobj):NULL)
#define THIS AST(self)
#define THAT AST(other)
#define ANOTHER AST(another)


#define MXDIM 20
#define MXATTR_LEN 50
#define ATTNORM(attrib) AttNorm(attrib,att_buf)

/*
*  Name:
*     TIDY

*  Purpose:
*     Clear the AST error status before returning to Python.

*  Synopsis:
*     TIDY

*  Description:
*     This macro tidies up before returning from an AST method to the
*     Python interpreter. It clears the AST error status. This is
*     because errors are communicated between methods by means of Python
*     exceptions. If an exception is raised by an AST method, responsibility
*     for execution of further methods is handed over to the Python
*     interpreter, which may choose to clear the exception allowing further
*     methods to execute. If this is done, the AST error status would
*     prevent further execution if it were not also cleared. It would be
*     nice to clear the AST status at the same time that the exception is
*     cleared, but I can;'t find out how to do this. So instead, clear
*     the AST error status before returning from every AST method.

*/

#define TIDY astClearStatus


/*
*  Name:
*     MAKE_ISA

*  Purpose:
*     Declare and define the "IsA" method for a named AST class.

*  Synopsis:
*     MAKE_ISA(class)

*  Description:
*     This macro expands to a prototype and definition of a function that
*     provides a Python interface to an AST class IsA method.

*  Parameters:
*     class
*        The AST Class name (e.g. Object, Mapping, etc).

*/

#define MAKE_ISA(class) \
\
static PyObject *isa_##class( Object *self ); \
static PyObject *isa_##class( Object *self ){ \
   PyObject *result = astIsA##class( self->ast_object ) ?  Py_True : Py_False; \
   TIDY; \
   Py_INCREF(result); \
   return result; \
}

/*
*  Name:
*     DEF_ISA

*  Purpose:
*     Expands to a description of the "IsA" method for a named AST class.

*  Synopsis:
*     DEF_ISA(class)

*  Description:
*     This macro expands to a static initialiser for PyMethodDef
*     structure that describes the "IsA" function for a named AST class.

*  Parameters:
*     class
*        The AST Class name (e.g. Object, Mapping, etc).
*     lclass
*        The lower case AST Class name (e.g. object, mapping, etc).

*/

#define DEF_ISA(class,lclass) \
   {"isa" #lclass, (PyCFunction)isa_##class, METH_NOARGS, "Test class membership"}


/*
*  Name:
*     MAKE_GET

*  Purpose:
*     Declare and define a get method for a generic AST attribute.

*  Synopsis:
*     MAKE_GET(class,attrib,getval)

*  Description:
*     This macro is intended to be used only within the macros that define
*     and declare get methods for AST attributes with specific data types.

*  Parameters:
*     class
*        The AST Class name (e.g. Object, Mapping, etc).
*     attrib
*        The AST attribute name (e.g. Nin, NFrame, etc ).
*     getval
*        The C expression to be returned by the getter function.

*/

#define MAKE_GET(class,attrib,getval) \
\
static PyObject *get##attrib( class *self, void *closure ); \
static PyObject *get##attrib( class *self, void *closure ){ \
   char att_buf[ MXATTR_LEN ]; \
   PyObject *result = (getval); \
   TIDY; \
   Py_INCREF( result ); \
   return result; \
}


/*
*  Name:
*     MAKE_SET

*  Purpose:
*     Declare and define a set method for a generic AST attribute.

*  Synopsis:
*     MAKE_SET(class,attrib,pytype,stype,setcode)

*  Description:
*     This macro is intended to be used only within the macros that define
*     and declare get and set methods for AST attributes with specific data
*     types.

*  Parameters:
*     class
*        The AST Class name (e.g. Object, Mapping, etc).
*     attrib
*        The AST attribute name (e.g. Nin, NFrame, etc ).
*     pytype
*        The Python type of the attribute (e.g. Unicode, Bool, etc).
*     stype
*        The human form of the attribute type (e.g. string, boolean, etc).
*     setcode
*        The C code that invokes the AST set method.

*/

#define MAKE_SET(class,attrib,pytype,stype,setcode) \
\
static int set##attrib( class *self, PyObject *value, void *closure ); \
static int set##attrib( class *self, PyObject *value, void *closure ){ \
   char att_buf[ MXATTR_LEN ]; \
   int result = -1; \
   if (value == NULL || value == Py_None ) { \
      astClear( ((Object*)self)->ast_object, ATTNORM(#attrib) ); \
      if( astOK ) result = 0; \
   } else { \
      setcode \
      if( result == -1 && !PyErr_Occurred()) { \
         char *rep = FormatObject( value ); \
         PyErr_Format( PyExc_TypeError, "Bad value (%s) supplied " \
                       "for " #class " attribute '" #attrib "'.", \
                       rep ); \
         rep = astFree( rep ); \
      } \
   } \
   TIDY; \
   return result; \
}

/*
*  Name:
*     MAKE_SETRO

*  Purpose:
*     Declare and define a set method for a read-only AST attribute.

*  Synopsis:
*     MAKE_SETRO(class,attrib,pytype,stype,setcode)

*  Description:
*     This macro creates a setter that simply throws an AttributeError
*     exception if an attempt is made to change the value of the attribute.
*     It is intended to be used only within the macros that define and
*     declare get and set methods for AST attributes with specific data
*     types.

*  Parameters:
*     class
*        The AST Class name (e.g. Object, Mapping, etc).
*     attrib
*        The AST attribute name (e.g. Nin, NFrame, etc ).

*/

#define MAKE_SETRO(class,attrib) \
\
static int set##attrib( class *self, PyObject *value, void *closure ); \
static int set##attrib( class *self, PyObject *value, void *closure ){ \
   if( ! PyErr_Occurred() ) { \
      PyErr_SetString( PyExc_AttributeError, \
                       "Can't set read-only attribute '" #attrib "'."); \
   } \
   return -1; \
}


/*
*  Name:
*     MAKE_GETC

*  Purpose:
*     Declare and define a get method for a string-valued AST attribute.

*  Synopsis:
*     MAKE_GETC(class,attrib)

*  Parameters:
*     class
*        The AST Class name (e.g. Object, etc).
*     attrib
*        The AST attribute name (e.g. Ident, ID, etc ).

*/

#define MAKE_GETC(class,attrib) \
MAKE_GET(class,attrib, \
   Py_BuildValue( "s", astGetC( ((Object*)self)->ast_object, ATTNORM(#attrib) ) ));


/*
*  Name:
*     MAKE_GETCOL

*  Purpose:
*     Declare and define a get method for a colour-valued AST attribute.

*  Synopsis:
*     MAKE_GETCOL(class,attrib)

*  Parameters:
*     class
*        The AST Class name (in ptactice, this should always be Plot).
*     attrib
*        The AST attribute name (e.g. Colour_Border ).

*/

#define MAKE_GETCOL(class,attrib) \
\
static PyObject *get##attrib( class *self, void *closure ); \
static PyObject *get##attrib( class *self, void *closure ){ \
   PyObject *result; \
   char att_buf[ MXATTR_LEN ]; \
   int icol = astGetI( ((Object*)self)->ast_object, ATTNORM(#attrib)); \
   const char *text = IntToColour( self, icol ); \
   if( text ) { \
      result = Py_BuildValue( "s", text ); \
   } else { \
      result = Py_BuildValue( "i", icol ); \
   } \
   TIDY; \
   Py_INCREF( result ); \
   return result; \
}

/*
*  Name:
*     MAKE_GETL

*  Purpose:
*     Declare and define a get method for a boolean-valued AST attribute.

*  Synopsis:
*     MAKE_GETL(class,attrib)

*  Parameters:
*     class
*        The AST Class name (e.g. Object, etc).
*     attrib
*        The AST attribute name (e.g. UseDefs, etc ).

*/

#define MAKE_GETL(class,attrib) \
MAKE_GET(class,attrib, \
   astGetI( ((Object*)self)->ast_object, ATTNORM(#attrib) ) ? Py_True : Py_False);


/*
*  Name:
*     MAKE_GETI

*  Purpose:
*     Declare and define a get method for a integer-valued AST attribute.

*  Synopsis:
*     MAKE_GETI(class,attrib)

*  Parameters:
*     class
*        The AST Class name (e.g. Mapping, etc).
*     attrib
*        The AST attribute name (e.g. Nin, etc ).

*/

#define MAKE_GETI(class,attrib) \
MAKE_GET(class,attrib, \
   PyLong_FromLong((long int) astGetI( ((Object*)self)->ast_object, ATTNORM(#attrib) )));

/*
*  Name:
*     MAKE_GETD

*  Purpose:
*     Declare and define a get method for a double-valued AST attribute.

*  Synopsis:
*     MAKE_GETD(class,attrib)

*  Parameters:
*     class
*        The AST Class name (e.g. Mapping, etc).
*     attrib
*        The AST attribute name (e.g. Nin, etc ).

*/

#define MAKE_GETD(class,attrib) \
MAKE_GET(class,attrib, \
   PyFloat_FromDouble(astGetD( ((Object*)self)->ast_object, ATTNORM(#attrib) )));


/*
*  Name:
*     MAKE_GETSETC

*  Purpose:
*     Declare and define get and set methods for a string-valued AST attribute.

*  Synopsis:
*     MAKE_GETSETC(class,attrib)

*  Parameters:
*     class
*        The AST Class name (e.g. Object, etc).
*     attrib
*        The AST attribute name (e.g. Ident, ID, etc ).

*/


#define SETCODEC(attrib) \
   char *cval = GetString(NULL,value); \
   if( cval ) { \
      astSetC( ((Object*)self)->ast_object, ATTNORM(#attrib), cval ); \
      if( astOK ) result = 0; \
      cval = astFree( cval ); \
   }

#define MAKE_GETSETC(class,attrib) \
   MAKE_GETC(class,attrib) \
   MAKE_SET(class,attrib, Unicode, string, SETCODEC(attrib));



/*
*  Name:
*     MAKE_GETSETL

*  Purpose:
*     Declare and define get and set methods for a boolean-valued AST
*     attribute.

*  Synopsis:
*     MAKE_GETSETL(class,attrib)

*  Parameters:
*     class
*        The AST Class name (e.g. Object, etc).
*     attrib
*        The AST attribute name (e.g. UseDefs, etc ).

*/

#define SETCODEL(attrib) \
   if( PyBool_Check( value ) ) { \
      astSetI( ((Object*)self)->ast_object, ATTNORM(#attrib), ( value == Py_True ) ); \
      if( astOK ) result = 0; \
   }


#define MAKE_GETSETL(class,attrib) \
   MAKE_GETL(class,attrib) \
   MAKE_SET(class,attrib, Bool, boolean, SETCODEL(attrib));


/*
*  Name:
*     MAKE_GETSETI

*  Purpose:
*     Declare and define get and set methods for a integer-valued AST
*     attribute.

*  Synopsis:
*     MAKE_GETSETI(class,attrib)

*  Parameters:
*     class
*        The AST Class name (e.g. Mapping, etc).
*     attrib
*        The AST attribute name (e.g. Nin, etc ).

*/

#define SETCODEI(attrib) \
   int ival = PyLong_AsLong( value ); \
   if( !PyErr_Occurred() ) { \
      astSetI( ((Object*)self)->ast_object, ATTNORM(#attrib), ival ); \
      if( astOK ) result = 0; \
   }

#define MAKE_GETSETI(class,attrib) \
   MAKE_GETI(class,attrib) \
   MAKE_SET(class,attrib, Long, integer, SETCODEI(attrib));


/*
*  Name:
*     MAKE_GETSETD

*  Purpose:
*     Declare and define get and set methods for a double-valued AST
*     attribute.

*  Synopsis:
*     MAKE_GETSETD(class,attrib)

*  Parameters:
*     class
*        The AST Class name (e.g. Mapping, etc).
*     attrib
*        The AST attribute name (e.g. Nin, etc ).

*/

#define SETCODED(attrib) \
   double dval = PyFloat_AsDouble( value ); \
   if( !PyErr_Occurred() ) { \
      astSetD( ((Object*)self)->ast_object, ATTNORM(#attrib), dval ); \
      if( astOK ) result = 0; \
   }


#define MAKE_GETSETD(class,attrib) \
   MAKE_GETD(class,attrib) \
   MAKE_SET(class,attrib, Float, floating point, SETCODED(attrib));


/*
*  Name:
*     MAKE_GETROC

*  Purpose:
*     Declare and define get and set methods for a read-only string-valued
*     AST attribute.

*  Synopsis:
*     MAKE_GETROC(class,attrib)

*  Parameters:
*     class
*        The AST Class name (e.g. Object, etc).
*     attrib
*        The AST attribute name (e.g. Ident, ID, etc ).

*/

#define MAKE_GETROC(class,attrib) \
   MAKE_GET(class,attrib, \
      Py_BuildValue( "s", astGetC( ((Object*)self)->ast_object, ATTNORM(#attrib) ) )); \
   MAKE_SETRO(class,attrib)


/*
*  Name:
*     MAKE_GETROL

*  Purpose:
*     Declare and define get and set methods for a read-only boolean-valued
*     AST attribute.

*  Synopsis:
*     MAKE_GETROL(class,attrib)

*  Parameters:
*     class
*        The AST Class name (e.g. Object, etc).
*     attrib
*        The AST attribute name (e.g. UseDefs, etc ).

*/

#define MAKE_GETROL(class,attrib) \
   MAKE_GET(class,attrib, \
      astGetI( ((Object*)self)->ast_object, ATTNORM(#attrib) ) ? Py_True : Py_False); \
   MAKE_SETRO(class,attrib)


/*
*  Name:
*     MAKE_GETROI

*  Purpose:
*     Declare and define get and set methods for a read-only integer-valued
*     AST attribute.

*  Synopsis:
*     MAKE_GETROI(class,attrib)

*  Parameters:
*     class
*        The AST Class name (e.g. Mapping, etc).
*     attrib
*        The AST attribute name (e.g. Nin, etc ).

*/

#define MAKE_GETROI(class,attrib) \
   MAKE_GET(class,attrib, \
      PyLong_FromLong((long int) astGetI( ((Object*)self)->ast_object, ATTNORM(#attrib) ))); \
   MAKE_SETRO(class,attrib)

/*
*  Name:
*     MAKE_GETROD

*  Purpose:
*     Declare and define get and set methods for a read-only double-valued
*     AST attribute.

*  Synopsis:
*     MAKE_GETROD(class,attrib)

*  Parameters:
*     class
*        The AST Class name (e.g. Mapping, etc).
*     attrib
*        The AST attribute name (e.g. Nin, etc ).

*/

#define MAKE_GETROD(class,attrib) \
   MAKE_GET(class,attrib, \
      PyFloat_FromDouble(astGetD( ((Object*)self)->ast_object, ATTNORM(#attrib) ))); \
   MAKE_SETRO(class,attrib)


/*
*  Name:
*     MAKE_GETSETCOL

*  Purpose:
*     Declare and define get and set methods for a colour-valued AST attribute.

*  Synopsis:
*     MAKE_GETSETCOL(class,attrib)

*  Parameters:
*     class
*        The AST Class name (in practice, this should always be Plot).
*     attrib
*        The AST attribute name (e.g. Colour_Border, etc ).

*/


#define SETCODECOL(attrib) \
   int icol; \
   char *cval = GetString(NULL,value); \
   if( cval ) { \
      icol = ColourToInt( self, cval ); \
      if( icol >= 0 ) { \
         astSetI(  ((Object*)self)->ast_object, ATTNORM(#attrib), icol ); \
      } \
      cval = astFree( cval ); \
   } else if( LONG_CHECK(value) ) { \
      icol = PyLong_AsLong( value ); \
      astSetI(  ((Object*)self)->ast_object, ATTNORM(#attrib), icol ); \
   } else if( ! PyErr_Occurred() ) { \
      char *rep = FormatObject(value); \
      PyErr_Format( PyExc_TypeError, "Cannot set attribute '" \
                    #attrib "' - value (%s) is not a known colour " \
                    "name or an integer.", rep ); \
      rep = astFree( rep ); \
   } \
   if( astOK ) result = 0;


#define MAKE_GETSETCOL(class,attrib) \
   MAKE_GETCOL(class,attrib) \
   MAKE_SET(class,attrib, Unicode, string, SETCODECOL(attrib));


/*
*  Name:
*     DEFATT

*  Purpose:
*     Define a literal PyGetSetDef structure to describe the get and set
*     functions for a single AST attribute.

*  Synopsis:
*     DEFATT(attrib,doc)

*  Description:
*     This macro is intended to be used within the declaration of the
*     static array of PyGetSetDef structures describing the get and set
*     functions for a class, a pointer to which is stored in the type
*     object for the class.

*  Parameters:
*     attrib
*        The AST attribute name (e.g. UseDefs, etc ).
*     doc
*        The quoted documentation string describing the attribute.

*/

#define DEFATT(attrib,doc) \
   { #attrib, (getter)get##attrib, (setter)set##attrib, doc, NULL}


/* The following section is used only in modules that use Ast's API */
/* ---------------------------------------------------------------- */
#else

static void **PyAst_API;

#define PyAst_FromString \
 (*(PyAst_FromString_RETURN (*)PyAst_FromString_PROTO) PyAst_API[PyAst_FromString_NUM])
#define PyAst_ToString \
 (*(PyAst_ToString_RETURN (*)PyAst_ToString_PROTO) PyAst_API[PyAst_ToString_NUM])

/*
 * Returns -1 on error, 0 on success.
 * Raises a PyExc_ImportError exception.
 */
static int import_pyast(void) {
   PyObject * pyast = NULL;
   if (PyErr_Occurred()) return -1;
   pyast = PyImport_ImportModule("starlink.Ast");
   PyAst_API = (void **) PyCapsule_Import( "starlink.Ast._C_API", 0 );
   if( ! PyAst_API && !PyErr_Occurred() ) {
       PyErr_Print();
       PyErr_SetString(PyExc_ImportError, "starlink.Ast failed to import");
   }
   Py_XDECREF(pyast);
   return PyAst_API ? 0 : -1;
}

#endif

#ifdef __cplusplus
}
#endif

#endif

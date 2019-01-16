#!python3

"""
Generates the file exceptions.c which encapsulates Python/AST
exception handling.

The environment variable AST_SOURCE should be set to point to
the folder containing the source distribution for the AST
library.
"""

from __future__ import print_function

import os
import os.path


def make_exceptions(dirname=None):

    if 'AST_SOURCE' not in os.environ:
        print("Please set AST_SOURCE environment variable to point to the AST source code directory")
        exit(1)

    # ensure that we have the error codes file
    errfile = os.path.join(os.environ['AST_SOURCE'], "ast_err.h")
    if not os.path.exists(errfile):
        print("Could not find the ast_err.h file in directory " + os.environ['AST_SOURCE'])
        exit(1)

    # Open an output C file
    cfilename = "exceptions.c"
    if dirname is not None:
        cfilename = os.path.join(dirname, cfilename)

    cfile = open(cfilename, "w")

    # Need a C header
    print(r"""/*
*  Name:
*     exceptions.c

*  Purpose:
*     Raise a Python exception when AST reports an error.

*  Description:
*     For each AST error code defined in ast_err.h, this file creates
*     a singleton instance of a corresponding Python Exception class. It
*     also provides an implementation of the astPutErr function that AST
*     uses to report errors. This implementation raises the corresponding
*     Python exception whenever AST reports an error.

*  Notes:
*     - This file is generated automatically by the "make_exceptions.py"
*       script, and should not be edited.

*/

/* Prototypes for functions defined in this file. */
static int RegisterErrors( PyObject *m );

/* Declare a static variable to hold an instance of a base AST Exception
   from which all the others are derived. */
static PyObject *AstError_err;

/* For each AST error code, declare a static variable to hold an instance
   of the corresponding Python Exception. */""", file=cfile)

    # Now read the ast_err.h file and create extract all the error codes
    # Note that AST__3DFSET is not currently supported because a
    # variable can not start with a number
    errcodes = []
    for line in open(errfile, "r"):
        words = line.split()
        if words and words[0] == "enum" and words[2][:5] == "AST__" and words[2][5:6].isalpha():
            errcodes.append(words[2][5:])

    if not errcodes:
        print("Could not find any error codes. Aborting")
        cfile.close()
        os.path.unlink(cfilename)
        exit(1)

    for code in errcodes:
        print("static PyObject *{0}_err;".format(code), file=cfile)

    print(r"""
/* Defines a function that creates a Python Exception object
   for each AST error code, and uses them to initialises the
   above static variables. It reurns 1 if successful, and zero
   if an error occurs. */

static int RegisterErrors( PyObject *m ){

   PyObject *dict = PyModule_GetDict(m);

/* First create an instance of a base AST exception class from which
   the other derive. */
   if( !( AstError_err = PyErr_NewException("Ast.AstError", NULL, NULL))) return 0;
   PyDict_SetItemString( dict, "AstError", AstError_err );

/* Now create an instance of each derived AST exception class. */""", file=cfile)

    for code in errcodes:
        print('   if( !({0}_err = PyErr_NewException("Ast.{0}", AstError_err, NULL))) return 0;'.format(code),
              file=cfile)
        print('   PyDict_SetItemString( dict, "{0}", {0}_err );'.format(code), file=cfile)
        print(" ", file=cfile)

    print(r"""   return 1;
}


/* The AST library calls this function to deliver an error message
   in the form of a Python Exception. For each AST error code, raise
   the corresponding Python Exception. */
void astPutErr_( int status_value, const char *message ) {

/* Local Variables: */
   PyObject *ex;
   PyObject *ptype;
   PyObject *pvalue;
   PyObject *ptraceback;
   char *text;
   int lstat;
   int nc;

/* Clear the AST status value so that AST memory functions will function
   properly. */
   lstat = astStatus;
   astClearStatus;

/* If an AST exception has already occurred, append the new message to it. */
   ex = PyErr_Occurred();
   if( ex ){
      if( PyErr_GivenExceptionMatches( ex, AstError_err ) ) {

/* Get the existing Exception text */
         PyErr_Fetch( &ptype, &pvalue, &ptraceback );
         PyObject *str = PyObject_Str( pvalue );
         text = GetString( NULL, str );
         Py_DECREF(str);
         if( text ) {

/* Ignore messages that give the C source file and line number since they are
   not useful for Python users. */
            if( strstr( text, "Ast.c" ) ) {
               text = astFree( text );
               nc = 0;
            } else {
               nc = strlen( text );
               text = astAppendString( text, &nc, "\n" );
            }

/* Append the new message and store the extended text with the Exception. */
            text = astAppendString( text, &nc, message );
            Py_DECREF( pvalue );
            pvalue = PyUnicode_FromString( text );
            PyErr_Restore( ptype, pvalue, ptraceback);
            text = astFree( text );
         }
      }

/* If an AST or non-AST exception has already occurred, restore the
   original AST status value then return without action. */
      astSetStatus( lstat );
      return;
   }

/* If no exception has already occurred, raise an appropriate AST exception now. */""", file=cfile)

    first = True
    for code in errcodes:
        if first:
            print("   if( status_value == AST__{0} ) {{".format(code), file=cfile)
            first = False
        else:
            print("   }} else if( status_value == AST__{0} ) {{".format(code), file=cfile)
        print("      PyErr_SetString( {0}_err, message );".format(code), file=cfile)

    print("""   } else {
      PyErr_SetString( AstError_err, message );
   }

/* restore the original AST status value. */
   astSetStatus( lstat );
}
""", file=cfile)
    cfile.close()


if __name__ == "__main__":
    make_exceptions()

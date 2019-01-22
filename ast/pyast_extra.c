/*
*+
*  Name:
*     pyast_extra.c

*  Purpose:
*     Provide pyast with access to some protected features of AST

*  Type of Module:
*     C source file.

*  Description:
*     This file provides public APIs for some protected functions in the
*     AST library, for use within pyast. The API is defined by the
*     pyast_extra.h include file.

*  Copyright:
*     Copyright (C) 2019 East Asian Observatory

*  Licence:
*     This program is free software: you can redistribute it and/or
*     modify it under the terms of the GNU Lesser General Public
*     License as published by the Free Software Foundation, either
*     version 3 of the License, or (at your option) any later
*     version.
*
*     This program is distributed in the hope that it will be useful,
*     but WITHOUT ANY WARRANTY; without even the implied warranty of
*     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*     GNU Lesser General Public License for more details.
*
*     You should have received a copy of the GNU Lesser General
*     License along with this program.  If not, see
*     <http://www.gnu.org/licenses/>.

*  Authors:
*     DSB: David S. Berry (Starlink)

*  History:
*     21-JAN-2019 (DSB):
*        Original version.
*/

/* Define the astCLASS macro so that this file has access to the
   protected API. */
#define astCLASS pyast

/* Include files: */
/* -------------- */
#include "mapping.h"
#include "memory.h"
#include "pyast_extra.h"
#include "ast_err.h"


/* Wrapper functions: */
/* ------------------ */
int astMapMergeID_( AstMapping *this, int where, int series, int *nmap,
                    AstMapping ***id_list, int **invert_list, int *status ) {

/* Local Variables: */
   AstMapping **map_list = NULL;
   int i;
   int result = -1;

/* Check inherited status */
   if( !astOK ) return result;

/* Get genuine pointers from the supplied ID values. */
   map_list = astMalloc( (*nmap)*sizeof(*map_list) );
   if( astOK ) {
      for( i = 0; i < *nmap; i++ ) {
         map_list[ i ] = astMakePointer( (*id_list)[ i ] );
      }
   }

/* Check the "here" value is correct. */
   if( this != map_list[ where ] && astOK ) {
      astError( AST__INVAR, "astMapMerge(%s): The supplied mapping 'this' "
                "(a %s) is not stored at the specified element (%d) of "
                "the map_list array.", status, astGetClass(this),
                astGetClass(this), where );
   }

/* The astMapMerge method call below may change annull one or more of the
   supplied Mapping pointers. This could cause the underlying Mapping object
   to be deleted. This can cause big problems for python. So ensure the
   supplied pointers are not modified by taking deep copies. */
   if( astOK ) {
      for( i = 0; i < *nmap; i++ ) {
         map_list[ i ] = astCopy( map_list[ i ] );
      }
   }

/* Call the protected method. This returns a list of genuine Mapping
   pointers in a newly allocated array. Call the function directly,
   rather than through the usual macro API, to avoid calling this
   function recursively. */
   result = astMapMerge_( map_list[ where ], where, series, nmap, &map_list,
                          invert_list, status );

/* Store the ID values corresponding to the returned Mapping pointers,
   adjusting the size of the supplied array as needed. These returned
   pointers are independent of the supplied pointers because deep copies
   of the supplied pointers were used above. */
   *id_list = astGrow( *id_list, *nmap, sizeof(**id_list) );
   if( astOK ) {
      for( i = 0; i < *nmap; i++ ) {
         (*id_list)[ i ] = astMakeId( map_list[ i ] );
      }
   }

/* Free the array of mapping pointers. */
   map_list = astFree( map_list );

/* Return the index of the first modified mapping */
   return result;
}


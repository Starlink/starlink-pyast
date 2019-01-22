#if !defined( PYAST_EXTRA_INCLUDED )   /* Include this file only once */
#define PYAST_EXTRA_INCLUDED
/*
*  Name:
*     pyast_extra.h

*  Type:
*     C include file.

*  Purpose:
*     Define the interface to the pyast_extra module.

*  Invocation:
*     #include "pyast_extra.h"

*  Description:
*     This include file defines the interface for the function in the
*     file pyast_extra.c. These provide pyast with access to some protected
*     features of AST.

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
*     DSB: B.S. Berry (EAO)

*  History:
*     21-JAN-2019 (DSB):
*        Original version.
*/

/* Macros. */
/* ------- */
#define STATUS_PTR astGetStatusPtr

/* Function prototypes. */
/* -------------------- */
int astMapMergeID_( AstMapping *, int, int, int *, AstMapping ***, int **, int * );

/* Function interfaces. */
/* -------------------- */
/* These macros are wrap-ups for the functions defined by this class to make
   them easier to invoke (e.g. to avoid type mis-matches when passing pointers
   to objects from derived classes). */

#define astMapMerge(this,where,series,nmap,id_list,invert_list) \
astMapMergeID_(astCheckMapping(this),where,series,nmap,id_list,invert_list,STATUS_PTR)



#endif




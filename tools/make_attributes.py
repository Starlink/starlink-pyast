#!python3

"""
This script reads the file "attributes.desc", which contains descriptions
of all multi-valued AST attributes, and creates a pair of files for each
attribute. For attribute "<attr>", the two files are:

<attr>_def.c: Contains the prototypes and definitions of methods to access
the attribute values.

<attr>_desc.c: Contains the description of the attribute getters and
setters that need to be included in th Python TypeObject for the class.
"""

from __future__ import print_function

import os
import os.path


def make_attributes(dirname=None):

    #  The hard-wired maximum number of dimensions supported by pyast.
    mxdim = 40

#  Open the input file.
    file = "attributes.desc"
    if dirname is not None:
        file = os.path.join(dirname, file)

    infile = open(file, "r")

#  Read the input file.
    for line in infile:

        #  Ignore blank lines or comment lines
        line = line.strip()
        if len(line) == 0 or line.startswith("#"):
            continue

#  Split the line into comma-separated fields
        (classname, attname, readonly, atype, desc, maxindex, minindex, items) = line.split(",")

#  Convert strings to numerical values
        minindex = int(minindex)

#  Initialise lists
        att_decs = list()
        att_descs = list()

#  Convert the fields to more useful types.
        items = items.split()
        if maxindex == "MXDIM":
            maxindex = mxdim
        else:
            maxindex = int(maxindex)

#  Loop over all indices for multi-valued attributes
        for i in range(minindex, maxindex + 1):

            #  Loop over all keys for multi-valued attributes
            for item in items:

                #  Construct the name of the attribute.
                aname = attname
                if item != "-":
                    aname += "_" + item

                else:
                    aname += "_" + str(i)

#  Form the C macro invocation that defines the attribute and append it
#  to the list.
                mac = "MAKE_GET" + readonly + atype + "(" + classname + "," + aname + ")"
                att_decs.append(mac)

#  Form the attribute description to store in the Python TypeObject.
                mac = "DEFATT(" + aname + ", \"" + desc + "\"),"
                att_descs.append(mac)

#  Multi-valued attributes can also (usually) be used without any index
#  of key. So add an unqualified attribute to the lists.
        mac = "MAKE_GET" + readonly + atype + "(" + classname + "," + attname + ")"
        att_decs.append(mac)
        mac = "DEFATT(" + attname + ", \"" + desc + "\"),"
        att_descs.append(mac)

#  Open the output def file
        cfilename = attname + "_def.c"
        if dirname is not None:
            cfilename = os.path.join(dirname, cfilename)
        cfile = open(cfilename, "w")

# Write out a prologue for the C file
        print(r"""/*
*  Name:
*     """ + attname + "_def.c" + r"""

*  Purpose:
*     Declare and define the Python accessor methods for the """ + attname + r"""
*     attribute.

*  Description:
*     This file uses the macros defined in pyast.h to define and declare
*     methods for accessing all the values within the multi-valued AST
*     attribute """ + attname + r""".

*  Notes:
*     - This file is generated automatically by the "make_attributes.py"
*     script, and should not be edited. Instead, edit the file
*     attributes.desc which is used as input by "make_attributes.py".

*/
""", file=cfile)

#  Write out the attribute declarations, then close the file.
        for att in att_decs:
            print(att, file=cfile)
        cfile.close()

#  Open the output desc file
        cfilename = attname + "_desc.c"
        if dirname is not None:
            cfilename = os.path.join(dirname, cfilename)
        cfile = open(cfilename, "w")

# Write out a prologue for the C file
        print(r"""/*
*  Name:
*     """ + attname + "_desc.c" + r"""

*  Purpose:
*     Add entries to the array of Python attribute accessors describing
*     the """ + attname + r""" attribute.

*  Description:
*     This file uses the macros defined in pyast.h to create a description
*     of the multi-valued AST attribute """ + attname + r""" for inclusion in
*     the array of attribute getters and setters store in the Python
*     TypeObject for the class.

*  Notes:
*     - This file is generated automatically by the "make_attributes.py"
*     script, and should not be edited. Instead, edit the file
*     attributes.desc which is used as input by "make_attributes.py".

*/
""", file=cfile)

#  Write out the attribute dedriptions, then close the file.
        for att in att_descs:
            print("   " + att, file=cfile)
        cfile.close()

#  Close the input file.
    infile.close()


if __name__ == "__main__":
    make_attributes()

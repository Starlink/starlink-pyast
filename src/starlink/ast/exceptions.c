/*
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
   of the corresponding Python Exception. */
static PyObject *ATGER_err;
static PyObject *ATSER_err;
static PyObject *ATTIN_err;
static PyObject *AXIIN_err;
static PyObject *BADAT_err;
static PyObject *BADBX_err;
static PyObject *BADIN_err;
static PyObject *BADNI_err;
static PyObject *BADNO_err;
static PyObject *BADPW_err;
static PyObject *BADSM_err;
static PyObject *BADWM_err;
static PyObject *BDBRK_err;
static PyObject *BDFMT_err;
static PyObject *BDFTS_err;
static PyObject *BDOBJ_err;
static PyObject *CLPAX_err;
static PyObject *CORNG_err;
static PyObject *CVBRK_err;
static PyObject *DIMIN_err;
static PyObject *DTERR_err;
static PyObject *ENDIN_err;
static PyObject *EOCHN_err;
static PyObject *EXPIN_err;
static PyObject *FCRPT_err;
static PyObject *FMTER_err;
static PyObject *FRMIN_err;
static PyObject *FRSIN_err;
static PyObject *FTCNV_err;
static PyObject *GRFER_err;
static PyObject *INHAN_err;
static PyObject *INNCO_err;
static PyObject *INTER_err;
static PyObject *INTRD_err;
static PyObject *KYCIR_err;
static PyObject *LDERR_err;
static PyObject *LUTII_err;
static PyObject *LUTIN_err;
static PyObject *MEMIN_err;
static PyObject *MTR23_err;
static PyObject *MTRAX_err;
static PyObject *MTRML_err;
static PyObject *MTRMT_err;
static PyObject *NAXIN_err;
static PyObject *NCHIN_err;
static PyObject *NCOIN_err;
static PyObject *NCPIN_err;
static PyObject *NELIN_err;
static PyObject *NOCTS_err;
static PyObject *NODEF_err;
static PyObject *NOFTS_err;
static PyObject *NOMEM_err;
static PyObject *NOPTS_err;
static PyObject *NOWRT_err;
static PyObject *NPTIN_err;
static PyObject *OBJIN_err;
static PyObject *OPT_err;
static PyObject *PDSIN_err;
static PyObject *PLFMT_err;
static PyObject *PRMIN_err;
static PyObject *PTRIN_err;
static PyObject *PTRNG_err;
static PyObject *RDERR_err;
static PyObject *REGIN_err;
static PyObject *REMIN_err;
static PyObject *SCSIN_err;
static PyObject *SELIN_err;
static PyObject *SLAIN_err;
static PyObject *TRNND_err;
static PyObject *UNMQT_err;
static PyObject *VSMAL_err;
static PyObject *WCSAX_err;
static PyObject *WCSNC_err;
static PyObject *WCSPA_err;
static PyObject *WCSTY_err;
static PyObject *XSOBJ_err;
static PyObject *ZOOMI_err;
static PyObject *BADCI_err;
static PyObject *ILOST_err;
static PyObject *ITFER_err;
static PyObject *ITFNI_err;
static PyObject *MBBNF_err;
static PyObject *MRITF_err;
static PyObject *OCLUK_err;
static PyObject *UNFER_err;
static PyObject *URITF_err;
static PyObject *GBDIN_err;
static PyObject *NGDIN_err;
static PyObject *PATIN_err;
static PyObject *SISIN_err;
static PyObject *SSPIN_err;
static PyObject *UINER_err;
static PyObject *UK1ER_err;
static PyObject *COMIN_err;
static PyObject *CONIN_err;
static PyObject *DUVAR_err;
static PyObject *INNTF_err;
static PyObject *MIOPA_err;
static PyObject *MIOPR_err;
static PyObject *MISVN_err;
static PyObject *MLPAR_err;
static PyObject *MRPAR_err;
static PyObject *NORHS_err;
static PyObject *UDVOF_err;
static PyObject *VARIN_err;
static PyObject *WRNFA_err;
static PyObject *BADUN_err;
static PyObject *NORSF_err;
static PyObject *NOSOR_err;
static PyObject *SPCIN_err;
static PyObject *XMLNM_err;
static PyObject *XMLCM_err;
static PyObject *XMLPT_err;
static PyObject *XMLIT_err;
static PyObject *XMLWF_err;
static PyObject *ZERAX_err;
static PyObject *BADOC_err;
static PyObject *MPGER_err;
static PyObject *MPIND_err;
static PyObject *REGCN_err;
static PyObject *NOVAL_err;
static PyObject *INCTS_err;
static PyObject *TIMIN_err;
static PyObject *STCKEY_err;
static PyObject *STCIND_err;
static PyObject *CNFLX_err;
static PyObject *TUNAM_err;
static PyObject *BDPAR_err;
static PyObject *PXFRRM_err;
static PyObject *BADSUB_err;
static PyObject *BADFLG_err;
static PyObject *LCKERR_err;
static PyObject *FUNDEF_err;
static PyObject *MPVIN_err;
static PyObject *OPRIN_err;
static PyObject *NONIN_err;
static PyObject *MPKER_err;
static PyObject *MPPER_err;
static PyObject *BADKEY_err;
static PyObject *BADTYP_err;
static PyObject *OLDCOL_err;
static PyObject *BADNULL_err;
static PyObject *BIGKEY_err;
static PyObject *BADCOL_err;
static PyObject *BIGTAB_err;
static PyObject *BADSIZ_err;
static PyObject *BADTAB_err;
static PyObject *NOTAB_err;
static PyObject *LEVMAR_err;
static PyObject *NOFIT_err;
static PyObject *ISNAN_err;
static PyObject *WRERR_err;
static PyObject *BDVNM_err;
static PyObject *MIRRO_err;
static PyObject *MNPCK_err;
static PyObject *EXSPIX_err;
static PyObject *NOCNV_err;
static PyObject *IMMUT_err;
static PyObject *NOBOX_err;
static PyObject *NOSKY_err;
static PyObject *BDSKY_err;
static PyObject *BDCLS_err;
static PyObject *NOIMP_err;
static PyObject *BGMOC_err;
static PyObject *INVAR_err;
static PyObject *INMOC_err;
static PyObject *SMBUF_err;
static PyObject *BGWRD_err;
static PyObject *TOOBG_err;
static PyObject *NOTSP_err;
static PyObject *TRUNC_err;
static PyObject *LYAML_err;
static PyObject *YAML_err;
static PyObject *NOYAML_err;
static PyObject *NOTMP_err;
static PyObject *BYAML_err;
static PyObject *BASDF_err;
static PyObject *UASDF_err;

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

/* Now create an instance of each derived AST exception class. */
   if( !(ATGER_err = PyErr_NewException("Ast.ATGER", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "ATGER", ATGER_err );
 
   if( !(ATSER_err = PyErr_NewException("Ast.ATSER", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "ATSER", ATSER_err );
 
   if( !(ATTIN_err = PyErr_NewException("Ast.ATTIN", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "ATTIN", ATTIN_err );
 
   if( !(AXIIN_err = PyErr_NewException("Ast.AXIIN", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "AXIIN", AXIIN_err );
 
   if( !(BADAT_err = PyErr_NewException("Ast.BADAT", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "BADAT", BADAT_err );
 
   if( !(BADBX_err = PyErr_NewException("Ast.BADBX", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "BADBX", BADBX_err );
 
   if( !(BADIN_err = PyErr_NewException("Ast.BADIN", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "BADIN", BADIN_err );
 
   if( !(BADNI_err = PyErr_NewException("Ast.BADNI", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "BADNI", BADNI_err );
 
   if( !(BADNO_err = PyErr_NewException("Ast.BADNO", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "BADNO", BADNO_err );
 
   if( !(BADPW_err = PyErr_NewException("Ast.BADPW", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "BADPW", BADPW_err );
 
   if( !(BADSM_err = PyErr_NewException("Ast.BADSM", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "BADSM", BADSM_err );
 
   if( !(BADWM_err = PyErr_NewException("Ast.BADWM", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "BADWM", BADWM_err );
 
   if( !(BDBRK_err = PyErr_NewException("Ast.BDBRK", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "BDBRK", BDBRK_err );
 
   if( !(BDFMT_err = PyErr_NewException("Ast.BDFMT", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "BDFMT", BDFMT_err );
 
   if( !(BDFTS_err = PyErr_NewException("Ast.BDFTS", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "BDFTS", BDFTS_err );
 
   if( !(BDOBJ_err = PyErr_NewException("Ast.BDOBJ", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "BDOBJ", BDOBJ_err );
 
   if( !(CLPAX_err = PyErr_NewException("Ast.CLPAX", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "CLPAX", CLPAX_err );
 
   if( !(CORNG_err = PyErr_NewException("Ast.CORNG", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "CORNG", CORNG_err );
 
   if( !(CVBRK_err = PyErr_NewException("Ast.CVBRK", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "CVBRK", CVBRK_err );
 
   if( !(DIMIN_err = PyErr_NewException("Ast.DIMIN", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "DIMIN", DIMIN_err );
 
   if( !(DTERR_err = PyErr_NewException("Ast.DTERR", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "DTERR", DTERR_err );
 
   if( !(ENDIN_err = PyErr_NewException("Ast.ENDIN", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "ENDIN", ENDIN_err );
 
   if( !(EOCHN_err = PyErr_NewException("Ast.EOCHN", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "EOCHN", EOCHN_err );
 
   if( !(EXPIN_err = PyErr_NewException("Ast.EXPIN", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "EXPIN", EXPIN_err );
 
   if( !(FCRPT_err = PyErr_NewException("Ast.FCRPT", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "FCRPT", FCRPT_err );
 
   if( !(FMTER_err = PyErr_NewException("Ast.FMTER", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "FMTER", FMTER_err );
 
   if( !(FRMIN_err = PyErr_NewException("Ast.FRMIN", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "FRMIN", FRMIN_err );
 
   if( !(FRSIN_err = PyErr_NewException("Ast.FRSIN", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "FRSIN", FRSIN_err );
 
   if( !(FTCNV_err = PyErr_NewException("Ast.FTCNV", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "FTCNV", FTCNV_err );
 
   if( !(GRFER_err = PyErr_NewException("Ast.GRFER", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "GRFER", GRFER_err );
 
   if( !(INHAN_err = PyErr_NewException("Ast.INHAN", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "INHAN", INHAN_err );
 
   if( !(INNCO_err = PyErr_NewException("Ast.INNCO", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "INNCO", INNCO_err );
 
   if( !(INTER_err = PyErr_NewException("Ast.INTER", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "INTER", INTER_err );
 
   if( !(INTRD_err = PyErr_NewException("Ast.INTRD", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "INTRD", INTRD_err );
 
   if( !(KYCIR_err = PyErr_NewException("Ast.KYCIR", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "KYCIR", KYCIR_err );
 
   if( !(LDERR_err = PyErr_NewException("Ast.LDERR", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "LDERR", LDERR_err );
 
   if( !(LUTII_err = PyErr_NewException("Ast.LUTII", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "LUTII", LUTII_err );
 
   if( !(LUTIN_err = PyErr_NewException("Ast.LUTIN", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "LUTIN", LUTIN_err );
 
   if( !(MEMIN_err = PyErr_NewException("Ast.MEMIN", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "MEMIN", MEMIN_err );
 
   if( !(MTR23_err = PyErr_NewException("Ast.MTR23", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "MTR23", MTR23_err );
 
   if( !(MTRAX_err = PyErr_NewException("Ast.MTRAX", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "MTRAX", MTRAX_err );
 
   if( !(MTRML_err = PyErr_NewException("Ast.MTRML", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "MTRML", MTRML_err );
 
   if( !(MTRMT_err = PyErr_NewException("Ast.MTRMT", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "MTRMT", MTRMT_err );
 
   if( !(NAXIN_err = PyErr_NewException("Ast.NAXIN", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "NAXIN", NAXIN_err );
 
   if( !(NCHIN_err = PyErr_NewException("Ast.NCHIN", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "NCHIN", NCHIN_err );
 
   if( !(NCOIN_err = PyErr_NewException("Ast.NCOIN", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "NCOIN", NCOIN_err );
 
   if( !(NCPIN_err = PyErr_NewException("Ast.NCPIN", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "NCPIN", NCPIN_err );
 
   if( !(NELIN_err = PyErr_NewException("Ast.NELIN", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "NELIN", NELIN_err );
 
   if( !(NOCTS_err = PyErr_NewException("Ast.NOCTS", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "NOCTS", NOCTS_err );
 
   if( !(NODEF_err = PyErr_NewException("Ast.NODEF", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "NODEF", NODEF_err );
 
   if( !(NOFTS_err = PyErr_NewException("Ast.NOFTS", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "NOFTS", NOFTS_err );
 
   if( !(NOMEM_err = PyErr_NewException("Ast.NOMEM", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "NOMEM", NOMEM_err );
 
   if( !(NOPTS_err = PyErr_NewException("Ast.NOPTS", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "NOPTS", NOPTS_err );
 
   if( !(NOWRT_err = PyErr_NewException("Ast.NOWRT", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "NOWRT", NOWRT_err );
 
   if( !(NPTIN_err = PyErr_NewException("Ast.NPTIN", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "NPTIN", NPTIN_err );
 
   if( !(OBJIN_err = PyErr_NewException("Ast.OBJIN", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "OBJIN", OBJIN_err );
 
   if( !(OPT_err = PyErr_NewException("Ast.OPT", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "OPT", OPT_err );
 
   if( !(PDSIN_err = PyErr_NewException("Ast.PDSIN", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "PDSIN", PDSIN_err );
 
   if( !(PLFMT_err = PyErr_NewException("Ast.PLFMT", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "PLFMT", PLFMT_err );
 
   if( !(PRMIN_err = PyErr_NewException("Ast.PRMIN", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "PRMIN", PRMIN_err );
 
   if( !(PTRIN_err = PyErr_NewException("Ast.PTRIN", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "PTRIN", PTRIN_err );
 
   if( !(PTRNG_err = PyErr_NewException("Ast.PTRNG", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "PTRNG", PTRNG_err );
 
   if( !(RDERR_err = PyErr_NewException("Ast.RDERR", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "RDERR", RDERR_err );
 
   if( !(REGIN_err = PyErr_NewException("Ast.REGIN", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "REGIN", REGIN_err );
 
   if( !(REMIN_err = PyErr_NewException("Ast.REMIN", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "REMIN", REMIN_err );
 
   if( !(SCSIN_err = PyErr_NewException("Ast.SCSIN", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "SCSIN", SCSIN_err );
 
   if( !(SELIN_err = PyErr_NewException("Ast.SELIN", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "SELIN", SELIN_err );
 
   if( !(SLAIN_err = PyErr_NewException("Ast.SLAIN", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "SLAIN", SLAIN_err );
 
   if( !(TRNND_err = PyErr_NewException("Ast.TRNND", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "TRNND", TRNND_err );
 
   if( !(UNMQT_err = PyErr_NewException("Ast.UNMQT", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "UNMQT", UNMQT_err );
 
   if( !(VSMAL_err = PyErr_NewException("Ast.VSMAL", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "VSMAL", VSMAL_err );
 
   if( !(WCSAX_err = PyErr_NewException("Ast.WCSAX", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "WCSAX", WCSAX_err );
 
   if( !(WCSNC_err = PyErr_NewException("Ast.WCSNC", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "WCSNC", WCSNC_err );
 
   if( !(WCSPA_err = PyErr_NewException("Ast.WCSPA", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "WCSPA", WCSPA_err );
 
   if( !(WCSTY_err = PyErr_NewException("Ast.WCSTY", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "WCSTY", WCSTY_err );
 
   if( !(XSOBJ_err = PyErr_NewException("Ast.XSOBJ", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "XSOBJ", XSOBJ_err );
 
   if( !(ZOOMI_err = PyErr_NewException("Ast.ZOOMI", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "ZOOMI", ZOOMI_err );
 
   if( !(BADCI_err = PyErr_NewException("Ast.BADCI", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "BADCI", BADCI_err );
 
   if( !(ILOST_err = PyErr_NewException("Ast.ILOST", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "ILOST", ILOST_err );
 
   if( !(ITFER_err = PyErr_NewException("Ast.ITFER", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "ITFER", ITFER_err );
 
   if( !(ITFNI_err = PyErr_NewException("Ast.ITFNI", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "ITFNI", ITFNI_err );
 
   if( !(MBBNF_err = PyErr_NewException("Ast.MBBNF", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "MBBNF", MBBNF_err );
 
   if( !(MRITF_err = PyErr_NewException("Ast.MRITF", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "MRITF", MRITF_err );
 
   if( !(OCLUK_err = PyErr_NewException("Ast.OCLUK", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "OCLUK", OCLUK_err );
 
   if( !(UNFER_err = PyErr_NewException("Ast.UNFER", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "UNFER", UNFER_err );
 
   if( !(URITF_err = PyErr_NewException("Ast.URITF", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "URITF", URITF_err );
 
   if( !(GBDIN_err = PyErr_NewException("Ast.GBDIN", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "GBDIN", GBDIN_err );
 
   if( !(NGDIN_err = PyErr_NewException("Ast.NGDIN", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "NGDIN", NGDIN_err );
 
   if( !(PATIN_err = PyErr_NewException("Ast.PATIN", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "PATIN", PATIN_err );
 
   if( !(SISIN_err = PyErr_NewException("Ast.SISIN", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "SISIN", SISIN_err );
 
   if( !(SSPIN_err = PyErr_NewException("Ast.SSPIN", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "SSPIN", SSPIN_err );
 
   if( !(UINER_err = PyErr_NewException("Ast.UINER", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "UINER", UINER_err );
 
   if( !(UK1ER_err = PyErr_NewException("Ast.UK1ER", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "UK1ER", UK1ER_err );
 
   if( !(COMIN_err = PyErr_NewException("Ast.COMIN", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "COMIN", COMIN_err );
 
   if( !(CONIN_err = PyErr_NewException("Ast.CONIN", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "CONIN", CONIN_err );
 
   if( !(DUVAR_err = PyErr_NewException("Ast.DUVAR", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "DUVAR", DUVAR_err );
 
   if( !(INNTF_err = PyErr_NewException("Ast.INNTF", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "INNTF", INNTF_err );
 
   if( !(MIOPA_err = PyErr_NewException("Ast.MIOPA", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "MIOPA", MIOPA_err );
 
   if( !(MIOPR_err = PyErr_NewException("Ast.MIOPR", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "MIOPR", MIOPR_err );
 
   if( !(MISVN_err = PyErr_NewException("Ast.MISVN", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "MISVN", MISVN_err );
 
   if( !(MLPAR_err = PyErr_NewException("Ast.MLPAR", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "MLPAR", MLPAR_err );
 
   if( !(MRPAR_err = PyErr_NewException("Ast.MRPAR", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "MRPAR", MRPAR_err );
 
   if( !(NORHS_err = PyErr_NewException("Ast.NORHS", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "NORHS", NORHS_err );
 
   if( !(UDVOF_err = PyErr_NewException("Ast.UDVOF", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "UDVOF", UDVOF_err );
 
   if( !(VARIN_err = PyErr_NewException("Ast.VARIN", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "VARIN", VARIN_err );
 
   if( !(WRNFA_err = PyErr_NewException("Ast.WRNFA", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "WRNFA", WRNFA_err );
 
   if( !(BADUN_err = PyErr_NewException("Ast.BADUN", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "BADUN", BADUN_err );
 
   if( !(NORSF_err = PyErr_NewException("Ast.NORSF", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "NORSF", NORSF_err );
 
   if( !(NOSOR_err = PyErr_NewException("Ast.NOSOR", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "NOSOR", NOSOR_err );
 
   if( !(SPCIN_err = PyErr_NewException("Ast.SPCIN", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "SPCIN", SPCIN_err );
 
   if( !(XMLNM_err = PyErr_NewException("Ast.XMLNM", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "XMLNM", XMLNM_err );
 
   if( !(XMLCM_err = PyErr_NewException("Ast.XMLCM", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "XMLCM", XMLCM_err );
 
   if( !(XMLPT_err = PyErr_NewException("Ast.XMLPT", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "XMLPT", XMLPT_err );
 
   if( !(XMLIT_err = PyErr_NewException("Ast.XMLIT", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "XMLIT", XMLIT_err );
 
   if( !(XMLWF_err = PyErr_NewException("Ast.XMLWF", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "XMLWF", XMLWF_err );
 
   if( !(ZERAX_err = PyErr_NewException("Ast.ZERAX", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "ZERAX", ZERAX_err );
 
   if( !(BADOC_err = PyErr_NewException("Ast.BADOC", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "BADOC", BADOC_err );
 
   if( !(MPGER_err = PyErr_NewException("Ast.MPGER", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "MPGER", MPGER_err );
 
   if( !(MPIND_err = PyErr_NewException("Ast.MPIND", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "MPIND", MPIND_err );
 
   if( !(REGCN_err = PyErr_NewException("Ast.REGCN", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "REGCN", REGCN_err );
 
   if( !(NOVAL_err = PyErr_NewException("Ast.NOVAL", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "NOVAL", NOVAL_err );
 
   if( !(INCTS_err = PyErr_NewException("Ast.INCTS", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "INCTS", INCTS_err );
 
   if( !(TIMIN_err = PyErr_NewException("Ast.TIMIN", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "TIMIN", TIMIN_err );
 
   if( !(STCKEY_err = PyErr_NewException("Ast.STCKEY", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "STCKEY", STCKEY_err );
 
   if( !(STCIND_err = PyErr_NewException("Ast.STCIND", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "STCIND", STCIND_err );
 
   if( !(CNFLX_err = PyErr_NewException("Ast.CNFLX", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "CNFLX", CNFLX_err );
 
   if( !(TUNAM_err = PyErr_NewException("Ast.TUNAM", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "TUNAM", TUNAM_err );
 
   if( !(BDPAR_err = PyErr_NewException("Ast.BDPAR", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "BDPAR", BDPAR_err );
 
   if( !(PXFRRM_err = PyErr_NewException("Ast.PXFRRM", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "PXFRRM", PXFRRM_err );
 
   if( !(BADSUB_err = PyErr_NewException("Ast.BADSUB", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "BADSUB", BADSUB_err );
 
   if( !(BADFLG_err = PyErr_NewException("Ast.BADFLG", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "BADFLG", BADFLG_err );
 
   if( !(LCKERR_err = PyErr_NewException("Ast.LCKERR", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "LCKERR", LCKERR_err );
 
   if( !(FUNDEF_err = PyErr_NewException("Ast.FUNDEF", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "FUNDEF", FUNDEF_err );
 
   if( !(MPVIN_err = PyErr_NewException("Ast.MPVIN", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "MPVIN", MPVIN_err );
 
   if( !(OPRIN_err = PyErr_NewException("Ast.OPRIN", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "OPRIN", OPRIN_err );
 
   if( !(NONIN_err = PyErr_NewException("Ast.NONIN", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "NONIN", NONIN_err );
 
   if( !(MPKER_err = PyErr_NewException("Ast.MPKER", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "MPKER", MPKER_err );
 
   if( !(MPPER_err = PyErr_NewException("Ast.MPPER", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "MPPER", MPPER_err );
 
   if( !(BADKEY_err = PyErr_NewException("Ast.BADKEY", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "BADKEY", BADKEY_err );
 
   if( !(BADTYP_err = PyErr_NewException("Ast.BADTYP", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "BADTYP", BADTYP_err );
 
   if( !(OLDCOL_err = PyErr_NewException("Ast.OLDCOL", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "OLDCOL", OLDCOL_err );
 
   if( !(BADNULL_err = PyErr_NewException("Ast.BADNULL", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "BADNULL", BADNULL_err );
 
   if( !(BIGKEY_err = PyErr_NewException("Ast.BIGKEY", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "BIGKEY", BIGKEY_err );
 
   if( !(BADCOL_err = PyErr_NewException("Ast.BADCOL", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "BADCOL", BADCOL_err );
 
   if( !(BIGTAB_err = PyErr_NewException("Ast.BIGTAB", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "BIGTAB", BIGTAB_err );
 
   if( !(BADSIZ_err = PyErr_NewException("Ast.BADSIZ", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "BADSIZ", BADSIZ_err );
 
   if( !(BADTAB_err = PyErr_NewException("Ast.BADTAB", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "BADTAB", BADTAB_err );
 
   if( !(NOTAB_err = PyErr_NewException("Ast.NOTAB", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "NOTAB", NOTAB_err );
 
   if( !(LEVMAR_err = PyErr_NewException("Ast.LEVMAR", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "LEVMAR", LEVMAR_err );
 
   if( !(NOFIT_err = PyErr_NewException("Ast.NOFIT", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "NOFIT", NOFIT_err );
 
   if( !(ISNAN_err = PyErr_NewException("Ast.ISNAN", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "ISNAN", ISNAN_err );
 
   if( !(WRERR_err = PyErr_NewException("Ast.WRERR", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "WRERR", WRERR_err );
 
   if( !(BDVNM_err = PyErr_NewException("Ast.BDVNM", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "BDVNM", BDVNM_err );
 
   if( !(MIRRO_err = PyErr_NewException("Ast.MIRRO", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "MIRRO", MIRRO_err );
 
   if( !(MNPCK_err = PyErr_NewException("Ast.MNPCK", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "MNPCK", MNPCK_err );
 
   if( !(EXSPIX_err = PyErr_NewException("Ast.EXSPIX", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "EXSPIX", EXSPIX_err );
 
   if( !(NOCNV_err = PyErr_NewException("Ast.NOCNV", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "NOCNV", NOCNV_err );
 
   if( !(IMMUT_err = PyErr_NewException("Ast.IMMUT", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "IMMUT", IMMUT_err );
 
   if( !(NOBOX_err = PyErr_NewException("Ast.NOBOX", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "NOBOX", NOBOX_err );
 
   if( !(NOSKY_err = PyErr_NewException("Ast.NOSKY", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "NOSKY", NOSKY_err );
 
   if( !(BDSKY_err = PyErr_NewException("Ast.BDSKY", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "BDSKY", BDSKY_err );
 
   if( !(BDCLS_err = PyErr_NewException("Ast.BDCLS", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "BDCLS", BDCLS_err );
 
   if( !(NOIMP_err = PyErr_NewException("Ast.NOIMP", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "NOIMP", NOIMP_err );
 
   if( !(BGMOC_err = PyErr_NewException("Ast.BGMOC", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "BGMOC", BGMOC_err );
 
   if( !(INVAR_err = PyErr_NewException("Ast.INVAR", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "INVAR", INVAR_err );
 
   if( !(INMOC_err = PyErr_NewException("Ast.INMOC", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "INMOC", INMOC_err );
 
   if( !(SMBUF_err = PyErr_NewException("Ast.SMBUF", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "SMBUF", SMBUF_err );
 
   if( !(BGWRD_err = PyErr_NewException("Ast.BGWRD", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "BGWRD", BGWRD_err );
 
   if( !(TOOBG_err = PyErr_NewException("Ast.TOOBG", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "TOOBG", TOOBG_err );
 
   if( !(NOTSP_err = PyErr_NewException("Ast.NOTSP", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "NOTSP", NOTSP_err );
 
   if( !(TRUNC_err = PyErr_NewException("Ast.TRUNC", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "TRUNC", TRUNC_err );
 
   if( !(LYAML_err = PyErr_NewException("Ast.LYAML", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "LYAML", LYAML_err );
 
   if( !(YAML_err = PyErr_NewException("Ast.YAML", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "YAML", YAML_err );
 
   if( !(NOYAML_err = PyErr_NewException("Ast.NOYAML", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "NOYAML", NOYAML_err );
 
   if( !(NOTMP_err = PyErr_NewException("Ast.NOTMP", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "NOTMP", NOTMP_err );
 
   if( !(BYAML_err = PyErr_NewException("Ast.BYAML", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "BYAML", BYAML_err );
 
   if( !(BASDF_err = PyErr_NewException("Ast.BASDF", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "BASDF", BASDF_err );
 
   if( !(UASDF_err = PyErr_NewException("Ast.UASDF", AstError_err, NULL))) return 0;
   PyDict_SetItemString( dict, "UASDF", UASDF_err );
 
   return 1;
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

/* If no exception has already occurred, raise an appropriate AST exception now. */
   if( status_value == AST__ATGER ) {
      PyErr_SetString( ATGER_err, message );
   } else if( status_value == AST__ATSER ) {
      PyErr_SetString( ATSER_err, message );
   } else if( status_value == AST__ATTIN ) {
      PyErr_SetString( ATTIN_err, message );
   } else if( status_value == AST__AXIIN ) {
      PyErr_SetString( AXIIN_err, message );
   } else if( status_value == AST__BADAT ) {
      PyErr_SetString( BADAT_err, message );
   } else if( status_value == AST__BADBX ) {
      PyErr_SetString( BADBX_err, message );
   } else if( status_value == AST__BADIN ) {
      PyErr_SetString( BADIN_err, message );
   } else if( status_value == AST__BADNI ) {
      PyErr_SetString( BADNI_err, message );
   } else if( status_value == AST__BADNO ) {
      PyErr_SetString( BADNO_err, message );
   } else if( status_value == AST__BADPW ) {
      PyErr_SetString( BADPW_err, message );
   } else if( status_value == AST__BADSM ) {
      PyErr_SetString( BADSM_err, message );
   } else if( status_value == AST__BADWM ) {
      PyErr_SetString( BADWM_err, message );
   } else if( status_value == AST__BDBRK ) {
      PyErr_SetString( BDBRK_err, message );
   } else if( status_value == AST__BDFMT ) {
      PyErr_SetString( BDFMT_err, message );
   } else if( status_value == AST__BDFTS ) {
      PyErr_SetString( BDFTS_err, message );
   } else if( status_value == AST__BDOBJ ) {
      PyErr_SetString( BDOBJ_err, message );
   } else if( status_value == AST__CLPAX ) {
      PyErr_SetString( CLPAX_err, message );
   } else if( status_value == AST__CORNG ) {
      PyErr_SetString( CORNG_err, message );
   } else if( status_value == AST__CVBRK ) {
      PyErr_SetString( CVBRK_err, message );
   } else if( status_value == AST__DIMIN ) {
      PyErr_SetString( DIMIN_err, message );
   } else if( status_value == AST__DTERR ) {
      PyErr_SetString( DTERR_err, message );
   } else if( status_value == AST__ENDIN ) {
      PyErr_SetString( ENDIN_err, message );
   } else if( status_value == AST__EOCHN ) {
      PyErr_SetString( EOCHN_err, message );
   } else if( status_value == AST__EXPIN ) {
      PyErr_SetString( EXPIN_err, message );
   } else if( status_value == AST__FCRPT ) {
      PyErr_SetString( FCRPT_err, message );
   } else if( status_value == AST__FMTER ) {
      PyErr_SetString( FMTER_err, message );
   } else if( status_value == AST__FRMIN ) {
      PyErr_SetString( FRMIN_err, message );
   } else if( status_value == AST__FRSIN ) {
      PyErr_SetString( FRSIN_err, message );
   } else if( status_value == AST__FTCNV ) {
      PyErr_SetString( FTCNV_err, message );
   } else if( status_value == AST__GRFER ) {
      PyErr_SetString( GRFER_err, message );
   } else if( status_value == AST__INHAN ) {
      PyErr_SetString( INHAN_err, message );
   } else if( status_value == AST__INNCO ) {
      PyErr_SetString( INNCO_err, message );
   } else if( status_value == AST__INTER ) {
      PyErr_SetString( INTER_err, message );
   } else if( status_value == AST__INTRD ) {
      PyErr_SetString( INTRD_err, message );
   } else if( status_value == AST__KYCIR ) {
      PyErr_SetString( KYCIR_err, message );
   } else if( status_value == AST__LDERR ) {
      PyErr_SetString( LDERR_err, message );
   } else if( status_value == AST__LUTII ) {
      PyErr_SetString( LUTII_err, message );
   } else if( status_value == AST__LUTIN ) {
      PyErr_SetString( LUTIN_err, message );
   } else if( status_value == AST__MEMIN ) {
      PyErr_SetString( MEMIN_err, message );
   } else if( status_value == AST__MTR23 ) {
      PyErr_SetString( MTR23_err, message );
   } else if( status_value == AST__MTRAX ) {
      PyErr_SetString( MTRAX_err, message );
   } else if( status_value == AST__MTRML ) {
      PyErr_SetString( MTRML_err, message );
   } else if( status_value == AST__MTRMT ) {
      PyErr_SetString( MTRMT_err, message );
   } else if( status_value == AST__NAXIN ) {
      PyErr_SetString( NAXIN_err, message );
   } else if( status_value == AST__NCHIN ) {
      PyErr_SetString( NCHIN_err, message );
   } else if( status_value == AST__NCOIN ) {
      PyErr_SetString( NCOIN_err, message );
   } else if( status_value == AST__NCPIN ) {
      PyErr_SetString( NCPIN_err, message );
   } else if( status_value == AST__NELIN ) {
      PyErr_SetString( NELIN_err, message );
   } else if( status_value == AST__NOCTS ) {
      PyErr_SetString( NOCTS_err, message );
   } else if( status_value == AST__NODEF ) {
      PyErr_SetString( NODEF_err, message );
   } else if( status_value == AST__NOFTS ) {
      PyErr_SetString( NOFTS_err, message );
   } else if( status_value == AST__NOMEM ) {
      PyErr_SetString( NOMEM_err, message );
   } else if( status_value == AST__NOPTS ) {
      PyErr_SetString( NOPTS_err, message );
   } else if( status_value == AST__NOWRT ) {
      PyErr_SetString( NOWRT_err, message );
   } else if( status_value == AST__NPTIN ) {
      PyErr_SetString( NPTIN_err, message );
   } else if( status_value == AST__OBJIN ) {
      PyErr_SetString( OBJIN_err, message );
   } else if( status_value == AST__OPT ) {
      PyErr_SetString( OPT_err, message );
   } else if( status_value == AST__PDSIN ) {
      PyErr_SetString( PDSIN_err, message );
   } else if( status_value == AST__PLFMT ) {
      PyErr_SetString( PLFMT_err, message );
   } else if( status_value == AST__PRMIN ) {
      PyErr_SetString( PRMIN_err, message );
   } else if( status_value == AST__PTRIN ) {
      PyErr_SetString( PTRIN_err, message );
   } else if( status_value == AST__PTRNG ) {
      PyErr_SetString( PTRNG_err, message );
   } else if( status_value == AST__RDERR ) {
      PyErr_SetString( RDERR_err, message );
   } else if( status_value == AST__REGIN ) {
      PyErr_SetString( REGIN_err, message );
   } else if( status_value == AST__REMIN ) {
      PyErr_SetString( REMIN_err, message );
   } else if( status_value == AST__SCSIN ) {
      PyErr_SetString( SCSIN_err, message );
   } else if( status_value == AST__SELIN ) {
      PyErr_SetString( SELIN_err, message );
   } else if( status_value == AST__SLAIN ) {
      PyErr_SetString( SLAIN_err, message );
   } else if( status_value == AST__TRNND ) {
      PyErr_SetString( TRNND_err, message );
   } else if( status_value == AST__UNMQT ) {
      PyErr_SetString( UNMQT_err, message );
   } else if( status_value == AST__VSMAL ) {
      PyErr_SetString( VSMAL_err, message );
   } else if( status_value == AST__WCSAX ) {
      PyErr_SetString( WCSAX_err, message );
   } else if( status_value == AST__WCSNC ) {
      PyErr_SetString( WCSNC_err, message );
   } else if( status_value == AST__WCSPA ) {
      PyErr_SetString( WCSPA_err, message );
   } else if( status_value == AST__WCSTY ) {
      PyErr_SetString( WCSTY_err, message );
   } else if( status_value == AST__XSOBJ ) {
      PyErr_SetString( XSOBJ_err, message );
   } else if( status_value == AST__ZOOMI ) {
      PyErr_SetString( ZOOMI_err, message );
   } else if( status_value == AST__BADCI ) {
      PyErr_SetString( BADCI_err, message );
   } else if( status_value == AST__ILOST ) {
      PyErr_SetString( ILOST_err, message );
   } else if( status_value == AST__ITFER ) {
      PyErr_SetString( ITFER_err, message );
   } else if( status_value == AST__ITFNI ) {
      PyErr_SetString( ITFNI_err, message );
   } else if( status_value == AST__MBBNF ) {
      PyErr_SetString( MBBNF_err, message );
   } else if( status_value == AST__MRITF ) {
      PyErr_SetString( MRITF_err, message );
   } else if( status_value == AST__OCLUK ) {
      PyErr_SetString( OCLUK_err, message );
   } else if( status_value == AST__UNFER ) {
      PyErr_SetString( UNFER_err, message );
   } else if( status_value == AST__URITF ) {
      PyErr_SetString( URITF_err, message );
   } else if( status_value == AST__GBDIN ) {
      PyErr_SetString( GBDIN_err, message );
   } else if( status_value == AST__NGDIN ) {
      PyErr_SetString( NGDIN_err, message );
   } else if( status_value == AST__PATIN ) {
      PyErr_SetString( PATIN_err, message );
   } else if( status_value == AST__SISIN ) {
      PyErr_SetString( SISIN_err, message );
   } else if( status_value == AST__SSPIN ) {
      PyErr_SetString( SSPIN_err, message );
   } else if( status_value == AST__UINER ) {
      PyErr_SetString( UINER_err, message );
   } else if( status_value == AST__UK1ER ) {
      PyErr_SetString( UK1ER_err, message );
   } else if( status_value == AST__COMIN ) {
      PyErr_SetString( COMIN_err, message );
   } else if( status_value == AST__CONIN ) {
      PyErr_SetString( CONIN_err, message );
   } else if( status_value == AST__DUVAR ) {
      PyErr_SetString( DUVAR_err, message );
   } else if( status_value == AST__INNTF ) {
      PyErr_SetString( INNTF_err, message );
   } else if( status_value == AST__MIOPA ) {
      PyErr_SetString( MIOPA_err, message );
   } else if( status_value == AST__MIOPR ) {
      PyErr_SetString( MIOPR_err, message );
   } else if( status_value == AST__MISVN ) {
      PyErr_SetString( MISVN_err, message );
   } else if( status_value == AST__MLPAR ) {
      PyErr_SetString( MLPAR_err, message );
   } else if( status_value == AST__MRPAR ) {
      PyErr_SetString( MRPAR_err, message );
   } else if( status_value == AST__NORHS ) {
      PyErr_SetString( NORHS_err, message );
   } else if( status_value == AST__UDVOF ) {
      PyErr_SetString( UDVOF_err, message );
   } else if( status_value == AST__VARIN ) {
      PyErr_SetString( VARIN_err, message );
   } else if( status_value == AST__WRNFA ) {
      PyErr_SetString( WRNFA_err, message );
   } else if( status_value == AST__BADUN ) {
      PyErr_SetString( BADUN_err, message );
   } else if( status_value == AST__NORSF ) {
      PyErr_SetString( NORSF_err, message );
   } else if( status_value == AST__NOSOR ) {
      PyErr_SetString( NOSOR_err, message );
   } else if( status_value == AST__SPCIN ) {
      PyErr_SetString( SPCIN_err, message );
   } else if( status_value == AST__XMLNM ) {
      PyErr_SetString( XMLNM_err, message );
   } else if( status_value == AST__XMLCM ) {
      PyErr_SetString( XMLCM_err, message );
   } else if( status_value == AST__XMLPT ) {
      PyErr_SetString( XMLPT_err, message );
   } else if( status_value == AST__XMLIT ) {
      PyErr_SetString( XMLIT_err, message );
   } else if( status_value == AST__XMLWF ) {
      PyErr_SetString( XMLWF_err, message );
   } else if( status_value == AST__ZERAX ) {
      PyErr_SetString( ZERAX_err, message );
   } else if( status_value == AST__BADOC ) {
      PyErr_SetString( BADOC_err, message );
   } else if( status_value == AST__MPGER ) {
      PyErr_SetString( MPGER_err, message );
   } else if( status_value == AST__MPIND ) {
      PyErr_SetString( MPIND_err, message );
   } else if( status_value == AST__REGCN ) {
      PyErr_SetString( REGCN_err, message );
   } else if( status_value == AST__NOVAL ) {
      PyErr_SetString( NOVAL_err, message );
   } else if( status_value == AST__INCTS ) {
      PyErr_SetString( INCTS_err, message );
   } else if( status_value == AST__TIMIN ) {
      PyErr_SetString( TIMIN_err, message );
   } else if( status_value == AST__STCKEY ) {
      PyErr_SetString( STCKEY_err, message );
   } else if( status_value == AST__STCIND ) {
      PyErr_SetString( STCIND_err, message );
   } else if( status_value == AST__CNFLX ) {
      PyErr_SetString( CNFLX_err, message );
   } else if( status_value == AST__TUNAM ) {
      PyErr_SetString( TUNAM_err, message );
   } else if( status_value == AST__BDPAR ) {
      PyErr_SetString( BDPAR_err, message );
   } else if( status_value == AST__PXFRRM ) {
      PyErr_SetString( PXFRRM_err, message );
   } else if( status_value == AST__BADSUB ) {
      PyErr_SetString( BADSUB_err, message );
   } else if( status_value == AST__BADFLG ) {
      PyErr_SetString( BADFLG_err, message );
   } else if( status_value == AST__LCKERR ) {
      PyErr_SetString( LCKERR_err, message );
   } else if( status_value == AST__FUNDEF ) {
      PyErr_SetString( FUNDEF_err, message );
   } else if( status_value == AST__MPVIN ) {
      PyErr_SetString( MPVIN_err, message );
   } else if( status_value == AST__OPRIN ) {
      PyErr_SetString( OPRIN_err, message );
   } else if( status_value == AST__NONIN ) {
      PyErr_SetString( NONIN_err, message );
   } else if( status_value == AST__MPKER ) {
      PyErr_SetString( MPKER_err, message );
   } else if( status_value == AST__MPPER ) {
      PyErr_SetString( MPPER_err, message );
   } else if( status_value == AST__BADKEY ) {
      PyErr_SetString( BADKEY_err, message );
   } else if( status_value == AST__BADTYP ) {
      PyErr_SetString( BADTYP_err, message );
   } else if( status_value == AST__OLDCOL ) {
      PyErr_SetString( OLDCOL_err, message );
   } else if( status_value == AST__BADNULL ) {
      PyErr_SetString( BADNULL_err, message );
   } else if( status_value == AST__BIGKEY ) {
      PyErr_SetString( BIGKEY_err, message );
   } else if( status_value == AST__BADCOL ) {
      PyErr_SetString( BADCOL_err, message );
   } else if( status_value == AST__BIGTAB ) {
      PyErr_SetString( BIGTAB_err, message );
   } else if( status_value == AST__BADSIZ ) {
      PyErr_SetString( BADSIZ_err, message );
   } else if( status_value == AST__BADTAB ) {
      PyErr_SetString( BADTAB_err, message );
   } else if( status_value == AST__NOTAB ) {
      PyErr_SetString( NOTAB_err, message );
   } else if( status_value == AST__LEVMAR ) {
      PyErr_SetString( LEVMAR_err, message );
   } else if( status_value == AST__NOFIT ) {
      PyErr_SetString( NOFIT_err, message );
   } else if( status_value == AST__ISNAN ) {
      PyErr_SetString( ISNAN_err, message );
   } else if( status_value == AST__WRERR ) {
      PyErr_SetString( WRERR_err, message );
   } else if( status_value == AST__BDVNM ) {
      PyErr_SetString( BDVNM_err, message );
   } else if( status_value == AST__MIRRO ) {
      PyErr_SetString( MIRRO_err, message );
   } else if( status_value == AST__MNPCK ) {
      PyErr_SetString( MNPCK_err, message );
   } else if( status_value == AST__EXSPIX ) {
      PyErr_SetString( EXSPIX_err, message );
   } else if( status_value == AST__NOCNV ) {
      PyErr_SetString( NOCNV_err, message );
   } else if( status_value == AST__IMMUT ) {
      PyErr_SetString( IMMUT_err, message );
   } else if( status_value == AST__NOBOX ) {
      PyErr_SetString( NOBOX_err, message );
   } else if( status_value == AST__NOSKY ) {
      PyErr_SetString( NOSKY_err, message );
   } else if( status_value == AST__BDSKY ) {
      PyErr_SetString( BDSKY_err, message );
   } else if( status_value == AST__BDCLS ) {
      PyErr_SetString( BDCLS_err, message );
   } else if( status_value == AST__NOIMP ) {
      PyErr_SetString( NOIMP_err, message );
   } else if( status_value == AST__BGMOC ) {
      PyErr_SetString( BGMOC_err, message );
   } else if( status_value == AST__INVAR ) {
      PyErr_SetString( INVAR_err, message );
   } else if( status_value == AST__INMOC ) {
      PyErr_SetString( INMOC_err, message );
   } else if( status_value == AST__SMBUF ) {
      PyErr_SetString( SMBUF_err, message );
   } else if( status_value == AST__BGWRD ) {
      PyErr_SetString( BGWRD_err, message );
   } else if( status_value == AST__TOOBG ) {
      PyErr_SetString( TOOBG_err, message );
   } else if( status_value == AST__NOTSP ) {
      PyErr_SetString( NOTSP_err, message );
   } else if( status_value == AST__TRUNC ) {
      PyErr_SetString( TRUNC_err, message );
   } else if( status_value == AST__LYAML ) {
      PyErr_SetString( LYAML_err, message );
   } else if( status_value == AST__YAML ) {
      PyErr_SetString( YAML_err, message );
   } else if( status_value == AST__NOYAML ) {
      PyErr_SetString( NOYAML_err, message );
   } else if( status_value == AST__NOTMP ) {
      PyErr_SetString( NOTMP_err, message );
   } else if( status_value == AST__BYAML ) {
      PyErr_SetString( BYAML_err, message );
   } else if( status_value == AST__BASDF ) {
      PyErr_SetString( BASDF_err, message );
   } else if( status_value == AST__UASDF ) {
      PyErr_SetString( UASDF_err, message );
   } else {
    PyErr_SetString( AstError_err, message );
}

/* restore the original AST status value. */
astSetStatus( lstat );
}


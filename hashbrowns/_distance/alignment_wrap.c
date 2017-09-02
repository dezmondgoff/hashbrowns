#if !defined(__clang__) && defined(__GNUC__) && defined(__GNUC_MINOR__)
#if __GNUC__ >= 5 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 4)
/* enable auto-vectorizer */
#pragma GCC optimize("tree-vectorize")
/* float associativity required to vectorize reductions */
#pragma GCC optimize("unsafe-math-optimizations")
/* maybe 5% gain, manual unrolling with more accumulators would be better */
#pragma GCC optimize("unroll-loops")
#endif
#endif

#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <float.h>
#include <unistr.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>

#include "alignment_impl.h"

#define MATRIX_TYPE(matrix_type) JOIN2(MATRIX_TYPE_, matrix_type)
#define MATRIX_TYPE_int8_t char
#define MATRIX_TYPE_long long
#define MATRIX_TYPE_double double

#define GAP(matrix_type) MATRIX_TYPE(matrix_type) gap_open, gap_ext

#define GAP_ARG_STRING(matrix_type) JOIN2(GAP_ARG_STRING_, matrix_type)
#define GAP_ARG_STRING_int8_t "bb"
#define GAP_ARG_STRING_long "ll"
#define GAP_ARG_STRING_double "dd"

#define STRING_LENGTHS1(array_type) JOIN2(STRING_LENGTHS1_, array_type)
#define STRING_LENGTHS1_str n = X_->dimensions[1] / 4
#define STRING_LENGTHS1_PyObject 											  \
do {																		  \
	npy_intp i, j; 															  \
	n = 0;																	  \
	for (i = 0; i < m; i++) {                                         	  	  \
		j = PyUnicode_GET_LENGTH(*(X + i));                          	  	  \
		if (j > n)                                                    	  	  \
			n = j;                                                    	  	  \
	}																		  \
} while(0)

#define STRING_LENGTHS2(array_type) JOIN2(STRING_LENGTHS2_, array_type)
#define STRING_LENGTHS2_str 												  \
nA = XA_->dimensions[1] / 4;														  \
nB = XB_->dimensions[1] / 4
#define STRING_LENGTHS2_PyObject 											  \
do {																		  \
	npy_intp i, j; 															  \
	nA = 0;																	  \
	for (i = 0; i < mA; i++) {                                         	  	  \
		j = PyUnicode_GET_LENGTH(*(XA + i));                          	  	  \
		if (j > nA)                                                    	  	  \
			nA = j;                                                    	  	  \
	}																		  \
} while(0)

#define ERROR_DICT(boolean) JOIN2(ERROR_DICT_, boolean)
#define ERROR_DICT_1														  \
else if (status < -255)														  \
	return PyErr_Format(PyExc_KeyError, "key pair (%c, %c) not found",		  \
						(char) ((-status) & 0xFF), (char)((-status) >> 8))
#define ERROR_DICT_0 else if (false)

#define ERROR_ARR(boolean) JOIN2(ERROR_ARR_, boolean)
#define ERROR_ARR_1															  \
else if (status < 0)														  \
	return PyErr_Format(PyExc_KeyError, "character '%c' not in matrix",		  \
						(char) (-status))
#define ERROR_ARR_0 else if (false)

#define BEGIN_THREADS(boolean) JOIN2(BEGIN_THREADS_, boolean)
#define BEGIN_THREADS_1 _BLANK
#define BEGIN_THREADS_0  													  \
NPY_BEGIN_THREADS_DEF;												  	  	  \
NPY_BEGIN_THREADS

#define END_THREADS(boolean) JOIN2(END_THREADS_, boolean)
#define END_THREADS_1 _BLANK
#define END_THREADS_0 NPY_END_THREADS

/* function_name: name of function
 * array_type: PyObject or char
 * score_type: long or double
 * matrix_type: int8_t, long, or double
 * has_dict: boolean
 * has_arr: boolean */

#define DEFINE_ALIGN_PDIST_WRAP(function_name, array_type, score_type, 		  \
								 matrix_type, has_dict, has_arr)	  		  						  \
static PyObject * JOIN5(pdist_, function_name, \
						PRE_HYPHEN(LOWER(array_type)),	  \
						ADD_TOKEN(has_dict, PRE_HYPHEN(score_type)), 		  \
						_wrap) 												  \
						(PyObject *self, PyObject *args)      		  		  \
{                                                                     	  	  \
	PyArrayObject *X_, *dm_;												  \
	GAP(matrix_type);														  \
	ADD_TOKEN(has_dict, PyObject *mat;)										  \
	int norm;																  \
	double tol;														  \
																			  \
	if (!PyArg_ParseTuple(args, "O!O!" GAP_ARG_STRING(matrix_type)			  \
						  ADD_TOKEN(has_dict, "O!")"p|d",  				  \
                          &PyArray_Type, &X_,                             	  \
                          &PyArray_Type, &dm_,								  \
                          &gap_open,						  	  			  \
						  &gap_ext,			  			  					  \
						  ADD_ARG(has_dict, &PyDict_Type)					  \
						  ADD_ARG(has_dict, &mat)							  \
						  &norm, &tol))							  	  \
		return NULL;                                                      	  \
																			  \
	npy_intp m, n;                                                  	  	  \
	int status;																  \
    double *dm;            												  	  \
    const ARRAY_TYPE(array_type)X;                         	  	  	  	  	  \
	const JOIN2(scoreptr, ADD_TOKEN(has_dict, _dict)) func = 	  			  \
		JOIN4(ARRAY_TYPE_ALIAS(array_type), 					  			  \
			  ADD_TOKEN(has_dict, PRE_HYPHEN(matrix_type)), 				  \
			  PRE_HYPHEN(function_name), _score);    	 					  \
                  															  \
	BEGIN_THREADS(has_dict);												  \
	X = (const ARRAY_TYPE(array_type))X_->data;								  \
	dm = (double *)dm_->data;                                         	  	  \
	m = X_->dimensions[0];													  \
	STRING_LENGTHS1(array_type);										  	  \
    																		  \
	if (norm)																  \
		status = JOIN4(LOWER(array_type), PRE_HYPHEN(score_type), 			  \
					   _align_pdist_norm, ADD_TOKEN(has_dict, _dict)) (		  \
								X, func, ADD_ARG(has_dict, mat) 			  \
							   (matrix_type) gap_open, 		  				  \
							   (matrix_type) gap_ext, dm, 					  \
							   m, n, tol); 			  				  		  \
	else																  	  \
		status = JOIN4(LOWER(array_type), PRE_HYPHEN(score_type), 			  \
					   _align_pdist, ADD_TOKEN(has_dict, _dict)) (		  	  \
							   X, func, ADD_ARG(has_dict, mat) 				  \
							   (matrix_type) gap_open, 		  				  \
							   (matrix_type) gap_ext, dm, m, n); 			  \
	END_THREADS(has_dict);												  	  \
																			  \
	if (status == NO_MEMORY)											  	  \
		return PyErr_NoMemory();										  	  \
	ERROR_DICT(has_dict);													  \
	ERROR_ARR(has_arr);                             						  \
    return Py_BuildValue("d", 0.);                                     	  	  \
}

DEFINE_ALIGN_PDIST_WRAP(levenshtein, PyObject, long, int8_t, false, false)
DEFINE_ALIGN_PDIST_WRAP(blosum62, PyObject, long, int8_t, false, true)
DEFINE_ALIGN_PDIST_WRAP(pam250, PyObject, long, int8_t, false, true)
DEFINE_ALIGN_PDIST_WRAP(needleman_wunsch, PyObject, double, double, true, false)
DEFINE_ALIGN_PDIST_WRAP(needleman_wunsch, PyObject, long, long, true, false)
DEFINE_ALIGN_PDIST_WRAP(levenshtein, str, long, int8_t, false, false)
DEFINE_ALIGN_PDIST_WRAP(blosum62, str, long, int8_t, false, true)
DEFINE_ALIGN_PDIST_WRAP(pam250, str, long, int8_t, false, true)
DEFINE_ALIGN_PDIST_WRAP(needleman_wunsch, str, long, long, true, false)
DEFINE_ALIGN_PDIST_WRAP(needleman_wunsch, str, double, double, true, false)

#define DEFINE_ALIGN_CDIST_WRAP(function_name, array_type, score_type, 		  \
								 matrix_type, has_dict, has_arr)	  		  \
static PyObject * JOIN5(cdist_, function_name, 								  \
						PRE_HYPHEN(LOWER(array_type)),	  					  \
						ADD_TOKEN(has_dict, PRE_HYPHEN(score_type)), 		  \
						_wrap) 												  \
						(PyObject *self, PyObject *args)      		  		  \
{                                                                   	  	  \
    PyArrayObject *XA_, *XB_, *dm_;                                      	  \
    GAP(matrix_type);														  \
    ADD_TOKEN(has_dict, PyObject *mat;)	                                      \
	int norm;																  \
	double tol;														  		  \
																			  \
	if (!PyArg_ParseTuple(args,							  				  	  \
					"O!O!O!" GAP_ARG_STRING(matrix_type) 			  		  \
					ADD_TOKEN(has_dict, "O!")"p|d",	  					  	  \
                    &PyArray_Type, &XA_,                             	  	  \
					&PyArray_Type, &XB_,								  	  \
                    &PyArray_Type, &dm_,				  	  				  \
                    &gap_open,					  							  \
					&gap_ext, 		  					  					  \
					ADD_ARG(has_dict, &PyDict_Type)						  	  \
					ADD_ARG(has_dict, &mat)									  \
					&norm, &tol))			  	  			  	  	  		  \
		return NULL;                                     	                  \
																			  \
	npy_intp mA, mB, nA ADD_COMMA(array_type) ADD_TOKEN(array_type, nB);   	  \
	int status;																  \
	double *dm;                                                           	  \
	const ARRAY_TYPE(array_type)XA;											  \
	const ARRAY_TYPE(array_type)XB;                                        	  \
	const JOIN2(scoreptr, ADD_TOKEN(has_dict, _dict)) func = 				  \
		JOIN4(ARRAY_TYPE_ALIAS(array_type), 								  \
			  ADD_TOKEN(has_dict, PRE_HYPHEN(matrix_type)),				  	  \
			  PRE_HYPHEN(function_name), _score);							  \
			  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  \
	BEGIN_THREADS(has_dict);											  	  \
	XA = (const ARRAY_TYPE(array_type))XA_->data;                     	  	  \
	XB = (const ARRAY_TYPE(array_type))XB_->data;                      	  	  \
	dm = (double *)dm_->data;                                         	  	  \
	mA = XA_->dimensions[0];                                          	  	  \
	mB = XB_->dimensions[0];                                              	  \
	STRING_LENGTHS2(array_type);										  	  \
																			  \
	if (norm)																  \
		status = JOIN4(LOWER(array_type), PRE_HYPHEN(score_type),			  \
					   _align_cdist_norm, ADD_TOKEN(has_dict, _dict)) (	  	  \
							   XA, XB, func, ADD_ARG(has_dict, mat) 		  \
							   (matrix_type) gap_open, (matrix_type) gap_ext, \
							   dm, mA, mB, nA ADD_COMMA(array_type) 		  \
							   ADD_TOKEN(array_type, nB), tol);   	  		  \
	else																	  \
		status = JOIN4(LOWER(array_type), PRE_HYPHEN(score_type),			  \
					   _align_cdist, ADD_TOKEN(has_dict, _dict)) (		  	  \
							   XA, XB, func, ADD_ARG(has_dict, mat) 		  \
							   (matrix_type) gap_open, (matrix_type) gap_ext, \
							   dm, mA, mB, nA ADD_COMMA(array_type) 		  \
							   ADD_TOKEN(array_type, nB));   	  			  \
	END_THREADS(has_dict);												  	  \
																			  \
	if (status == NO_MEMORY)											  	  \
		return PyErr_NoMemory();										  	  \
	ERROR_DICT(has_dict);												  	  \
	ERROR_ARR(has_arr);														  \
	return Py_BuildValue("d", 0.);											  \
}

DEFINE_ALIGN_CDIST_WRAP(levenshtein, PyObject, long, int8_t, false, false)
DEFINE_ALIGN_CDIST_WRAP(blosum62, PyObject, long, int8_t, false, true)
DEFINE_ALIGN_CDIST_WRAP(pam250, PyObject, long, int8_t, false, true)
DEFINE_ALIGN_CDIST_WRAP(needleman_wunsch, PyObject, double, double, true, false)
DEFINE_ALIGN_CDIST_WRAP(needleman_wunsch, PyObject, long, long, true, false)
DEFINE_ALIGN_CDIST_WRAP(levenshtein, str, long, int8_t, false, false)
DEFINE_ALIGN_CDIST_WRAP(blosum62, str, long, int8_t, false, true)
DEFINE_ALIGN_CDIST_WRAP(pam250, str, long, int8_t, false, true)
DEFINE_ALIGN_CDIST_WRAP(needleman_wunsch, str, long, long, true, false)
DEFINE_ALIGN_CDIST_WRAP(needleman_wunsch, str, double, double, true, false)

#define DEFINE_ALIGN_SSDIST_WRAP(function_name, array_type, score_type, 	  \
								 matrix_type, has_dict, has_arr)	  		  \
static PyObject * JOIN5(ssdist_, function_name, 							  \
						PRE_HYPHEN(LOWER(array_type)),	  	  				  \
						ADD_TOKEN(has_dict, PRE_HYPHEN(score_type)), 		  \
						_wrap) 												  \
						(PyObject *self, PyObject *args)      		  		  \
{                                                                   	  	  \
	PyArrayObject *XA_, *XB_, *indicesA_, *indicesB_, *indptr_, *dm_;   	  \
	GAP(matrix_type);														  \
	ADD_TOKEN(has_dict, PyObject *mat;)										  \
	int norm;																  \
	double tol;														  		  \
																			  \
	if (!PyArg_ParseTuple(args,								  				  \
				"O!O!O!O!O!O!" GAP_ARG_STRING(matrix_type)					  \
				ADD_TOKEN(has_dict, "O!")"p|d",    							  \
                &PyArray_Type, &XA_, 									  	  \
				&PyArray_Type, &XB_, 									  	  \
				&PyArray_Type, &indicesA_, 								  	  \
				&PyArray_Type, &indicesB_, 								  	  \
				&PyArray_Type, &indptr_, 								  	  \
				&PyArray_Type, &dm_,			  			  	  			  \
                &gap_open,							  	  					  \
				&gap_ext, 				  	  								  \
				ADD_ARG(has_dict, &PyDict_Type) 							  \
				ADD_ARG(has_dict, &mat) &norm, &tol))	  			  		  \
		return NULL;                                                      	  \
																			  \
	npy_intp mA, nA ADD_COMMA(array_type) ADD_TOKEN(array_type, nB);   	  	  \
	int status;																  \
    double *dm;                                                           	  \
    const ARRAY_TYPE(array_type)XA;											  \
    const ARRAY_TYPE(array_type)XB;                                        	  \
	const npy_intp *indicesA, *indicesB, *indptr;  							  \
	const JOIN2(scoreptr, ADD_TOKEN(has_dict, _dict)) func = 				  \
		JOIN4(ARRAY_TYPE_ALIAS(array_type),									  \
			  ADD_TOKEN(has_dict, PRE_HYPHEN(matrix_type)), 				  \
			  PRE_HYPHEN(function_name), _score);			  				  \
																			  \
	BEGIN_THREADS(has_dict);												  \
	XA = (const ARRAY_TYPE(array_type))XA_->data;                             \
	XB = (const ARRAY_TYPE(array_type))XB_->data;                             \
	indicesA = (const npy_intp *)indicesA_->data;                     	  	  \
	indicesB = (const npy_intp *)indicesB_->data;                     	  	  \
	indptr = (const npy_intp *)indptr_->data;                         	  	  \
	dm = (double *)dm_->data;                                     		  	  \
	mA = indicesA_->dimensions[0];                                    	  	  \
	STRING_LENGTHS2(array_type);										  	  \
																			  \
	if (norm)																  \
		status = JOIN4(LOWER(array_type), PRE_HYPHEN(score_type),			  \
					   _align_ssdist_norm, ADD_TOKEN(has_dict, _dict)) (	  \
							   XA, XB, indicesA, indicesB, indptr, func, 	  \
							   ADD_ARG(has_dict, mat) gap_open, gap_ext, 	  \
							   dm, mA, nA ADD_COMMA(array_type) 			  \
							   ADD_TOKEN(array_type, nB), tol); 			  \
	else 	   																  \
		status = JOIN4(LOWER(array_type), PRE_HYPHEN(score_type),			  \
					   _align_ssdist, ADD_TOKEN(has_dict, _dict)) (			  \
							   XA, XB, indicesA, indicesB, indptr, func, 	  \
							   ADD_ARG(has_dict, mat) gap_open, gap_ext, 	  \
							   dm, mA, nA ADD_COMMA(array_type) 			  \
							   ADD_TOKEN(array_type, nB));					  \
	END_THREADS(has_dict);												  	  \
																			  \
	if (status == NO_MEMORY)											  	  \
		return PyErr_NoMemory();										  	  \
	ERROR_DICT(has_dict);													  \
	ERROR_ARR(has_arr);													  	  \
	return Py_BuildValue("d", 0.0);											  \
}                                                                      	  	  \

DEFINE_ALIGN_SSDIST_WRAP(levenshtein, PyObject, long, int8_t, false, false)
DEFINE_ALIGN_SSDIST_WRAP(blosum62, PyObject, long, int8_t, false, true)
DEFINE_ALIGN_SSDIST_WRAP(pam250, PyObject, long, int8_t, false, true)
DEFINE_ALIGN_SSDIST_WRAP(needleman_wunsch, PyObject, double, double, true, false)
DEFINE_ALIGN_SSDIST_WRAP(needleman_wunsch, PyObject, long, long, true, false)
DEFINE_ALIGN_SSDIST_WRAP(levenshtein, str, long, int8_t, false, false)
DEFINE_ALIGN_SSDIST_WRAP(blosum62, str, long, int8_t, false, true)
DEFINE_ALIGN_SSDIST_WRAP(pam250, str, long, int8_t, false, true)
DEFINE_ALIGN_SSDIST_WRAP(needleman_wunsch, str, long, long, true, false)
DEFINE_ALIGN_SSDIST_WRAP(needleman_wunsch, str, double, double, true, false)

static PyMethodDef _alignmentWrapMethods[] = {
  {"pdist_levenshtein_pyobject_wrap", pdist_levenshtein_pyobject_wrap, METH_VARARGS},
  {"pdist_levenshtein_str_wrap", pdist_levenshtein_str_wrap, METH_VARARGS},
  {"pdist_blosum62_pyobject_wrap", pdist_blosum62_pyobject_wrap, METH_VARARGS},
  {"pdist_blosum62_str_wrap", pdist_blosum62_str_wrap, METH_VARARGS},
  {"pdist_pam250_pyobject_wrap", pdist_pam250_pyobject_wrap, METH_VARARGS},
  {"pdist_pam250_str_wrap", pdist_pam250_str_wrap, METH_VARARGS},
  {"pdist_needleman_wunsch_pyobject_long_wrap", pdist_needleman_wunsch_pyobject_long_wrap, METH_VARARGS },
  {"pdist_needleman_wunsch_str_long_wrap", pdist_needleman_wunsch_str_long_wrap, METH_VARARGS },
  {"pdist_needleman_wunsch_pyobject_double_wrap", pdist_needleman_wunsch_pyobject_double_wrap, METH_VARARGS },
  {"pdist_needleman_wunsch_str_double_wrap", pdist_needleman_wunsch_str_double_wrap, METH_VARARGS },
  {"cdist_levenshtein_pyobject_wrap", cdist_levenshtein_pyobject_wrap, METH_VARARGS},
  {"cdist_levenshtein_str_wrap", cdist_levenshtein_str_wrap, METH_VARARGS},
  {"cdist_blosum62_pyobject_wrap", cdist_blosum62_pyobject_wrap, METH_VARARGS},
  {"cdist_blosum62_str_wrap", cdist_blosum62_str_wrap, METH_VARARGS},
  {"cdist_pam250_pyobject_wrap", cdist_pam250_pyobject_wrap, METH_VARARGS},
  {"cdist_pam250_str_wrap", cdist_pam250_str_wrap, METH_VARARGS},
  {"cdist_needleman_wunsch_pyobject_long_wrap", cdist_needleman_wunsch_pyobject_long_wrap, METH_VARARGS },
  {"cdist_needleman_wunsch_str_long_wrap", cdist_needleman_wunsch_str_long_wrap, METH_VARARGS },
  {"cdist_needleman_wunsch_pyobject_double_wrap", cdist_needleman_wunsch_pyobject_double_wrap, METH_VARARGS },
  {"cdist_needleman_wunsch_str_double_wrap", cdist_needleman_wunsch_str_double_wrap, METH_VARARGS },
  {"ssdist_levenshtein_pyobject_wrap", ssdist_levenshtein_pyobject_wrap, METH_VARARGS},
  {"ssdist_levenshtein_str_wrap", ssdist_levenshtein_str_wrap, METH_VARARGS},
  {"ssdist_blosum62_pyobject_wrap", ssdist_blosum62_pyobject_wrap, METH_VARARGS},
  {"ssdist_blosum62_str_wrap", ssdist_blosum62_str_wrap, METH_VARARGS},
  {"ssdist_pam250_pyobject_wrap", ssdist_pam250_pyobject_wrap, METH_VARARGS},
  {"ssdist_pam250_str_wrap", ssdist_pam250_str_wrap, METH_VARARGS},
  {"ssdist_needleman_wunsch_pyobject_long_wrap", ssdist_needleman_wunsch_pyobject_long_wrap, METH_VARARGS },
  {"ssdist_needleman_wunsch_str_long_wrap", ssdist_needleman_wunsch_str_long_wrap, METH_VARARGS },
  {"ssdist_needleman_wunsch_pyobject_double_wrap", ssdist_needleman_wunsch_pyobject_double_wrap, METH_VARARGS },
  {"ssdist_needleman_wunsch_str_double_wrap", ssdist_needleman_wunsch_str_double_wrap, METH_VARARGS },
  {NULL, NULL}     /* Sentinel - marks the end of this structure */
};

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_alignment_wrap",
    NULL,
    -1,
    _alignmentWrapMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyObject *PyInit__alignment_wrap(void)
{
    PyObject *m;

    m = PyModule_Create(&moduledef);
    import_array();

    return m;
}
#else
PyMODINIT_FUNC init_distance_wrap(void)
{
  (void) Py_InitModule("_alignment_wrap", _alignmentWrapMethods);
  import_array();  // Must be present for NumPy.  Called first after above line.
}
#endif

/**
 * Author: Dezmond Goff

**/

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
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>

#include "ssdist_impl.h"

#define DEFINE_WRAP_ssdist(name, type)                                        \
    static PyObject *                                                         \
    ssdist_ ## name ## _wrap(PyObject *self, PyObject *args)                  \
    {                                                                         \
        PyArrayObject *XA_, *XB_, *indicesA_, *indicesB_, *indptr_, *ssdm_;   \
        npy_intp mA, n;                                                       \
        double *ssdm;                                                         \
        const type *XA, *XB;                                                  \
        const npy_intp *indicesA, *indicesB, *indptr;                         \
        if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!",                           \
                              &PyArray_Type, &XA_,                            \
                              &PyArray_Type, &XB_,                            \
                              &PyArray_Type, &indicesA_,                      \
                              &PyArray_Type, &indicesB_,                      \
                              &PyArray_Type, &indptr_,                        \
                              &PyArray_Type, &ssdm_)) {                       \
            return NULL;                                                      \
        }                                                                     \
        else {                                                                \
            NPY_BEGIN_ALLOW_THREADS;                                          \
            XA = (const type *)XA_->data;                                     \
            XB = (const type *)XB_->data;                                     \
            indicesA = (const npy_intp *)indicesA_->data;                     \
            indicesB = (const npy_intp *)indicesB_->data;                     \
            indptr = (const npy_intp *)indptr_->data;                         \
            ssdm = (double *)ssdm_->data;                                     \
            mA = indicesA_->dimensions[0];                                    \
            n = XA_->dimensions[1];                                           \
                                                                              \
            ssdist_ ## name ## _ ## type(XA, XB, indicesA, indicesB, indptr,  \
                                         ssdm, mA, n);                        \
            NPY_END_ALLOW_THREADS;                                            \
        }                                                                     \
        return Py_BuildValue("d", 0.);                                        \
    }                                                                         \

DEFINE_WRAP_ssdist(bray_curtis, double)
DEFINE_WRAP_ssdist(canberra, double)
DEFINE_WRAP_ssdist(chebyshev, double)
DEFINE_WRAP_ssdist(city_block, double)
DEFINE_WRAP_ssdist(euclidean, double)
DEFINE_WRAP_ssdist(hamming, double)
DEFINE_WRAP_ssdist(jaccard, double)
DEFINE_WRAP_ssdist(sqeuclidean, double)
DEFINE_WRAP_ssdist(yule_bool, char)

static NPY_INLINE double *mahalanobis_dimbuf(npy_intp n) {
    double *dimbuf;
    dimbuf = calloc(n, 2 * sizeof(double));
    if (!dimbuf) {
        PyErr_Format(PyExc_MemoryError, "could not allocate %zd * %zd bytes",
                     n, 2 * sizeof(double));
    }
    return dimbuf;
}


static PyObject *ssdist_mahalanobis_wrap(PyObject *self, PyObject *args, 
                                         PyObject *kwargs) {
    PyArrayObject *XA_, *XB_, *indicesA_, *indicesB_, *indptr_, *ssdm_;
    PyArrayObject *covinv_;
    npy_intp mA, n;
    double *ssdm, *dimbuf;
    const double *XA, *XB;
    const double *covinv;
    const npy_intp *indicesA, *indicesB, *indptr;
    static char *kwlist[] = {"XA", "XB", "indicesA", "indicesB", "indptr", "ssdm", "VI", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, 
                                    "O!O!O!O!O!O!O!:ssdist_mahalanobis_wrap", 
                                    kwlist,
                                    &PyArray_Type, &XA_, 
                                    &PyArray_Type, &XB_, 
                                    &PyArray_Type, &indicesA_, 
                                    &PyArray_Type, &indicesB_, 
                                    &PyArray_Type, &indptr_, 
                                    &PyArray_Type, &ssdm_, 
                                    &PyArray_Type, &covinv_)) {
        return 0;
    }
    else {
        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS;
        XA = (const double *)XA_->data;
        XB = (const double *)XB_->data;
        indicesA = (const npy_intp *)indicesA_->data;
        indicesB = (const npy_intp *)indicesB_->data;
        indptr = (const npy_intp *)indptr_->data;
        covinv = (const double *)covinv_->data;
        ssdm = (double *)ssdm_->data;
        mA = indicesA_->dimensions[0];
        n = XA_->dimensions[1];
    
        dimbuf = mahalanobis_dimbuf(n);
        if (!dimbuf) {
          NPY_END_THREADS;
          return NULL;
        }
    
        ssdist_mahalanobis(XA, XB, indicesA, indicesB, indptr, covinv, dimbuf, 
                       ssdm, mA, n);
        free(dimbuf);
        NPY_END_THREADS;
    }
    return Py_BuildValue("d", 0.0);
}

static PyObject *ssdist_seuclidean_wrap(PyObject *self, PyObject *args, PyObject *kwargs) {
  PyArrayObject *XA_, *XB_, *indicesA_, *indicesB_, *indptr_, *ssdm_, *var_;
  npy_intp mA, n;
  double *ssdm;
  const double *XA, *XB, *var;
  const npy_intp *indicesA, *indicesB, *indptr; 
  static char *kwlist[] = {"XA", "XB", "indicesA", "indicesB", "indptr", "ssdm", "V", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, 
            "O!O!O!O!O!O!O!:ssdist_seuclidean_wrap", kwlist,
            &PyArray_Type, &XA_, 
            &PyArray_Type, &XB_,
            &PyArray_Type, &indicesA_, 
            &PyArray_Type, &indicesB_, 
            &PyArray_Type, &indptr_, 
            &PyArray_Type, &ssdm_, 
            &PyArray_Type, &var_)) {
    return 0;
  }
  else {
    NPY_BEGIN_ALLOW_THREADS;
    XA = (const double *)XA_->data;
    XB = (const double *)XB_->data;
    indicesA = (const npy_intp *)indicesA_->data;
    indicesB = (const npy_intp *)indicesB_->data; 
    indptr = (const npy_intp *)indptr_->data;
    ssdm = (double *)ssdm_->data;
    var = (double *)var_->data;
    mA = indicesA_->dimensions[0];
    n = XA_->dimensions[1];

    ssdist_seuclidean(XA, XB, indicesA, indicesB, indptr, var, ssdm, mA, n);
    NPY_END_ALLOW_THREADS;
  }
  return Py_BuildValue("d", 0.0);
}

static PyObject *ssdist_hamming_bool_wrap(PyObject *self, PyObject *args) {
  PyArrayObject *XA_, *XB_, *indicesA_, *indicesB_, *indptr_, *ssdm_;
  npy_intp mA, n;
  double *ssdm;
  const char *XA, *XB;
  const npy_intp *indicesA, *indicesB, *indptr; 
  if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!",
            &PyArray_Type, &XA_, 
            &PyArray_Type, &XB_, 
            &PyArray_Type, &indicesA_, 
            &PyArray_Type, &indicesB_, 
            &PyArray_Type, &indptr_, 
            &PyArray_Type, &ssdm_)) {
    return 0;
  }
  else {
    NPY_BEGIN_ALLOW_THREADS;
    XA = (const char*)XA_->data;
    XB = (const char*)XB_->data;
    indicesA = (const npy_intp *)indicesA_->data;
    indicesB = (const npy_intp *)indicesB_->data;
    indptr = (const npy_intp *)indptr_->data;
    ssdm = (double *)ssdm_->data;
    mA = indicesA_->dimensions[0];   
    n = XA_->dimensions[1];

    ssdist_hamming_char(XA, XB, indicesA, indicesB, indptr, ssdm, mA, n);
    NPY_END_ALLOW_THREADS;
  }
  return Py_BuildValue("d", 0.0);
}

static PyObject *ssdist_jaccard_bool_wrap(PyObject *self, PyObject *args) {
  PyArrayObject *XA_, *XB_, *indicesA_, *indicesB_, *indptr_, *ssdm_;
  npy_intp mA, n;
  double *ssdm;
  const char *XA, *XB;
  const npy_intp *indicesA, *indicesB, *indptr; 
  if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!",
            &PyArray_Type, &XA_, 
            &PyArray_Type, &XB_, 
            &PyArray_Type, &indicesA_, 
            &PyArray_Type, &indicesB_, 
            &PyArray_Type, &indptr_, 
            &PyArray_Type, &ssdm_)) {
    return 0;
  }
  else {
    NPY_BEGIN_ALLOW_THREADS;
    XA = (const char*)XA_->data;
    XB = (const char*)XB_->data;
    indicesA = (const npy_intp *)indicesA_->data; 
    indicesB = (const npy_intp *)indicesB_->data; 
    indptr = (const npy_intp *)indptr_->data;
    ssdm = (double *)ssdm_->data;
    mA = indicesA_->dimensions[0];
    n = XA_->dimensions[1];

    ssdist_jaccard_char(XA, XB, indicesA, indicesB, indptr, ssdm, mA, n);
    NPY_END_ALLOW_THREADS;
  }
  return Py_BuildValue("d", 0.0);
}

static PyObject *ssdist_minkowski_wrap(PyObject *self, PyObject *args, PyObject *kwargs) {
  PyArrayObject *XA_, *XB_, *indicesA_, *indicesB_, *indptr_, *ssdm_;
  npy_intp mA, n;
  double *ssdm;
  const double *XA, *XB;
  const npy_intp *indicesA, *indicesB, *indptr; 
  double p;
  static char *kwlist[] = {"XA", "XB", "indicesA", "indicesB", "indptr", "ssdm", "p", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, 
            "O!O!O!O!O!O!d:ssdist_minkowski_wrap", kwlist,
            &PyArray_Type, &XA_, 
            &PyArray_Type, &XB_, 
            &PyArray_Type, &indicesA_, 
            &PyArray_Type, &indicesB_, 
            &PyArray_Type, &indptr_, 
            &PyArray_Type, &ssdm_, 
            &p)) {
    return 0;
  }
  else {
    NPY_BEGIN_ALLOW_THREADS;
    XA = (const double *)XA_->data;
    XB = (const double *)XB_->data;
    indicesA = (const npy_intp *)indicesA_->data;
    indicesB = (const npy_intp *)indicesB_->data;
    indptr = (const npy_intp *)indptr_->data;
    ssdm = (double *)ssdm_->data;
    mA = indicesA_->dimensions[0];
    n = XA_->dimensions[1];

    ssdist_minkowski(XA, XB, indicesA, indicesB, indptr, ssdm, mA, n, p);
    NPY_END_ALLOW_THREADS;
  }
  return Py_BuildValue("d", 0.0);
}

static PyObject *ssdist_weighted_minkowski_wrap(PyObject *self, PyObject *args, PyObject *kwargs) {
  PyArrayObject *XA_, *XB_, *indicesA_, *indicesB_, *indptr_, *ssdm_, *w_;
  npy_intp mA, n;
  double *ssdm;
  const double *XA, *XB, *w;
  const npy_intp *indicesA, *indicesB, *indptr; 
  double p;
  static char *kwlist[] = {"XA", "XB", "indicesA", "indicesB", "indptr", "ssdm", "p", "w", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, 
            "O!O!O!O!O!O!dO!:ssdist_weighted_minkowski_wrap", kwlist,
            &PyArray_Type, &XA_, 
            &PyArray_Type, &XB_, 
            &PyArray_Type, &indicesA_, 
            &PyArray_Type, &indicesB_, 
            &PyArray_Type, &indptr_, 
            &PyArray_Type, &ssdm_,
            &p, 
            &PyArray_Type, &w_)) {
    return 0;
  }
  else {
    NPY_BEGIN_ALLOW_THREADS;
    XA = (const double *)XA_->data;
    XB = (const double *)XB_->data;
    indicesA = (const npy_intp *)indicesA_->data;
    indicesB = (const npy_intp *)indicesB_->data; 
    indptr = (const npy_intp *)indptr_->data;
    w = (const double *)w_->data;
    ssdm = (double *)ssdm_->data;
    mA = indicesA_->dimensions[0];
    n = XA_->dimensions[1];

    ssdist_weighted_minkowski(XA, XB, indicesA, indicesB, indptr, ssdm, mA, n, p, w);
    NPY_END_ALLOW_THREADS;
  }
  return Py_BuildValue("d", 0.0);
}

static PyObject *ssdist_dice_bool_wrap(PyObject *self, PyObject *args) {
  PyArrayObject *XA_, *XB_, *indicesA_, *indicesB_, *indptr_, *ssdm_;
  npy_intp mA, n;
  double *ssdm;
  const char *XA, *XB;
  const npy_intp *indicesA, *indicesB, *indptr; 
  if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!",
            &PyArray_Type, &XA_, 
            &PyArray_Type, &XB_, 
            &PyArray_Type, &indicesA_, 
            &PyArray_Type, &indicesB_, 
            &PyArray_Type, &indptr_, 
            &PyArray_Type, &ssdm_)) {
    return 0;
  }
  else {
    NPY_BEGIN_ALLOW_THREADS;
    XA = (const char*)XA_->data;
    XB = (const char*)XB_->data;
    indicesA = (const npy_intp *)indicesA_->data;
    indicesB = (const npy_intp *)indicesB_->data;
    indptr = (const npy_intp *)indptr_->data;
    ssdm = (double *)ssdm_->data;
    mA = indicesA_->dimensions[0];    
    n = XA_->dimensions[1];

    ssdist_dice_char(XA, XB, indicesA, indicesB, indptr, ssdm, mA, n);
    NPY_END_ALLOW_THREADS;
  }
  return Py_BuildValue("");
}

static PyObject *ssdist_rogerstanimoto_bool_wrap(PyObject *self, PyObject *args) {
  PyArrayObject *XA_, *XB_, *indicesA_, *indicesB_, *indptr_, *ssdm_;
  npy_intp mA, n;
  double *ssdm;
  const char *XA, *XB;
  const npy_intp *indicesA, *indicesB, *indptr; 
  if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!",
            &PyArray_Type, &XA_, 
            &PyArray_Type, &XB_, 
            &PyArray_Type, &indicesA_, 
            &PyArray_Type, &indicesB_, 
            &PyArray_Type, &indptr_, 
            &PyArray_Type, &ssdm_)) {
    return 0;
  }
  else {
    NPY_BEGIN_ALLOW_THREADS;
    XA = (const char*)XA_->data;
    XB = (const char*)XB_->data;
    indicesA = (const npy_intp *)indicesA_->data;
    indicesB = (const npy_intp *)indicesB_->data;
    indptr = (const npy_intp *)indptr_->data;
    ssdm = (double *)ssdm_->data;
    mA = indicesA_->dimensions[0];    
    n = XA_->dimensions[1];

    ssdist_rogerstanimoto_char(XA, XB, indicesA, indicesB, indptr, ssdm, mA, n);
    NPY_END_ALLOW_THREADS;
  }
  return Py_BuildValue("");
}

static PyObject *ssdist_russellrao_bool_wrap(PyObject *self, PyObject *args) {
  PyArrayObject *XA_, *XB_, *indicesA_, *indicesB_, *indptr_, *ssdm_;
  npy_intp mA, n;
  double *ssdm;
  const char *XA, *XB;
  const npy_intp *indicesA, *indicesB, *indptr; 
  if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!",
            &PyArray_Type, &XA_, 
            &PyArray_Type, &XB_, 
            &PyArray_Type, &indicesA_, 
            &PyArray_Type, &indicesB_, 
            &PyArray_Type, &indptr_, 
            &PyArray_Type, &ssdm_)) {
    return 0;
  }
  else {
    NPY_BEGIN_ALLOW_THREADS;
    XA = (const char*)XA_->data;
    XB = (const char*)XB_->data;
    indicesA = (const npy_intp *)indicesA_->data;
    indicesB = (const npy_intp *)indicesB_->data; 
    indptr = (const npy_intp *)indptr_->data;
    ssdm = (double *)ssdm_->data;
    mA = indicesA_->dimensions[0];
    n = XA_->dimensions[1];

    ssdist_russellrao_char(XA, XB, indicesA, indicesB, indptr, ssdm, mA, n);
    NPY_END_ALLOW_THREADS;
  }
  return Py_BuildValue("");
}

static PyObject *ssdist_kulsinski_bool_wrap(PyObject *self, PyObject *args) {
  PyArrayObject *XA_, *XB_, *indicesA_, *indicesB_, *indptr_, *ssdm_;
  npy_intp mA, n;
  double *ssdm;
  const char *XA, *XB;
  const npy_intp *indicesA, *indicesB, *indptr; 
  if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!",
            &PyArray_Type, &XA_, 
            &PyArray_Type, &XB_, 
            &PyArray_Type, &indicesA_, 
            &PyArray_Type, &indicesB_, 
            &PyArray_Type, &indptr_, 
            &PyArray_Type, &ssdm_)) {
    return 0;
  }
  else {
    NPY_BEGIN_ALLOW_THREADS;
    XA = (const char*)XA_->data;
    XB = (const char*)XB_->data;
    indicesA = (const npy_intp *)indicesA_->data; 
    indicesB = (const npy_intp *)indicesB_->data;
    indptr = (const npy_intp *)indptr_->data;
    ssdm = (double *)ssdm_->data;
    mA = indicesA_->dimensions[0];
    n = XA_->dimensions[1];

    ssdist_kulsinski_char(XA, XB, indicesA, indicesB, indptr, ssdm, mA, n);
    NPY_END_ALLOW_THREADS;
  }
  return Py_BuildValue("");
}

static PyObject *ssdist_sokalmichener_bool_wrap(PyObject *self, PyObject *args) {
  PyArrayObject *XA_, *XB_, *indicesA_, *indicesB_, *indptr_, *ssdm_;
  npy_intp mA, n;
  double *ssdm;
  const char *XA, *XB;
  const npy_intp *indicesA, *indicesB, *indptr; 
  if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!",
            &PyArray_Type, &XA_, 
            &PyArray_Type, &XB_, 
            &PyArray_Type, &indicesA_, 
            &PyArray_Type, &indicesB_, 
            &PyArray_Type, &indptr_, 
            &PyArray_Type, &ssdm_)) {
    return 0;
  }
  else {
    NPY_BEGIN_ALLOW_THREADS;
    XA = (const char*)XA_->data;
    XB = (const char*)XB_->data;
    indicesA = (const npy_intp *)indicesA_->data;
    indicesB = (const npy_intp *)indicesB_->data;
    indptr = (const npy_intp *)indptr_->data;
    ssdm = (double *)ssdm_->data;
    mA = indicesA_->dimensions[0];   
    n = XA_->dimensions[1];

    ssdist_sokalmichener_char(XA, XB, indicesA, indicesB, indptr, ssdm, mA, n);
    NPY_END_ALLOW_THREADS;
  }
  return Py_BuildValue("");
}

static PyObject *ssdist_sokalsneath_bool_wrap(PyObject *self, PyObject *args) {
  PyArrayObject *XA_, *XB_, *indicesA_, *indicesB_, *indptr_, *ssdm_;
  npy_intp mA, n;
  double *ssdm;
  const char *XA, *XB;
  const npy_intp *indicesA, *indicesB, *indptr; 
  if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!",
            &PyArray_Type, &XA_, 
            &PyArray_Type, &XB_, 
            &PyArray_Type, &indicesA_, 
            &PyArray_Type, &indicesB_, 
            &PyArray_Type, &indptr_, 
            &PyArray_Type, &ssdm_)) {
    return 0;
  }
  else {
    NPY_BEGIN_ALLOW_THREADS;
    XA = (const char*)XA_->data;
    XB = (const char*)XB_->data;
    indicesA = (const npy_intp *)indicesA_->data;
    indicesB = (const npy_intp *)indicesB_->data;
    indptr = (const npy_intp *)indptr_->data;
    ssdm = (double *)ssdm_->data;
    mA = indicesA_->dimensions[0];
    n = XA_->dimensions[1];

    ssdist_sokalsneath_char(XA, XB, indicesA, indicesB, indptr, ssdm, mA, n);
    NPY_END_ALLOW_THREADS;
  }
  return Py_BuildValue("");
}

static PyMethodDef _ssdistWrapMethods[] = {
  {"ssdist_bray_curtis_wrap", ssdist_bray_curtis_wrap, METH_VARARGS},
  {"ssdist_canberra_wrap", ssdist_canberra_wrap, METH_VARARGS},
  {"ssdist_chebyshev_wrap", ssdist_chebyshev_wrap, METH_VARARGS},
  {"ssdist_city_block_wrap", ssdist_city_block_wrap, METH_VARARGS},
  {"ssdist_dice_bool_wrap", ssdist_dice_bool_wrap, METH_VARARGS},
  {"ssdist_euclidean_wrap", ssdist_euclidean_wrap, METH_VARARGS},
  {"ssdist_sqeuclidean_wrap", ssdist_sqeuclidean_wrap, METH_VARARGS},
  {"ssdist_hamming_wrap", ssdist_hamming_wrap, METH_VARARGS},
  {"ssdist_hamming_bool_wrap", ssdist_hamming_bool_wrap, METH_VARARGS},
  {"ssdist_jaccard_wrap", ssdist_jaccard_wrap, METH_VARARGS},
  {"ssdist_jaccard_bool_wrap", ssdist_jaccard_bool_wrap, METH_VARARGS},
  {"ssdist_kulsinski_bool_wrap", ssdist_kulsinski_bool_wrap, METH_VARARGS},
  {"ssdist_mahalanobis_wrap", ssdist_mahalanobis_wrap, METH_VARARGS | METH_KEYWORDS},
  {"ssdist_minkowski_wrap", ssdist_minkowski_wrap, METH_VARARGS | METH_KEYWORDS},
  {"ssdist_weighted_minkowski_wrap", ssdist_weighted_minkowski_wrap, METH_VARARGS | METH_KEYWORDS},
  {"ssdist_rogerstanimoto_bool_wrap", ssdist_rogerstanimoto_bool_wrap, METH_VARARGS},
  {"ssdist_russellrao_bool_wrap", ssdist_russellrao_bool_wrap, METH_VARARGS},
  {"ssdist_seuclidean_wrap", ssdist_seuclidean_wrap, METH_VARARGS | METH_KEYWORDS},
  {"ssdist_sokalmichener_bool_wrap", ssdist_sokalmichener_bool_wrap, METH_VARARGS},
  {"ssdist_sokalsneath_bool_wrap", ssdist_sokalsneath_bool_wrap, METH_VARARGS},
  {"ssdist_yule_bool_wrap", ssdist_yule_bool_wrap, METH_VARARGS},
  {NULL, NULL}     /* Sentinel - marks the end of this structure */
};

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_ssdist_wrap",
    NULL,
    -1,
    _ssdistWrapMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyObject *PyInit__ssdist_wrap(void)
{
    PyObject *m;

    m = PyModule_Create(&moduledef);
    import_array();

    return m;
}
#else
PyMODINIT_FUNC init_distance_wrap(void)
{
  (void) Py_InitModule("_ssdist_wrap", _ssdistWrapMethods);
  import_array();  // Must be present for NumPy.  Called first after above line.
}
#endif
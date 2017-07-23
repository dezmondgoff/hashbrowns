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
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>

#include "alignment_impl.h"

#define DEFINE_ALIGN_WRAP_pdist(style, name)                                  \
    static PyObject *                                                         \
    pdist ## style ## _ ## name ## _wrap(PyObject *self, PyObject *args)      \
    {                                                                         \
        PyArrayObject *X_, *dm_;                                              \
        npy_intp i, j, n, m;                                                  \
        double *dm;                                                           \
        int *buf;                                                             \
        const PyBytesObject *X;                                               \
        PyObject *mat;                                                        \
        int gap_open, gap_ext;                                                \
        scorePtr func = &(name ## _score);                                    \
                                                                              \
        if (!PyArg_ParseTuple(args, "O!O!ii|O!",                              \
                              &PyArray_Type, &X_,                             \
                              &PyArray_Type, &dm_,                            \
                              &gap_open, &gap_ext,                            \
                              &PyDict_Type, &mat)) {                          \
            return NULL;                                                      \
        }                                                                     \
        else {                                                                \
            NPY_BEGIN_ALLOW_THREADS;                                          \
            X = (const PyBytesObject *)X_->data;                              \
            dm = (double *)dm_->data;                                         \
            m = X_->dimensions[0];                                            \
                                                                              \
            n = 0;                                                            \
            for (i = 0; i < m; i++) {                                         \
                j = PyBytes_GET_SIZE(X + i);                                  \
                if (j > n)                                                    \
                    n = j;                                                    \
            }                                                                 \
                                                                              \
            buf = align_buffer(n);                                            \
            if (!buf) {                                                       \
                NPY_END_THREADS;                                              \
                return NULL;                                                  \
            }                                                                 \
                                                                              \
            pdist ## style ## _alignment(X, func, mat, gap_open, gap_ext, dm, \
                                         m, buf, n);                          \
            free(buf);                                                        \
            NPY_END_ALLOW_THREADS;                                            \
        }                                                                     \
        return Py_BuildValue("d", 0.);                                        \
    }                                                                         \

DEFINE_ALIGN_WRAP_pdist(, levenshtein)
DEFINE_ALIGN_WRAP_pdist(_normalized, levenshtein)
DEFINE_ALIGN_WRAP_pdist(, blosum62)
DEFINE_ALIGN_WRAP_pdist(_normalized, blosum62)
DEFINE_ALIGN_WRAP_pdist(, alignment)
DEFINE_ALIGN_WRAP_pdist(_normalized, alignment)                             

#define DEFINE_ALIGN_WRAP_cdist(style, name)                                  \
    static PyObject *                                                         \
    cdist ## style ## _ ## name ## _wrap(PyObject *self, PyObject *args)      \
    {                                                                         \
        PyArrayObject *XA_, *XB_, *dm_;                                       \
        npy_intp i, j, mA, mB, n;                                             \
        double *dm;                                                           \
        int *buf;                                                             \
        const PyBytesObject *XA, *XB;                                         \
        PyObject *mat;                                                        \
        int gap_open, gap_ext;                                                \
        scorePtr func = &(name ## _score);                                    \
                                                                              \
        if (!PyArg_ParseTuple(args, "O!O!O!ii|O!",                            \
                              &PyArray_Type, &XA_,                            \
                              &PyArray_Type, &XB_,                            \
                              &PyArray_Type, &dm_,                             \
                              &gap_open, &gap_ext,                            \
                              &PyDict_Type, &mat)) {                          \
            return NULL;                                                      \
        }                                                                     \
        else {                                                                \
            NPY_BEGIN_ALLOW_THREADS;                                          \
            XA = (const PyBytesObject *)XA_->data;                            \
            XB = (const PyBytesObject *)XB_->data;                            \
            dm = (double *)dm_->data;                                         \
            mA = XA_->dimensions[0];                                          \
            mB = XB_->dimensions[0];                                          \
                                                                              \
            n = 0;                                                            \
            for (i = 0; i < mA; i++) {                                        \
                j = PyBytes_GET_SIZE(XA + i);                                 \
                if (j > n)                                                    \
                    n = j;                                                    \
            }                                                                 \
                                                                              \
            buf = align_buffer(n);                                            \
            if (!buf) {                                                       \
                NPY_END_THREADS;                                              \
                return NULL;                                                  \
            }                                                                 \
                                                                              \
            cdist ## style ## _alignment(XA, XB, func, mat, gap_open,         \
                                         gap_ext, dm, mA, mB, buf, n);        \
            free(buf);                                                        \
            NPY_END_ALLOW_THREADS;                                            \
        }                                                                     \
        return Py_BuildValue("d", 0.);                                        \
    }                                                                         \

DEFINE_ALIGN_WRAP_cdist(, levenshtein)
DEFINE_ALIGN_WRAP_cdist(_normalized, levenshtein)
DEFINE_ALIGN_WRAP_cdist(, blosum62)
DEFINE_ALIGN_WRAP_cdist(_normalized, blosum62)
DEFINE_ALIGN_WRAP_cdist(, alignment)
DEFINE_ALIGN_WRAP_cdist(_normalized, alignment)

#define DEFINE_ALIGN_WRAP_ssdist(style, name)                                 \
    static PyObject *                                                         \
    ssdist ## style ## _ ## name ## _wrap(PyObject *self, PyObject *args)     \
    {                                                                         \
        PyArrayObject *XA_, *XB_, *indicesA_, *indicesB_, *indptr_, *ssdm_;   \
        npy_intp i, j, mA, n;                                                 \
        double *ssdm;                                                         \
        int *buf;                                                             \
        const PyBytesObject *XA, *XB;                                         \
        PyObject *mat;                                                        \
        const npy_intp *indicesA, *indicesB, *indptr;                         \
        int gap_open, gap_ext;                                                \
        const scorePtr func = name ## _score;                                 \
                                                                              \
        if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!ii|O!",                      \
                              &PyArray_Type, &XA_,                            \
                              &PyArray_Type, &XB_,                            \
                              &PyArray_Type, &indicesA_,                      \
                              &PyArray_Type, &indicesB_,                      \
                              &PyArray_Type, &indptr_,                        \
                              &PyArray_Type, &ssdm_,                          \
                              &gap_open, &gap_ext,                            \
                              &PyDict_Type, &mat)) {                          \
            return NULL;                                                      \
        }                                                                     \
        else {                                                                \
            NPY_BEGIN_ALLOW_THREADS;                                          \
            XA = (const PyBytesObject *)XA_->data;                            \
            XB = (const PyBytesObject *)XB_->data;                            \
            indicesA = (const npy_intp *)indicesA_->data;                     \
            indicesB = (const npy_intp *)indicesB_->data;                     \
            indptr = (const npy_intp *)indptr_->data;                         \
            ssdm = (double *)ssdm_->data;                                     \
            mA = indicesA_->dimensions[0];                                    \
                                                                              \
            n = 0;                                                            \
            for (i = 0; i < mA; i++) {                                        \
                j = PyBytes_GET_SIZE(XA + i);                                 \
                if (j > n)                                                    \
                    n = j;                                                    \
            }                                                                 \
                                                                              \
            buf = align_buffer(n);                                            \
            if (!buf) {                                                       \
                NPY_END_THREADS;                                              \
                return NULL;                                                  \
            }                                                                 \
                                                                              \
            ssdist ## style ## _alignment(XA, XB, indicesA, indicesB, indptr, \
                                          func, mat, gap_open, gap_ext, ssdm, \
                                          mA, buf, n);                        \
            free(buf);                                                        \
            NPY_END_ALLOW_THREADS;                                            \
        }                                                                     \
        return Py_BuildValue("d", 0.);                                        \
    }                                                                         \

DEFINE_ALIGN_WRAP_ssdist(, levenshtein)
DEFINE_ALIGN_WRAP_ssdist(_normalized, levenshtein)
DEFINE_ALIGN_WRAP_ssdist(, blosum62)
DEFINE_ALIGN_WRAP_ssdist(_normalized, blosum62)
DEFINE_ALIGN_WRAP_ssdist(, alignment)
DEFINE_ALIGN_WRAP_ssdist(_normalized, alignment)

static PyMethodDef _alignmentWrapMethods[] = {
  {"pdist_levenshtein_wrap", pdist_levenshtein_wrap, METH_VARARGS},
  {"pdist_normalized_levenshtein_wrap", pdist_normalized_levenshtein_wrap, METH_VARARGS},
  {"pdist_blosum62_wrap", pdist_blosum62_wrap, METH_VARARGS},
  {"pdist_normalized_blosum62_wrap", pdist_normalized_blosum62_wrap, METH_VARARGS},  
  {"pdist_alignment_wrap", pdist_alignment_wrap, METH_VARARGS},
  {"pdist_normalized_alignment_wrap", pdist_normalized_alignment_wrap, METH_VARARGS},
  {"cdist_levenshtein_wrap", cdist_levenshtein_wrap, METH_VARARGS},
  {"cdist_normalized_levenshtein_wrap", cdist_normalized_levenshtein_wrap, METH_VARARGS},
  {"cdist_blosum62_wrap", cdist_blosum62_wrap, METH_VARARGS},
  {"cdist_normalized_blosum62_wrap", cdist_normalized_blosum62_wrap, METH_VARARGS},  
  {"cdist_alignment_wrap", cdist_alignment_wrap, METH_VARARGS},
  {"cdist_normalized_alignment_wrap", cdist_normalized_alignment_wrap, METH_VARARGS},
  {"ssdist_levenshtein_wrap", ssdist_levenshtein_wrap, METH_VARARGS},
  {"ssdist_normalized_levenshtein_wrap", ssdist_normalized_levenshtein_wrap, METH_VARARGS},
  {"ssdist_blosum62_wrap", ssdist_blosum62_wrap, METH_VARARGS},
  {"ssdist_normalized_blosum62_wrap", ssdist_normalized_blosum62_wrap, METH_VARARGS},  
  {"ssdist_alignment_wrap", ssdist_alignment_wrap, METH_VARARGS},
  {"ssdist_normalized_alignment_wrap", ssdist_normalized_alignment_wrap, METH_VARARGS},
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
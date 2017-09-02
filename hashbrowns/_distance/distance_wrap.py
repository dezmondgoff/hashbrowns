# distutils: language = c
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False
# cython: profile = True

import numpy as np
from . import _alignment_wrap
from scipy.spatial.distance import _distance_wrap

from functools import partial
from scipy._lib.six import xrange, string_types
from .distance import cdist, _row_norms
from .ssdist import (ssdist, _validate_alignment_args,
                     _validate_score_matrix, _filter_deprecated_kwargs)
from .alignment import needleman_wunsch, blosum62, levenshtein, pam250
from scipy.sparse import csr_matrix

def _copy_array_if_base_present(a):
    """g
    Copies the array if its base points to a parent array.
    """
    if a.base is not None:
        return a.copy()
    return a

def _convert_to_double(a):
    if a.dtype != np.double:
        return np.ascontiguousarray(a, np.double)
    return a

def _convert_to_str(a):
    assert(isinstance(a, np.ndarray))
    if a.dtype != np.object_ and not np.issubdtype(a.dtype, np.str):
        return np.ascontiguousarray(a, dtype=np.object_)
    return a

def _validate_string_dtypes(a, b):
    if a.dtype != b.dtype:
        if not np.issubdtype(a.dtype, b.dtype):
            raise ValueError("String arrays must have the same dtype")
    if a.dtype == np.object_:
        if not all(isinstance(x, str) for x in a):
            raise ValueError("All objects in XA must be strings")
        if not all(isinstance(x, str) for x in b):
            raise ValueError("All objects in XB must be strings")

def _validate_ouput(a, b, out):
    n, m = a.shape[0], b.shape[0]
    size = n * m
    if np.prod(out.shape) < size:
        raise ValueError("Not enough space in provided output array.")
    if not out.flags.contiguous:
        raise ValueError("Output matrix must be contiguous.")
    return size, (n, m)

_string_names = []
_STRING_METRICS_NAMES = ["blosum62", "levenshtein", "needleman_wunsch", "pam250"]
_STRING_cdist = {}
_TEST_STRING_METRICS = {'test_' + name: eval(name) for name in _STRING_METRICS_NAMES}

for wrap_name, names, has_dict in [
    ("levenshtein", ["edit", "levenshtein"], False),
    ("blosum62", ["blosum62", "blosum"], False),
    ("pam250", ["pam250", "pam"], False),
    ("needleman_wunsch", ["needleman_wunsch", "needleman-wunsch", "nw", "align",
                          "alignment"], True)
]:
    _string_names.extend(names)

    for typ in ["o", "s"]:
        for mtyp in ["l", "d"] if has_dict else [None]:
            fmt = ("%s_" % wrap_name,
                   {"o": "pyobject_", "s": "str_"}[typ],
                   {None:"", "l":"long_", "d":"double_"}[mtyp])
            fn_name = "%s%s%swrap" % fmt
            fn = getattr(_alignment_wrap, "%s_%s" % ("cdist", fn_name))
            for name in names:
                key = (name, typ)
                if mtyp is not None:
                    key += (mtyp,)
                _STRING_cdist[key] = fn

def _cdist_string_callable(XA, XB, metric, dm, gap_open=None, gap_ext=None,
                           mat=None, normalized=None, tol=None):
    # metrics that expects multiple args
    if metric in [blosum62, levenshtein, pam250]:
        (gap_open, gap_ext, normalized, tol), _ = _validate_alignment_args(
            gap_open, gap_ext, mat, normalized, tol)
        _filter_deprecated_kwargs(mat=mat)
        metric = partial(metric, gap_open=gap_open, gap_ext=gap_ext,
                         normalized=normalized, tol=tol)
    elif metric == needleman_wunsch:
        (gap_open, gap_ext, mat, normalized, tol), _ = _validate_alignment_args(
            gap_open, gap_ext, mat, normalized, tol)
        metric = partial(needleman_wunsch, gap_open=gap_open, gap_ext=gap_ext,
                         mat=mat, normalized=normalized, tol=tol)

    for i in xrange(0, XA.shape[0]):
            for j in xrange(0, XB.shape[0]):
                dm[i, j] = metric(XA[i], XB[j])

def cdist_wrap(XA, XB, metric, p=None, V=None, VI=None, w=None, gap_open=None,
               gap_ext=None, mat=None, normalized=None, tol=None, out=None):
    if out is None:
        dm = np.empty((XA.shape[0], XB.shape[0]))
    else:
        size, shape = _validate_ouput(XA, XB, out)
        dm = out.ravel()[:size].reshape(shape)

    try:
        return cdist(XA, XB, metric, p, V, VI, w, dm)
    except ValueError as e:
        if not str(e).startswith(("XA must be a 2-dimensional array",
                                  "XB must be a 2-dimensional array",
                                  "Unknown Distance Metric",
                                  "Unknown \"Test\" Distance Metric")):
            raise e
    if callable(metric):
        _filter_deprecated_kwargs(p=p, V=V, w=w, VI=VI)
        kwargs = {}
        _cdist_string_callable(XA, XB, metric, dm, gap_open=gap_open,
                               gap_ext=gap_ext, mat=mat, normalized=normalized,
                               tol=tol)
    elif isinstance(metric, string_types):
        mstr = metric.lower()
        if mstr in ["aitchison", "simplex", "clr"]:
            _filter_deprecated_kwargs(p=p, V=V, w=w, VI=VI, gap_open=gap_open,
                                      gap_ext=gap_ext, mat=mat)
            XA = _copy_array_if_base_present(XA)
            XB = _copy_array_if_base_present(XB)
            XA = _convert_to_double(XA)
            XB = _convert_to_double(XB)
            # check positive
            if not np.all(XA >= 0):
                raise ValueError("Values in XA must be greater than or equal "
                                 "to zero.")
            if not np.all(XB >= 0):
                raise ValueError("Values in XB must be greater than or equal "
                                 "to zero.")
            # check if on simplex
            sumA = XA.sum(axis=-1)
            sumB = XB.sum(axis=-1)
            if not np.allclose(sumA, 1):
                XA = XA * (1 / sumA[:, np.newaxis])
            if not np.allclose(sumB, 1):
                XB = XB * (1 / sumB[:, np.newaxis])
            XA = np.log(XA, out=XA)
            XB = np.log(XB, out=XB)
            XA -= XA.mean(axis=-1)[:, np.newaxis]
            XB -= XB.mean(axis=-1)[:, np.newaxis]
            _distance_wrap.cdist_euclidean_wrap(XA, XB, dm)
        elif mstr in _string_names:
            if mstr not in ["needleman_wunsch", "needleman-wunsch", "nw",
                            "align", "alignment"]:
                args, dtype = _validate_alignment_args(gap_open, gap_ext, mat,
                                                       normalized, tol)
                _filter_deprecated_kwargs(p=p, V=V, w=w, VI=VI, mat=mat)
                matrix_dtype = None
            else:
                if mat is None:
                    raise ValueError("Must provide scoring matrix")
                args, dtype = _validate_alignment_args(gap_open, gap_ext, mat,
                                                       normalized, tol)
                _filter_deprecated_kwargs(p=p, V=V, w=w, VI=VI)
            XA = _convert_to_str(XA)
            XB = _convert_to_str(XB)
            _validate_string_dtypes(XA, XB)

            key = (mstr, "o" if XA.dtype == np.object_ else "s")
            if dtype is not None:
                key += ({int:"l", float:"d"}[dtype],)
            _cdist_fn = _STRING_cdist[key]

            args = (XA, XB, dm) + tuple(arg for arg in args if arg is not None)
            _cdist_fn(*args)
        elif mstr.startswith("test_"):
            if mstr in _TEST_STRING_METRICS:
                metric = _TEST_STRING_METRICS[mstr]
                _filter_deprecated_kwargs(p=p, V=V, w=w, VI=VI)
                kwargs = {"gap_open":gap_open, "gap_ext":gap_ext, "mat":mat,
                          "normalized":normalized}
                _cdist_string_callable(XA, XB, metric, dm, **kwargs)
            else:
                raise ValueError('Unknown "Test" Distance Metric: %s' % mstr[5:])
        else:
            raise ValueError('Unknown Distance Metric: %s' % mstr)
    else:
        raise TypeError('2nd argument metric must be a string identifier '
                        'or a function.')
    return dm

def ssdist_wrap(x, y, indices_x, indices_y, indptr, mstr, p=None, w=None,
                   VI=None, V=None, gap_open=None, gap_ext=None, mat=None,
                   normalized=True, out=None):
    if indices_y is None:
        raise ValueError("Must provide indices for second array.")
    if out is None:
        dm = np.empty(indices_y.size)
    else:
        if not out.flags.contiguous:
            raise ValueError("Output matrix must be contiguous.")
        if np.prod(out.shape) < indices_y.size:
            raise ValueError("Not enough space in provided output array.")
        dm = out.ravel()[:indices_y.size]
    return ssdist(x, y, indices_y, indptr, indices_x, mstr, p, w, VI, V,
                  gap_open, gap_ext, mat, normalized, dm)

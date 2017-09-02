#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================
Distance computations (:mod:`scipy.spatial.distance`)
=====================================================

Function Reference
------------------

Distance computation within/between collections of raw observation vectors
using index pointers.

   ssdist  -- sparse matrix of pairwise distances between subsets of one two
   collections of vectors.
"""

from __future__ import division, print_function, absolute_import

import numpy as np
import warnings

from functools import partial
from scipy._lib.six import callable, string_types
from scipy._lib.six import xrange
from scipy.spatial.distance import *
from scipy.sparse import csr_matrix

from .alignment import blosum62, levenshtein, needleman_wunsch, pam250
from . import _ssdist_wrap
from . import _alignment_wrap
from .score_matrices import load_blosum62

_blosum62 = load_blosum62()

# load preset alignment matrices

def _subset_array_if_base_present(a, indices):
    """
    Copies the array if its base points to a parent array.
    """
    assert(isinstance(a, np.ndarray))
    if indices is None:
        return a, indices
    elif a.base is not None:
        unique, indices = np.unique(indices, return_inverse=True)
        return np.ascontiguousarray(a[unique]), indices
    return a, indices

def _subset_array_if_is_parent(parent, a, indices):
    """
    Copies the array if its base points to a parent array.
    """
    assert(isinstance(a, np.ndarray))
    if indices is None:
        return a, indices
    elif a is parent:
        unique, indices = np.unique(indices, return_inverse=True)
        return np.ascontiguousarray(a[unique]), indices
    return a, indices

def _convert_to_bool(a, indices):
    assert(isinstance(a, np.ndarray))
    if a.dtype != np.bool:
        if indices is None:
            return np.ascontiguousarray(a, dtype=np.bool), indices
        else:
            unique, indices = np.unique(indices, return_inverse=True)
            return np.ascontiguousarray(a[unique], dtype=np.bool), indices
    return a, indices

def _convert_to_double(a, indices):
    assert(isinstance(a, np.ndarray))

    if a.dtype != np.double:
        if indices is None:
            return np.ascontiguousarray(a, dtype=np.bool), indices
        else:
            unique, indices = np.unique(indices, return_inverse=True)
            return np.ascontiguousarray(a[unique], dtype=np.double), indices
    return a, indices

def _convert_to_str(a, indices):
    assert(isinstance(a, np.ndarray))
    if a.dtype != np.object and not np.issubdtype(a.dtype, np.str):
        if indices is None:
            return np.ascontiguousarray(a, dtype="unicode"), indices
        else:
            unique, indices = np.unique(indices, return_inverse=True)
            return np.ascontiguousarray(a[unique], dtype="unicode"), indices
    return a, indices

def _validate_string_dtypes(a, b, indicesa, indicesb):
    if a.dtype != b.dtype:
        raise ValueError("String arrays must have the same dtype")
    if a.dtype == np.object_:
        if not np.all((isinstance(a[i], str) for i in indicesa)):
            raise ValueError("All indexed objects in XA must be strings")
        if not np.all((isinstance(b[i], str) for i in indicesb)):
            raise ValueError("All indexed objects in XB must be strings")

def _validate_alignment_args(gap_open, gap_ext, mat, normalized, tol):
    if gap_open is None:
        gap_open = 1
    if gap_ext is None:
        gap_ext = 1
    if normalized is None:
        normalized = False
    elif normalized and tol is None:
        tol = 1e-07

    if mat is None:
        return (gap_open, gap_ext, normalized, tol), None
    mat, dtype = _validate_score_matrix(mat)
    return (gap_open, gap_ext, mat, normalized, tol), dtype

def _validate_score_matrix(mat):
    if not isinstance(mat, dict):
        raise ValueError("Alignment matrix must be a dict object.")
    matrix_dtype = next(iter(mat.items()))[1].__class__
    if matrix_dtype not in [int, float]:
        raise ValueError("Values must be integer or float types")
    for k in mat:
        if not isinstance(k, str) or len(k) != 2:
            raise ValueError("Keys must two character strings")
    for v in mat.values():
        if not isinstance(v, matrix_dtype):
            raise ValueError("Values must all be of the same type.")
    return mat, matrix_dtype

def _filter_deprecated_kwargs(**kwargs):
    for k in ["p", "V", "w", "VI", "gap_open", "gap_ext", "mat"]:
        kw = kwargs.pop(k, None)
        if kw is not None:
            warnings.warn("Got unexpected kwarg %s. This will raise an error"
                          " in a future version." % k, DeprecationWarning)

def _ssdist_cosine(XA, XB, indicesA, indicesB, indptr, dm):
    uniqueA, reverseA = np.unique(indicesA, return_inverse=True)
    uniqueB, reverseB = np.unique(indicesB, return_inverse=True)

    num = np.diff(indptr)
    np.einsum("ij, ij -> i", XA[indicesA].repeat(num), XB[indicesB], out=dm)

    normsA = _row_norms(XA[uniqueA])
    normsB = _row_norms(XB[uniqueB]).reshape(-1, 1)
    dm /= normsA[reverseA].repeat(num)
    dm /= normsB[reverseB]
    dm *= -1
    dm += 1

def aitchison(u, v):
    """
    Computes the Aitchison distance between two 1-D arrays.

    The Aitchison distance between `u` and `v`, defined as

    .. math::

       \\left(\\sum_{i < j}{(\\log\\left(\\frac{u_i}{u_j}\\right) -\\log\\left(\\frac{v_i}{v_j}\\right))^2)}\\right)^{1/2}.

    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.

    Returns
    -------
    wminkowski : double
        The weighted Minkowski distance between vectors `u` and `v`.

    """
    u = _validate_vector(u)
    v = _validate_vector(v)
    # check positive
    if not np.all(u >= 0):
        raise ValueError("All values in u must be greater than or equal to "
                         "zero.")
    if not np.all(v >= 0):
        raise ValueError("All values in XB must be greater than or equal to "
                         "zero.")
    # check if all values sum to 1
    norm_u = u.sum()
    norm_v = v.sum()
    if not np.isclose(norm_u, 1):
        u = (1 / norm_u) * u
    if not np.isclose(norm_v, 1):
        v = (1 / norm_v) * v
    u = np.log(u)
    v = np.log(v)
    u -= u.mean()
    v -= v.mean()
    dist = norm(u - v)
    return dist

# Registry of "simple" distance metrics" pdist and cdist implementations,
# meaning the ones that accept one dtype and have no additional arguments.
_SIMPLE_ssdist = {}

for wrap_name, names, typ in [
    ("bray_curtis", ["braycurtis"], "double"),
    ("canberra", ["canberra"], "double"),
    ("chebyshev", ["chebychev", "chebyshev", "cheby", "cheb", "ch"], "double"),
    ("city_block", ["cityblock", "cblock", "cb", "c"], "double"),
    ("euclidean", ["euclidean", "euclid", "eu", "e"], "double"),
    ("sqeuclidean", ["sqeuclidean", "sqe", "sqeuclid"], "double"),
    ("dice", ["dice"], "bool"),
    ("kulsinski", ["kulsinski"], "bool"),
    ("rogerstanimoto", ["rogerstanimoto"], "bool"),
    ("russellrao", ["russellrao"], "bool"),
    ("sokalmichener", ["sokalmichener"], "bool"),
    ("sokalsneath", ["sokalsneath"], "bool"),
    ("yule", ["yule"], "bool"),
]:
    converter = {"bool": _convert_to_bool,
                 "double": _convert_to_double}[typ]
    fn_name = {"bool": "%s_bool_wrap",
               "double": "%s_wrap"}[typ] % wrap_name
    ssdist_fn = getattr(_ssdist_wrap, "ssdist_%s" % fn_name)
    for name in names:
        _SIMPLE_ssdist[name] = converter, ssdist_fn

# Registry of string distance metrics" pdist and cdist implementations,
# meaning the ones that accept one dtype and have no additional arguments.
_string_names = []
_STRING_ssdist = {}

for wrap_name, names, has_dict in [
    ("levenshtein", ["edit", "levenshtein"], False),
    ("blosum62", ["blosum62", "blosum"], False),
    ("pam250", ["pam250", "pam"], False),
    ("needleman_wunsch", ["needleman_wunsch", "needleman-wunsch", "align",
                          "alignment"], True)
]:
    _string_names.extend(names)

    for typ in ["o", "s"]:
        for mtyp in ["l", "d"] if has_dict else [None]:
            fmt = ("%s_" % wrap_name,
                   {"o": "pyobject_", "s": "str_"}[typ],
                   {None:"", "l":"long_", "d":"double_"}[mtyp])
            fn_name = "%s%s%swrap" % fmt
            fn = getattr(_alignment_wrap, "%s_%s" % ("ssdist", fn_name))
            for name in names:
                _STRING_ssdist[(name, typ) +
                               (mtyp,) if mtyp is not None else ()] = fn

_METRICS_NAMES = ["aitchison", "braycurtis", "canberra", "chebyshev",
                  "cityblock", "correlation", "cosine", "dice", "euclidean",
                  "hamming", "jaccard", "kulsinski", "mahalanobis", "matching",
                  "minkowski", "rogerstanimoto", "russellrao", "seuclidean",
                  "sokalmichener", "sokalsneath", "sqeuclidean", "yule",
                  "wminkowski"]

_STRING_METRICS_NAMES = ["blosum62", "levenshtein", "needleman_wunsch", "pam250"]

_TEST_METRICS = {"test_" + name: eval("" + name) for name in _METRICS_NAMES +
                 _STRING_METRICS_NAMES}

def ssdist(XA, XB, indicesB, indptr, indicesA=None, metric="euclidean", p=None,
           V=None, VI=None, w=None, gap_open=None, gap_ext=None, mat=None,
           normalized=None, out=None):
    """
    Computes sparse submatrix of distances between within/between collections
    of observations

    See Notes for common calling conventions.

    Parameters
    ----------
    XA : ndarray
        An :math:`m_A` by :math:`n` array of :math:`m_A`
        original observations in an :math:`n`-dimensional space.
        Inputs are converted to float type.
    XB : ndarray
        An :math:`m_B` by :math:`n` array of :math:`m_B`
        original observations in an :math:`n`-dimensional space.
        Inputs are converted to float type.
    indices: ndarray
        An array of index pointers to rows in XB
    indptr: ndarray
        An array of length :math:`m_A + 1`
    metric : str or callable, optional
        The distance metric to use.  If a string, the distance function can be
        "braycurtis", "canberra", "chebyshev", "cityblock", "correlation",
        "cosine", "dice", "euclidean", "hamming", "jaccard", "kulsinski",
        "mahalanobis", "matching", "minkowski", "rogerstanimoto", "russellrao",
        "seuclidean", "sokalmichener", "sokalsneath", "sqeuclidean",
        "wminkowski", "yule".
    p : double, optional
        The p-norm to apply
        Only for Minkowski, weighted and unweighted. Default: 2.
    w : ndarray, optional
        The weight vector.
        Only for weighted Minkowski. Mandatory
    V : ndarray, optional
        The variance vector
        Only for standardized Euclidean. Default: var(vstack([XA, XB]), axis=0, ddof=1)
    VI : ndarray, optional
        The inverse of the covariance matrix
        Only for Mahalanobis. Default: inv(cov(vstack([XA, XB]).T)).T

    Returns
    -------
    Y : sparse matrix
        Sparse submatrix of distance matrix in csr format.

    Raises
    ------
    ValueError
        An exception is thrown if `XA` and `XB` do not have
        the same number of columns.

    Notes
    -----
    The following are common calling conventions:

    1. ``Y = cdist(XA, XB, indptr, "euclidean")``

       Computes the distance between points using Euclidean distance (2-norm) as the distance metric between the
       points.

    2. ``Y = cdist(XA, XB, "minkowski", p)``

       Computes the distances using the Minkowski distance
       :math:`||u-v||_p` (:math:`p`-norm) where :math:`p \\geq 1`.

    3. ``Y = cdist(XA, XB, "cityblock")``

       Computes the city block or Manhattan distance between the
       points.

    4. ``Y = cdist(XA, XB, "seuclidean", V=None)``

       Computes the standardized Euclidean  The standardized
       Euclidean distance between two n-vectors ``u`` and ``v`` is

       .. math::

          \\sqrt{\\sum {(u_i-v_i)^2 / V[x_i]}}.

       V is the variance vector; V[i] is the variance computed over all
       the i"th components of the points. If not passed, it is
       automatically computed.

    5. ``Y = cdist(XA, XB, "sqeuclidean")``

       Computes the squared Euclidean distance :math:`||u-v||_2^2` between
       the vectors.

    6. ``Y = cdist(XA, XB, "cosine")``

       Computes the cosine distance between vectors u and v,

       .. math::

          1 - \\frac{u \\cdot v}
                   {{||u||}_2 {||v||}_2}

       where :math:`||*||_2` is the 2-norm of its argument ``*``, and
       :math:`u \\cdot v` is the dot product of :math:`u` and :math:`v`.

    7. ``Y = cdist(XA, XB, indptr, indptr, "correlation")``

       Computes the correlation distance between vectors u and v. This is

       .. math::

          1 - \\frac{(u - \\bar{u}) \\cdot (v - \\bar{v})}
                   {{||(u - \\bar{u})||}_2 {||(v - \\bar{v})||}_2}

       where :math:`\\bar{v}` is the mean of the elements of vector v,
       and :math:`x \\cdot y` is the dot product of :math:`x` and :math:`y`.


    8. ``Y = cdist(XA, XB, indptr, indptr, "hamming")``

       Computes the normalized Hamming distance, or the proportion of
       those vector elements between two n-vectors ``u`` and ``v``
       which disagree. To save memory, the matrix ``X`` can be of type
       boolean.

    9. ``Y = cdist(XA, XB, indptr, "jaccard")``

       Computes the Jaccard distance between the points. Given two
       vectors, ``u`` and ``v``, the Jaccard distance is the
       proportion of those elements ``u[i]`` and ``v[i]`` that
       disagree where at least one of them is non-zero.

    10. ``Y = cdist(XA, XB, indptr, "chebyshev")``

       Computes the Chebyshev distance between the points. The
       Chebyshev distance between two n-vectors ``u`` and ``v`` is the
       maximum norm-1 distance between their respective elements. More
       precisely, the distance is given by

       .. math::

          d(u,v) = \\max_i {|u_i-v_i|}.

    11. ``Y = cdist(XA, XB, indptr, "canberra")``

       Computes the Canberra distance between the points. The
       Canberra distance between two points ``u`` and ``v`` is

       .. math::

         d(u,v) = \\sum_i \\frac{|u_i-v_i|}
                              {|u_i|+|v_i|}.

    12. ``Y = cdist(XA, XB, indptr, "braycurtis")``

       Computes the Bray-Curtis distance between the points. The
       Bray-Curtis distance between two points ``u`` and ``v`` is


       .. math::

            d(u,v) = \\frac{\\sum_i (|u_i-v_i|)}
                          {\\sum_i (|u_i+v_i|)}

    13. ``Y = cdist(XA, XB, indptr, "mahalanobis", VI=None)``

       Computes the Mahalanobis distance between the points. The
       Mahalanobis distance between two points ``u`` and ``v`` is
       :math:`\\sqrt{(u-v)(1/V)(u-v)^T}` where :math:`(1/V)` (the ``VI``
       variable) is the inverse covariance. If ``VI`` is not None,
       ``VI`` will be used as the inverse covariance matrix.

    14. ``Y = cdist(XA, XB, indptr, "yule")``

       Computes the Yule distance between the boolean
       vectors. (see `yule` function documentation)

    15. ``Y = cdist(XA, XB, indptr, "matching")``

       Synonym for "hamming".

    16. ``Y = cdist(XA, XB, indptr, "dice")``

       Computes the Dice distance between the boolean vectors. (see
       `dice` function documentation)

    17. ``Y = cdist(XA, XB, indptr, "kulsinski")``

       Computes the Kulsinski distance between the boolean
       vectors. (see `kulsinski` function documentation)

    18. ``Y = cdist(XA, XB, indptr, "rogerstanimoto")``

       Computes the Rogers-Tanimoto distance between the boolean
       vectors. (see `rogerstanimoto` function documentation)

    19. ``Y = cdist(XA, XB, indptr, "russellrao")``

       Computes the Russell-Rao distance between the boolean
       vectors. (see `russellrao` function documentation)

    20. ``Y = cdist(XA, XB, indptr, "sokalmichener")``

       Computes the Sokal-Michener distance between the boolean
       vectors. (see `sokalmichener` function documentation)

    21. ``Y = cdist(XA, XB, indptr, "sokalsneath")``

       Computes the Sokal-Sneath distance between the vectors. (see
       `sokalsneath` function documentation)


    22. ``Y = cdist(XA, XB, indptr, "wminkowski")``

       Computes the weighted Minkowski distance between the
       vectors. (see `wminkowski` function documentation)

    22. ``Y = cdist(..., "edit")``

       Computes the edit distance between the strings.
       (see `edit` function documentation)

    22. ``Y = ssdist(..., "blosum")``

       Computes a positive alignment distance between the strings
       based on the blosum62 matrix.(see `blosum` function documentation)

    22. ``Y = ssdist(..., "pam250")``

       Computes a positive alignment distance between the strings
       based on the pam250 matrix. (see `pam250` function documentation)

    22. ``Y = ssdist(..., "align")``

       Computes a positive alignment distance between the strings
       using a user specified align matrix. (see `align` function documentation)

    23. ``Y = cdist(XA, XB, indptr, f)``

       Computes the distance between all pairs of vectors in X
       using the user supplied 2-arity function f. For example,
       Euclidean distance between the vectors could be computed
       as follows::

         dm = cdist(XA, XB, indptr, lambda u, v: np.sqrt(((u-v)**2).sum()))

       Note that you should avoid passing a reference to one of
       the distance functions defined in this library. For example,::

         dm = cdist(XA, XB, indptr, sokalsneath)

       would calculate the pair-wise distances between the vectors in
       X using the Python function `sokalsneath`. This would result in
       sokalsneath being called :math:`{n \\choose 2}` times, which
       is inefficient. Instead, the optimized C version is more
       efficient, and we call it using the following syntax::

         dm = cdist(XA, XB, indptr, "sokalsneath")

    Examples
    --------
    Find the Euclidean distances between four 2-D coordinates:

    >>> from scipy.spatial import distance
    >>> coords = [(35.0456, -85.2672),
    ...           (35.1174, -89.9711),
    ...           (35.9728, -83.9422),
    ...           (36.1667, -86.7833)]
    >>> cdist(coords, coords, "euclidean")
    array([[ 0.    ,  4.7044,  1.6172,  1.8856],
           [ 4.7044,  0.    ,  6.0893,  3.3561],
           [ 1.6172,  6.0893,  0.    ,  2.8477],
           [ 1.8856,  3.3561,  2.8477,  0.    ]])


    Find the Manhattan distance from a 3-D point to the corners of the unit
    cube:

    >>> a = np.array([[0, 0, 0],
    ...               [0, 0, 1],
    ...               [0, 1, 0],
    ...               [0, 1, 1],
    ...               [1, 0, 0],
    ...               [1, 0, 1],
    ...               [1, 1, 0],
    ...               [1, 1, 1]])
    >>> b = np.array([[ 0.1,  0.2,  0.4]])
    >>> cdist(a, b, "cityblock")
    array([[ 0.7],
           [ 0.9],
           [ 1.3],
           [ 1.5],
           [ 1.5],
           [ 1.7],
           [ 2.1],
           [ 2.3]])

    """
    # You can also call this as:
    #     Y = cdist(XA, XB, indptr, "test_abc")
    # where "abc" is the metric being tested.  This computes the distance
    # between all pairs of vectors in XA and XB using the distance metric "abc"
    # but with a more succinct, verifiable, but less efficient implementation.

    # Store input arguments to check whether we can modify later.
    input_XA, input_XB = XA, XB
    input_indicesB = indicesB

    if indicesA is None:
        indicesA = np.arange(XA.shape[0])
    if not isinstance(XA, np.ndarray):
        XA = np.asarray(XA, order="C")
    if not isinstance(XB, np.ndarray):
        XB = np.asarray(XB, order="C")
    if not isinstance(indicesB, np.ndarray):
        indicesB = np.asarray(indicesB)
    if not isinstance(indicesA, np.ndarray):
        indicesA = np.asarray(indicesA)
    if not isinstance(indptr, np.ndarray):
        indptr = np.asarray(indptr)

    # The C code doesn"t do striding.
    XA, indicesA = _subset_array_if_base_present(XA, indicesA)
    XB, indicesB = _subset_array_if_base_present(XB, indicesB)

    sA = XA.shape
    sB = XB.shape
    sindptr = indptr.shape

    if XA.dtype != np.str:
        if len(sA) != 2:
            raise ValueError("XA must be a 2-dimensional array.")
        if len(sB) != 2:
            raise ValueError("XB must be a 2-dimensional array.")
        if sA[1] != sB[1]:
            raise ValueError("XA and XB must have the same number of columns "
                             "(i.e. feature dimension.)")
        if len(sindptr) != 1:
            raise ValueError("Index pointers must be a 1-dimensional array.")
        if (indptr[-1] != indicesB.shape[0] or
            sindptr[0] != indicesA.shape[0] + 1):
            raise ValueError("Index pointer array has incorrect/incompatible "
                             "shape.")
    else:
        if len(sindptr) != 1:
            raise ValueError("Index pointers must be a 1-dimensional array.")
        if sindptr[0] != indicesA.shape[0] or indptr[-1] != indicesB.shape[0]:
            raise ValueError("Index pointer array has incorrect/incompatible "
                             "shape.")

    mA = indicesA.shape[0]
    mB = indicesB.shape[0]
    n = sA[1]
    if out is None:
        dm = np.zeros(mB, dtype=np.double)
    else:
        if not out.flags.contiguous:
            raise ValueError("Output array must be contiguous")
        if out.shape != (mB,):
            raise ValueError("Output array has incorrect shape")
        if out.dtype != np.double:
            raise ValueError("Output array must have double dtype")
        dm = out

    # validate input for multi-args metrics
    if(metric in ["minkowski", "mi", "m", "pnorm", "test_minkowski"] or
       metric == minkowski):
        p = _validate_minkowski_args(p)
        _filter_deprecated_kwargs(w=w, V=V, VI=VI, gap_open=gap_open,
                                  gap_ext=gap_ext, mat=mat,
                                  normalized=normalized)
    elif(metric in ["wminkowski", "wmi", "wm", "wpnorm", "test_wminkowski"] or
         metric == wminkowski):
        p, w = _validate_wminkowski_args(p, w)
        _filter_deprecated_kwargs(V=V, VI=VI, gap_open=gap_open,
                                           gap_ext=gap_ext, mat=mat,
                                           normalized=normalized)
    elif(metric in ["seuclidean", "se", "s", "test_seuclidean"] or
         metric == seuclidean):
        V = _validate_seuclidean_args(np.vstack([XA, XB]), n, V)
        _filter_deprecated_kwargs(p=p, w=w, VI=VI, gap_open=gap_open,
                                  gap_ext=gap_ext, mat=mat,
                                  normalized=normalized)
    elif(metric in ["mahalanobis", "mahal", "mah", "test_mahalanobis"] or
         metric == mahalanobis):
        VI = _validate_mahalanobis_args(np.vstack([XA, XB]), mA + mB,
                                                           n, VI)
        _filter_deprecated_kwargs(p=p, w=w, V=V, gap_open=gap_open,
                                           gap_ext=gap_ext, mat=mat,
                                           normalized=normalized)
    elif(metric in ["edit", "levenshtein", "blosum", "pam", "blosum62", "pam250"]
         or metric in [levenshtein, blosum62, pam250]):
        gap_open, gap_ext = _validate_alignment_args(gap_open, gap_ext,
                                                     normalized)
        _filter_deprecated_kwargs(p=p, V=V, w=w, VI=VI, mat=mat)
        matrix_dtype = None
    elif(metric in ["align", "alignment"] or metric == alignment):
        gap_open, gap_ext = _validate_alignment_args(gap_open, gap_ext,
                                                     normalized)
        mat, matrix_dtype = _validate_score_matrix(mat)
        _filter_deprecated_kwargs(p=p, V=V, w=w, VI=VI)
    else:
        _filter_deprecated_kwargs(p=p, w=w, V=V, VI=VI, gap_open=gap_open,
                                  gap_ext=gap_ext, mat=mat,
                                  normalized=normalized)

    if callable(metric):
        # metrics that expects only doubles:
        if metric in [braycurtis, canberra, chebyshev,
                      cityblock, correlation,
                      cosine, euclidean, mahalanobis,
                      minkowski, sqeuclidean,
                      seuclidean, wminkowski]:
            XA, indicesA = _convert_to_double(XA)
            XB, indicesB = _convert_to_double(XB)
        # metrics that expects only bools:
        elif metric in [dice, kulsinski, rogerstanimoto,
                        russellrao, sokalmichener, sokalsneath,
                        yule]:
            XA, indicesA = _convert_to_bool(XA)
            XB, indicesB = _convert_to_bool(XB)
        # metrics that may receive multiple types:
        elif metric in [matching, hamming, jaccard]:
            if XA.dtype == bool:
                XA, indicesA = _convert_to_bool(XA)
                XB, indicesB = _convert_to_bool(XB)
            else:
                XA, indicesA = _convert_to_double(XA)
                XB, indicesB = _convert_to_double(XB)

        # metrics that expects multiple args
        if metric == minkowski:
            metric = partial(minkowski, p=p)
        elif metric == wminkowski:
            metric = partial(wminkowski, p=p, w=w)
        elif metric == seuclidean:
            metric = partial(seuclidean, V=V)
        elif metric == mahalanobis:
            metric = partial(mahalanobis, VI=VI)

        for i in range(indicesA.shape[0]):
            ii = indicesA[i]
            for j in range(indptr[i], indptr[i + 1]):
                dm[j] = metric(XA[ii, :], XB[indicesB[j], :])

    elif isinstance(metric, string_types):
        mstr = metric.lower()

        try:
            validate, ssdist_fn = _SIMPLE_ssdist[mstr]
            XA, indicesA = validate(XA, indicesA)
            XB, indicesB = validate(XB, indicesB)
            ssdist_fn(XA, XB, indicesA, indicesB, indptr, dm)
            indices = input_indicesB - np.min(input_indicesB)
            return csr_matrix((dm, indices, indptr), copy=False)
        except KeyError:
            pass

        if mstr in ["matching", "hamming", "hamm", "ha", "h"]:
            if XA.dtype == bool:
                XA = _convert_to_bool(XA, indicesA)
                XB = _convert_to_bool(XB, indicesB)
                _ssdist_wrap.ssdist_hamming_bool_wrap(XA, XB, indicesA, indicesB,
                                                    indptr, dm)
            else:
                XA = _convert_to_double(XA)
                XB = _convert_to_double(XB)
                _ssdist_wrap.ssdist_hamming_wrap(XA, XB, indicesA, indicesB,
                                                 indptr, dm)
        elif mstr in ["jaccard", "jacc", "ja", "j"]:
            if XA.dtype == bool:
                XA = _convert_to_bool(XA)
                XB = _convert_to_bool(XB)
                _ssdist_wrap.ssdist_jaccard_bool_wrap(XA, XB, indicesA, indicesB, indptr, dm)
            else:
                XA = _convert_to_double(XA)
                XB = _convert_to_double(XB)
                _ssdist_wrap.ssdist_jaccard_wrap(XA, XB, indicesA, indicesB, indptr, dm)
        elif mstr in ["minkowski", "mi", "m", "pnorm"]:
            XA = _convert_to_double(XA)
            XB = _convert_to_double(XB)
            _ssdist_wrap.ssdist_minkowski_wrap(XA, XB, indicesA, indicesB, indptr, dm, p=p)
        elif mstr in ["wminkowski", "wmi", "wm", "wpnorm"]:
            XA = _convert_to_double(XA)
            XB = _convert_to_double(XB)
            _ssdist_wrap.ssdist_weighted_minkowski_wrap(XA, XB, indicesA, indicesB, indptr, dm, p=p, w=w)
        elif mstr in ["seuclidean", "se", "s"]:
            XA = _convert_to_double(XA)
            XB = _convert_to_double(XB)
            _ssdist_wrap.ssdist_seuclidean_wrap(XA, XB, indicesA, indicesB, indptr, dm, V=V)
        elif mstr in ["cosine", "cos"]:
            XA = _convert_to_double(XA)
            XB = _convert_to_double(XB)
            _ssdist_cosine(XA, XB, indicesA, indicesB, indptr, dm)
        elif mstr in ["correlation", "co"]:
            XA = _convert_to_double(XA)
            XB = _convert_to_double(XB)
            XA, indicesA = _subset_array_if_is_parent(input_XA, XA, indicesA)
            XB, indicesB = _subset_array_if_is_parent(input_XB, XB, indicesB)
            XA -= XA.mean(axis=1)[:, np.newaxis]
            XB -= XB.mean(axis=1)[:, np.newaxis]
            _ssdist_cosine(XA, XB, indicesA, indicesB, indptr, dm)
        elif mstr in ["aitchison", "simplex", "clr"]:
            XA = _convert_to_double(XA)
            XB = _convert_to_double(XB)
            XA, indicesA = _subset_array_if_is_parent(input_XA, XA, indicesA)
            XB, indicesB = _subset_array_if_is_parent(input_XB, XB, indicesB)
            # check positive
            if not (XA >= 0).all():
                raise ValueError("All values in XA must be greater than or "
                                 "equal to zero.")
            if not (XB >= 0).all():
                raise ValueError("All values in XB must be greater than or "
                                 "equal to zero.")
            # check normalized
            sumA = XA.sum(axis=-1)
            sumB = XA.sum(axis=-1)
            if not np.allclose(sumA, 1.):
                XA /= sumA[:, np.newaxis]
            if not np.allclose(sumB, 1.):
                XB /= sumB[:, np.newaxis]
            XA = np.log(XA, out=XA)
            XB = np.log(XB, out=XB)
            XA -= XA.mean(axis=-1)[:, np.newaxis]
            XB -= XB.mean(axis=-1)[:, np.newaxis]
            _ssdist_wrap.ssdist_euclidean(XA, XB, indicesA, indicesB, indptr, dm)
        elif mstr in ["mahalanobis", "mahal", "mah"]:
            XA = _convert_to_double(XA)
            XB = _convert_to_double(XB)
            _ssdist_wrap.ssdist_mahalanobis_wrap(XA, XB, indicesA, indicesB,
                                                 indptr, dm, VI=VI)
        elif mstr in _string_names:
            XA = _convert_to_str(XA)
            XB = _convert_to_str(XB)
            _validate_string_dtypes(XA, XB, indicesA, indicesB)
            key = (mstr, normalized, "o" if XA.dtype == np.object_ else "s")
            if matrix_dtype is not None:
                key += ({int:"l", float:"d"}[matrix_dtype],)
            _ssdist_fn = _STRING_ssdist[key]
            args = (XA, XB, indicesA, indicesB, indptr, dm)
            kwargs = [gap_open, gap_ext, mat]
            args += tuple(kwarg for kwarg in kwargs if kwarg is not None)
            _ssdist_fn(*args)
        elif mstr.startswith("test_"):
            if mstr in _TEST_METRICS:
                kwargs = {"p":p, "w":w, "V":V, "VI":VI}
                dm = ssdist(XA, XB, indicesB, indicesA, _TEST_METRICS[mstr],
                              **kwargs)
            else:
                raise ValueError("Unknown \"Test\" Distance Metric: %s" % mstr[5:])
        else:
            raise ValueError("Unknown Distance Metric: %s" % mstr)
    else:
        raise TypeError("2nd argument metric must be a string identifier "
                        "or a function.")
    indices = input_indicesB - np.min(input_indicesB)
    return csr_matrix((dm, indices, indptr), copy=False)

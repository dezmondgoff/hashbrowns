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
from scipy.spatial import distance
from scipy.sparse import csr_matrix

import hashbrowns.distance._ssdist_wrap as _ssdist_wrap
import hashbrowns.distance._alignment_wrap as _alignment_wrap
from hashbrowns.distance.score_matrices import blosum62

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
    if a.dtype != np.object:
        if indices is None:
            return np.ascontiguousarray(a, dtype=np.bool), indices
        else:
            unique, indices = np.unique(indices, return_inverse=True)
            return np.ascontiguousarray(a[unique], dtype=str), indices
    return a, indices

def _validate_score_matrix(mat):
    if mat is None:
        mat = blosum62
    elif not isinstance(mat, dict):
        raise ValueError("Alignment matrix must be a dict object.")
    return mat

def _validate_alignment_args(gap_open, gap_ext, normalized):
    if gap_open is None:
        gap_open = 1
    if gap_ext is None:
        gap_ext = 1
    if normalized is None:
        normalized = False
    return gap_open, gap_ext, normalized

def _validate_preset_align_args(gap_open, gap_ext, normalized):
    if gap_open is None:
        gap_open = 1
    if gap_ext is None:
        gap_ext = 1
    if normalized is None:
        normalized = False
    return gap_open, gap_ext, normalized

def _validate_align_args(gap_open, gap_ext, smat, normalized):
    if gap_open is None:
        gap_open = 1
    if gap_ext is None:
        gap_ext = 1
    if smat is None:
        smat = blosum62
    elif not isinstance(smat, dict):
        raise ValueError("Alignment matrix must be a dict object.")
    if normalized is None:
        normalized = False
    return gap_open, gap_ext, smat, normalized

def _filter_deprecated_kwargs(**kwargs):
    for k in ["p", "V", "w", "VI", "gap_open", "gap_ext", "smat"]:
        kw = kwargs.pop(k, None)
        if kw is not None:
            warnings.warn("Got unexpected kwarg %s. This will raise an error"
                          " in a future version." % k, DeprecationWarning)

def _ssdist_cosine(XA, XB, indicesA, indicesB, indptr, ssdm):
    uniqueA, reverseA = np.unique(indicesA, return_inverse=True)
    uniqueB, reverseB = np.unique(indicesB, return_inverse=True)

    num = np.diff(indptr)
    np.einsum("ij, ij -> i", XA[indicesA].repeat(num),
              XB[indicesB].T, out=ssdm)

    normsA = distance._row_norms(XA[uniqueA])
    normsB = distance._row_norms(XB[uniqueB]).reshape(-1, 1)
    ssdm /= normsA[reverseA].repeat(num)
    ssdm /= normsB[reverseB]
    ssdm *= -1
    ssdm += 1

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

_METRICS_NAMES = ["braycurtis", "canberra", "chebyshev", "cityblock",
                  "correlation", "cosine", "dice", "euclidean", "hamming",
                  "jaccard", "kulsinski", "mahalanobis", "matching",
                  "minkowski", "rogerstanimoto", "russellrao", "seuclidean",
                  "sokalmichener", "sokalsneath", "sqeuclidean", "yule", "wminkowski"]

_TEST_METRICS = {"test_" + name: eval("distance." + name) for name in _METRICS_NAMES}

def ssdist(XA, XB, indicesB, indptr, indicesA=None, metric="euclidean", p=None,
           V=None, VI=None, w=None, gap_open=None, gap_ext=None, smat=None,
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

       Computes the standardized Euclidean distance. The standardized
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

    23. ``Y = cdist(XA, XB, indptr, f)``

       Computes the distance between all pairs of vectors in X
       using the user supplied 2-arity function f. For example,
       Euclidean distance between the vectors could be computed
       as follows::

         ssdm = cdist(XA, XB, indptr, lambda u, v: np.sqrt(((u-v)**2).sum()))

       Note that you should avoid passing a reference to one of
       the distance functions defined in this library. For example,::

         ssdm = cdist(XA, XB, indptr, sokalsneath)

       would calculate the pair-wise distances between the vectors in
       X using the Python function `sokalsneath`. This would result in
       sokalsneath being called :math:`{n \\choose 2}` times, which
       is inefficient. Instead, the optimized C version is more
       efficient, and we call it using the following syntax::

         ssdm = cdist(XA, XB, indptr, "sokalsneath")

    Examples
    --------
    Find the Euclidean distances between four 2-D coordinates:

    >>> from scipy.spatial import distance
    >>> coords = [(35.0456, -85.2672),
    ...           (35.1174, -89.9711),
    ...           (35.9728, -83.9422),
    ...           (36.1667, -86.7833)]
    >>> distance.cdist(coords, coords, "euclidean")
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
    >>> distance.cdist(a, b, "cityblock")
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
            print(indptr[-1], indicesB.shape)
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
        ssdm = np.zeros(mB, dtype=np.double)
    else:
        if not out.flags.contiguous:
            raise ValueError("Output array must be contiguous.")
        if out.shape != (mB,):
            raise ValueError("Output array has wrong dimension.")
        ssdm = out

    # validate input for multi-args metrics
    if(metric in ["minkowski", "mi", "m", "pnorm", "test_minkowski"] or
       metric == distance.minkowski):
        p = distance._validate_minkowski_args(p)
        _filter_deprecated_kwargs(w=w, V=V, VI=VI, gap_open=gap_open,
                                  gap_ext=gap_ext, smat=smat,
                                  normalized=normalized)
    elif(metric in ["wminkowski", "wmi", "wm", "wpnorm", "test_wminkowski"] or
         metric == distance.wminkowski):
        p, w = distance._validate_wminkowski_args(p, w)
        _filter_deprecated_kwargs(V=V, VI=VI, gap_open=gap_open,
                                           gap_ext=gap_ext, smat=smat,
                                           normalized=normalized)
    elif(metric in ["seuclidean", "se", "s", "test_seuclidean"] or
         metric == distance.seuclidean):
        V = distance._validate_seuclidean_args(np.vstack([XA, XB]), n, V)
        _filter_deprecated_kwargs(p=p, w=w, VI=VI, gap_open=gap_open,
                                  gap_ext=gap_ext, smat=smat,
                                  normalized=normalized)
    elif(metric in ["mahalanobis", "mahal", "mah", "test_mahalanobis"] or
         metric == distance.mahalanobis):
        VI = distance._validate_mahalanobis_args(np.vstack([XA, XB]), mA + mB,
                                                           n, VI)
        _filter_deprecated_kwargs(p=p, w=w, V=V, gap_open=gap_open,
                                           gap_ext=gap_ext, smat=smat,
                                           normalized=normalized)
    elif(metric in ["blosum", "pam", "blosum62", "pam250"]):
        gap_open, gap_ext = _validate_preset_align_args(gap_open,
                                                              gap_ext,
                                                              normalized)
        _filter_deprecated_kwargs(p=p, V=V, w=w, VI=VI, smat=smat)
    elif(metric in ["align", "alignment"]):
        gap_open, gap_ext. smat = _validate_align_args(gap_open, gap_ext, smat,
                                                      normalized)
        _filter_deprecated_kwargs(p=p, V=V, w=w, VI=VI)

    else:
        _filter_deprecated_kwargs(p=p, w=w, V=V, VI=VI,
                                           gap_open=gap_open, gap_ext=gap_ext,
                                           smat=smat)

    if callable(metric):
        # metrics that expects only doubles:
        if metric in [distance.braycurtis, distance.canberra, distance.chebyshev,
                      distance.cityblock, distance.correlation,
                      distance.cosine, distance.euclidean, distance.mahalanobis,
                      distance.minkowski, distance.sqeuclidean,
                      distance.seuclidean, distance.wminkowski]:
            XA, indicesA = _convert_to_double(XA)
            XB, indicesB = _convert_to_double(XB)
        # metrics that expects only bools:
        elif metric in [distance.dice, distance.kulsinski, distance.rogerstanimoto,
                        distance.russellrao, distance.sokalmichener, distance.sokalsneath,
                        distance.yule]:
            XA, indicesA = _convert_to_bool(XA)
            XB, indicesB = _convert_to_bool(XB)
        # metrics that may receive multiple types:
        elif metric in [distance.matching, distance.hamming, distance.jaccard]:
            if XA.dtype == bool:
                XA, indicesA = _convert_to_bool(XA)
                XB, indicesB = _convert_to_bool(XB)
            else:
                XA, indicesA = _convert_to_double(XA)
                XB, indicesB = _convert_to_double(XB)

        # metrics that expects multiple args
        if metric == distance.minkowski:
            metric = partial(distance.minkowski, p=p)
        elif metric == distance.wminkowski:
            metric = partial(distance.wminkowski, p=p, w=w)
        elif metric == distance.seuclidean:
            metric = partial(distance.seuclidean, V=V)
        elif metric == distance.mahalanobis:
            metric = partial(distance.mahalanobis, VI=VI)

        for i in indicesA:
            for j in xrange(indptr[i], indptr[i + 1]):
                ssdm[j] = metric(XA[i, :], XB[indicesB[j], :])

    elif isinstance(metric, string_types):
        mstr = metric.lower()

        try:
            validate, ssdist_fn = _SIMPLE_ssdist[mstr]
            XA, indicesA = validate(XA, indicesA)
            XB, indicesB = validate(XB, indicesB)
            ssdist_fn(XA, XB, indicesA, indicesB, indptr, ssdm)
            indices = input_indicesB - np.min(input_indicesB)
            return csr_matrix((ssdm, indices, indptr), copy=False)
        except KeyError:
            pass

        if mstr in ["matching", "hamming", "hamm", "ha", "h"]:
            if XA.dtype == bool:
                XA = _convert_to_bool(XA, indicesA)
                XB = _convert_to_bool(XB, indicesB)
                _ssdist_wrap.ssdist_hamming_bool_wrap(XA, XB, indicesA, indicesB,
                                                    indptr, ssdm)
            else:
                XA = _convert_to_double(XA)
                XB = _convert_to_double(XB)
                _ssdist_wrap.ssdist_hamming_wrap(XA, XB, indicesA, indicesB,
                                                 indptr, ssdm)
        elif mstr in ["jaccard", "jacc", "ja", "j"]:
            if XA.dtype == bool:
                XA = _convert_to_bool(XA)
                XB = _convert_to_bool(XB)
                _ssdist_wrap.ssdist_jaccard_bool_wrap(XA, XB, indicesA, indicesB, indptr, ssdm)
            else:
                XA = _convert_to_double(XA)
                XB = _convert_to_double(XB)
                _ssdist_wrap.ssdist_jaccard_wrap(XA, XB, indicesA, indicesB, indptr, ssdm)
        elif mstr in ["minkowski", "mi", "m", "pnorm"]:
            XA = _convert_to_double(XA)
            XB = _convert_to_double(XB)
            _ssdist_wrap.ssdist_minkowski_wrap(XA, XB, indicesA, indicesB, indptr, ssdm, p=p)
        elif mstr in ["wminkowski", "wmi", "wm", "wpnorm"]:
            XA = _convert_to_double(XA)
            XB = _convert_to_double(XB)
            _ssdist_wrap.ssdist_weighted_minkowski_wrap(XA, XB, indicesA, indicesB, indptr, ssdm, p=p, w=w)
        elif mstr in ["seuclidean", "se", "s"]:
            XA = _convert_to_double(XA)
            XB = _convert_to_double(XB)
            _ssdist_wrap.ssdist_seuclidean_wrap(XA, XB, indicesA, indicesB, indptr, ssdm, V=V)
        elif mstr in ["cosine", "cos"]:
            XA = _convert_to_double(XA)
            XB = _convert_to_double(XB)
            _ssdist_cosine(XA, XB, indicesA, indicesB, indptr, ssdm)
        elif mstr in ["correlation", "co"]:
            XA = _convert_to_double(XA)
            XB = _convert_to_double(XB)
            XA, indicesA = _subset_array_if_is_parent(input_XA, XA, indicesA)
            XB, indicesB = _subset_array_if_is_parent(input_XB, XB, indicesB)
            XA -= XA.mean(axis=1)[:, np.newaxis]
            XB -= XB.mean(axis=1)[:, np.newaxis]
            _ssdist_cosine(XA, XB, indicesA, indicesB, indptr, ssdm)
        elif mstr in ["simplex", "clr"]:
            XA = _convert_to_double(XA)
            XB = _convert_to_double(XB)
            XA, indicesA = _subset_array_if_is_parent(input_XA, XA, indicesA)
            XB, indicesB = _subset_array_if_is_parent(input_XB, XB, indicesB)
            # check positive
            if not np.all(XA >= 0):
                raise ValueError("Values in XA must be greater than or equal to zero.")
            if not np.all(XB >= 0):
                raise ValueError("Values in XB must be greater than or equal to zero.")
            # check normalized
            normsA = distance._row_norms(XA)
            normsB = distance._row_norms(XB)
            if not np.allclose(normsA, 1.):
                XA /= normsA[:, None]
            if not np.allclose(normsB, 1.):
                XB /= normsB[:, None]
            XA = np.log(XA)
            XB = np.log(XB)
            sumsA = np.sum(XA, axis=-1)
            sumsB = np.sum(XA, axis=-1)
            XA -= sumsA[:, None]
            XB -= sumsB[:, None]
            _ssdist_wrap.ssdist_euclidean(XA, XB, indicesA, indicesB, indptr,
                                          ssdm)
        elif mstr in ["mahalanobis", "mahal", "mah"]:
            XA = _convert_to_double(XA)
            XB = _convert_to_double(XB)
            _ssdist_wrap.ssdist_mahalanobis_wrap(XA, XB, indicesA, indicesB,
                                                 indptr, ssdm, VI=VI)
        elif mstr in ["blosum", "blosum62"]:
            XA = _convert_to_str(XA)
            XB = _convert_to_str(XB)
            args = (XA, XB, indicesA, indicesB, indptr, ssdm, gap_open, gap_ext)
            if normalized:
                _alignment_wrap.ssdist_normalized_blosum62_wrap(*args)
            else:
                _ssdist_wrap.ssdist_blosum_wrap(*args)
        elif mstr in ["pam", "pam250"]:
            XA = _convert_to_str(XA)
            XB = _convert_to_str(XB)
            args = (XA, XB, indicesA, indicesB, indptr, ssdm, gap_open, gap_ext)
            if normalized:
                _alignment_wrap.ssdist_normalized_pam250_wrap(*args)
            else:
                _alignment_wrap.ssdist_pam250_wrap(*args)
        elif mstr in ["edit", "levenshtein"]:
            XA = _convert_to_str(XA)
            XB = _convert_to_str(XB)
            args = (XA, XB, indicesA, indicesB, indptr, ssdm, gap_open, gap_ext)
            if normalized:
                _alignment_wrap.ssdist_normalized_levenshtein_wrap(*args)
            else:
                _alignment_wrap.ssdist_levenshtein_wrap(*args)
        elif mstr in ["align", "alignment"]:
            XA = _convert_to_str(XA)
            XB = _convert_to_str(XB)
            args = (XA, XB, indicesA, indicesB, indptr, ssdm, gap_open, gap_ext,
                    smat)
            if normalized:
                _alignment_wrap.ssdist_normalized_alignment_wrap(*args)
            else:
                _alignment_wrap.ssdist_alignment_wrap(*args)
        elif mstr.startswith("test_"):
            if mstr in _TEST_METRICS:
                kwargs = {"p":p, "w":w, "V":V, "VI":VI}
                ssdm = ssdist(XA, XB, indicesB, indicesA, _TEST_METRICS[mstr],
                              **kwargs)
            else:
                raise ValueError("Unknown \"Test\" Distance Metric: %s" % mstr[5:])
        else:
            raise ValueError("Unknown Distance Metric: %s" % mstr)
    else:
        raise TypeError("2nd argument metric must be a string identifier "
                        "or a function.")
    indices = input_indicesB - np.min(input_indicesB)
    return csr_matrix((ssdm, indices, indptr), copy=False)

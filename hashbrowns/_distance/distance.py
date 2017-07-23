"""
=====================================================
Distance computations (:mod:`scipy.spatial.distance`)
=====================================================

.. sectionauthor:: Damian Eads

Function Reference
------------------

Distance matrix computation from a collection of raw observation vectors
stored in a rectangular array.

.. autosummary::
   :toctree: generated/

   pdist   -- pairwise distances between observation vectors.
   cdist   -- distances between two collections of observation vectors
   squareform -- convert distance matrix to a condensed one and vice versa
   directed_hausdorff -- directed Hausdorff distance between arrays

Predicates for checking the validity of distance matrices, both
condensed and redundant. Also contained in this module are functions
for computing the number of observations in a distance matrix.

.. autosummary::
   :toctree: generated/

   is_valid_dm -- checks for a valid distance matrix
   is_valid_y  -- checks for a valid condensed distance matrix
   num_obs_dm  -- # of observations in a distance matrix
   num_obs_y   -- # of observations in a condensed distance matrix

Distance functions between two numeric vectors ``u`` and ``v``. Computing
distances over a large collection of vectors is inefficient for these
functions. Use ``pdist`` for this purpose.

.. autosummary::
   :toctree: generated/

   braycurtis       -- the Bray-Curtis distance.
   canberra         -- the Canberra distance.
   chebyshev        -- the Chebyshev distance.
   cityblock        -- the Manhattan distance.
   correlation      -- the Correlation distance.
   cosine           -- the Cosine distance.
   euclidean        -- the Euclidean distance.
   mahalanobis      -- the Mahalanobis distance.
   minkowski        -- the Minkowski distance.
   seuclidean       -- the normalized Euclidean distance.
   sqeuclidean      -- the squared Euclidean distance.
   wminkowski       -- the weighted Minkowski distance.

Distance functions between two boolean vectors (representing sets) ``u`` and
``v``.  As in the case of numerical vectors, ``pdist`` is more efficient for
computing the distances between all pairs.

.. autosummary::
   :toctree: generated/

   dice             -- the Dice dissimilarity.
   hamming          -- the Hamming distance.
   jaccard          -- the Jaccard distance.
   kulsinski        -- the Kulsinski distance.
   matching         -- the matching dissimilarity.
   rogerstanimoto   -- the Rogers-Tanimoto dissimilarity.
   russellrao       -- the Russell-Rao dissimilarity.
   sokalmichener    -- the Sokal-Michener dissimilarity.
   sokalsneath      -- the Sokal-Sneath dissimilarity.
   yule             -- the Yule dissimilarity.

:func:`hamming` also operates over discrete numerical vectors.
"""

# Copyright (C) Damian Eads, 2007-2008. New BSD License.

from __future__ import division, print_function, absolute_import

__all__ = [
    'braycurtis',
    'canberra',
    'cdist',
    'chebyshev',
    'cityblock',
    'correlation',
    'cosine',
    'dice',
    'directed_hausdorff',
    'euclidean',
    'hamming',
    'is_valid_dm',
    'is_valid_y',
    'jaccard',
    'kulsinski',
    'mahalanobis',
    'matching',
    'minkowski',
    'num_obs_dm',
    'num_obs_y',
    'pdist',
    'rogerstanimoto',
    'russellrao',
    'seuclidean',
    'sokalmichener',
    'sokalsneath',
    'sqeuclidean',
    'squareform',
    'wminkowski',
    'yule'
]


import warnings
import numpy as np

from functools import partial
from scipy._lib.six import callable, string_types
from scipy._lib.six import xrange

from scipy.spatial.distance import _distance_wrap, _hausdorff
from scipy.linalg import norm

def _copy_array_if_base_present(a):
    """
    Copies the array if its base points to a parent array.
    """
    if a.base is not None:
        return a.copy()
    return a


def _convert_to_bool(X):
    return np.ascontiguousarray(X, dtype=bool)


def _convert_to_double(X):
    return np.ascontiguousarray(X, dtype=np.double)


def _filter_deprecated_kwargs(**kwargs):
    # Filtering out old default keywords
    for k in ["p", "V", "w", "VI"]:
        kw = kwargs.pop(k, None)
        if kw is not None:
            warnings.warn('Got unexpected kwarg %s. This will raise an error'
                          ' in a future version.' % k, DeprecationWarning)


def _nbool_correspond_all(u, v):
    if u.dtype != v.dtype:
        raise TypeError("Arrays being compared must be of the same data type.")

    if u.dtype == int or u.dtype == np.float_ or u.dtype == np.double:
        not_u = 1.0 - u
        not_v = 1.0 - v
        nff = (not_u * not_v).sum()
        nft = (not_u * v).sum()
        ntf = (u * not_v).sum()
        ntt = (u * v).sum()
    elif u.dtype == bool:
        not_u = ~u
        not_v = ~v
        nff = (not_u & not_v).sum()
        nft = (not_u & v).sum()
        ntf = (u & not_v).sum()
        ntt = (u & v).sum()
    else:
        raise TypeError("Arrays being compared have unknown type.")

    return (nff, nft, ntf, ntt)


def _nbool_correspond_ft_tf(u, v):
    if u.dtype == int or u.dtype == np.float_ or u.dtype == np.double:
        not_u = 1.0 - u
        not_v = 1.0 - v
        nft = (not_u * v).sum()
        ntf = (u * not_v).sum()
    else:
        not_u = ~u
        not_v = ~v
        nft = (not_u & v).sum()
        ntf = (u & not_v).sum()
    return (nft, ntf)


def _validate_mahalanobis_args(X, m, n, VI):
    if VI is None:
        if m <= n:
            # There are fewer observations than the dimension of
            # the observations.
            raise ValueError("The number of observations (%d) is too "
                             "small; the covariance matrix is "
                             "singular. For observations with %d "
                             "dimensions, at least %d observations "
                             "are required." % (m, n, n + 1))
        CV = np.atleast_2d(np.cov(X.astype(np.double).T))
        VI = np.linalg.inv(CV).T.copy()
    VI = _copy_array_if_base_present(_convert_to_double(VI))
    return VI


def _validate_minkowski_args(p):
    if p is None:
        p = 2.
    return p


def _validate_seuclidean_args(X, n, V):
    if V is None:
        V = np.var(X.astype(np.double), axis=0, ddof=1)
    else:
        V = np.asarray(V, order='c')
        if V.dtype != np.double:
            raise TypeError('Variance vector V must contain doubles.')
        if len(V.shape) != 1:
            raise ValueError('Variance vector V must '
                             'be one-dimensional.')
        if V.shape[0] != n:
            raise ValueError('Variance vector V must be of the same '
                             'dimension as the vectors on which the distances '
                             'are computed.')
    return _convert_to_double(V)


def _validate_vector(u, dtype=None):
    # XXX Is order='c' really necessary?
    u = np.asarray(u, dtype=dtype, order='c').squeeze()
    # Ensure values such as u=1 and u=[1] still return 1-D arrays.
    u = np.atleast_1d(u)
    if u.ndim > 1:
        raise ValueError("Input vector should be 1-D.")
    return u


def _validate_wminkowski_args(p, w):
    if w is None:
        raise ValueError('weighted minkowski requires a weight '
                         'vector `w` to be given.')
    w = _convert_to_double(w)
    if p is None:
        p = 2.
    return p, w


def directed_hausdorff(u, v, seed=0):
    """
    Computes the directed Hausdorff distance between two N-D arrays.

    Distances between pairs are calculated using a Euclidean metric.

    Parameters
    ----------
    u : (M,N) ndarray
        Input array.
    v : (O,N) ndarray
        Input array.
    seed : int or None
        Local `np.random.RandomState` seed. Default is 0, a random shuffling of
        u and v that guarantees reproducibility.

    Returns
    -------
    d : double
        The directed Hausdorff distance between arrays `u` and `v`,

    index_1 : int
        index of point contributing to Hausdorff pair in `u`

    index_2 : int
        index of point contributing to Hausdorff pair in `v`

    Notes
    -----
    Uses the early break technique and the random sampling approach
    described by [1]_. Although worst-case performance is ``O(m * o)``
    (as with the brute force algorithm), this is unlikely in practice
    as the input data would have to require the algorithm to explore
    every single point interaction, and after the algorithm shuffles
    the input points at that. The best case performance is O(m), which
    is satisfied by selecting an inner loop distance that is less than
    cmax and leads to an early break as often as possible. The authors
    have formally shown that the average runtime is closer to O(m).

    .. versionadded:: 0.19.0

    References
    ----------
    .. [1] A. A. Taha and A. Hanbury, "An efficient algorithm for
           calculating the exact Hausdorff distance." IEEE Transactions On
           Pattern Analysis And Machine Intelligence, vol. 37 pp. 2153-63,
           2015.

    See Also
    --------
    scipy.spatial.procrustes : Another similarity test for two data sets

    Examples
    --------
    Find the directed Hausdorff distance between two 2-D arrays of
    coordinates:

    >>> from scipy.spatial.distance import directed_hausdorff
    >>> u = np.array([(1.0, 0.0),
    ...               (0.0, 1.0),
    ...               (-1.0, 0.0),
    ...               (0.0, -1.0)])
    >>> v = np.array([(2.0, 0.0),
    ...               (0.0, 2.0),
    ...               (-2.0, 0.0),
    ...               (0.0, -4.0)])

    >>> directed_hausdorff(u, v)[0]
    2.23606797749979
    >>> directed_hausdorff(v, u)[0]
    3.0

    Find the general (symmetric) Hausdorff distance between two 2-D
    arrays of coordinates:

    >>> max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
    3.0

    Find the indices of the points that generate the Hausdorff distance
    (the Hausdorff pair):

    >>> directed_hausdorff(v, u)[1:]
    (3, 3)

    """
    u = np.asarray(u, dtype=np.float64, order='c')
    v = np.asarray(v, dtype=np.float64, order='c')
    result = _hausdorff.directed_hausdorff(u, v, seed)
    return result


def minkowski(u, v, p):
    """
    Computes the Minkowski distance between two 1-D arrays.

    The Minkowski distance between 1-D arrays `u` and `v`,
    is defined as

    .. math::

       {||u-v||}_p = (\\sum{|u_i - v_i|^p})^{1/p}.

    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.
    p : int
        The order of the norm of the difference :math:`{||u-v||}_p`.

    Returns
    -------
    d : double
        The Minkowski distance between vectors `u` and `v`.

    """
    u = _validate_vector(u)
    v = _validate_vector(v)
    if p < 1:
        raise ValueError("p must be at least 1")
    dist = norm(u - v, ord=p)
    return dist


def wminkowski(u, v, p, w):
    """
    Computes the weighted Minkowski distance between two 1-D arrays.

    The weighted Minkowski distance between `u` and `v`, defined as

    .. math::

       \\left(\\sum{(|w_i (u_i - v_i)|^p)}\\right)^{1/p}.

    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.
    p : int
        The order of the norm of the difference :math:`{||u-v||}_p`.
    w : (N,) array_like
        The weight vector.

    Returns
    -------
    wminkowski : double
        The weighted Minkowski distance between vectors `u` and `v`.

    """
    u = _validate_vector(u)
    v = _validate_vector(v)
    w = _validate_vector(w)
    if p < 1:
        raise ValueError("p must be at least 1")
    dist = norm(w * (u - v), ord=p)
    return dist


def euclidean(u, v):
    """
    Computes the Euclidean distance between two 1-D arrays.

    The Euclidean distance between 1-D arrays `u` and `v`, is defined as

    .. math::

       {||u-v||}_2

    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.

    Returns
    -------
    euclidean : double
        The Euclidean distance between vectors `u` and `v`.

    """
    u = _validate_vector(u)
    v = _validate_vector(v)
    dist = norm(u - v)
    return dist


def sqeuclidean(u, v):
    """
    Computes the squared Euclidean distance between two 1-D arrays.

    The squared Euclidean distance between `u` and `v` is defined as

    .. math::

       {||u-v||}_2^2.


    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.

    Returns
    -------
    sqeuclidean : double
        The squared Euclidean distance between vectors `u` and `v`.

    """
    # Preserve float dtypes, but convert everything else to np.float64
    # for stability.
    utype, vtype = None, None
    if not (hasattr(u, "dtype") and np.issubdtype(u.dtype, np.inexact)):
        utype = np.float64
    if not (hasattr(v, "dtype") and np.issubdtype(v.dtype, np.inexact)):
        vtype = np.float64

    u = _validate_vector(u, dtype=utype)
    v = _validate_vector(v, dtype=vtype)
    u_v = u - v

    return np.dot(u_v, u_v)


def cosine(u, v):
    """
    Computes the Cosine distance between 1-D arrays.

    The Cosine distance between `u` and `v`, is defined as

    .. math::

       1 - \\frac{u \\cdot v}
                {||u||_2 ||v||_2}.

    where :math:`u \\cdot v` is the dot product of :math:`u` and
    :math:`v`.

    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.

    Returns
    -------
    cosine : double
        The Cosine distance between vectors `u` and `v`.

    """
    u = _validate_vector(u)
    v = _validate_vector(v)
    dist = 1.0 - np.dot(u, v) / (norm(u) * norm(v))
    return dist


def correlation(u, v):
    """
    Computes the correlation distance between two 1-D arrays.

    The correlation distance between `u` and `v`, is
    defined as

    .. math::

       1 - \\frac{(u - \\bar{u}) \\cdot (v - \\bar{v})}
               {{||(u - \\bar{u})||}_2 {||(v - \\bar{v})||}_2}

    where :math:`\\bar{u}` is the mean of the elements of `u`
    and :math:`x \\cdot y` is the dot product of :math:`x` and :math:`y`.

    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.

    Returns
    -------
    correlation : double
        The correlation distance between 1-D array `u` and `v`.

    """
    u = _validate_vector(u)
    v = _validate_vector(v)
    umu = u.mean()
    vmu = v.mean()
    um = u - umu
    vm = v - vmu
    dist = 1.0 - np.dot(um, vm) / (norm(um) * norm(vm))
    return dist


def hamming(u, v):
    """
    Computes the Hamming distance between two 1-D arrays.

    The Hamming distance between 1-D arrays `u` and `v`, is simply the
    proportion of disagreeing components in `u` and `v`. If `u` and `v` are
    boolean vectors, the Hamming distance is

    .. math::

       \\frac{c_{01} + c_{10}}{n}

    where :math:`c_{ij}` is the number of occurrences of
    :math:`\\mathtt{u[k]} = i` and :math:`\\mathtt{v[k]} = j` for
    :math:`k < n`.

    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.

    Returns
    -------
    hamming : double
        The Hamming distance between vectors `u` and `v`.

    """
    u = _validate_vector(u)
    v = _validate_vector(v)
    if u.shape != v.shape:
        raise ValueError('The 1d arrays must have equal lengths.')
    return (u != v).mean()


def jaccard(u, v):
    """
    Computes the Jaccard-Needham dissimilarity between two boolean 1-D arrays.

    The Jaccard-Needham dissimilarity between 1-D boolean arrays `u` and `v`,
    is defined as

    .. math::

       \\frac{c_{TF} + c_{FT}}
            {c_{TT} + c_{FT} + c_{TF}}

    where :math:`c_{ij}` is the number of occurrences of
    :math:`\\mathtt{u[k]} = i` and :math:`\\mathtt{v[k]} = j` for
    :math:`k < n`.

    Parameters
    ----------
    u : (N,) array_like, bool
        Input array.
    v : (N,) array_like, bool
        Input array.

    Returns
    -------
    jaccard : double
        The Jaccard distance between vectors `u` and `v`.

    """
    u = _validate_vector(u)
    v = _validate_vector(v)
    dist = (np.double(np.bitwise_and((u != v),
                                     np.bitwise_or(u != 0, v != 0)).sum()) /
            np.double(np.bitwise_or(u != 0, v != 0).sum()))
    return dist


def kulsinski(u, v):
    """
    Computes the Kulsinski dissimilarity between two boolean 1-D arrays.

    The Kulsinski dissimilarity between two boolean 1-D arrays `u` and `v`,
    is defined as

    .. math::

         \\frac{c_{TF} + c_{FT} - c_{TT} + n}
              {c_{FT} + c_{TF} + n}

    where :math:`c_{ij}` is the number of occurrences of
    :math:`\\mathtt{u[k]} = i` and :math:`\\mathtt{v[k]} = j` for
    :math:`k < n`.

    Parameters
    ----------
    u : (N,) array_like, bool
        Input array.
    v : (N,) array_like, bool
        Input array.

    Returns
    -------
    kulsinski : double
        The Kulsinski distance between vectors `u` and `v`.

    """
    u = _validate_vector(u)
    v = _validate_vector(v)
    n = float(len(u))
    (nff, nft, ntf, ntt) = _nbool_correspond_all(u, v)

    return (ntf + nft - ntt + n) / (ntf + nft + n)


def seuclidean(u, v, V):
    """
    Returns the standardized Euclidean distance between two 1-D arrays.

    The standardized Euclidean distance between `u` and `v`.

    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.
    V : (N,) array_like
        `V` is an 1-D array of component variances. It is usually computed
        among a larger collection vectors.

    Returns
    -------
    seuclidean : double
        The standardized Euclidean distance between vectors `u` and `v`.

    """
    u = _validate_vector(u)
    v = _validate_vector(v)
    V = _validate_vector(V, dtype=np.float64)
    if V.shape[0] != u.shape[0] or u.shape[0] != v.shape[0]:
        raise TypeError('V must be a 1-D array of the same dimension '
                        'as u and v.')
    return np.sqrt(((u - v) ** 2 / V).sum())


def cityblock(u, v):
    """
    Computes the City Block (Manhattan) distance.

    Computes the Manhattan distance between two 1-D arrays `u` and `v`,
    which is defined as

    .. math::

       \\sum_i {\\left| u_i - v_i \\right|}.

    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.

    Returns
    -------
    cityblock : double
        The City Block (Manhattan) distance between vectors `u` and `v`.

    """
    u = _validate_vector(u)
    v = _validate_vector(v)
    return abs(u - v).sum()


def mahalanobis(u, v, VI):
    """
    Computes the Mahalanobis distance between two 1-D arrays.

    The Mahalanobis distance between 1-D arrays `u` and `v`, is defined as

    .. math::

       \\sqrt{ (u-v) V^{-1} (u-v)^T }

    where ``V`` is the covariance matrix.  Note that the argument `VI`
    is the inverse of ``V``.

    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.
    VI : ndarray
        The inverse of the covariance matrix.

    Returns
    -------
    mahalanobis : double
        The Mahalanobis distance between vectors `u` and `v`.

    """
    u = _validate_vector(u)
    v = _validate_vector(v)
    VI = np.atleast_2d(VI)
    delta = u - v
    m = np.dot(np.dot(delta, VI), delta)
    return np.sqrt(m)


def chebyshev(u, v):
    """
    Computes the Chebyshev distance.

    Computes the Chebyshev distance between two 1-D arrays `u` and `v`,
    which is defined as

    .. math::

       \\max_i {|u_i-v_i|}.

    Parameters
    ----------
    u : (N,) array_like
        Input vector.
    v : (N,) array_like
        Input vector.

    Returns
    -------
    chebyshev : double
        The Chebyshev distance between vectors `u` and `v`.

    """
    u = _validate_vector(u)
    v = _validate_vector(v)
    return max(abs(u - v))


def braycurtis(u, v):
    """
    Computes the Bray-Curtis distance between two 1-D arrays.

    Bray-Curtis distance is defined as

    .. math::

       \\sum{|u_i-v_i|} / \\sum{|u_i+v_i|}

    The Bray-Curtis distance is in the range [0, 1] if all coordinates are
    positive, and is undefined if the inputs are of length zero.

    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.

    Returns
    -------
    braycurtis : double
        The Bray-Curtis distance between 1-D arrays `u` and `v`.

    """
    u = _validate_vector(u)
    v = _validate_vector(v, dtype=np.float64)
    return abs(u - v).sum() / abs(u + v).sum()


def canberra(u, v):
    """
    Computes the Canberra distance between two 1-D arrays.

    The Canberra distance is defined as

    .. math::

         d(u,v) = \\sum_i \\frac{|u_i-v_i|}
                              {|u_i|+|v_i|}.

    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.

    Returns
    -------
    canberra : double
        The Canberra distance between vectors `u` and `v`.

    Notes
    -----
    When `u[i]` and `v[i]` are 0 for given i, then the fraction 0/0 = 0 is
    used in the calculation.

    """
    u = _validate_vector(u)
    v = _validate_vector(v, dtype=np.float64)
    olderr = np.seterr(invalid='ignore')
    try:
        d = np.nansum(abs(u - v) / (abs(u) + abs(v)))
    finally:
        np.seterr(**olderr)
    return d


def yule(u, v):
    """
    Computes the Yule dissimilarity between two boolean 1-D arrays.

    The Yule dissimilarity is defined as

    .. math::

         \\frac{R}{c_{TT} * c_{FF} + \\frac{R}{2}}

    where :math:`c_{ij}` is the number of occurrences of
    :math:`\\mathtt{u[k]} = i` and :math:`\\mathtt{v[k]} = j` for
    :math:`k < n` and :math:`R = 2.0 * c_{TF} * c_{FT}`.

    Parameters
    ----------
    u : (N,) array_like, bool
        Input array.
    v : (N,) array_like, bool
        Input array.

    Returns
    -------
    yule : double
        The Yule dissimilarity between vectors `u` and `v`.

    """
    u = _validate_vector(u)
    v = _validate_vector(v)
    (nff, nft, ntf, ntt) = _nbool_correspond_all(u, v)
    return float(2.0 * ntf * nft) / float(ntt * nff + ntf * nft)


def matching(u, v):
    """
    Computes the Hamming distance between two boolean 1-D arrays.

    This is a deprecated synonym for :func:`hamming`.
    """
    return hamming(u, v)


def dice(u, v):
    """
    Computes the Dice dissimilarity between two boolean 1-D arrays.

    The Dice dissimilarity between `u` and `v`, is

    .. math::

         \\frac{c_{TF} + c_{FT}}
              {2c_{TT} + c_{FT} + c_{TF}}

    where :math:`c_{ij}` is the number of occurrences of
    :math:`\\mathtt{u[k]} = i` and :math:`\\mathtt{v[k]} = j` for
    :math:`k < n`.

    Parameters
    ----------
    u : (N,) ndarray, bool
        Input 1-D array.
    v : (N,) ndarray, bool
        Input 1-D array.

    Returns
    -------
    dice : double
        The Dice dissimilarity between 1-D arrays `u` and `v`.

    """
    u = _validate_vector(u)
    v = _validate_vector(v)
    if u.dtype == bool:
        ntt = (u & v).sum()
    else:
        ntt = (u * v).sum()
    (nft, ntf) = _nbool_correspond_ft_tf(u, v)
    return float(ntf + nft) / float(2.0 * ntt + ntf + nft)


def rogerstanimoto(u, v):
    """
    Computes the Rogers-Tanimoto dissimilarity between two boolean 1-D arrays.

    The Rogers-Tanimoto dissimilarity between two boolean 1-D arrays
    `u` and `v`, is defined as

    .. math::
       \\frac{R}
            {c_{TT} + c_{FF} + R}

    where :math:`c_{ij}` is the number of occurrences of
    :math:`\\mathtt{u[k]} = i` and :math:`\\mathtt{v[k]} = j` for
    :math:`k < n` and :math:`R = 2(c_{TF} + c_{FT})`.

    Parameters
    ----------
    u : (N,) array_like, bool
        Input array.
    v : (N,) array_like, bool
        Input array.

    Returns
    -------
    rogerstanimoto : double
        The Rogers-Tanimoto dissimilarity between vectors
        `u` and `v`.

    """
    u = _validate_vector(u)
    v = _validate_vector(v)
    (nff, nft, ntf, ntt) = _nbool_correspond_all(u, v)
    return float(2.0 * (ntf + nft)) / float(ntt + nff + (2.0 * (ntf + nft)))


def russellrao(u, v):
    """
    Computes the Russell-Rao dissimilarity between two boolean 1-D arrays.

    The Russell-Rao dissimilarity between two boolean 1-D arrays, `u` and
    `v`, is defined as

    .. math::

      \\frac{n - c_{TT}}
           {n}

    where :math:`c_{ij}` is the number of occurrences of
    :math:`\\mathtt{u[k]} = i` and :math:`\\mathtt{v[k]} = j` for
    :math:`k < n`.

    Parameters
    ----------
    u : (N,) array_like, bool
        Input array.
    v : (N,) array_like, bool
        Input array.

    Returns
    -------
    russellrao : double
        The Russell-Rao dissimilarity between vectors `u` and `v`.

    """
    u = _validate_vector(u)
    v = _validate_vector(v)
    if u.dtype == bool:
        ntt = (u & v).sum()
    else:
        ntt = (u * v).sum()
    return float(len(u) - ntt) / float(len(u))


def sokalmichener(u, v):
    """
    Computes the Sokal-Michener dissimilarity between two boolean 1-D arrays.

    The Sokal-Michener dissimilarity between boolean 1-D arrays `u` and `v`,
    is defined as

    .. math::

       \\frac{R}
            {S + R}

    where :math:`c_{ij}` is the number of occurrences of
    :math:`\\mathtt{u[k]} = i` and :math:`\\mathtt{v[k]} = j` for
    :math:`k < n`, :math:`R = 2 * (c_{TF} + c_{FT})` and
    :math:`S = c_{FF} + c_{TT}`.

    Parameters
    ----------
    u : (N,) array_like, bool
        Input array.
    v : (N,) array_like, bool
        Input array.

    Returns
    -------
    sokalmichener : double
        The Sokal-Michener dissimilarity between vectors `u` and `v`.

    """
    u = _validate_vector(u)
    v = _validate_vector(v)
    if u.dtype == bool:
        ntt = (u & v).sum()
        nff = (~u & ~v).sum()
    else:
        ntt = (u * v).sum()
        nff = ((1.0 - u) * (1.0 - v)).sum()
    (nft, ntf) = _nbool_correspond_ft_tf(u, v)
    return float(2.0 * (ntf + nft)) / float(ntt + nff + 2.0 * (ntf + nft))


def sokalsneath(u, v):
    """
    Computes the Sokal-Sneath dissimilarity between two boolean 1-D arrays.

    The Sokal-Sneath dissimilarity between `u` and `v`,

    .. math::

       \\frac{R}
            {c_{TT} + R}

    where :math:`c_{ij}` is the number of occurrences of
    :math:`\\mathtt{u[k]} = i` and :math:`\\mathtt{v[k]} = j` for
    :math:`k < n` and :math:`R = 2(c_{TF} + c_{FT})`.

    Parameters
    ----------
    u : (N,) array_like, bool
        Input array.
    v : (N,) array_like, bool
        Input array.

    Returns
    -------
    sokalsneath : double
        The Sokal-Sneath dissimilarity between vectors `u` and `v`.

    """
    u = _validate_vector(u)
    v = _validate_vector(v)
    if u.dtype == bool:
        ntt = (u & v).sum()
    else:
        ntt = (u * v).sum()
    (nft, ntf) = _nbool_correspond_ft_tf(u, v)
    denom = ntt + 2.0 * (ntf + nft)
    if denom == 0:
        raise ValueError('Sokal-Sneath dissimilarity is not defined for '
                         'vectors that are entirely false.')
    return float(2.0 * (ntf + nft)) / denom


# Registry of "simple" distance metrics' pdist and cdist implementations,
# meaning the ones that accept one dtype and have no additional arguments.
_SIMPLE_CDIST = {}
_SIMPLE_PDIST = {}

for wrap_name, names, typ in [
    ("bray_curtis", ['braycurtis'], "double"),
    ("canberra", ['canberra'], "double"),
    ("chebyshev", ['chebychev', 'chebyshev', 'cheby', 'cheb', 'ch'], "double"),
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
    cdist_fn = getattr(_distance_wrap, "cdist_%s" % fn_name)
    pdist_fn = getattr(_distance_wrap, "pdist_%s" % fn_name)
    for name in names:
        _SIMPLE_CDIST[name] = converter, cdist_fn
        _SIMPLE_PDIST[name] = converter, pdist_fn

_METRICS_NAMES = ['braycurtis', 'canberra', 'chebyshev', 'cityblock',
                  'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
                  'jaccard', 'kulsinski', 'mahalanobis', 'matching',
                  'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
                  'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule', 'wminkowski']

_TEST_METRICS = {'test_' + name: eval(name) for name in _METRICS_NAMES}

def _cosine_cdist(XA, XB, dm):
    XA = _convert_to_double(XA)
    XB = _convert_to_double(XB)

    np.dot(XA, XB.T, out=dm)

    dm /= _row_norms(XA).reshape(-1, 1)
    dm /= _row_norms(XB)
    dm *= -1
    dm += 1


def cdist(XA, XB, metric='euclidean', p=None, V=None, VI=None, w=None, 
          out=None):
    """
    Computes distance between each pair of the two collections of inputs.

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
    metric : str or callable, optional
        The distance metric to use.  If a string, the distance function can be
        'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
        'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski',
        'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao',
        'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
        'wminkowski', 'yule'.
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
    out : ndarray, optional
        If not None, the distance matrix Y is stored in this array

    Returns
    -------
    Y : ndarray
        A :math:`m_A` by :math:`m_B` distance matrix is returned.
        For each :math:`i` and :math:`j`, the metric
        ``dist(u=XA[i], v=XB[j])`` is computed and stored in the
        :math:`ij` th entry.

    Raises
    ------
    ValueError
        An exception is thrown if `XA` and `XB` do not have
        the same number of columns.

    Notes
    -----
    The following are common calling conventions:

    1. ``Y = cdist(XA, XB, 'euclidean')``

       Computes the distance between :math:`m` points using
       Euclidean distance (2-norm) as the distance metric between the
       points. The points are arranged as :math:`m`
       :math:`n`-dimensional row vectors in the matrix X.

    2. ``Y = cdist(XA, XB, 'minkowski', p)``

       Computes the distances using the Minkowski distance
       :math:`||u-v||_p` (:math:`p`-norm) where :math:`p \\geq 1`.

    3. ``Y = cdist(XA, XB, 'cityblock')``

       Computes the city block or Manhattan distance between the
       points.

    4. ``Y = cdist(XA, XB, 'seuclidean', V=None)``

       Computes the standardized Euclidean distance. The standardized
       Euclidean distance between two n-vectors ``u`` and ``v`` is

       .. math::

          \\sqrt{\\sum {(u_i-v_i)^2 / V[x_i]}}.

       V is the variance vector; V[i] is the variance computed over all
       the i'th components of the points. If not passed, it is
       automatically computed.

    5. ``Y = cdist(XA, XB, 'sqeuclidean')``

       Computes the squared Euclidean distance :math:`||u-v||_2^2` between
       the vectors.

    6. ``Y = cdist(XA, XB, 'cosine')``

       Computes the cosine distance between vectors u and v,

       .. math::

          1 - \\frac{u \\cdot v}
                   {{||u||}_2 {||v||}_2}

       where :math:`||*||_2` is the 2-norm of its argument ``*``, and
       :math:`u \\cdot v` is the dot product of :math:`u` and :math:`v`.

    7. ``Y = cdist(XA, XB, 'correlation')``

       Computes the correlation distance between vectors u and v. This is

       .. math::

          1 - \\frac{(u - \\bar{u}) \\cdot (v - \\bar{v})}
                   {{||(u - \\bar{u})||}_2 {||(v - \\bar{v})||}_2}

       where :math:`\\bar{v}` is the mean of the elements of vector v,
       and :math:`x \\cdot y` is the dot product of :math:`x` and :math:`y`.


    8. ``Y = cdist(XA, XB, 'hamming')``

       Computes the normalized Hamming distance, or the proportion of
       those vector elements between two n-vectors ``u`` and ``v``
       which disagree. To save memory, the matrix ``X`` can be of type
       boolean.

    9. ``Y = cdist(XA, XB, 'jaccard')``

       Computes the Jaccard distance between the points. Given two
       vectors, ``u`` and ``v``, the Jaccard distance is the
       proportion of those elements ``u[i]`` and ``v[i]`` that
       disagree where at least one of them is non-zero.

    10. ``Y = cdist(XA, XB, 'chebyshev')``

       Computes the Chebyshev distance between the points. The
       Chebyshev distance between two n-vectors ``u`` and ``v`` is the
       maximum norm-1 distance between their respective elements. More
       precisely, the distance is given by

       .. math::

          d(u,v) = \\max_i {|u_i-v_i|}.

    11. ``Y = cdist(XA, XB, 'canberra')``

       Computes the Canberra distance between the points. The
       Canberra distance between two points ``u`` and ``v`` is

       .. math::

         d(u,v) = \\sum_i \\frac{|u_i-v_i|}
                              {|u_i|+|v_i|}.

    12. ``Y = cdist(XA, XB, 'braycurtis')``

       Computes the Bray-Curtis distance between the points. The
       Bray-Curtis distance between two points ``u`` and ``v`` is


       .. math::

            d(u,v) = \\frac{\\sum_i (|u_i-v_i|)}
                          {\\sum_i (|u_i+v_i|)}

    13. ``Y = cdist(XA, XB, 'mahalanobis', VI=None)``

       Computes the Mahalanobis distance between the points. The
       Mahalanobis distance between two points ``u`` and ``v`` is
       :math:`\\sqrt{(u-v)(1/V)(u-v)^T}` where :math:`(1/V)` (the ``VI``
       variable) is the inverse covariance. If ``VI`` is not None,
       ``VI`` will be used as the inverse covariance matrix.

    14. ``Y = cdist(XA, XB, 'yule')``

       Computes the Yule distance between the boolean
       vectors. (see `yule` function documentation)

    15. ``Y = cdist(XA, XB, 'matching')``

       Synonym for 'hamming'.

    16. ``Y = cdist(XA, XB, 'dice')``

       Computes the Dice distance between the boolean vectors. (see
       `dice` function documentation)

    17. ``Y = cdist(XA, XB, 'kulsinski')``

       Computes the Kulsinski distance between the boolean
       vectors. (see `kulsinski` function documentation)

    18. ``Y = cdist(XA, XB, 'rogerstanimoto')``

       Computes the Rogers-Tanimoto distance between the boolean
       vectors. (see `rogerstanimoto` function documentation)

    19. ``Y = cdist(XA, XB, 'russellrao')``

       Computes the Russell-Rao distance between the boolean
       vectors. (see `russellrao` function documentation)

    20. ``Y = cdist(XA, XB, 'sokalmichener')``

       Computes the Sokal-Michener distance between the boolean
       vectors. (see `sokalmichener` function documentation)

    21. ``Y = cdist(XA, XB, 'sokalsneath')``

       Computes the Sokal-Sneath distance between the vectors. (see
       `sokalsneath` function documentation)


    22. ``Y = cdist(XA, XB, 'wminkowski')``

       Computes the weighted Minkowski distance between the
       vectors. (see `wminkowski` function documentation)

    23. ``Y = cdist(XA, XB, f)``

       Computes the distance between all pairs of vectors in X
       using the user supplied 2-arity function f. For example,
       Euclidean distance between the vectors could be computed
       as follows::

         dm = cdist(XA, XB, lambda u, v: np.sqrt(((u-v)**2).sum()))

       Note that you should avoid passing a reference to one of
       the distance functions defined in this library. For example,::

         dm = cdist(XA, XB, sokalsneath)

       would calculate the pair-wise distances between the vectors in
       X using the Python function `sokalsneath`. This would result in
       sokalsneath being called :math:`{n \\choose 2}` times, which
       is inefficient. Instead, the optimized C version is more
       efficient, and we call it using the following syntax::

         dm = cdist(XA, XB, 'sokalsneath')

    Examples
    --------
    Find the Euclidean distances between four 2-D coordinates:

    >>> from scipy.spatial import distance
    >>> coords = [(35.0456, -85.2672),
    ...           (35.1174, -89.9711),
    ...           (35.9728, -83.9422),
    ...           (36.1667, -86.7833)]
    >>> distance.cdist(coords, coords, 'euclidean')
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
    >>> distance.cdist(a, b, 'cityblock')
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
    #     Y = cdist(XA, XB, 'test_abc')
    # where 'abc' is the metric being tested.  This computes the distance
    # between all pairs of vectors in XA and XB using the distance metric 'abc'
    # but with a more succinct, verifiable, but less efficient implementation.

    # Store input arguments to check whether we can modify later.
    input_XA, input_XB = XA, XB

    XA = np.asarray(XA, order='c')
    XB = np.asarray(XB, order='c')

    # The C code doesn't do striding.
    XA = _copy_array_if_base_present(XA)
    XB = _copy_array_if_base_present(XB)

    s = XA.shape
    sB = XB.shape

    if len(s) != 2:
        raise ValueError('XA must be a 2-dimensional array.')
    if len(sB) != 2:
        raise ValueError('XB must be a 2-dimensional array.')
    if s[1] != sB[1]:
        raise ValueError('XA and XB must have the same number of columns '
                         '(i.e. feature dimension.)')

    mA = s[0]
    mB = sB[0]
    n = s[1]
    if out is None:
        dm = np.zeros((mA, mB), dtype=np.double)
    else:
      if out.shape != (mA, mB):
        raise ValueError("Output array has wrong dimension.")
      dm = out  

    # validate input for multi-args metrics
    if(metric in ['minkowski', 'mi', 'm', 'pnorm', 'test_minkowski'] or
       metric == minkowski):
        p = _validate_minkowski_args(p)
        _filter_deprecated_kwargs(w=w, V=V, VI=VI)
    elif(metric in ['wminkowski', 'wmi', 'wm', 'wpnorm', 'test_wminkowski'] or
         metric == wminkowski):
        p, w = _validate_wminkowski_args(p, w)
        _filter_deprecated_kwargs(V=V, VI=VI)
    elif(metric in ['seuclidean', 'se', 's', 'test_seuclidean'] or
         metric == seuclidean):
        V = _validate_seuclidean_args(np.vstack([XA, XB]), n, V)
        _filter_deprecated_kwargs(p=p, w=w, VI=VI)
    elif(metric in ['mahalanobis', 'mahal', 'mah', 'test_mahalanobis'] or
         metric == mahalanobis):
        VI = _validate_mahalanobis_args(np.vstack([XA, XB]), mA + mB, n, VI)
        _filter_deprecated_kwargs(p=p, w=w, V=V)
    else:
        _filter_deprecated_kwargs(p=p, w=w, V=V, VI=VI)

    if callable(metric):
        # metrics that expects only doubles:
        if metric in [braycurtis, canberra, chebyshev, cityblock, correlation,
                      cosine, euclidean, mahalanobis, minkowski, sqeuclidean,
                      seuclidean, wminkowski]:
            XA = _convert_to_double(XA)
            XB = _convert_to_double(XB)
        # metrics that expects only bools:
        elif metric in [dice, kulsinski, rogerstanimoto, russellrao,
                        sokalmichener, sokalsneath, yule]:
            XA = _convert_to_bool(XA)
            XB = _convert_to_bool(XB)
        # metrics that may receive multiple types:
        elif metric in [matching, hamming, jaccard]:
            if XA.dtype == bool:
                XA = _convert_to_bool(XA)
                XB = _convert_to_bool(XB)
            else:
                XA = _convert_to_double(XA)
                XB = _convert_to_double(XB)

        # metrics that expects multiple args
        if metric == minkowski:
            metric = partial(minkowski, p=p)
        elif metric == wminkowski:
            metric = partial(wminkowski, p=p, w=w)
        elif metric == seuclidean:
            metric = partial(seuclidean, V=V)
        elif metric == mahalanobis:
            metric = partial(mahalanobis, VI=VI)

        for i in xrange(0, mA):
            for j in xrange(0, mB):
                dm[i, j] = metric(XA[i, :], XB[j, :])

    elif isinstance(metric, string_types):
        mstr = metric.lower()

        try:
            validate, cdist_fn = _SIMPLE_CDIST[mstr]
            XA = validate(XA)
            XB = validate(XB)
            cdist_fn(XA, XB, dm)
            return dm
        except KeyError:
            pass

        if mstr in ['matching', 'hamming', 'hamm', 'ha', 'h']:
            if XA.dtype == bool:
                XA = _convert_to_bool(XA)
                XB = _convert_to_bool(XB)
                _distance_wrap.cdist_hamming_bool_wrap(XA, XB, dm)
            else:
                XA = _convert_to_double(XA)
                XB = _convert_to_double(XB)
                _distance_wrap.cdist_hamming_wrap(XA, XB, dm)
        elif mstr in ['jaccard', 'jacc', 'ja', 'j']:
            if XA.dtype == bool:
                XA = _convert_to_bool(XA)
                XB = _convert_to_bool(XB)
                _distance_wrap.cdist_jaccard_bool_wrap(XA, XB, dm)
            else:
                XA = _convert_to_double(XA)
                XB = _convert_to_double(XB)
                _distance_wrap.cdist_jaccard_wrap(XA, XB, dm)
        elif mstr in ['minkowski', 'mi', 'm', 'pnorm']:
            XA = _convert_to_double(XA)
            XB = _convert_to_double(XB)
            _distance_wrap.cdist_minkowski_wrap(XA, XB, dm, p=p)
        elif mstr in ['wminkowski', 'wmi', 'wm', 'wpnorm']:
            XA = _convert_to_double(XA)
            XB = _convert_to_double(XB)
            _distance_wrap.cdist_weighted_minkowski_wrap(XA, XB, dm, p=p, w=w)
        elif mstr in ['seuclidean', 'se', 's']:
            XA = _convert_to_double(XA)
            XB = _convert_to_double(XB)
            _distance_wrap.cdist_seuclidean_wrap(XA, XB, dm, V=V)
        elif mstr in ['cosine', 'cos']:
            XA = _convert_to_double(XA)
            XB = _convert_to_double(XB)
            _cosine_cdist(XA, XB, dm)
        elif mstr in ['correlation', 'co']:
            XA = _convert_to_double(XA)
            XB = _convert_to_double(XB)
            XA = XA.copy() if XA is input_XA else XA
            XB = XB.copy() if XB is input_XB else XB
            XA -= XA.mean(axis=1)[:, np.newaxis]
            XB -= XB.mean(axis=1)[:, np.newaxis]
            _cosine_cdist(XA, XB, dm)
        elif mstr in ['mahalanobis', 'mahal', 'mah']:
            XA = _convert_to_double(XA)
            XB = _convert_to_double(XB)
            # sqrt((u-v)V^(-1)(u-v)^T)
            _distance_wrap.cdist_mahalanobis_wrap(XA, XB, dm, VI=VI)
        elif mstr.startswith("test_"):
            if mstr in _TEST_METRICS:
                kwargs = {"p":p, "w":w, "V":V, "VI":VI}
                dm = cdist(XA, XB, _TEST_METRICS[mstr], **kwargs)
            else:
                raise ValueError('Unknown "Test" Distance Metric: %s' % mstr[5:])
        else:
            raise ValueError('Unknown Distance Metric: %s' % mstr)
    else:
        raise TypeError('2nd argument metric must be a string identifier '
                        'or a function.')
    return dm
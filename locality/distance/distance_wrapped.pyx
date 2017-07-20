# distutils: language = c
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False
# cython: profile = True

import numpy as np
import locality.distance._alignment_wrap as _alignment_wrap

from distance import cdist
from locality.distance.ssdist import ssdist
from locality.distance.score_matrices import blosum62, pam250
from scipy.sparse import csr_matrix

_metric_names = ["aitchison", "braycurtis", "canberra", "chebyshev", 
                 "cityblock", "correlation", "cosine", "dice", "euclidean", 
                 "hamming", "jaccard", "kulsinski", "mahalanobis", "matching",
                 "minkowski", "rogerstanimoto", "russellrao", "seuclidean",
                 "sokalmichener", "sokalsneath", "sqeuclidean", "yule", 
                 "wminkowski"]

blosum62 = blosum62()
pam250 = pam250()

def _convert_to_str(x):
    return np.ascontiguousarray(x, np.string_)

def _validate_ouput(x, y, out):
    size = x.shape[0] * y.shape[0]
    if np.prod(out.shape) < size:
        raise ValueError("Not enough space in provided output array.")
    if not out.flag.contiguous:
        raise ValueError("Output matrix must be contiguous.")

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

def pnorm(x, y, i, j, p=2):
    return np.sum(np.abs(x[i]-y[j])**p)

def cdist_callable(x, y, mstr, double[:,::1] dm, **kwargs):
    
    cdef npy_intp i, j
    
    if not callable(mstr):
        raise ValueError("Metric must be a function.")
    if x.shape[1] != y.shape[1]:
        raise ValueError("Observations must have same dimensions.")
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            dm[i,j] = mstr(x, y, i, j, **kwargs)
    
    return np.asarray(dm)

def ssdist_callable(x, y, npy_intp[::1] indices_x, npy_intp[::1] indices_y, 
                    npy_intp[::1] indptr, mstr, double[::1] dm, **kwargs):
    
    cdef npy_intp i, j, k, ii
    
    if not callable(mstr):
        raise ValueError("Metric must be a function.")
    if x.shape[1] != y.shape[1]:
        raise ValueError("Observations must have same dimensions.")
    
    if indices_x is None:
        k = 0
        for i in range(x.shape[0]):
            for j in range(indptr[i],indptr[i+1]):
                dm[k] = mstr(x, y, i, indices_y[j], **kwargs)
                k += 1
    else:
        k = 0
        for i in range(indices_x.shape[0]):
            ii = indices_x[i]
            for j in range(indptr[i],indptr[i+1]):
                dm[k] = mstr(x, y, ii, indices_y[j], **kwargs)
                k += 1
    
    indices = indices_y - np.min(indices_y)
    return csr_matrix((np.asarray(dm), indices, indptr), copy=False)

def cdist_wrapped(x, y, mstr, p=None, w=None, VI=None, V=None, gap_open=None,
                  gap_ext=None, mat=None, normalized=True, out=None, **kwargs):
    if out is None:
        dm = np.empty((x.shape[0], y.shape[0]))
    else:
        mx = x.shape[0]
        my = y.shape[0]
        size = mx * my
        if np.prod(out.shape) < size:
            raise ValueError("Not enough space in provided output array.")
        if not out.flags.contiguous:
            raise ValueError("Output matrix must be contiguous.")
        dm = out.ravel()[:size].reshape(mx, my)
    if callable(mstr):
        return cdist_callable(x, y, mstr, dm, **kwargs)
    elif mstr in ["lnorm", "pnorm"]:
        if kwargs["p"] >= 1:
            return cdist(x, y, "minkowski", p = kwargs["p"])
        else:
            return cdist_callable(x, y, pnorm, p) 
    elif mstr in ["blosum", "blosum62"]:
        gap_open, gap_ext, normalized = _validate_alignment_args(gap_open, 
                                                                 gap_ext, 
                                                                 normalized)
        _validate_ouput(x, y, dm)
        x = _convert_to_str(x)
        y = _convert_to_str(y)
        if normalized:
            _alignment_wrap.cdist_normalized_blosum62_wrap(x, y, dm, gap_open, 
                                                           gap_ext)
        else:
            _alignment_wrap.cdist_blosum62_wrap(x, y, dm, gap_open, gap_ext)
    
    elif mstr in ["pam", "pam250"]:
        gap_open, gap_ext, normalized = _validate_alignment_args(gap_open, 
                                                                 gap_ext, 
                                                                 normalized)
        _validate_ouput(x, y, dm)
        x = _convert_to_str(x)
        y = _convert_to_str(y)
        if normalized:
            _alignment_wrap.cdist_normalized_pam250_wrap(x, y, dm, gap_open, 
                                                         gap_ext)
        else:
            _alignment_wrap.cdist_pam250(x, y, dm, gap_open, gap_ext)
    elif mstr in ["edit", "levenshtein"]:
        gap_open, gap_ext, normalized = _validate_alignment_args(gap_open, 
                                                                 gap_ext, 
                                                                 normalized)
        _validate_ouput(x, y, dm)
        x = _convert_to_str(x)
        y = _convert_to_str(y)
        if normalized:
            _alignment_wrap.cdist_normalized_levenshtein_wrap(x, y, dm, 
                                                              gap_open, gap_ext)
        else:
            _alignment_wrap.cdist_levenshtein_wrap(x, y, dm, gap_open, gap_ext) 
    elif mstr in ["align", "alignment"]:
        gap_open, gap_ext, normalized = _validate_alignment_args(gap_open, 
                                                                 gap_ext, 
                                                                 normalized)
        _validate_ouput(x, y, dm)
        mat = _validate_score_matrix(mat)
        x = _convert_to_str(x)
        y = _convert_to_str(y)
        if normalized:
            _alignment_wrap.cdist_normalized_alignment_wrap(x, y, dm, gap_open, 
                                                      gap_ext, mat)
        else:
            _alignment_wrap.cdist_alignment_wrap(x, y, dm, gap_open, gap_ext, 
                                                 mat)                               
    else:
        return cdist(x, y, mstr, p, w, VI, V)

def ssdist_wrapped(x, y, indices_x, indices_y, indptr, mstr,  p=None, w=None, 
                   VI=None, V=None, gap_open=None, gap_ext=None, mat=None, 
                   normalized=True, out=None, **kwargs):
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
    if callable(mstr):
        return ssdist_callable(x, y, indices_x, indices_y, indptr, mstr, 
                               out=dm, **kwargs)
    elif mstr == "pnorm":
        if kwargs["p"] >= 1:
            return ssdist(x, y, indices_y, indptr, indices_x, "minkowski", p=p,
                          out=dm)
        else:
            return ssdist_callable(x, y, indices_y, indices_x, indptr, 
                                   pnorm, p, out=dm) 
    else:
        return ssdist(x, y, indices_y, indptr, indices_x, mstr, p, w, VI, V, 
                      gap_open, gap_ext, mat, normalized, dm)
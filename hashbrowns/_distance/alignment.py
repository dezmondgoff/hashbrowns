"""
Pure Python version of dag based affine alignment
"""
from .score_matrices import load_blosum62, load_pam250

_blosum62 = load_blosum62()
_pam250 = load_pam250()

def blosum62(u, v, gap_open=10, gap_ext=1, normalized=False, tol=1e-07):
    """
    Computes a positive alignment distance between two string using the
    Needleman-Wunsch algorithm and the blosum62 scoring matrix.

    Parameters
    ----------
    u : str
        Input string
    v : str
        Input string
    gap_open : int, optional
        Gap opening penalty
        Default 1
    gap_ext : int, optional
        Gap extension penalty
        Default 1
    normalized : boolean, optional
        Flag to return un-normalized or normalized distances
        Default False
    tol : float, optional
        Error tolerance for normalization iteration
        Default 1e-07

    Returns
    -------
    blosum62 : int or double
        The (normalized) alignment distance between strings `u` and `v`.
    """
    func = lambda a, b : _blosum62[a + b]
    max_score = _get_max_score(u, v, func)

    if not normalized:
        return max_score - _nw(u, v, gap_open, gap_ext, func)

    score, length = _nw_norm_init(u, v, gap_open, gap_ext, func)
    old_lam = 0
    lam = (max_score - score) / length

    while True:
        score, length = _nw_norm(u, v, gap_open, gap_ext, func, lam)
        old_lam = lam
        lam = (max_score - score) / length

        if abs(lam - old_lam) < tol:
            break

    return lam

def pam250(u, v, gap_open=10, gap_ext=1, normalized=False, tol=1e-07):
    """
    Computes a positive alignment distance between two string using the
    Needleman-Wunsch algorithm and the pam250 scoring matrix.

    Parameters
    ----------
    u : str
        Input string
    v : str
        Input string
    gap_open : int, optional
        Gap opening penalty
        Default 1
    gap_ext : int, optional
        Gap extension penalty
        Default 1
    normalized : boolean, optional
        Flag to return un-normalized or normalized distances
        Default False
    tol : float, optional
        Error tolerance for normalization iteration
        Default 1e-07

    Returns
    -------
    pam250 : int or double
        The (normalized) alignment distance between strings `u` and `v`.

    """
    func = lambda a, b : _pam250[a + b]
    max_score = _get_max_score(u, v, func)

    if not normalized:
        return max_score - _nw(u, v, gap_open, gap_ext, func)

    score, length = _nw_norm_init(u, v, gap_open, gap_ext, func)
    old_lam = 0
    lam = (max_score - score) / length

    while True:
        score, length = _nw_norm(u, v, gap_open, gap_ext, func, lam)
        old_lam = lam
        lam = (max_score - score) / length
        if abs(lam - old_lam) < tol:
            break

    return lam

def levenshtein(u, v, gap_open=1, gap_ext=1, normalized=False, tol=1e-07):
    """
    Computes the levenshtein/edit distance between two strings.

    Parameters
    ----------
    u : str
        Input string
    v : str
        Input string
    gap_open : int, optional
        Gap opening penalty
        Default 1
    gap_ext : int, optional
        Gap extension penalty
        Default 1
    normalized : boolean, optional
        Flag to return un-normalized or normalized distances
        Default False
    tol : float, optional
        Error tolerance for normalization iteration
        Default 1e-07

    Returns
    -------
    levenshtein : int or double
        The (normalized) levenshtein distance between strings `u` and `v`.

    """
    func = lambda a, b : 0 if a == b else -1

    if not normalized:
        return -_nw(u, v, gap_open, gap_ext, func)

    score, length = _nw_norm_init(u, v, gap_open, gap_ext, func)
    old_lam = 0
    lam = -score / length

    while True:
        score, length = _nw_norm(u, v, gap_open, gap_ext, func, lam)
        old_lam = lam
        lam = -score / length
        if abs(lam - old_lam) < tol:
            break

    return lam

def needleman_wunsch(u, v, gap_open=10, gap_ext=1, mat=None, normalized=False,
                     tol=1e-07):
    """
    Computes a positive alignment distance between two strings using the
    Needleman-Wunsch algorithm and a user specificied scoring matrix.

    Parameters
    ----------
    u : str
        Input string
    v : str
        Input string
    gap_open : int, optional
        Gap opening penalty
        Default 1
    gap_ext : int, optional
        Gap extension penalty
        Default 1
    mat : dict, optional
        Score matrix as a Python dictionary
    normalized : boolean, optional
        Flag to return un-normalized or normalized distances
        Default False
    tol : float, optional
        Error tolerance for normalization iteration
        Default 1e-07

    Returns
    -------
    needleman_wunsch : int or double
        The (normalized) normalized distance between strings `u` and `v`.

    """
    if mat is None or not isinstance(mat, dict):
        raise ValueError("Must provide valid score matrix")

    func = lambda a, b : mat[a + b]
    max_score = _get_max_score(u, v, func)

    if not normalized:
        return max_score - _nw(u, v, gap_open, gap_ext, func)

    score, length = _nw_norm_init(u, v, gap_open, gap_ext, func)
    old_lam = 0
    lam = (max_score - score) / length

    while True:
        score, length = _nw_norm(u, v, gap_open, gap_ext, func, lam)
        old_lam = lam
        lam = (max_score - score) / length
        if abs(lam - old_lam) < tol:
            break

    return lam

def _get_max_score(u, v, func):
    return max(sum(func(k, k) for k in seq) for seq in (u, v))

def _nw(u, v, gap_open, gap_ext, func):
    nu = len(u)
    nv = len(v)

    lo = [[0] * (nu + 1), [0] * (nu + 1)]
    lo[0][0] = -float("inf")
    lo[1][0] = -float("inf")
    lo[0][1] = -gap_open

    mid = [[0] * (nu + 1), [0] * (nu + 1)]
    mid[0][0] = 0
    mid[1][0] = -gap_open
    mid[0][1] = -gap_open

    up = [[0] * (nu + 1), [0] * (nu + 1)]
    up[0][0] = -float("inf")
    up[1][0] = -gap_open
    up[0][1] = -float("inf")

    for j in range(2, nu + 1):
        lo[0][j] = lo[0][j - 1] - gap_ext
        mid[0][j] = lo[0][j]
        up[0][j] = -float("inf")

    index = 0;

    for i in range(1, nv + 1):
        index = 1 - index

        if i > 1:
            up[index][0] = up[1 - index][0] - gap_ext
            mid[index][0] = up[index][0]                                                                                                 \

        for j in range(1, nu + 1):
            v1 = mid[index][j - 1] - gap_open
            v2 = lo[index][j - 1] - gap_ext
            lo[index][j] = v1 if v1 > v2  else v2

            v1 = mid[1 - index][j] - gap_open
            v2 = up[1 - index][j] - gap_ext
            up[index][j] = v1 if v1 > v2 else v2

            try:
                s = func(u[j - 1], v[i - 1])
            except Exception:
                raise KeyError("character pair ({}, {}) ".format(u[j - 1],
                               v[i - 1]) + "not found")

            v1 = mid[1 - index][j - 1] + s
            v2 = lo[index][j]
            v3 = up[index][j]
            mid[index][j] = v1 if (v1 > v2 and v1 > v3) else (v2 if
                                    v2 > v3 else v3)

    return mid[index][nu]

def _nw_norm(u, v, gap_open, gap_ext, func, lam):
    nu = len(u)
    nv = len(v)

    nlo = [[0] * (nu + 1), [0] * (nu + 1)]
    nlo[0][0] = -float("inf")
    nlo[0][1] = -gap_open
    nlo[1][0] = -float("inf")

    nmid = [[0] * (nu + 1), [0] * (nu + 1)]
    nmid[0][0] = 0
    nmid[1][0] = -gap_open
    nmid[0][1] = -gap_open

    nup = [[0] * (nu + 1), [0] * (nu + 1)]
    nup[0][0] = -float("inf")
    nup[1][0] = -gap_open
    nup[0][1] = -float("inf")

    slo = [[0] * (nu + 1), [0] * (nu + 1)]
    slo[0][0] = -float("inf")
    slo[1][0] = -float("inf")
    slo[0][1] = -gap_open

    smid = [[0] * (nu + 1), [0] * (nu + 1)]
    smid[0][0] = 0
    smid[1][0] = -gap_open
    smid[0][1] = -gap_open

    sup = [[0] * (nu + 1), [0] * (nu + 1)]
    sup[0][0] = -float("inf")
    sup[1][0] = -gap_open
    sup[0][1] = -float("inf")

    llo = [[0] * (nu + 1), [0] * (nu + 1)]
    llo[0][0] = 0
    llo[0][1] = 1
    llo[1][0] = 0

    lmid = [[0] * (nu + 1), [0] * (nu + 1)]
    lmid[0][0] = 0
    lmid[0][1] = 1
    lmid[1][0] = 1

    lup = [[0] * (nu + 1), [0] * (nu + 1)]
    lup[0][0] = 0
    lup[0][1] = 0
    lup[1][0] = 1

    for j in range(2, nu + 1):
        nlo[0][j] = nlo[0][j - 1] - gap_ext
        nmid[0][j] = nlo[0][j]
        nup[0][j] = -float("inf")

        slo[0][j] = slo[0][j - 1] - gap_ext
        smid[0][j] = slo[0][j]
        sup[0][j] = -float("inf")

        llo[0][j] = llo[0][j - 1] + 1
        lmid[0][j] = llo[0][j]
        lup[0][j] = 0

    index = 0;

    for i in range(1, nv + 1):
        index = 1 - index

        if i > 1:
            nup[index][0] = nup[1 - index][0] - gap_ext
            nmid[index][0] = nup[index][0]

            sup[index][0] = sup[1 - index][0] - gap_ext
            smid[index][0] = sup[index][0]

            lup[index][0] = lup[1 - index][0] + 1
            lmid[index][0] = lup[index][0]                                                                                               \

        for j in range(1, nu + 1):
            v1 = nmid[index][j - 1] - gap_open
            v2 = nlo[index][j - 1] - gap_ext

            if v1 > v2:
                nlo[index][j] = v1
                slo[index][j] = smid[index][j - 1] - gap_open
                llo[index][j] = lmid[index][j - 1] + 1
            else:
                nlo[index][j] = v2
                slo[index][j] = slo[index][j - 1] - gap_ext
                llo[index][j] = llo[index][j - 1] + 1

            v1 = nmid[1 - index][j] - gap_open
            v2 = nup[1 - index][j] - gap_ext

            if v1 > v2:
                nup[index][j] = v1
                sup[index][j] = smid[1 - index][j] - gap_open
                lup[index][j] = lmid[1 - index][j] + 1
            else:
                nup[index][j] = v2
                sup[index][j] = sup[1 - index][j] - gap_ext
                lup[index][j] = lup[1 - index][j] + 1

            try:
                s = func(u[j - 1], v[i - 1])
            except Exception:
                raise KeyError("character pair ({}, {}) ".format(u[j - 1],
                               v[i - 1]) + "not found")

            v1 = nmid[1 - index][j - 1] + s - lam
            v2 = nlo[index][j]
            v3 = nup[index][j]

            if v1 > v2 and v1 > v3:
            	nmid[index][j] = v1
            	smid[index][j] = smid[1 - index][j - 1] + s
            	lmid[index][j] = lmid[1 - index][j - 1] + 1
            elif v2 > v3:
            	nmid[index][j] = v2
            	smid[index][j] = slo[index][j]
            	lmid[index][j] = llo[index][j]
            else:
            	nmid[index][j] = v3
            	smid[index][j] = sup[index][j]
            	lmid[index][j] = lup[index][j]

    return smid[index][nu], lmid[index][nu]

def _nw_norm_init(u, v, gap_open, gap_ext, func):
    nu = len(u)
    nv = len(v)

    slo = [[0] * (nu + 1), [0] * (nu + 1)]
    slo[0][0] = -float("inf")
    slo[1][0] = -float("inf")
    slo[0][1] = -gap_open
    smid = [[0] * (nu + 1), [0] * (nu + 1)]
    smid[0][0] = 0
    smid[1][0] = -gap_open
    smid[0][1] = -gap_open
    sup = [[0] * (nu + 1), [0] * (nu + 1)]
    sup[0][0] = -float("inf")
    sup[1][0] = -gap_open
    sup[0][1] = -float("inf")

    llo = [[0] * (nu + 1), [0] * (nu + 1)]
    llo[0][0] = 0
    llo[1][0] = 0
    llo[0][1] = 1
    lmid = [[0] * (nu + 1), [0] * (nu + 1)]
    lmid[0][0] = 0
    lmid[1][0] = 1
    lmid[0][1] = 1
    lup = [[0] * (nu + 1), [0] * (nu + 1)]
    lup[0][0] = 0
    lup[1][0] = 1
    lup[0][1] = 0

    for j in range(2, nu + 1):
        slo[0][j] = slo[0][j - 1] - gap_ext
        smid[0][j] = slo[0][j]
        sup[0][j] = -float("inf")
        llo[0][j] = llo[0][j - 1] + 1
        lmid[0][j] = llo[0][j]
        lup[0][j] = 0

    index = 0;

    for i in range(1, nv + 1):
        index = 1 - index

        if i > 1:
            sup[index][0] = sup[1 - index][0] - gap_ext
            smid[index][0] = sup[index][0]
            lup[index][0] = lup[1 - index][0] + 1
            lmid[index][0] = lup[index][0]                                                                                                    \

        for j in range(1, nu + 1):
            w1 = smid[index][j - 1] - gap_open
            el1 = lmid[index][j - 1] + 1

            w2 = slo[index][j - 1] - gap_ext
            el2 = llo[index][j - 1] + 1

            r1 = w1 / el1
            r2 = w2 / el2

            if r1 > r2:
                slo[index][j] = w1
                llo[index][j] = el1
            else:
                slo[index][j] = w2
                llo[index][j] = el2

            w1 = smid[1 - index][j] - gap_open
            el1 = lmid[1 - index][j] + 1

            w2 = sup[1 - index][j] - gap_ext
            el2 = lup[1 - index][j] + 1

            r1 = w1 / el1
            r2 = w2 / el2

            if r1 > r2:
                sup[index][j] = w1
                lup[index][j] = el1
            else:
                sup[index][j] = w2
                lup[index][j] = el2

            try:
                s = func(u[j - 1], v[i - 1])
            except Exception:
                raise KeyError("character pair ({}, {}) ".format(u[j - 1],
                               v[i - 1]) + "not found")

            w1 = smid[1 - index][j - 1] + s
            el1 = lmid[1 - index][j - 1] + 1

            w2 = slo[index][j]
            el2 = llo[index][j]

            w3 = sup[index][j]
            el3 = lup[index][j]

            r1 = w1 / el1
            r2 = w2 / el2
            r3 = w3 / el3

            if r1 > r2 and r1 > r3:
                smid[index][j] = w1
                lmid[index][j] = el1
            elif r2 > r3:
                smid[index][j] = w2
                lmid[index][j] = el2
            else:
                smid[index][j] = w3
                lmid[index][j] = el3

    return smid[index][nu], lmid[index][nu]

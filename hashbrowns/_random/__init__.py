"""
================================
Random Number Generators for LSH
================================

==============================================================================
Utility functions
==============================================================================
sample_indices       Sets of uniformly distributed integers for indexing
sample_intervals     Sets of uniformly distributed integers for indexing
                     within contiguous intervals
==============================================================================

==============================================================================
Univariate distributions
==============================================================================
stable               General stable distributions.
==============================================================================
"""
from __future__ import division, absolute_import, print_function

import warnings

# To get sub-modules
from .info import __doc__, __all__


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="numpy.ndarray size changed")
    from .stable import stable
    from .sample import sample_indices, sample_intervals

def __RandomStateInferface_ctor():
    """Return a RandomState instance.

    This function exists solely to assist (un)pickling.

    Note that the state of the RandomStateInterface returned here is irrelevant, as this function"s
    entire purpose is to return a newly allocated RandomState whose state pickle can set.
    Consequently the RandomStateInterface returned by this function is a freshly allocated copy
    with a seed=0.

    See https://github.com/numpy/numpy/issues/4763 for a detailed discussion

    """
    return RandomStateInferace(np.random.RandomState(seed=0))

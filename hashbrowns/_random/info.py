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

==============================================================================
Internal functions
==============================================================================
retreive_state          Get rk_state from RandomState.
return_state            Return rk_state to RandomState.
==============================================================================

"""
from __future__ import division, absolute_import, print_function

__all__ = ["sample_indices", "sample_intervals", "stable"]

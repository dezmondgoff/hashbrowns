"""
================================
Helper Functions for LSH
================================

==============================================================================
Utility functions
==============================================================================
hash_argmin
encode_by_bits
encode_by_place
decode_by_bits
decode_by_place
get_hash_distances
rank
bitdot
==============================================================================

"""
from __future__ import division, absolute_import, print_function

import warnings

# To get sub-modules
from .info import __doc__, __all__

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="numpy.ndarray size changed")
    from .hashing import (hash_argmin, encode_by_bits, encode_by_place,
                          decode_by_bits, decode_by_place, get_hash_distances)
    from .rank import rank
    from .bitarray import bitdot

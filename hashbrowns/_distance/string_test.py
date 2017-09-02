import numpy as np
import sys
import os

path = "/home/dezmond/git/hashbrowns"

try:
    from hashbrowns._distance.distance_wrap import cdist_wrap
    from hashbrowns._distance.ssdist import _blosum62
except ModuleNotFoundError:
    sys.path.append(path)
    from hashbrowns._distance.distance_wrap import cdist_wrap
    from hashbrowns._distance.ssdist import _blosum62

X1 = np.loadtxt(os.path.join(path, "tests/test_data/random_amino1.txt"), np.str)
X2 = np.loadtxt(os.path.join(path, "tests/test_data/random_amino2.txt"), np.str)
#x = np.array(['NEFITWYTSFLP', 'WTHKF*CHSYRAR'], dtype=np.str)
D1 = cdist_wrap(X1, X2, "blosum62", gap_open=1, gap_ext=1, normalized=True)
D2 = cdist_wrap(X1, X2, "test_blosum62", gap_open=1, gap_ext=1, normalized=True)
print(np.allclose(D1, D2))

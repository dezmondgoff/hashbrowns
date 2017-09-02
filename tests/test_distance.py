from __future__ import division, print_function, absolute_import

import os
import sys
import numpy as np
from numpy.testing import (verbose, run_module_suite, assert_,
                           assert_raises, assert_array_equal, assert_equal,
                           assert_almost_equal, assert_allclose)
try:
    from hashbrowns._distance.distance_wrap import (cdist_wrap, ssdist_wrap,
                                                    _STRING_METRICS_NAMES)
    from hashbrowns._distance.ssdist import _METRICS_NAMES, _blosum62
except ModuleNotFoundError:
    top_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(top_path)
    from hashbrowns._distance.distance_wrap import (cdist_wrap, ssdist_wrap,
                                                    _STRING_METRICS_NAMES)
    from hashbrowns._distance.ssdist import  _METRICS_NAMES, _blosum62

test_path = os.path.dirname(os.path.abspath(__file__))

eo = {}
filenames = ["random_amino1.txt", "random_amino2.txt"]
for filename in filenames:
    tmp = np.loadtxt(os.path.join(test_path, "data", filename), np.str)
    eo[filename.strip(".txt")] = np.array([x.upper() for x in tmp])

class TestSsdist(object):
    pass

class TestCdist(object):

    def test_cdist_out(self):
        eps = 1e-07
        out1 = np.empty((10, 10), dtype=np.double)
        out2 = np.empty((1, 1), dtype=np.double)
        out3 = np.empty((100, 100), dtype=np.double)[::10, ::10]
        out4 = np.empty((10, 10), dtype=np.int64)
        X1 = np.random.random((10, 10))
        X2 = np.random.random((10, 10))
        for metric in _METRICS_NAMES:
            kwargs = {'p': None, 'w': None, 'V': None, 'VI': None}
            if metric in ['minkowski', 'wminkowski']:
                kwargs['p'] = 1.23
            if metric == 'wminkowski':
                kwargs['w'] = 1.0 / np.concatenate((X1, X2)).std(axis=0)
            Y1 = cdist_wrap(X1, X2, metric, out=out1, **kwargs)
            Y2 = cdist_wrap(X1, X2, metric, **kwargs)
            assert_(Y1.base is out1)
            assert_allclose(Y1, Y2)
            assert_raises(ValueError, cdist_wrap, X1, X2, metric, out=out2, **kwargs)
            assert_raises(ValueError, cdist_wrap, X1, X2, metric, out=out3, **kwargs)
            assert_raises(ValueError, cdist_wrap, X1, X2, metric, out=out4, **kwargs)

    def test_string_metrics(self):
        X1 = eo["random_amino1"]
        X2 = eo["random_amino2"]
        for metric in _STRING_METRICS_NAMES:
            mat = None if metric != 'needleman_wunsch' else _blosum62
            for normalized in [True, False]:
                Y1 = cdist_wrap(X1.astype(np.object_), X2.astype(np.object_),
                                metric, mat=mat, normalized=normalized)
                Y2 = cdist_wrap(X1.astype(np.object_), X2.astype(np.object_),
                                "test_" + metric, mat=mat,
                                normalized=normalized)
                assert_allclose(Y1, Y2)
                Y1 = cdist_wrap(X1.astype(np.str), X2.astype(np.str), metric,
                                mat=mat, normalized=normalized)
                Y2 = cdist_wrap(X1.astype(np.str), X2.astype(np.str),
                                "test_" + metric, mat=mat,
                                normalized=normalized)
                assert_allclose(Y1, Y2)

    def test_string_array_type(self):
        X1 = eo["random_amino1"]
        X2 = eo["random_amino2"]
        for metric in _STRING_METRICS_NAMES:
            mat = None if metric != 'needleman_wunsch' else _blosum62
            for normalized in [True, False]:
                Y1 = cdist_wrap(X1.astype(np.object_), X2.astype(np.object_),
                                metric, mat=mat, normalized=normalized)
                Y2 = cdist_wrap(X1.astype(np.str), X2.astype(np.str), metric,
                                mat=mat, normalized=normalized)
                assert_allclose(Y1, Y2)

class TestPdist(object):
    pass

if __name__ == '__main__':
    run_module_suite()

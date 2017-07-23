import numpy as np
import unittest

from locality.helpers.rank import rank

class Rank_Tests(uniittest.TestCase):

    def main_test(self):
        q = 5
        n = 10
        each = np.random.randint(2, 10, num_points)
        indptr = np.empty(num_points + 1, dtype=np.int)
        indptr[0] = 0
        indptr[1:] = np.cumsum(each)
        dist = np.random.random(indptr[-1])
        indices = np.random.randint(0, 100, indptr[-1])
        confirm = np.empty(q * n, dtype=np.int)
        j = 0
        for i in range(n):
            a, b = indptr[i], indptr[i + 1]
            sort = np.argsort(dist[a:b])
            confirm[j:j + q] = indices[a:b][sort[:q]]
            j += q
        test = rank(q, n, dist, indices, indptr)
        self.assertEqual(confirm, test)
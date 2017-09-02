from __future__ import division, print_function, absolute_import

import os
import sys
from scipy._lib.six import xrange, u

import numpy as np
from numpy.testing import (verbose, run_module_suite, assert_,
                           assert_raises, assert_array_equal, assert_equal,
                           assert_almost_equal, assert_allclose)

try:
    from hashbrowns._distance.distance import cdist
except ModuleNotFoundError:
    top_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(top_path)
    from hashbrowns._distance.distance import cdist

from hashbrowns.storage import storage
from multiprocessing import Pool

def _get_filepath(self, filename=None):
    if filename is None:
        return os.path.dirname(__file__)
    return os.path.join(os.path.dirname(__file__), filename)

def _assert_unique(a):
    assert_(len(a) == len(set(a)))

class TestDictStorage(object):
   
    def test_pack_hashes(self):
        L = 4
        n = 10
        hashlist = np.random.randint(0, 2, (n,L)).astype(np.uint) 
        indices = np.arange(n)
        bytes = np.array([np.random.bytes(1) for _ in range(n)])
        indices_output = {i for i, x in zip(indices, hashlist[:,3]) if x == 1}
        hytes_output = {b for b, x in zip(bytes, hashlist[:,3]) if x == 1}
        storage_config = {"dict":None}
        dict_storage = storage(L, storage_config)

        def task(args):
            storage, ids, hashlist = args
            storage.pack_hashes(ids, hashlist)
            return storage

        pool = Pool(2)
        out1, out2 = pool.map(task, ((dict_storage, ids, hashlist) 
                                        for ids in (indices, bytes)))
        assert_(out1 is not out2)
        tmp1 = out1.storage[3]
        tmp2 = out2.storage[3]
        _assert_unique(tmp1)
        _assert_unique(tmp2)
        assert_equal(indices_output, set(tmp1))
        assert_equal(bytes_output, set(tmp2))
    
    def test_clear(self):
        n = 10
        L = 4
        indices = np.arange(n)
        hashlist = np.random.randint(0, 2, (n,L)).astype(np.uint) 
        dict_storage = storage(L, storage_config)
        dict_storage.pack_hashes(indices, hashlist)
        for storage in dict_storage.storage:
            assert_(not not storage)
        dict_storage.clear(2)
        assert_(not dict_storage.storage[2])
        dict_storage.clear()
        for storage in dict_storage.storage:
            assert_(not storage)
        
    def test_keys(self):
        n = 10
        L = 4
        indices = np.arange(n)
        hashlist = np.random.randint(0, 2, (n,L)).astype(np.uint) 
        dict_storage = storage(L, storage_config)
        dict_storage.pack_hashes(indices, hashlist)
        view = dict_storage.keys()
        view_list = list(view)
        dict_storage.clear(0)
        assert_(view != view_list)
        assert_(not (view ^ view_list))
       
    def test_get_lists(self): 
        
    
        

    def test_redis_storage(self):
        try:
            num_hashtables = 4
            num_points = 1000
            hashlist = np.load(self.get_filepath("test_data/sample_hashlist.npy"))
            output1 = open(self.get_filepath("test_data/hash_output1.txt")).read()
            output1 = [int(x) for x in output1.split()]

            output2 = open(self.get_filepath("test_data/hash_output2.txt")).read()
            output2 = output2.split()
            ids1 = np.arange(num_points)
            ids2 = np.load(self.get_filepath("test_data/string_ids.npy"))
            storage_config = {"redis":{}}
            redis_storage = storage(num_hashtables, storage_config)
            pool = Pool(2)
            res1 = pool.apply_async(redis_storage.store_hashes,
                                    (ids1, hashlist))
            res1.get()
            self.assertEqual(output1, redis_storage.get_list(0, 18))
            redis_storage.clear()
            redis_storage.close()
            res2 = pool.apply_async(redis_storage.store_hashes,
                                    (ids2, hashlist))
            res2.get()
            self.assertEqual(output2, redis_storage.get_list(0, 18))
            redis_storage.clear()
        except Exception as e:
            redis_storage.clear()
            raise e

    def test_shelve_storage(self):
        try:
            num_hashtables = 4
            num_points = 1000
            hashlist = np.load(self.get_filepath("test_data/sample_hashlist.npy"))
            output1 = open(self.get_filepath("test_data/hash_output1.txt")).read()
            output1 = [int(x) for x in output1.split()]
            output2 = open(self.get_filepath("test_data/hash_output2.txt")).read()
            output2 = output2.split()
            ids1 = np.arange(num_points)
            ids2 = np.load(self.get_filepath("test_data/string_ids.npy"))
            filename1 = self.get_filepath("shelve00")
            filename2 = self.get_filepath("shelve01")
            storage_config = {"shelve":{}}
            shelve_storage1 = storage(num_hashtables, storage_config, filename1)
            shelve_storage2 = storage(num_hashtables, storage_config, filename2)
            pool = Pool(2)
            res1 = pool.apply_async(shelve_storage1.store_hashes,
                                    (ids1, hashlist))
            res2 = pool.apply_async(shelve_storage2.store_hashes,
                                    (ids2, hashlist))
            res1.get()
            res2.get()
            self.assertEqual(output1, shelve_storage1.get_list(0, 18))
            self.assertEqual(output2, shelve_storage2.get_list(0, 18))
            os.remove(filename1)
            os.remove(filename2)
        except Exception as e:
            os.remove(filename1)
            os.remove(filename2)
            raise e

    def test_dbm_storage(self):
        try:
            num_hashtables = 4
            num_points = 1000
            hashlist = np.load(self.get_filepath("test_data/sample_hashlist.npy"))
            output1 = open(self.get_filepath("test_data/hash_output1.txt")).read()
            output1 = [int(x) for x in output1.split()]
            output2 = open(self.get_filepath("test_data/hash_output2.txt")).read()
            output2 = output2.split()
            ids1 = np.arange(num_points)
            ids2 = np.load(self.get_filepath("test_data/string_ids.npy"))
            filename1 = self.get_filepath("dbm00")
            filename2 = self.get_filepath("dbm01")
            storage_config = {"dbm":{}}
            dbm_storage1 = storage(num_hashtables, storage_config, filename1)
            dbm_storage2 = storage(num_hashtables, storage_config, filename2)
            pool = Pool(2)
            res1 = pool.apply_async(dbm_storage1.store_hashes,
                                    (ids1, hashlist))
            res2 = pool.apply_async(dbm_storage2.store_hashes,
                                    (ids2, hashlist))
            res1.get()
            res2.get()
            self.assertEqual(output1, dbm_storage1.get_list(0, 18))
            self.assertEqual(output2, dbm_storage2.get_list(0, 18))
            os.remove(filename1)
            os.remove(filename2)
        except Exception as e:
            os.remove(filename1)
            os.remove(filename2)
            raise e

    def test_sqlite_storage(self):
        try:
            num_hashtables = 4
            num_points = 1000
            hashlist = np.load(self.get_filepath("test_data/sample_hashlist.npy"))
            output1 = open(self.get_filepath("test_data/hash_output1.txt")).read()
            output1 = [int(x) for x in output1.split()]
            output2 = open(self.get_filepath("test_data/hash_output2.txt")).read()
            output2 = output2.split()
            ids1 = np.arange(num_points)
            ids2 = np.load(self.get_filepath("test_data/string_ids.npy"))
            filename = self.get_filepath("sql.db")
            storage_config = {"sqlite":{}}
            sqlite_storage1 = storage(num_hashtables, storage_config, filename)
            sqlite_storage2 = storage(num_hashtables, storage_config, filename)
            pool = Pool(2)
            res1 = pool.apply_async(sqlite_storage1.store_hashes,
                                    (ids1, hashlist))
            res2 = pool.apply_async(sqlite_storage2.store_hashes,
                                    (ids2, hashlist))
            res1.get()
            res2.get()
            self.assertEqual(output1, sqlite_storage1.get_list(0, 18))
            self.assertEqual(output2, sqlite_storage2.get_list(0, 18))
            os.remove(filename)
        except Exception as e:
            os.remove(filename)
            raise e

if __name__ == '__main__':
    unittest.main()

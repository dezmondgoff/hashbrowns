import os
import unittest
import numpy as np
from locality.storage import storage
from multiprocessing import Pool


class Storage_Tests(unittest.TestCase):

    def get_filepath(self, filename=None):
        if filename is None:
            return os.path.dirname(__file__)
        return os.path.join(os.path.dirname(__file__), filename)

    def test_dict_storage(self):
        L = 4
        n = 1000
        hashlist = np.load(self.get_filepath("test_data/sample_hashlist.npy"))
        output = open(self.get_filepath("test_data/hash_output.txt")).read()
        output = [int(x) for x in output.split()]
        ids = np.arange(n)
        storage_config = {"dict":None}
        dict_storage = storage(L, storage_config)
        pool = Pool(2)

        def f(storage, ids, hashlist):
            storage.pack_hashes(ids, hashlist)
            return storage

        res1 = pool.apply_async(f, (dict_storage, ids, hashlist))
        res2 = pool.apply_async(f, (dict_storage, ids, hashlist))
        dict_storage1 = res1.get()
        dict_storage2 = res2.get()
        self.assertEqual(output, dict_storage1.get_list(0, 18))
        self.assertEqual(output, dict_storage2.get_list(0, 18))

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

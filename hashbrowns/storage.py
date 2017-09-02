# -*- coding: utf-8 -*-
import os
import json
import shelve
import dbm
import sqlite3
import ntpath
import numpy as np
from array import array
from copy import deepcopy
from _helpers import get_hash_distances

from pathlib import Path

try:
    import redis
except ImportError:
    redis = False

__all__ = ['storage']

def storage(L, storage_config, filename=None):
    """ Given the configuration for storage and the index, return the
    configured storage instance.
    """
    if 'dict' in storage_config:
        return DictStorage(L)
    elif 'redis' in storage_config:
        return RedisStorage(L, storage_config['redis'])
    elif 'shelve' in storage_config:
        return ShelveStorage(L, filename, storage_config['shelve'])
    elif 'dbm' in storage_config:
        return DBMStorage(L, filename, storage_config['dbm'])
    elif "sqlite" in storage_config:
        return SQLiteStorage(L, filename,
                             storage_config['sqlite'])
    else:
        raise NotImplementedError("Storage type not supported.")


def _validate_dtype(arrays, dtype):
    for array in arrays:
        if array.dtype != dtype:
            raise ValueError("Array has incorrect dtype.")

def _fill_array(it, array=None):
    if array is None:
        return np.fromiter(it)
    for i, elem in enumerate(it):
        array[i] = elem
    return array[:i + 1]


class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSONEncoder to properly encode numpy data types.
    """
    def default(self, obj):

        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


class MultiDictKeyView(object):

    def __init__(self, *parents, **kwargs):
        self.parents = parents
        self.n = len(parents)
        self.pos = -1

    def __iter__(self):
        self.pos = 0
        self.current_iter = iter(self.parents[self.pos].keys(*kwargs))
        return self

    def __next__(self):
        if self.pos < 0 or self.pos > self.n:
            raise StopIteration
        try:
            return next(self.current_iter)
        except StopIteration:
            self.pos += 1
            try:
                self.current_iter = iter(self.parents[self.pos].keys(*kwargs))
                return next(self.current_iter)
            except IndexError:
                raise StopIteration

    def __leq__(self, other):
        if isinstance(other, MultiDictKeyView):
            pass


class BaseStorage(object):
    """An abstract class used as an adapter for storage formats.
    """
    def __init__(self, config):

        raise NotImplementedError

    def _set_attrs(self, other, copy=False):
        if copy:
            other = other.copy()

        for key, item in other.__dict__.items():
            self.__dict__[key] = item

    def copy(self):
        return deepcopy(self)

    def clear(self, index=None):
        """
        Clear hash table(s) specified by index. If index is None, this method
        should clear all hash tables.
        """
        raise NotImplementedError

    def keys(self, index):
        """
        Returns an view-like object that produces the hash keys used in the hash table
        specified by index.
        """
        raise NotImplementedError

    def pack_hashes(self, ids, hashlist):
        """
        Append data identifiers into the hash tables using hashes in 2D hash
        array, where rows represent individual data points and columns represent
        individual hash tables. This method should return None.
        """
        raise NotImplementedError

    def get_list(self, index, hashkey):
        raise NotImplementedError

    def get_close(self, index, key, b=1, max_dist=None, max_size=None,
                  hash_stores=None, dist_stores=None):
        """
        Return set ...
        """
        if hash_stores is not None:
            if isinstance(hash_stores, array):
                hash_stores = (hash_stores,)
            elif isinstance(hash_stores, tuple):
                pass
            else:
                raise ValueError("hash_stores must be a python array or a tuple "
                                 "of python arrays")
            if len(hash_arrays) < self.L:
                raise ValueError("must be at least {} ".format(self.L) +
                                 "arrays for storing hash keys")
        if dist_stores is not None:
            if isinstance(dict_stores, array):
               dict_stores = (dict_stores,)
           elif isinstance(dict_stores, tuple):
                pass
            else:
                raise ValueError("dist_stores must be a python array or a tuple "
                                 "of python arrays")
            if len(hash_arrays) < self.L:
                raise ValueError("there must be at least {} ".format(self.L) +
                                 "arrays for storing distances")

        if max_dist is None and max_size is None:
            raise ValueError("must specify at least one of max_dist and/or "
                             "max_size")
        elif max_dist is None:
            max_dist = np.inf
        elif max_size is None:
            max_size = np.inf

        m = self.L

        if m == 1:
            close = []
            it =
            if hash_stores is not None:
                hash_store = hash_stores[0]
            if dist_stores is not None:
                dist_store = dist_stores[0]
            hashkeys = _fill_array((k for k in self.keys(0) if k != hashkey),
                                    hash_store)
            distances = get_hash_distances(key, hashkeys, b, dist_store)
            visited = [True] * hashkeys.size
            d = 0
            size = 0

            while d <= max_dist and size <= max_size:
                for i in range(hashkeys.used):
                    if flags[i] and distances[i] <= d:
                        visited[i] = False
                        tmp = self.get_list(0, hashkeys[i])
                        close.extend(tmp)
                        size += len(tmp)
                d += 1
        else:
            close = set()
            retrieved = [False] * m
            visited_flags = [None] * m
            d = 0
            size = 0

            while d <= max_dist and size <= max_size:
                for i in range(m):
                    if size == max_size:
                        break
                    if not retrieved[i]:
                        it = (k for k in self.keys(0) if k != hashkey[i])
                        hashkeys = _fill_array(it, hash_stores[i])
                        distances = get_hash_distances(key, hashkeys, b, dist_stores[i])
                        visited = [False] * hashkeys.used
                        visited_flags[i] = visited
                    else:
                        hashkeys = hash_stores[i]
                        distances = dist_stores[i]
                        visited = visited_flags[i]

                    for j in range(hashkeys.used):
                        if distances[j] <= d and not visited[j]:
                            tmp = self.get_list(i, hashkeys[j])
                            close.update(tmp)
                            visited[j] = True
                            size += len(tmp)
                d += 1

        return close

    def get_candidates(self, hashlist):
        """
        Returns a list of values stored at hash *key* in the table specified by
        *index*. This method should return a list of values stored at *key*.
        If the list is empty or if *key* is not present in the specified hash
        table, the method should return an empty list.
        """
        raise NotImplementedError


class DictStorage(BaseStorage):

    def __init__(self, arg1, copy=True):
        self._name = 'dict'

        if isinstance(arg1, DictStorage):
            self._set_attrs(arg1, copy)
        else:
            self.L = arg1
            self.storage = [dict() for _ in range(arg1)]


    def clear(self, index=None):
        if index is not None:
            try:
                self.storage[index].clear()
            except IndexError:
                _storage = self.storage

                for i in index:
                    _storage[i].clear()
        else:
            for table in self.storage:
                table.clear()

    def keys(self, index):
        return self.storage[index].keys()

    def pack_hashes(self, index, hashlist):
        local_storage = self.storage
        n = hashlist.shape[0]
        m = self.L

        if m == 1:
            local_setdefault = _storage[0].setdefault
            for i in range(n):
                local_setdefault(hashlist[i, 0], []).append(index[i])
        else:
            setters = [local_storage[j].setdefault for j in range(_m)]
            for i in range(n):
                for j in range(m):
                    setters[j](hashlist[i, j], []).append(ids[i])

    def get_list(self, index, hashkey):
        return self.storage[index].get(hashkey, [])

    def get_candidates(self, hashlist, q, b, indptr=None, cand=None,
                       dtype=None, max_dist=2, hash_stores=None,
                       dist_stores=None):
        local_storage = self.storage
        n = hashlist.shape[0]
        m = self.L

        if indptr is None:
            indptr = np.empty(n + 1, dtype=np.intp)
        else:
            assert indptr.size >= n + 1
        if cand is None:
            if dtype is None:
                raise ValueError("Must provide dtype of candidates")
            cand = np.empty(4, dtype=dtype)

        indptr[0] = 0

        if m == 1:
            local_get = _storage[0].get
            for i in range(n):
                tmp = local_get(hashlist[i, 0], [])
                if len(tmp) < q:
                    tmp.extend(self.get_close(0, hashlist[i, 0], b, max_dist=2,
                                              max_size=q, hash_stores,
                                              dist_stores))
                ii = indptr[i]
                indptr[i + 1] = ii + len(tmp)
                for j, c in enumerate(tmp):
                    cand_out[ii + j] = c
        else:
            getters = [local_storage[j].get for j in range(m)]
            for i in range(n):
                tmp = set()
                for j in range(m):
                    tmp.update(getters[j](hashlist[i, j], []))
                if len(tmp) < q:
                    tmp.extend(self.get_close(0, hashlist[i, :], b, max_dist=2,
                                              max_size=q, hash_stores
                                              dist_stores))
                ii = indptr[i]
                indptr[i + 1] = ii + len(tmp)
                for j, c in enumerate(s):
                    cand[ii + j] = c

        return indptr[:n + 1], cand[:indptr[n]]


class RedisStorage(BaseStorage):

    def __init__(self, L, config):
        if not redis:
            raise ImportError("redis-py is required to use Redis as storage.")
        self.L = L
        self.storage = None
        self._config = config
        self._name = 'redis'

    def connect(self):
        """
        Creates connections to Redis server. Used to created connection objects
        after data has been passed to a child process.
        """
        _n = self.L
        _storage = [0] * _n
        _config = dict(self._config)

        for i in range(_n):
            _config["db"] = i
            _storage[i] = redis.StrictRedis(**_config)

        self.storage = _storage

    def close(self):
        """
        Close connections to Redis server by deleting connection objects.
        """
        self.storage = None

    def clear(self, index=None):

        if self.storage is None:
            self.connect()
        if index is not None:
            self.storage[index].flushdb()
        else:
            self.storage[0].flushall()

    def keys(self, index=None, pattern="*"):
        if self.storage is None:
            self.connect()
        if index is not None:
            try:
                return self.storage[index].keys(pattern)
            except IndexError:
                parents = tuple(self.storage[i] for i in index)
                return MultiDictKeyView(*parents, **{"pattern":pattern})
        return MultiDictKeyView(*self.storage, **{"pattern":pattern})

    def pack_hashes(self, ids, hashlist):
        if self.storage is None:
            self.connect()
        local_storage = self.storage

        n = hashlist.shape[0]
        m = self.L

        if m == 1:
            local_append = local_storage[0].rpush
            for i in range(_n):
                local_append(hashlist[i, 0], ids[i])
        else:
            appenders = [local_storage[j].rpush for j in range(m)]
            for i in range(n):
                for j in range(m):
                    appenders[j](hashlist[i, j], ids[i])

    def get_lists(self, hashlist, indptr=None, cand=None, dtype=None):
        if self.storage is None:
            self.connect()
        local_storage = self.storage

        n = hashlist.shape[0]
        m = self.L

        if indptr is None:
            indptr = np.empty(n + 1, dtype=np.intp)
        else:
            assert indptr.size >= n + 1
        if cand is None:
            if dtype is None:
                raise ValueError("Must provide dtype of candidates")
            cand = np.empty(4, dtype=dtype)

        if m == 1:
            local_get = local_storage[0].lrange
            for i in range(n):
                tmp = local_get(hashlist[i, 0], 0, -1)
                ii = indptr[i]
                indptr[i + 1] = ii + len(tmp)
                for j, c in enumerate(tmp):
                    while True:
                        try:
                            cand_out[ii + j] = c
                            break
                        except IndexError:
                            cand.resize((2 * cand.size,), refcheck=False)

        else:
            getters = [local_storage[j].lrange for j in range(m)]
            for i in range(_n):
                tmp = set()
                for j in range(_m):
                    tmp.update(getters[j](hashlist[i, j], 0, -1))
                ii = indptr[i]
                indptr[i + 1] = ii + len(tmp)
                for j, c in enumerate(s):
                    while True:
                        try:
                            cand_out[ii + j] = c
                            break
                        except IndexError:
                            cand.resize((2 * cand.size,), refcheck=False)

        return indptr[:n + 1], cand[:indptr[n]]


class KeyValueStorage(BaseStorage):
    """
    Abstract class for several types of persistent key-value storage.
    """

    def __init__(self, L, filename, config):
        self.L = L
        self._config = config

        if filename is None:
            self.filename = self._name + "_0"
            self.ext = ""
        else:
            self.filename = filename
            head, tail = ntpath.split(self.filename)
            if tail is not None:
                try:
                    self.ext = "." + tail.split(".")[1]
                except IndexError:
                    self.ext = ""
            else:
                try:
                    self.ext = "." + head.split(".")[1]
                except IndexError:
                    self.ext = ""

        i = 1
        while Path(self.filename).is_file():
            s = "_" + str(i)
            head, tail = ntpath.split(self.filename)
            if tail is not None:
                tail = tail.split(".")[0].split("_")[0]
                self.filename = os.path.join(head, tail + s + self.ext)
            else:
                head = head.split(".")[0].split("_")[0]
                self.filename = head + s + self.ext
            i += 1

        self.storage = self.open(self.filename, **self._config)

    def open(self, flag="c", **config):
        filenames = (self.filename.strip(self.ext) + "t" + str(i) + self.ext
                     for i in range(self.L))
        open_args = dict(self.config)
        open_args["flag"] = flag
        self.storage = [self.__open_call(fn, **open_args) for fn in filenames]

    def close(self):
        for d in self.storage:
            d.close()
        self.storage = None

    def clear(self, index=None):
        # This method is very slow
        if self.storage is None:
            self.open()
        if index is not None:
            try:
                self.storage[index].clear()
            except IndexError:
                for i in index:
                    self.storage[i].clear()
        else:
            for i in range(self.L):
                self.storage[i].clear()

    def keys(self, index=None):
        if self.storage is None:
            self.open()
        if index is not None:
            try:
                return self.storage[index].keys()
            except IndexError:
                parents = tuple(self.storage[i] for i in index)
                return MultiDictKeyView(*parents)
        return MultiDictKeyView(*self.storage)


class ShelveStorage(KeyValueStorage):

    def __init__(self, *args):
        self._open_call = shelve.open
        self._name = 'shelve'
        KeyValueStorage.__init__(self, *args)

    def pack_hashes(self, ids, hashlist):
        if self.storage is None:
            self.connect()
        _storage = self.storage

        _n = hashlist.shape[0]
        _m = self.L
        _key_to_str = self.key_to_str
        _get = _storage.get

        if _m == 1:
            for i in range(_n):
                key = _key_to_str(0, hashlist[i, 0])
                copy = _get(key, [])
                copy.append(ids[i])
                _storage[key] = copy
        else:
            for i in range(_n):
                for j in range(_m):
                    key = _key_to_str(j, hashlist[i, j])
                    copy = _get(key, [])
                    copy.append(ids[i])
                    _storage[key] = copy

    def get_lists(self, hashlist, indptr_out, cand_out):
        if self.storage is None:
            self.connect()
        _storage = self.storage

        _n = hashlist.shape[0]
        _m = self.L
        _key_to_str = self.key_to_str
        _get = _storage[0].get

        if _m == 1:
            for i in range(_n):
                ii = indptr_out[i]
                key = _key_to_str(0, hashlist[i, 0])
                tmp = _get(key, [])
                for j, c in enumerate(tmp):
                    while True:
                        try:
                            cand_out[ii + j] = c
                            break
                        except IndexError:
                            cand_out.resize((2 * cand_out.size,),
                                            refcheck=False)
                indptr_out[i + 1] = indptr_out[i] + len(tmp)
        else:
            for i in range(_n):
                ii = indptr_out[i]
                s = set()
                for j in range(_m):
                    key = _key_to_str(j, hashlist[i, j])
                    s.update(_get(key, []))
                for j, c in enumerate(s):
                    while True:
                        try:
                            cand_out[ii + j] = c
                            break
                        except IndexError:
                            cand_out.resize((2 * cand_out.size,),
                                            refcheck=False)
                indptr_out[i + 1] = indptr_out[i] + len(s)

        return indptr_out[:_n + 1], cand_out[:indptr_out[_n]]

class DBMStorage(KeyValueStorage):

    def __init__(self, *args):
        self._open_call = dbm.open
        self._name = "dbm"
        self._ext = ""
        KeyValueStorage.__init__(self, *args)

    def pack_hashes(self, ids, hashlist):
        if self.storage is None:
            self.open(flag="w")
        _storage = self.storage

        _key_to_str = self.key_to_str
        _n = hashlist.shape[0]
        _m = self.L
        _loads = json.loads
        _dumps = json.dumps

        for i in range(_n):
            for j in range(_m):
                key = _key_to_str(j, hashlist[i, j])
                b = _storage.get(key, None)

                if b is None:
                    b = []
                else:
                    b = _loads(b)
                b.append(ids[i])

                _storage[key] = _dumps(b, cls=NumpyEncoder)

    def get_lists(self, hashlist):
        _n = hashlist.shape[0]
        _m = self.L
        _key_to_str = self.key_to_str
        _loads = json.loads

        sets = [0] * _n
        cumsum = np.empty(_n + 1, dtype=np.int)
        cumsum[0] = 0

        with self._open_call(self.filename, flag='w') as _storage:
            for i in range(_n):
                s = set()
                for j in range(_m):
                    key = _key_to_str(j, hashlist[i, j])
                    b = _storage.get(key, None)
                    if b is not None:
                        s.update(_loads(b))
                sets[i] = s
                cumsum[i + 1] = cumsum[i] + len(s)

        return cumsum, np.array([x for s in sets for x in s])


class SQLiteStorage(BaseStorage):

    def __init__(self, L, filename, config):

        self.L = L
        self._config = config
        self._name = "sqlite"
        self.conn = None
        self.cur = None

        if filename is None:
            self.filename = self._name + '00.db'
            self.ext = "db"
        else:
            self.filename = filename
            head, tail = ntpath.split(self.filename)
            if tail is not None:
                try:
                    self.ext = tail.split(".")[1]
                except IndexError:
                    self.ext = ""
            else:
                try:
                    self.ext = head.split(".")[1]
                except IndexError:
                    self.ext = ""

        i = 0
        while Path(self.filename).is_file():
            i += 1
            s = str(i)
            if len(s) == 1:
                s = '0' + s
            head, tail = ntpath.split(self.filename)
            if tail is not None:
                tail = tail.split(".")[0][:-2]
                self.filename = os.path.join(head, tail + s + "." + self.ext)
            else:
                head = head.split(".")[0][:-2]
                self.filename = head + s + "." + self.ext

        self.connect()
        self.cur.execute("""create table if not exists kvs
                        (table_id int, key blob, value blob)""")

    def __getstate__(self):
        """
        Return state values to be pickled.
        """
        return self.filename, self.L, self._config, self._name

    def __setstate__(self, state):
        """
        Restore state from the unpickled state values.
        """
        self.filename, self.L, self._config, self._name = state
        self.conn = None
        self.curr = None

    def connect(self):
        """
        Open a connection to the SQLite database.
        """
        self.conn = sqlite3.connect(self.filename, **self._config)
        self.cur = self.conn.cursor()

    def close(self):
        """
        Close the connection to the SQLite database.
        """
        if self.conn is not None:
            if self.conn.in_transaction:
                self.conn.commit()
            self.cur.close()
            self.conn.close()
            self.conn = None
            self.cur = None

    def keys(self, index=None):

        if self.conn is None:
            self.connect()
        _cur = self.cur

        if index is not None:
            _cur.execute("select distinct key from kvs where table_id=?", index)
        else:
            _cur.execute("select distinct key from kvs")

        for x in _cur:
            yield json.loads(x[0])

    def clear(self, index=None):

        if self.conn is None:
            self.connect()
        _cur = self.cur

        if index is not None:
            _cur.execute("delete * from kvs where table_id=?", index)
        else:
            _cur.execute("drop table if exists kvs")
            _cur.execute("""create table if not exists kvs
                        (table_id int, key blob, value blob)""")

    def pack_hashes(self, ids, hashlist):
        if self.conn is None:
            self.connect()
        _cur = self.cur

        _dumps = json.dumps
        _n = hashlist.shape[0]
        _m = self.L
        _dumps = json.dumps

        g = ((table, _dumps(key, cls=NumpyEncoder), ids[i])
             for i in range (_n)
             for table, key in zip(range(_m), hashlist[i, :]))

        _cur.execute("drop index if exists table_key_value")
        _cur.executemany("insert into kvs (table_id, key, value) values " +
                      "(?,?,?)", g)

    def get_lists(self, hashlist):
        if self.conn is None:
            self.connect()
        _cur = self.cur

        _n = hashlist.shape[0]
        _m = self.L
        _dumps = json.dumps

        cumsum = np.empty(_n + 1, dtype=np.int)
        cumsum[0] = 0

        _cur.execute("""create index if not exists table_key_value on
                     kvs(table_id, key, value)""")

        g = (tuple(x for t in zip(range(_m), hashlist[i, :]) for x in t)
             for i in range (_n))
        s = ",".join(["(?,?)"] * _m)
        _cur.executemany("""select count(distinct value) from kvs where
                        (table_id, key) in ({})""".format(s), g)

        for i, x in enumerate(_cur):
            cumsum[i + 1] = cumsum[i] + x

        out = np.empty(cumsum[-1], dtype=np.int)

        _cur.executemany("select distinct value from kvs where (table_id, key)" +
                      "in ({})".format(s), g)

        for i, x in enumerate(_cur):
            out[i] = x

        return cumsum, out

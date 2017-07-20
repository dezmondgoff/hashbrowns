# lshash/storage.py

import os
import json
import shelve
import dbm
import sqlite3
import ntpath
import numpy as np
from copy import deepcopy

from pathlib import Path

try:
    import redis
except ImportError:
    redis = False

__all__ = ['storage']

def storage(num_hashtables, storage_config, fid=None):
    """ Given the configuration for storage and the index, return the
    configured storage instance.
    """
    if 'dict' in storage_config:
        return InMemoryStorage(num_hashtables)
    elif 'redis' in storage_config:
        return RedisStorage(num_hashtables, storage_config['redis'])
    elif 'shelve' in storage_config:
        return ShelveStorage(num_hashtables, fid, storage_config['shelve'])
    elif 'dbm' in storage_config:
        return DBMStorage(num_hashtables, fid, storage_config['dbm'])
    elif "sqlite" in storage_config:
        return SQLiteStorage(num_hashtables, fid, 
                             storage_config['sqlite'])
    else:
        raise NotImplementedError("Storage type not supported.")

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

class BaseStorage(object):
    
    def __init__(self, config):
        """
        An abstract class used as an adapter for storage formats. 
        """
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
    
    def keys(self, index=None):
        """
        Returns an iterator that produces the hash keys used in the hash table 
        specified by index. If index is None, this method should return keys 
        used in all hash tables. 
        """
        raise NotImplementedError
    
    def pack_hashes(self, ids, hashlist):
        """
        Append data identifiers into the hash tables using hashes in 2D hash 
        array, where rows represent individual data points and columns represent 
        individual hash tables. This method should return None.
        """
        raise NotImplementedError
    
    def get_lists(self, hashlist):
        """ 
        Returns a list of values stored at hash *key* in the table specified by
        *index*. This method should return a list of values stored at *key*. 
        If the list is empty or if *key* is not present in the specified hash
        table, the method should return an empty list. 
        """
        raise NotImplementedError


class InMemoryStorage(BaseStorage):
    
    def __init__(self, arg1, copy=True):
        self._name = 'dict'
        
        if isinstance(arg1, InMemoryStorage):
            self._set_attrs(arg1, copy)
        else:
            self.num_hashtables = arg1
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
                
    def keys(self, index=None):
        _storage = self.storage
        
        if index is not None:
            try:
                return _storage[index].keys()
            except IndexError:
                return (key for i in index for key in _storage[i].keys())
        return (key for table in _storage for key in table.keys())
    
    def pack_hashes(self, ids, hashlist):
        _storage = self.storage
        _n = hashlist.shape[0]
        _m = self.num_hashtables
        
        if _m == 1:
            _set = _storage[0].setdefault
            for i in range(_n):
                _set(hashlist[i,0], []).append(ids[i]) 
        else:
            setters = [_storage[j].setdefault for j in range(_m)]
            for i in range(_n):
                for j in range(_m):
                    setters[j](hashlist[i,j], []).append(ids[i])
    
    def get_lists(self, hashlist, indptr_out, cand_out):
        _storage = self.storage
        _n = hashlist.shape[0]
        _m = self.num_hashtables
        
        assert(indptr_out.size >= _n + 1)
        indptr_out[0] = 0
        
        if _m == 1:
            _get = _storage[0].get
            for i in range(_n):
                ii = indptr_out[i]
                tmp = _get(hashlist[i,0], [])
                for j, c in enumerate(tmp):
                    while True:
                        try:
                            cand_out[ii + j] = c
                            break
                        except IndexError:
                            cand_out.resize((2 * cand_out.size, ), 
                                            refcheck=False)
                indptr_out[i + 1] = indptr_out[i] + len(tmp)
        else:
            getters = [_storage[j].get for j in range(_m)]
            for i in range(_n):
                s = set()
                for j in range(_m):
                    s.update(getters[j](hashlist[i,j], []))
                ii = indptr_out[i]
                for j, c in enumerate(s):
                    while True:
                        try:
                            cand_out[ii + j] = c
                            break
                        except IndexError:
                            cand_out.resize((2 * cand_out.size, ), 
                                            refcheck=False)
                indptr_out[i + 1] = indptr_out[i] + len(s)
        
        return indptr_out[:_n + 1], cand_out[:indptr_out[_n]]


class RedisStorage(BaseStorage):
    
    def __init__(self, num_hashtables, config):
        if not redis:
            raise ImportError("redis-py is required to use Redis as storage.")
        self.num_hashtables = num_hashtables
        self.storage = None
        self._config = config
        self._name = 'redis'
    
    def connect(self):
        """
        Creates connections to Redis server. Used to created connection objects
        after data has been passed to a child process.
        """
        _n = self.num_hashtables
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
        
        _storage = self.storage
        
        if _storage is None:
            self.connect()
        if index is not None:
            try:
                return _storage[index].keys(pattern)
            except IndexError:
                return (key for i in index for key in _storage[i].keys())
        return (key for table in _storage for key in table.keys(pattern))
    
    def pack_hashes(self, ids, hashlist):
        if self.storage is None:
            self.connect()
        _storage = self.storage
        
        _n = hashlist.shape[0]
        _m = self.num_hashtables
        
        if _m == 1:
            _append = _storage[0].rpush
            for i in range(_n):
                _append(hashlist[i,0], ids[i]) 
        else:
            appenders = [_storage[j].rpush for j in range(_m)]
            for i in range(_n):
                for j in range(_m):
                    appenders[j](hashlist[i,j], ids[i])
               
    def get_lists(self, hashlist, indptr_out, cand_out):
        if self.storage is None:
            self.connect()
        _storage = self.storage
        
        _n = hashlist.shape[0]
        _m = self.num_hashtables
        
        if _m == 1:
            _get = _storage[0].lrange
            for i in range(_n):
                ii = indptr_out[i]
                tmp = _get(hashlist[i,0], 0, -1)
                for j, c in enumerate(tmp):
                    while True:
                        try:
                            cand_out[ii + j] = c
                            break
                        except IndexError:
                            cand_out.resize((2 * cand_out.size, ), 
                                            refcheck=False)
                indptr_out[i + 1] = indptr_out[i] + len(tmp)
        else:
            getters = [_storage[j].lrange for j in range(_m)]
            for i in range(_n):
                s = set()
                for j in range(_m):
                    s.update(getters[j](hashlist[i,j], 0, -1))
                ii = indptr_out[i]
                for j, c in enumerate(s):
                    while True:
                        try:
                            cand_out[ii + j] = c
                            break
                        except IndexError:
                            cand_out.resize((2 * cand_out.size, ), 
                                            refcheck=False)
                indptr_out[i + 1] = indptr_out[i] + len(s)
        
        return indptr_out[:_n + 1], cand_out[:indptr_out[_n]]


class KeyValueStorage(BaseStorage):
    """
    Abstract class for several types of persistent key-value storage.
    """
    
    def __init__(self, num_hashtables, fid, config):    
        self.num_hashtables = num_hashtables
        
        if fid is None:
            self.fid = self._name + '00'
            self.ext = ""
        else:
            self.fid = fid
            head, tail = ntpath.split(self.fid)
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
        while Path(self.fid).is_file():
            i += 1
            s = str(i)
            if len(s) == 1:
                s = '0' + s
            head, tail = ntpath.split(self.fid)
            if tail is not None:
                tail = tail.split(".")[0][:-2]
                self.fid = os.path.join(head, tail + s + "." + self.ext)
            else:
                head = head.split(".")[0][:-2]
                self.fid = head + s + "." + self.ext
        
        self.storage = self.open(self.fid, **config)
    
    def open(self, flag="_cur", **config):
        self.storage = self._open_call(self.fid, flag=flag, **config)
        
    def close(self):
        self.storage.close()
        self.storage = None
        
    def clear(self, index=None):
        # This method is very slow for shelves
        if self.storage is None:
            self.open()
        if index is not None:
            _storage = self.storage
            for key in (key for key in _storage.keys() 
                        if key.startswith("(" + str(index))):
                del _storage[key]
        else:
            self.open(self.fid, flag="n")
            
    def key_to_str(self, index, key):
        """
        Convert index, key pair to Python string object.
        """
        return str((index, key))
    
    def keys(self, index=None):
        if self.storage is None:
            self.open()
        _storage = self.storage
        if index is not None:
            try:
                s = tuple("(" + str(i) for i in index)
                it = (key for key in _storage.keys() if key.startswith(s))
                while True:
                    yield next(it)
            except TypeError:
                s = "(" + str(index)
                it = (key for key in _storage.keys() if key.startswith(s))
                while True:
                    yield next(it)
        else:
            it = _storage.keys()
            while True:
                yield next(it)
                    
                
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
        _m = self.num_hashtables
        _key_to_str = self.key_to_str
        _get = _storage.get
        
        if _m == 1:
            for i in range(_n):
                key = _key_to_str(0, hashlist[i,0])
                copy = _get(key, [])
                copy.append(ids[i])
                _storage[key] = copy 
        else:
            for i in range(_n):
                for j in range(_m):
                    key = _key_to_str(j, hashlist[i,j])
                    copy = _get(key, [])
                    copy.append(ids[i])
                    _storage[key] = copy
               
    def get_lists(self, hashlist, indptr_out, cand_out):
        if self.storage is None:
            self.connect()
        _storage = self.storage
        
        _n = hashlist.shape[0]
        _m = self.num_hashtables
        _key_to_str = self.key_to_str
        _get = _storage[0].get
        
        if _m == 1:
            for i in range(_n):
                ii = indptr_out[i]
                key = _key_to_str(0, hashlist[i,0])
                tmp = _get(key, [])
                for j, c in enumerate(tmp):
                    while True:
                        try:
                            cand_out[ii + j] = c
                            break
                        except IndexError:
                            cand_out.resize((2 * cand_out.size, ), 
                                            refcheck=False)
                indptr_out[i + 1] = indptr_out[i] + len(tmp)
        else:
            for i in range(_n):
                ii = indptr_out[i]
                s = set()
                for j in range(_m):
                    key = _key_to_str(j, hashlist[i,j])
                    s.update(_get(key, []))
                for j, c in enumerate(s):
                    while True:
                        try:
                            cand_out[ii + j] = c
                            break
                        except IndexError:
                            cand_out.resize((2 * cand_out.size, ), 
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
        _m = self.num_hashtables
        _loads = json.loads
        _dumps = json.dumps
        
        for i in range(_n):
            for j in range(_m):
                key = _key_to_str(j, hashlist[i,j])
                b = _storage.get(key, None)
                
                if b is None:
                    b = []
                else:
                    b = _loads(b)
                b.append(ids[i])
                
                _storage[key] = _dumps(b, cls=NumpyEncoder)
       
    def get_lists(self, hashlist):        
        _n = hashlist.shape[0]
        _m = self.num_hashtables
        _key_to_str = self.key_to_str
        _loads = json.loads
        
        sets = [0] * _n
        cumsum = np.empty(_n + 1, dtype=np.int)
        cumsum[0] = 0
        
        with self._open_call(self.fid, flag='w') as _storage:
            for i in range(_n):
                s = set()
                for j in range(_m):
                    key = _key_to_str(j, hashlist[i,j])
                    b = _storage.get(key, None)
                    if b is not None:
                        s.update(_loads(b))
                sets[i] = s
                cumsum[i + 1] = cumsum[i] + len(s)
        
        return cumsum, np.array([x for s in sets for x in s])


class SQLiteStorage(BaseStorage):
    
    def __init__(self, num_hashtables, fid, config):
        
        self.num_hashtables = num_hashtables
        self._config = config
        self._name = "sqlite"
        self.conn = None
        self.cur = None
        
        if fid is None:
            self.fid = self._name + '00.db' 
            self.ext = "db"
        else:
            self.fid = fid
            head, tail = ntpath.split(self.fid)
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
        while Path(self.fid).is_file():
            i += 1
            s = str(i)
            if len(s) == 1:
                s = '0' + s
            head, tail = ntpath.split(self.fid)
            if tail is not None:
                tail = tail.split(".")[0][:-2]
                self.fid = os.path.join(head, tail + s + "." + self.ext)
            else:
                head = head.split(".")[0][:-2]
                self.fid = head + s + "." + self.ext
        
        self.connect()
        self.cur.execute("""create table if not exists kvs 
                        (table_id int, key blob, value blob)""")
    
    def __getstate__(self):
        """
        Return state values to be pickled.
        """
        return self.fid, self.num_hashtables, self._config, self._name

    def __setstate__(self, state):
        """
        Restore state from the unpickled state values.
        """
        self.fid, self.num_hashtables, self._config, self._name = state
        self.conn = None
        self.curr = None
    
    def connect(self):
        """
        Open a connection to the SQLite database.
        """
        self.conn = sqlite3.connect(self.fid, **self._config)
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
        _m = self.num_hashtables
        _dumps = json.dumps
        
        g = ((table, _dumps(key, cls=NumpyEncoder), ids[i]) 
             for i in range (_n) 
             for table, key in zip(range(_m), hashlist[i,:]))
        
        _cur.execute("drop index if exists table_key_value")
        _cur.executemany("insert into kvs (table_id, key, value) values " +
                      "(?,?,?)", g)
       
    def get_lists(self, hashlist):
        if self.conn is None:
            self.connect()
        _cur = self.cur
        
        _n = hashlist.shape[0]
        _m = self.num_hashtables
        _dumps = json.dumps
        
        cumsum = np.empty(_n + 1, dtype=np.int)
        cumsum[0] = 0
        
        _cur.execute("""create index if not exists table_key_value on 
                     kvs(table_id, key, value)""")
        
        g = (tuple(x for t in zip(range(_m), hashlist[i,:]) for x in t)
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
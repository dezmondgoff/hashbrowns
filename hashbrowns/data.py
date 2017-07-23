# locality/data.py

import numpy as np
import zipfile
from copy import deepcopy
from random import randint
from locality.random.sample import sample

def check_if_indices(ids):
    if ids.dtype != np.int:
        return False
    if ids[0] != 0:
        return False
    return np.all(ids == np.arange(ids.size))

def wrap_data(arg1):
    if isinstance(arg1, BaseData):
        return arg1
    if isinstance(arg1, np.ndarray):
        return MemoryArray(arg1)
    if isinstance(arg1, np.memmap):
        return MemMapArray(arg1)
    if isinstance(arg1, str):
        raise NotImplementedError

class BaseData(object):
    def __init__(self, arg1, copy=False):
        """
        An abstract class used as an adapter for read-only data structures.
        """
        pass

    def _set_attrs(self, other, copy=False):

        if copy:
            other = other.copy()

        for key, item in other.__dict__.iteritems():
            self.__dict__[key] = item

    def open(self):
        pass

    def close(self):
        pass

    def sample(self, k, rstate=None):
        """ Return k random samples from data structure."""
        pass

    def get_data(self):
        """Returns data in an array"""
        pass

    def get_ids(self):
        """Returns ids in an array"""
        pass

    def get_data_by_ids(self, item_ids):
        """
        Returns item(s) in data associated with id unique within the data
        structure.
        """
        pass

    def get_data_by_id_range(self, lo, hi):
        """
        Returns item(s) in data within a particular range.
        """
        pass

    def parition(self, max_size):
        """
        Iterator that returns slice objects for large blocks of data.
        """
        pass

    def copy(self):
        return deepcopy(self)


class MemoryArray(BaseData):
    """Wrapper for numpy ndarray."""

    def __init__(self, arg1, copy=False):
        self._name = "numpy ndarray"

        if isinstance(arg1, MemoryArray):
            self._set_atrrs(arg1, copy)
        else:
            try:
                self.data = np.ascontiguousarray(arg1)
                self.shape = self.data.shape
            except ValueError:
                raise ValueError("Data object could not be converted to numpy "
                                 "array.")

    def __getitem__(self, index):
        return self.data.__getitem__(index)

    def sample(self, m, s, rstate=None):
        idx = sample(self.shape[0], m, s, rstate=rstate)
        unique_idx, inverse = np.unique(idx, return_inverse=True)
        data = self.data[unique_idx]
        return inverse, data

    def get_data(self, slice_obj=None):
        if slice_obj is None:
            return self.data
        return self.data[slice_obj[0]:slice_obj[1]]

    def get_ids(self, index):
        if isinstance(index, int):
            return index
        if isinstance(index, slice):
            return np.arange(index.start, index.stop, index.step)
        if isinstance(index, np.ndarray):
            return index

    def get_data_by_id(self, item_ids):
        return self.data[item_ids]

    def partition(self, num_workers, max_size):
        _n = self.n
        block_size = _n // num_workers
        if block_size > max_size:
            block_size = max_size

        i = 0
        j = block_size
        while i < self.data.shape[0]:
            k = j + block_size
            if k > _n:
                yield slice(i, _n, 1)
                break
            else:
                yield slice(i, j, 1)
            i = j
            j = k

    @property
    def ids(self):
        return np.arange(self.shape[0])

    @property
    def n(self):
        return self.shape[0]

    @property
    def dim(self):
        return self.shape[1]

    @property
    def data_dtype(self):
        return self.data.dtype

    @property
    def ids_dtype(self):
        return np.int


class MemMapArray(MemoryArray):

    def __init__(self, arg1, shape=None, dtype=None, copy=False):
        if isinstance(arg1, MemMapArray):
            self._set_self(arg1, copy)
        elif isinstance(arg1, np.memmap):
            self.data = arg1
            self.fid = arg1.filename
            self.shape = arg1.shape
            self.data_dtype = arg1.dtype
        elif isinstance(arg1, str):
            self.fid = arg1
            if self.fid.endswith(".npy"):
                npy = open(self.fid)
                version = np.lib.format.read_magic(npy)
                header = np.lib.format._read_array_header(npy, version)
                self.shape = header[0]
                self.data_dtype = header[2]
            elif self.fid.endswith(["dat", "txt"]):
                if shape is None or dtype is None:
                    raise ValueError("Shape and dtype must be specified for "
                                     "binary files.")
                self.shape = shape
                self.data_dtype = dtype
            else:
                raise NotImplementedError("File format not supported.")

    def open(self):
        if self.fid.endswith(".npy"):
            self.data = np.load(self.fid, mmap_mode="r")
        elif self.fid.endswith(".dat", ".txt"):
            self.data = np.memmap(self.fid, self.data_dtype, "r", self.offset,
                                  self.shape, "C")

    def close(self):
        self.data = None


class MemoryFileData(BaseData):
    pass

class MemMapFileData(BaseData):
    pass

class InFileData(BaseData):
    """
    Data in file on disk.
    """
    _name = 'in file'

    def __init__(self, arg1, header=True, rownames=True,
                 sep='\t', dtype=np.double, copy=False):

        if isinstance(arg1, InFileData):
            self._set_self(arg1, copy)
        elif isinstance(arg1, str):
            self.filepath = arg1
            self.header = header
            self.rownames = rownames
            self.cached = False
            self.linecache = []
            with open(self.filepath) as f:
                if header:
                    f.readline()
                line1 = f.readline().rstrip(sep).split(sep)
                self.dim = len(line1)

    def _set_self(self, other, copy=False):

        if copy:
            other = other.copy()

        self.filepath = other.filepath
        self.header = other.header
        self.rownames = other.rownames
        self.cached = other.cached
        self.linecache = other.linecache
        self.dim = other.dim

    def sample(self, k):

        reservoir = [0] * k
        c = 0
        if self.cached:
            # sample from
            s = sample(len(self.linecache) - 1, 1, k)
            with open(self.filepath) as f:
                for i, j in enumerate(s):
                    a = self.linecache[j]
                    b = self.linecache[j + 1]
                    f.seek(a - c, 1)
                    reservoir[i] = f.read(b - a)
                    c = b
        else:
            # use standard reservoir sampling while caching file line positions
            self.linecache.append(0)
            with open(self.filepath) as f:

                s = np.empty(k, dtype=np.int)

                for i in range(k):
                    s[k] = i
                    line = f.readline()
                    if line is '':
                        break
                    self.linecache.append(f.tell())
                    reservoir.append(line)

                if i < k - 1:
                    return reservoir

                i = k
                while True:
                    line = f.readline()
                    if line is '':
                        break
                    self.linecache.append(f.tell())
                    j = randint(0, i)
                    if j < k:
                        s[j] = i
                        reservoir[j] = line
                    i += 1

            self.cached = True

        return np.fromstring(''.join(reservoir), self.dtype, self.sep)

    def get_items(self, item_ids):

        if self.cached:

            n = item_ids.size
            sorted_ids = np.sort(item_ids)
            lines = [0] * n

            with open(self.filepath) as f:

                c = 0
                for i in range(n):
                    j = sorted_ids[i]
                    a = self.linecache[j]
                    b = self.linecache[j + 1]
                    f.seek(a - c, 1)
                    lines.append(f.read(b - a))
                    c = b

            return np.fromstring(''.join(lines), dtype=self.dtype,
                                 sep=self.sep)
        else:
            raise ValueError("File line positions have not been cached")

    def get_block(self, num_workers, MAX_BLOCK_SIZE=1000):

        if self.cached:

            n = len(self.linecache) - 1

            with open(self.filepath) as f:

                if self.rownames:
                    f.readline()

                block_size = n // num_workers
                if block_size >= MAX_BLOCK_SIZE:
                    block_size = MAX_BLOCK_SIZE

                i = 0
                while i < len(self.filecache):
                    try:
                        lines = f.read(self.linecache[i + block_size])
                        out = np.fromstring(''.join(lines), dtype=self.dtype,
                                            sep=self.sep)
                        ids = np.arange(i, i + block_size)
                    except IndexError:
                        lines = f.read()
                        out = np.fromstring(''.join(lines), dtype=self.dtype,
                                            sep=self.sep)
                        ids = np.arange(i, n)

                    i += block_size

                    yield (ids, out)


        else:

            k = 0
            self.filecache.append(0)

            with open(self.filepath) as f:

                if self.rownames:
                    f.readline()

                lines = [0] * MAX_BLOCK_SIZE

                while True:

                    line = f.readline()

                    if line is '':
                        break

                    lines[i] = line
                    self.linecache.append(f.tell())
                    i += 1

                    if i % MAX_BLOCK_SIZE == 0:

                        out = np.fromstring(''.join(lines), dtype=self.dtype,
                                            sep=self.sep)
                        ids = np.arange(i - MAX_BLOCK_SIZE, i)

                        yield (ids, out)

                if i % MAX_BLOCK_SIZE != 0:

                    num_lines = (i % MAX_BLOCK_SIZE)
                    block_size = num_lines // num_workers

                    n = i - num_lines
                    j = 0

                    for _ in range(num_workers - 1):

                        out = np.fromstring(''.join(lines[j:j + block_size]),
                                            dtype=self.dtype, sep=self.sep)
                        ids = np.arange(n, n + block_size)

                        j += block_size
                        n += block_size

                        yield (ids, out)

                    out = np.fromstring(''.join(lines[j:num_lines]))
                    ids = np.arange(n, i)

                    yield(ids, out)

            self.cached = True

def HDF5Data(BaseData):
    pass

def SQLiteData(BaseData):
    pass

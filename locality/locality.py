import numpy as np
import os

from copy import deepcopy
from locality.data import BaseData, wrap_data
from locality.exception_wrap import ExceptionWrapper
from multiprocessing import Pipe, Process, Queue
from queue import Full, Empty
from itertools import chain
from warnings import warn
from sortedcontainers.sortedlist import SortedList
from locality.distance.distance_wrapped import _metric_names, cdist_wrapped, ssdist_wrapped
from locality.random.sample import sample_intervals
from locality.storage import storage
from locality.random.stable import random_stable
from locality.helpers.bitarray import bitdot
from locality.helpers.encode import encode_by_bits, encode_by_place
from locality.helpers.hash_argmin import hash_argmin
from locality.helpers.rank import rank


class HashWarning(UserWarning):
    pass


class LSH(object):

    def __init__(self, arg1, L=1, storage_type="dict", storage_config=None,
                 storage_fid=None, copy=False, **kwargs):
        """Abstract constructor, calls subclass method _init_build.

        Args:
            arg1 (object): Python object
            L (int): number of hashtables
            storage_type (str): type of storage
            storage_config (dict): dictionary of storage-specific configuration
                options
            storage_fid (str): filepath for persistent storage formats
            copy (boolean): If arg1 is same type as self, whether to make copy
            **kwargs: LSH type specific keyword arguments, see _init_build

        Raises:
            NotImplementedError: If one tries to use LSH constructor directly
        """
        if self.__class__ is LSH:
            raise NotImplementedError
        if isinstance(arg1, self.__class__):
            self._set_attrs(arg1, copy)
        else:
            self.data = wrap_data(arg1)
            storage_config = {storage_type: storage_config}
            self._init_hashtables(L, storage_config, storage_fid)
            self._init_build(**kwargs)

    def _set_attrs(self, other, copy=False):

        if copy:
            other = other.copy()

        for key, item in other.__dict__.iteritems():
            self.__dict__[key] = item

    def copy(self):
        """Returns a copy of this object."""
        return deepcopy(self)

    def _init_hashtables(self, L, storage_config, storage_fid=None):
        """Initializes hashtables."""
        self.L = L
        self.hashtables = storage(L, storage_config, storage_fid)

    def _rebuild(self, block_size):
        if self._change_flag:
            self._build(block_size)
            self._change_flag = False

    def _hash_in_serial(self):
        _L = self.L
        _selected = self._selected
        _b = self.b
        _data = self.data
        _pack = self.hashtables.pack_hashes
        _hash = self._hash
        _encode = self._encode

        tmp = _hash(_data.get_data())
        hashlist = _encode(tmp, _b).reshape(-1, _L)
        _pack(_data.get_ids(), hashlist)

    def _hash_in_parallel(self, error_queue, kill_queue, in_queue, out_queue,
                          hash_func, encode_func, max_size, data):
        try:
            _L = self.L
            _b = self.b
            _kill_get = kill_queue.get
            _in_get = in_queue.get
            _out_put = out_queue.put
            _get_data = data.get_data

            dist_calc, hash_out, encode_out = self._init_hash_memory(max_size)

            while True:
                try:
                    _kill_get(False)
                    return
                except Empty:
                    pass

                slice_obj = _in_get()
                if slice_obj == "kill":
                    break

                tmp1 = hash_func(_get_data(slice_obj), dist_calc, hash_out)
                tmp2 = encode_out[:tmp1.shape[0]]
                hashlist = encode_func(tmp1, _b, out=tmp2)
                hashlist = hashlist.reshape(-1, _L)
                result = (slice_obj, hashlist)
                _out_put(result)
        except Exception as e:
            error_queue.put(ExceptionWrapper(os.getpid(), e))

    def _query_in_serial(self, y, q):

        _L = self.L
        _b = self.b
        _metric = self.metric
        _margs = self.margs
        _where = np.where
        _empty = np.empty
        _unique = np.unique
        _get_data = self.data.get_data
        _get_data_by_id = self.data.get_data_by_id
        _get_lists = storage.get_lists
        _hash = self._hash
        _encode = self._encode

        result = SortedList()
        _add = result.add

        q_data = y.get_data()
        num_points = q_data.shape[0]

        tmp = _hash(q_data)
        hashlist = _encode(tmp, _b).reshape(-1, _L)
        indptr, cand = _get_lists(hashlist)

        c_ids, c_indices = _unique(cand, return_inverse=True)
        c_data = _get_data_by_id(c_ids)

        dist = ssdist_wrapped(q_data, c_data, None, c_indices, indptr, _metric,
                              _margs)
        nn_indices = rank(q, num_points, dist.data, c_indices, indptr)

        q_ids = y.get_ids()
        nn_ids = _where(nn_indices > 0, c_ids[nn_indices], -1)

        _m = q_ids.shape[0]

        j = 0
        for i in range(_m):
            _add((q_ids[i], tuple(nn_ids[j:j + q])))
            j += q

        return result

    def _query_in_parallel(self, error_queue, kill_queue, in_queue, out_queue,
                           hash_func, encode_func, max_size, y,
                           q, data, storage):
        try:
            _L = self.L
            _b = self.b
            _metric = self.metric
            _margs = self.margs
            _where = np.where
            _empty = np.empty
            _unique = np.unique
            _get_data = data.get_data
            _get_data_by_id = data.get_data_by_id
            _get_lists = storage.get_lists

            dist_calc, hash_out, encode_out = self._init_hash_memory(max_size)
            indptr_out, cand_out = self._init_query_memory(max_size, data.n,
                                                           data.ids_dtype)
            while True:
                try:
                    kill_queue.get(False)
                    return
                except Empty:
                    pass

                slice_obj = in_queue.get()
                if slice_obj == "kill":
                    break

                q_data = y.get_data(slice_obj)
                num_points = q_data.shape[0]

                tmp1 = hash_func(q_data, dist_calc, hash_out)
                tmp2 = encode_out[:tmp1.shape[0]]
                hashlist = encode_func(tmp1, _b, out=tmp2)
                hashlist = hashlist.reshape(-1, _L)
                indptr, cand = _get_lists(hashlist, indptr_out, cand_out)

                c_ids, c_indices = _unique(cand, return_inverse=True)
                c_data = _get_data_by_id(c_ids)

                dist = ssdist_wrapped(q_data, c_data, None, c_indices, indptr,
                                      _metric, _margs)
                nn_indices = rank(q, num_points, dist.data, c_indices,
                                  indptr)

                result = (slice_obj,
                          _where(nn_indices > 0, c_ids[nn_indices], -1))
                out_queue.put(result)
        except Exception as e:
            error_queue.put(ExceptionWrapper(os.getpid(), e))

    def _store_when_available(self, error_queue, kill_queue, in_queue,
                              result_send, data, storage):
        _kill_get = kill_queue.get
        _in_get = in_queue.get
        _pack = storage.pack_hashes
        _get_ids = data.get_ids

        try:
            while True:
                try:
                    _kill_get(False)
                    result_send.send(None)
                    return
                except Empty:
                    pass

                item = _in_get()
                if item == "kill":
                    break

                slice_obj, hashlist = item
                _pack(_get_ids(slice_obj), hashlist)

            result_send.send(storage)
        except Exception as e:
            result_send.send(None)
            error_queue.put(ExceptionWrapper(os.getpid(), e))

    def _merge_when_available(self, error_queue, kill_queue, in_queue,
                              result_send, y, q):
        try:
            result = SortedList()
            _in_get = in_queue.get
            _kill_get = kill_queue.get
            _get_ids = y.get_ids
            _add = result.add

            while True:

                try:
                    _kill_get(False)
                    result_send.send(None)
                    return
                except Empty:
                    pass

                item = _in_get()
                if item == "kill":
                    break

                slice_obj, nn_ids = item
                q_ids = _get_ids(slice_obj)
                _n = q_ids.shape[0]

                j = 0
                for i in range(_n):
                    _add((q_ids[i], tuple(nn_ids[j:j + q])))
                    j += q

            result_send.send(result)
        except Exception as e:
            result_send.send(None)
            error_queue.put(ExceptionWrapper(os.getpid(), e))

    def _distribute_and_wait(self, error_queue, error_send, kill_recv,
                             kill_queue, in_queue, block_size, num_processes,
                             data):
        try:
            it = chain(data.partition(num_processes - 2, block_size),
                       ["kill"] * (num_processes - 2))
            distributed = False
            finished = False
            elem = None

            while not finished:
                if kill_recv.poll():
                    error_send.send(None)
                    break

                try:
                    e = error_queue.get(False)
                    for _ in range(num_processes - 1):
                        kill_queue.put(None)
                    error_send.send(e)
                    break
                except Empty:
                    pass

                if not distributed:
                    try:
                        while True:
                            if elem is None:
                                elem = next(it)
                            in_queue.put(elem, False)
                            elem = None
                    except StopIteration:
                        distributed = True
                    except Full:
                        pass
        except Exception as e:
            for _ in range(num_processes - 1):
                kill_queue.put(None)
            error_send.send(ExceptionWrapper(os.getpid(), e))

    def _mp_template(self, listener_args, aggregator_func, aggregator_args,
                     worker_func, worker_args, num_processes, block_size):
        try:
            error_queue = Queue(num_processes)
            in_queue = Queue(num_processes - 2)
            out_queue = Queue(num_processes - 2)
            kill_queue = Queue(num_processes - 1)
            kill_recv, kill_send = Pipe(duplex=False)
            error_recv, error_send = Pipe(duplex=False)
            result_recv, result_send = Pipe(duplex=False)

            max_size = self.data.n // (num_processes - 2)
            max_size = block_size if max_size > block_size else max_size
            max_size = max_size + self.data.n % max_size

            listener_args = (error_queue, error_send, kill_recv, kill_queue,
                             in_queue, block_size,
                             num_processes) + listener_args
            listener = Process(target=self._distribute_and_wait,
                               args=listener_args)
            listener.start()

            aggregator_args = (error_queue, kill_queue, out_queue,
                               result_send) + aggregator_args
            aggregator = Process(target=aggregator_func, args=aggregator_args)
            aggregator.start()

            workers = []
            worker_args = (error_queue, kill_queue, in_queue, out_queue,
                           self._hash, self._encode, max_size) + worker_args
            for _ in range(num_processes - 2):
                w = Process(target=worker_func, args=worker_args)
                w.start()
                workers.append(w)

            for w in workers:
                w.join()

            if result_recv.poll():
                h = result_recv.recv()
                assert(h is None)
                kill_send.send(None)
                e = error_recv.recv()
                assert(e is not None)
                e.re_raise()

            out_queue.put("kill")
            result = result_recv.recv()
            aggregator.join()

            kill_send.send(None)
            if result is None:
                e = error_recv.recv()
                assert(e is not None)
                e.re_raise()
            listener.join()

            return result

        except Exception as e:
            listener.terminate()
            aggregator.terminate()
            for w in workers:
                w.terminate()
            raise e

    def _hash_all(self, parallel=True, num_processes=os.cpu_count() + 2,
                  block_size=1500,):
        """Hash and store references for all objects in data set in serial or
        parallel."""
        if not parallel:
            self._hash_in_serial()
        else:
            listener_args = (self.data,)
            aggregator_func = self._store_when_available
            aggregator_args = (self.data, self.hashtables)
            worker_func = self._hash_in_parallel
            worker_args = listener_args
            self.hashtables = self._mp_template(listener_args, aggregator_func,
                                                aggregator_args, worker_func,
                                                worker_args, num_processes,
                                                block_size)

#     def query_all(self, k):
#         hash_func = self._get_hash_func()
#         encode_func = self._get_encode_func()
#         return self._lquery(k, self.data.all_ids(), hash_func, encode_func)
#
#     def _query_by_id(self, q, query_ids, hash_func, encode_func, *,
#                 num_processes=4,
#                 MAX_PER_WORKER=1000, wait_time=0.001):
#
#         # create queues
#         manager = Manager()
#         error_queue = manager.Queue(num_processes)
#         kill_switches = [manager.Event() for _ in range(num_processes)]
#         meta_in_queue = manager.Queue(num_processes - 1)
#         meta_out_queue = manager.Queue(num_processes - 1)
#         result_in, result_out = Pipe()
#
#         # listener function
#         def merger_func(error_queue, kill_switch, in_queue, result_in):
#
#             try:
#                 while True:
#
#                     if kill_switch.is_set():
#                         break
#
#                     item = in_queue.get()
#                     if item == "kill":
#                         break
#
#                     result += item
#
#                 result_in.send(result)
#
#             except Exception as e:
#                 error_queue.put(ExceptionWrapper(e))
#
#         # worker function
#         def worker_func(error_queue, kill_switch, in_queue, out_queue,
#                         q, hash_func, encode_func):
#             try:
#                 while True:
#
#                     if kill_switch.is_set():
#                         break
#
#                     item = in_queue.get()
#                     if item == "kill":
#                         break
#
#                     num_points, q_ids, q_data = item
#
#                     result = self._squery(q, num_points, q_ids,
# q_data,
#                                           hash_func, encode_func)
#                     out_queue.append(result)
#
#             except Exception as e:
#                 error_queue.put(ExceptionWrapper(e))
#
#         # start merger
#         merger_args = (error_queue, kill_switches[0], meta_out_queue,
# result_in)
#         merger = Process(merger_func, merger_args)
#         merger.start()
#
#         # start workers
#         pool = []
#         for kill_switch in kill_switches[1:]:
#             worker_args = (error_queue, kill_switch, meta_in_queue,
#                            meta_out_queue, q, hash_func, encode_func)
#             p = Process(worker_func, worker_args)
#             p.daemon = False
#             p.start()
#             pool.append(p)
#
#         def mini_blocks(query_ids, data, num_processes,
#                         MAX_PER_WORKER=MAX_PER_WORKER):
#
#             num_points = query_ids.size
#             per_worker = num_points // num_processes
#             if per_worker >= MAX_PER_WORKER:
#                 per_worker = MAX_PER_WORKER
#
#             i = 0
#             while True:
#                 if i == num_points:
#                     break
#
#                 #  block to in_queue
#                 q_ids = query_ids[i:i + per_worker]
#                 q_data = data.get(q_ids)
#
#                 yield (q_ids.shape[0], q_ids, q_data)
#                 i += per_worker
#
#         it = mini_blocks(query_ids, self.data, num_processes - 1)
#         poll_func = lambda x: x.is_alive()
#
#         self._distribute_and_wait(it, error_queue, kill_switches,
#                              meta_in_queue, pool, poll_func, num_processes,
#                              wait_time)
#
#         # kill merger
#         meta_out_queue.put("kill")
#
#         # wait for result
#         result = result_out.recv()
#         return result

    def _query(self, y, q, parallel=True,
               num_processes=os.cpu_count() + 2, block_size=1500):
        if not parallel:
            return self._query_in_serial()
        else:
            listener_args = (y,)
            aggregator_func = self._merge_when_available
            aggregator_args = (self.data, self.hashtables)
            worker_func = self._query_in_parallel
            worker_args = (y, q, self.data, storage)
            return self._mp_template(listener_args, aggregator_func,
                                     aggregator_args, worker_func, worker_args,
                                     num_processes, block_size)

    def query(self, y, q, by_id=False, block_size=1e6,
              num_processes=os.cpu_count() + 2):
        """
        Args:
            y (object): Data points to be queried
            q (int): Number of neighbors to return of each point
            num_processes (int): (optional) Number of subprocesses, defaults to
                number of cores plus 2.

        Returns:
            A SortedList mapping each identifier in `y` to a tuple of
            identifiers from the data set representing its `q` nearest
            neighbors.

            For example:

            SortedList([(0, (32266, 49414, 2423, 42492, 75357)),
                        (1, (35063, 42229, 70142, 40291, -1)),
                        (2, (70687, 38807, 40571, 59077, 89853))], load=1000)

        """
        if by_id:
            if y.shape[0] < block_size:
                query_ids = y
                y = self.data.get_data_by_id(query_ids)
            else:
                query_ids = y
                return self._lquery(q, query_ids)
        else:
            query_ids = None

        if not issubclass(y.__class__, BaseData):
            try:
                y.shape[1]
            except IndexError:
                y = y.reshape(1, -1)
            y = wrap_data(y)

        _data_dim = self.data.shape[1]
        if y.shape[1] != _data_dim:
            raise ValueError(
                "Points {} dimensions.".format(
                        "have too many" if y.shape[1] > _data_dim
                        else "are missing"))

        return self._query(y, q, num_processes)


class VoroniLSH(LSH):

    def _init_build(self, k=4, m=10, w=3, metric="euclidean", **kwargs):
        """
        Args:
            k (int): number of hash functions
            m (int): number of medoids
            w (int): number of medoids per hash function
            metric (string or callable): name of or reference to metric
                function
            **kwargs: Keyword arguments used by cdist/ssdist or user defined
                metric function

        Raises:
            ValueError: If any of the arguments are invalid (see _build)
            NotImplementedError: If `metric` is not supported
        """
        self.k = k
        self.m = m
        self.w = w
        self.metric = metric
        self.margs = kwargs

        self._init_encoding()
        self._init_medoids()
        self._init_hash_funcs()
        self._build(1500)

    def _init_encoding(self):
        self._encode = encode_by_bits
        b = int(np.floor(np.log(self.w - 1) / np.log(2)) + 1)
        d = int(np.ceil(np.log(self.k * b) / np.log(8)))
        self.b = b
        if d > 8:
            warn("Expected hash is longer than 64bit integer, will likely "
                 "result in increased collisions and decreased performance.",
                 HashWarning)

    def _init_medoids(self):
        self._medoid_indices, self.medoids = self.data.sample(self.L, self.m)

    def _init_hash_funcs(self):
        _m = self.m
        n = _m * self.L
        tmp = sample_intervals(n, _m, self.k, self.w)
        tmp = self._medoid_indices.take(tmp)
        s, i = np.unique(tmp, return_inverse=True)
        if s.size == self.medoids.shape[0]:
            self._selected = None
        else:
            self._selected = s
        self._selected_inverse = i.astype(np.intp)

    def _build(self, block_size):
        if self.L < 1:
            raise ValueError("There must be at least one hashtable.")
        if self.k < 1:
            raise ValueError("There must be at least one hash function.")
        if self.w < 2:
            raise ValueError("There must be at least two medoids per hash "
                             "function.")
        if self.w > self.m:
            raise ValueError("The number of medoids used in each hash "
                             "function is greater than the total number "
                             "of chosen medoids.")
        if isinstance(self.metric, str):
            if self.metric not in _metric_names:
                raise NotImplementedError("Metric not available using "
                                          "built-in distance functions. Try "
                                          "passing a custom function to "
                                          "\"metric\".")
        elif not callable(self.metric):
            raise ValueError("Metric must be a string or a callable.")
        self._hash_all(block_size)

    def _hash(self, points, dist_calc=None, out=None):
        size = points.shape[0]
        size *= self._selected_inverse.size // self.w
        if out is None:
            out = np.empty(size, dtype=np.int)
        else:
            assert out.flags.contiguous, "Output array must be contiguous."
            assert np.prod(out.shape) >= size, (
                "Not enough space in provided output array.")
            out = out.ravel()[:size]

        if self._selected is None:
            tmp = self.medoids
        else:
            tmp = self.medoids.take(self._selected, axis=0)
        dist = cdist_wrapped(points, tmp, self.metric, out=dist_calc,
                             **self.margs)
        result = hash_argmin(dist, self._selected_inverse, self.w, out)
        return result.reshape(-1, self.k)

    def _init_hash_memory(self, max_size):
        hash_out_size = max_size
        hash_out_size *= (self._selected_inverse.size // self.w)
        if self._selected is None:
            dist_calc_size = self.m * self.L
        else:
            dist_calc_size = self._selected.size
        dist_calc = np.empty((max_size, dist_calc_size), dtype=np.double)
        hash_out = np.empty(hash_out_size, dtype=np.int)
        encode_out = np.empty(max_size * self.L, dtype=np.uint)
        return dist_calc, hash_out, encode_out

    def _init_query_memory(self, max_size, num_points, dtype):
        tmp = self.L * self.w ** self.k
        exp_size = num_points / tmp
        indptr_out = np.empty(max_size + 1, dtype=np.intp)
        cand_out = np.empty(int(np.ceil(exp_size)), dtype=dtype)
        return indptr_out, cand_out


class CosineLSH(LSH):

    def _init_build(self, k=4):
        """
        Args:
            k (int): number of hash functions

        Raises:
            ValueError: If any of the arguments is invalid (see _build)
        """
        self.k = k
        self.w = 2
        self.metric = "cosine"

        self._init_encoding()
        self._init_vectors()
        self._build(1500)

    def _init_encoding(self):
        self._encode = encode_by_bits
        self.b = 1
        d = int(np.ceil(np.log(self.k) / np.log(8)))
        if d > 8:
            warn("Expected hash is longer than 64bit integer, will likely "
                 "result in increased collisions and decreased performance.",
                 HashWarning)

    def _init_vectors(self):
        n = self.L * self.k
        max_uint = np.iinfo(np.uint).max
        size = (n, int(np.ceil(self.data.dim / 64)))
        self.bitvecs = np.random.randint(max_uint, size=size, dtype=np.uint)

    def _build(self, block_size):
        if self.L < 1:
            raise ValueError("There must be at least one hashtable.")
        if self.k < 1:
            raise ValueError("There must be at least one hash function.")
        self._hash_all(block_size)

    def _hash(self, points, dist_calc=None, out=None):
        result = bitdot(points, self.vecs, out)
        return result.reshape(-1, self.k)

    def _init_hash_memory(self, max_size):
        dist_calc = None
        hash_out = np.empty(max_size * self.vecs.shape[0], dtype=np.int)
        encode_out = np.empty(max_size * self.L, dtype=np.uint)
        return dist_calc, hash_out, encode_out

    def _init_query_memory(self, max_size, num_points, dtype):
        exp_size = num_points / 2 ** self.k
        indptr_out = np.empty(max_size + 1, dtype=np.intp)
        cand_out = np.empty(int(np.ceil(exp_size)), dtype=dtype)
        return indptr_out, cand_out


class StableLSH(LSH):

    def _init_build(self, k=4, r=1, el=2, metric="euclidean"):
        """
        Args:
            k (int): number of hash functions
            r (int): radius/scale of hashes
            el (float): parameter on (0,2]
            metric (string or callable): name of or reference to metric
                function
            **kwargs: Keyword arguments used by cdist/ssdist or user defined
                metric function

        Raises:
            ValueError: If any parameter are invalid (see _build)
            NotImplementedError: If metric is not supported
        """
        self.k = k
        self.r = r
        self.el = el
        self.metric = metric

        self._init_encoding()
        self._init_vectors()
        self._build(1500)

    def _init_encoding(self):
        self._encode = encode_by_place
        self.b = 8

    def _init_vectors(self):
        """Initiates random vectors."""
        n = self.L * self.k
        self.alpha = random_stable(self.el, shape=(n, self.data.dim))
        self.beta = np.random.uniform(0, self.r, n)

    def _build(self, block_size):
        if self.el is None and self.metric is None:
            raise ValueError("Must specify el or metric.")
        elif self.el is None:
            if self.metric == "euclidean":
                self.el = 2
            elif self.metric in {"lnorm", "l-norm", "pnorm", "p-norm",
                                 "minkowski"}:
                raise ValueError("Must specify el.")
            else:
                raise NotImplementedError("Metric not supported for random "
                                          "projection. Try VoroniLSH.")
        elif self.metric is None:
            if self.el == 2:
                self.metric = "euclidean"
            if self.el < 0 or self.el > 2:
                raise ValueError("Parameter el must be on [0,2].")
            else:
                self.metric = "lnorm"
        self._hash_all(block_size)

    def _hash(self, points, dist_calc=None, out=None):
        n, m = points.shape[0], self.alpha.shape[0]
        size = n * m
        if dist_calc is not None:
            assert dist_calc.flags.contiguous, (
                "Output array must be contiguous.")
            assert np.prod(dist_calc.shape) >= size, (
                "Not enough space in provided output array.")
            dist_calc = dist_calc.ravel()[:size].reshape(n, m)
        if out is not None:
            assert out.flags.contiguous, "Output array must be contiguous."
            assert np.prod(out.shape) >= size, (
                "Not enough space in provided output array.")
            out = out.ravel()[:size].reshape(n, m)

        result = np.dot(points, self.alpha.T, out=dist_calc)
        result += self.beta
        result = np.floor((1 / self.r) * result, out=out)
        return result.reshape(-1, self.k)

    def _init_hash_memory(self, max_size):
        size = max_size * self.vecs.shape[0]
        dist_calc = np.empty(size, dtype=np.double)
        hash_out = np.empty(size, dtype=np.int)
        encode_out = np.empty(max_size * self.L, dtype=np.uint)
        return dist_calc, hash_out, encode_out

    def _init_query_memory(self, max_size, num_points, dtype):
        exp_size = num_points / self.r
        indptr_out = np.empty(max_size + 1, dtype=np.intp)
        cand_out = np.empty(int(np.ceil(exp_size)), dtype=dtype)
        return indptr_out, cand_out

import numpy as np
import locality.data as dat
import os, time

from locality.exception_wrap import ExceptionWrapper
from multiprocessing import Manager, Pipe, Process, Queue, active_children
from queue import Full, Empty
from itertools import chain
from locality.distance.distance_wrapped import cdist_wrapped, ssdist_wrapped
from locality.random.sample import sample_intervals
from locality.storage import storage, InMemoryStorage
from locality.random.stable import random_stable
from locality.helpers.bitarray import bitdot
from locality.helpers.encode  import encode_to_int, encode_to_string
from locality.helpers.rank import rank
from math import ceil, floor, log
from sortedcontainers.sortedlist import SortedList

class LSH(object):

    def __init__(self, data, num_hashtables=1, num_func=4, method='voroni',
                 metric='euclidean', num_medoids=5, hash_size=3, radius=1, el=2,
                 max_size=1000, 
                 storage_filepath=None, storage_config=None, test=False, **kwargs):
        
        # initialize data object
        if not isinstance(data.__class__, dat.BaseData):
            if isinstance(data, np.ndarray):
                self.data = dat.MemoryArrayData(data,max_size=max_size)
            elif isinstance(data, str):
                try:
                    self.data = dat.InFileData(data)
                except OSError:
                    pass
            else:
                try:
                    self.data = dat.MemoryArrayData(np.asarray(data))
                except ValueError:
                    raise ValueError('Data object is not a filepath nor can be convered to ndarray')
        
        self.num_func = num_func
        self.num_medoids = num_medoids
        self.hash_size = hash_size
        self.num_hashtables = num_hashtables
        self.method = method
        self.metric = metric
        self.radius = radius
        self.el = el
        self.margs = kwargs
        
        if storage_config is None:
            storage_config = {'dict': None}
        
        if not test:
            self._init_hashtables(storage_config, storage_filepath)
            self._init_build()  

    def _init_build(self):
        
        self._set_encoding_method()
        
        if self.method == 'random':
            self._init_vecs()
        
        if self.method == 'voroni':
            self._init_medoids()
            self._init_hash_funcs()
        
        self._build()
            
    def _init_hashtables(self, storage_config, storage_filepath=None):
        """
        Initialize hashtables
        """
        self.hashtables = storage(self.num_hashtables, storage_config, 
                               storage_filepath)
             
    def _init_medoids(self):
        """
        
        """
        self._medoid_indices, self.medoids = self.data.sample(self.num_hashtables, 
                                                         self.num_medoids)

    def _init_hash_funcs(self):
        """
        
        """
        n = self.num_medoids * self.num_hashtables
        self.hash_funcs = sample_intervals(n, self.num_medoids, self.num_func, 
                                           self.hash_size)
        tmp = self._medoid_indices.take(self.hash_funcs)
        self._selected, self._selected_inverse = np.unique(tmp, 
                                                           return_inverse=True) 
        if self._selected.size == self.medoids.shape[0]:
            self._selected = 'all'
        
    def _init_vectors(self):
        """
        
        """
        n = self.num_hashtables * self.num_func * self.hash_size
       
        if self.metric == 'cosine':
            self.vecs = np.random.randint((n, ceil(self.ndim / 64)))
        else:
            self.vecs = random_stable(self.el, shape=(n, self.dims))
            self.b = np.random.uniform(0, self.radius, n)
        
    def _build(self):
        if self.num_func < 1:
            raise ValueError("There must be at least one hash function.")
        if self.method == "voroni" and self.hash_size > self.num_medoids:
            raise ValueError("Medoids used in each hash function is greater " + 
                             "than total number of medoids.")
        elif self.metric not in {'p-norm', 'pnorm','minkowski',
                                 'euclidean','cosine'}:
            raise ValueError("Metric not available using random projections. Try method = 'voroni'.")
        self._hash_all()
        
    def _build_random(self):
        if self.num_func < 1:
            raise ValueError('There must be at least one hash function.')
        
        self._random_hash_all()
        
    def _set_encoding_method(self):
        
        b = int(floor(log(self.hash_size)/log(2)) + 1)
        
        if self.method == 'voroni' and self.num_func * b <= 64:
            self._bitnum = b
            self._encode_method = 'int'
        else:
            self._bytenum = None
            self._encode_method = 'str'
    
    def _get_encode_func(self):
        if self._encode_method == 'int':
            return encode_to_int
        else:
            return encode_to_bytes
    
    def _voroni_hash(self, points):
        
        if self._selected is "all":
            tmp = self.medoids
        else:
            tmp = self.medoids.take(self._selected, axis=0)
        dist = cdist_wrapped(points, tmp, self.metric, **self.margs)
        dist = dist[:, self._selected_inverse]
        result = np.argmin(dist.reshape(-1, self.hash_size), axis=-1)
        return result.reshape(-1, self.num_func)
    
    def _cosine_hash(self, points):
        
        result = bitdot(points, self.vecs)
        
        return result.reshape(-1, self.num_hashtables)
    
    def _lnorm_hash(self, points):
        
        result = np.dot(points, self.vecs.T) + self.b 
        result = np.floor((1 / self.radius) * result)
        
        return result.reshape(-1, self.num_hashtables)

    def _get_hash_func(self):
        if self.method == 'voroni':
            return self._voroni_hash
        elif self.metric == 'cosine':
            return self._cosine_hash
        else:
            return self._lnorm_hash
         
    def rehash(self):
        self._init_medoids()
        self._init_hash_funcs()
        self._param_update = True
        self._rebuild()
     
    def _rebuild(self):
        if self._param_update:
            if self.method == 'random':
                self._build_random()
            if self.method == 'voroni':
                self._build_voroni()
            self._param_update = False
    
    def _hash_in_serial(self):
        
        _n = self.num_hashtables
        _b = self._bitnum
        _data = self.data
        _storage = self.storage
        
        hash_func = self._get_hash_func()
        encode_func = self._get_encode_func()
            
        hashlist = encode_func(hash_func(_data.get_data()), _b)
        hashlist = hashlist.reshape(-1, _n)
        _storage.pack_hashes(_data.get_ids(), hashlist)
        
        return
    
    def _hash_in_parallel(self, error_queue, kill_switch, in_queue, out_queue, 
                          hash_func, encode_func, data):
        
        try:
            
            _n = self.num_hashtables
            _b = self._bitnum
        
            while True:
                
                if kill_switch.is_set():
                    return
                
                slice_obj = in_queue.get()
                if slice_obj == "kill":
                    break
                
                hashlist = encode_func(hash_func(data.get_data(slice_obj)), _b)
                hashlist = hashlist.reshape(-1, _n)
                result = (slice_obj, hashlist)
                out_queue.put(result)
                
        except Exception as e:
            error_queue.put(ExceptionWrapper(os.getpid(), e))
            
    def _query_in_parallel(self, error_queue, kill_switch, in_queue, out_queue, 
                           query_data, query_num, hash_func, encode_func, data, 
                           storage):

        try:
            
            _n = self.num_hashtables
            _b = self._bitnum
            _metric = self.metric
            _margs = self.margs
            _where = np.where
            _empty = np.empty
            _unique = np.unique
            _get_data = data.get_data
            _get_data_by_id = data.get_data_by_id
            _get_cumsum  = storage.get_cumsum
            _get_lists = storage.get_lists
            
            while True:
                
                if kill_switch.is_set():
                    return
                
                slice_obj = in_queue.get()
                if slice_obj == "kill":
                    break
              
                q_data = query_data.get_data(slice_obj)
                q_ids = query_data.get_ids(slice_obj)
                num_points = q_data.shape[0]
                
                hashlist = encode_func(hash_func(q_data), _b)
                hashlist = hashlist.reshape(-1, _n)
                
                indptr, cand = _get_lists(hashlist)
                
                c_ids, c_indices = _unique(cand, return_inverse=True)
                c_data = _get_data_by_id(c_ids)
                
                dist = ssdist_wrapped(q_data, c_data, None, c_indices, indptr, 
                                      _metric, _margs)
                nn_indices = rank(query_num, num_points, dist.data, c_indices, 
                                  indptr)
            
                result = (q_ids, _where(nn_indices > 0, c_ids[nn_indices], -1))
                out_queue.put(result)
        
        except Exception as e:   
            error_queue.put(ExceptionWrapper(os.getpid(), e))
            
    def _store_when_available(self, error_queue, kill_switch, in_queue, 
                              result_out, data, storage):
       
        try:
            
            while True:
                
                if kill_switch.is_set():
                    return
                
                item = in_queue.get()
                if item == "kill":
                    break
                
                slice_obj, hashlist = item
                storage.pack_hashes(data.get_ids(slice_obj), hashlist)
               
            #result_out.put(storage)
        
        except Exception as e:    
            error_queue.put(ExceptionWrapper(os.getpid(), e))
            
    
    def _merge_when_available(self, error_queue, kill_switch, ready_switch,
                              in_queue, result_out, query_num):
        
        try:
            
            result = SortedList()
            
            while True:
                
                if kill_switch.is_set():
                    return
                
                item = in_queue.get()
                if item == "kill":
                    break
                
                q_ids, c_ids = item
                _n = q_ids.shape[0]
                
                j = 0
                for i in range(_n):
                    result.add((q_ids[i], tuple(c_ids[j:j + query_num])))
                    j += query_num
            
            ready_switch.set()
            result_out.put(result) 
        
        except Exception as e:
            error_queue.put(ExceptionWrapper(os.getpid(), e))
            
    
    def _distribute_and_wait(self, error_queue, error_send, self_kill_switch,
                             other_kill_switches, in_queue, data, 
                             num_processes):
        
        try:
            generator = chain(data.partition(num_processes - 2),
                              ["kill"] * (num_processes - 2))
            distributed = False
            #workers_finished = False
            finished = False
            elem = None
           
            while not finished:
                
                if self_kill_switch.is_set():
                    break
                
                try:
                    e = error_queue.get(False)
                    for k in other_kill_switches:
                        k.set()                   
                    error_send.send(e)
                    break
                except Empty:
                    pass
            
                if not distributed:
                    try:
                        while True:
                            if elem is None:
                                elem = next(generator)
                            in_queue.put(elem, False)
                            elem = None
                    except StopIteration:
                        distributed = True
                    except Full:
                        pass 
            
#                 elif not workers_finished:
#                     
#                     workers_finished = True
#                     
#                     for r in ready_switches[1:]:
#                         if not r.is_set():
#                             workers_finished = False
#                             break
#                     
#                     if workers_finished:
#                         out_queue.put("kill")
#                     
#                 elif ready_switches[0].is_set():
#                     print("exit")
#                     break
                    
        except Exception as e:
            for k in other_kill_switches:
                k.set()
            error_send.send(ExceptionWrapper(os.getpid(), e))
            return  
        
    def _hash_all(self, num_processes=os.cpu_count()+2):
        """
        Hash and store references for all objects in data set in parallel.
        """
        
        try:
            t1 = time.time()
            hash_func = self._get_hash_func()
            encode_func = self._get_encode_func()
        
            manager = Manager()
            manager.register(InMemoryStorage)
            tmpstorage = manager.InMemoryStorage
            error_queue = manager.Queue(num_processes)
            hash_in_queue = manager.Queue(num_processes - 2)
            hash_out_queue = manager.Queue(num_processes - 2)
            listener_kill_switch = manager.Event()
            other_kill_switches = [manager.Event() 
                                   for _ in range(num_processes - 1)]
            error_out = manager.Queue(1)
            result_out = manager.Queue(1)
            
            # start error listener
            listener_args =  (error_queue, error_out, listener_kill_switch, 
                              other_kill_switches, hash_in_queue, self.data, 
                              num_processes)
            listener = Process(target=self._distribute_and_wait, 
                               args=listener_args)
            listener.start()
            
            # start storage process
            packer_args = (error_queue, other_kill_switches[0], hash_out_queue, 
                           result_out, self.data, tmpstorage)
            packer = Process(target=self._store_when_available, 
                              args=packer_args)
            packer.start()

            # start worker processes
            workers = []
            for k in other_kill_switches[1:]:
                worker_args = (error_queue, k, hash_in_queue, hash_out_queue, 
                               hash_func, encode_func, self.data)
                w = Process(target=self._hash_in_parallel, args=worker_args)
                w.start()
                workers.append(w)
            
            for w in workers:
                w.join()
            t2 = time.time()
            hash_out_queue.put("kill")
            listener_kill_switch.set()  
            packer.join()
            t3 = time.time()
            listener.join()
            
            try:
                e = error_out.get(False)
                e.re_raise()
            except Empty:
                pass
            
            self.hashtables.storage = dict(tmpstorage.storage) 
            t4 = time.time()
            manager.shutdown()
            
            print(t2-t1, t3-t1, t4-t3)
        except Exception as e:
            
            manager.shutdown()
            listener.terminate()
            packer.terminate()
            for w in workers:
                w.terminate()
            raise e
            
    def query(self, query_data, query_num, by_id=False, MAX_SIZE=1e6):
        """
        Wrapper for _query and _query_by_id
        """
        if by_id:
            if query_data.shape[0] < MAX_SIZE:
                query_ids = query_data
                query_data = self.data.get_data_by_id(query_ids)
            else:
                query_ids = query_data
                return self._lquery(query_num, query_ids) 
        else:
            query_ids = None
        
        if not issubclass(query_data.__class__, dat.BaseData):
            try:
                query_data.shape[1]
            except IndexError:
                query_data = query_data.reshape(1,-1)
            query_data = dat.MemoryArrayData(query_data)
  
        _data_dim = self.data.shape[1]
        if query_data.shape[1] != _data_dim:
            raise ValueError(
                'Points {} dimensions.'.format(
                        'have too many' if query_data.shape[1] > _data_dim
                        else 'are missing'))
            
        return self._query(query_data, query_num)
                
    def query_all(self, k):
        hash_func = self._get_hash_func()
        encode_func = self._get_encode_func()
        return self._lquery(k, self.data.all_ids(), hash_func, encode_func)
    
    def _query_by_id(self, query_num, query_ids, hash_func, encode_func, *, num_processes=4, 
                MAX_PER_WORKER=1000, wait_time=0.001):
        
        # create queues
        manager = Manager()
        error_queue = manager.Queue(num_processes)
        kill_switches = [manager.Event() for _ in range(num_processes)]
        meta_in_queue = manager.Queue(num_processes - 1)
        meta_out_queue = manager.Queue(num_processes - 1)
        result_in, result_out = Pipe()
        
        # listener function
        def merger_func(error_queue, kill_switch, in_queue, result_in):
            
            try:
                while True:
                    
                    if kill_switch.is_set():
                        break
                    
                    item = in_queue.get()
                    if item == "kill":
                        break
                    
                    result += item
                    
                result_in.send(result)
            
            except Exception as e:
                error_queue.put(ExceptionWrapper(e))
        
        # worker function
        def worker_func(error_queue, kill_switch, in_queue, out_queue, 
                        query_num, hash_func, encode_func):
            try:
                while True:
                    
                    if kill_switch.is_set():
                        break
                    
                    item = in_queue.get()
                    if item == "kill":
                        break
                
                    num_points, q_ids, q_data = item  
                    
                    result = self._squery(query_num, num_points, q_ids, q_data, 
                                          hash_func, encode_func)
                    out_queue.append(result)
            
            except Exception as e:
                error_queue.put(ExceptionWrapper(e))
        
        # start merger
        merger_args = (error_queue, kill_switches[0], meta_out_queue, result_in)
        merger = Process(merger_func, merger_args)
        merger.start()
        
        # start workers 
        pool = []
        for kill_switch in kill_switches[1:]:
            worker_args = (error_queue, kill_switch, meta_in_queue, 
                           meta_out_queue, query_num, hash_func, encode_func)
            p = Process(worker_func, worker_args)
            p.daemon = False
            p.start()
            pool.append(p)
       
        def mini_blocks(query_ids, data, num_processes, 
                        MAX_PER_WORKER=MAX_PER_WORKER):
            
            num_points = query_ids.size 
            per_worker = num_points // num_processes
            if per_worker >= MAX_PER_WORKER:
                per_worker = MAX_PER_WORKER
            
            i = 0
            while True:
                if i == num_points:
                    break
                
                #  block to in_queue
                q_ids = query_ids[i:i + per_worker]
                q_data = data.get(q_ids)
            
                yield (q_ids.shape[0], q_ids, q_data)
                i += per_worker
        
        generator = mini_blocks(query_ids, self.data, num_processes - 1)
        poll_func = lambda x: x.is_alive()
        
        self._distribute_and_wait(generator, error_queue, kill_switches, 
                             meta_in_queue, pool, poll_func, num_processes, 
                             wait_time)
            
        # kill merger
        meta_out_queue.put("kill")
        
        # wait for result
        result = result_out.recv()
        
        return result
    
    def _query(self, query_data, query_num, num_processes=os.cpu_count()+2):
        """
        Args:
            query_data (BaseData): Data points to be queried
            query_num (int): Number of neighbors to return of each point
            num_processes (int): (optional) Number of subprocesses, defaults to 
                number of cores plus 2.  
        
        Returns:
            A SortedList mapping each identifier in *query_data* to a tuple of 
            identifiers from the data set representing its nearest neighbors. 
            
            For example:
                
            SortedList([(0, (32266, 49414, 2423, 42492, 75357)), 
                        (1, (35063, 42229, 70142, 40291, -1)), 
                        (2, (70687, 38807, 40571, 59077, 89853))], load=1000)

        """
        
        try:
            
            hash_func = self._get_hash_func()
            encode_func = self._get_encode_func()
            
            manager = Manager()
            error_queue = manager.Queue(num_processes)
            query_in_queue = manager.Queue(num_processes - 2)
            query_out_queue = manager.Queue(num_processes - 2)
            kill_switches = [manager.Event() for _ in range(num_processes - 1)]
            ready_switches = [manager.Event() for _ in range(num_processes - 1)]
            error_out = manager.Queue(1)
            result_out = manager.Queue(1)
            
            # start manager process
            listener_args =  (error_queue, error_out, kill_switches, 
                              ready_switches, query_in_queue, query_out_queue,
                              query_data, num_processes)
            listener = Process(target=self._distribute_and_wait, 
                               args=listener_args)
            listener.start()
            
            # start storage process
            merger_args = (error_queue, kill_switches[0], ready_switches[0], 
                           query_out_queue, result_out, query_num)
            merger = Process(target=self._merge_when_available, 
                             args=merger_args)
            merger.start()

            # start worker processes
            workers = []
            for k, r in zip(kill_switches[1:], ready_switches[1:]):
                worker_args = (error_queue, k, r, query_in_queue, 
                               query_out_queue, query_data, query_num, 
                               hash_func, encode_func, self.data, 
                               self.hashtables)
                w = Process(target=self._query_in_parallel, args=worker_args)
                w.start()
                workers.append(w)
            
            for w in workers:
                w.join()   
            merger.join()
            listener.join()
            try:
                e = error_out.get(False)
                e.re_raise()
            except Empty:
                pass
            
            out = result_out.get()
            
            manager.shutdown()
            
            return out
        
        except Exception as e:
            
            manager.shutdown()
            listener.terminate()
            merger.terminate()
            for w in workers:
                w.terminate()
            raise e
import numpy as np
import locality.data as dat
import os, time

from locality.exception_wrap import ExceptionWrapper
from multiprocessing import Pool, Manager, Pipe, Process
from threading import Thread, Event
from queue import Full, Empty, Queue
from locality.distance.distance_wrapped import cdist_wrapped, ssdist_wrapped
from locality.random.sample import sample_intervals
from locality.storage import storage
from locality.random.stable import random_stable
from locality.helpers.bitarray import bitdot
from locality.helpers.encode  import encode_to_int, encode_to_string
from locality.helpers.query_helpers import rank, collect_pass1, collect_pass2
from math import ceil, floor, log

class LSH(object):

    def __init__(self, data, num_hashtables=1, num_func=4, method='voroni',
                 metric='euclidean', num_medoids=5, hash_size=3, radius=1, el=2, 
                 storage_filepath=None, storage_config=None, test=False, **kwargs):
        
        # initialize data object
        if not isinstance(data.__class__, dat.BaseData):
            if isinstance(data, np.ndarray):
                self.data = dat.InMemoryData(data)
            elif isinstance(data, str):
                try:
                    self.data = dat.InFileData(data)
                except OSError:
                    pass
            else:
                try:
                    self.data = dat.InMemoryData(np.asarray(data))
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
        
    def _hash_in_parallel(self, error_queue, kill_switch, ready_switch, 
                          in_queue, out_queue, hash_func, encode_func, data):
        
        try:
            while True:
                
                if kill_switch.is_set():
                    return
                
                slice_obj = in_queue.get()
                print(os.getpid(), "received")
                if slice_obj == "kill":
                    print("dead")
                    break
                
                hashlist = encode_func(hash_func(data.get_data(slice_obj)), 
                                       self._bitnum)
                hashlist = hashlist.reshape(-1, self.num_hashtables)
                result = (slice_obj, hashlist)
                out_queue.put(result)
            
            ready_switch.set()
            print(ready_switch.is_set())
            
        except Exception as e:
            
            error_queue.put(ExceptionWrapper(os.getpid(), e))
            return
            
    def _query_in_parallel(self, error_queue, kill_switch, in_queue, out_queue, 
                           query_num, hash_func, encode_func, data, storage):

        try:
            
            # list of getters
            table_getters = [table.get_list for table in storage]
            
            while True:
                
                if kill_switch.is_set():
                    break
                
                item = in_queue.get()
                if item == "kill":
                    break
              
                num_points, q_ids, q_data = item
                
                # hash inputs
                hashlist = encode_func(hash_func(q_data), self.bitnum)
                
                # allocate memory for indptr
                indptr = np.empty(num_points + 1, dtype=np.int)
                indptr[0] = 0 
                
                # get length for array of possible neighbors
                indptr_size = collect_pass1(num_points, self.num_hashtables, 
                                            hashlist, table_getters)
                
                # allocate memory for indptr
                cand = np.empty(indptr_size, dtype=q_ids.dtype)
                
                # collect possible neighbors 
                collect_pass2(num_points, self.num_hashtables, hashlist, cand, 
                              indptr, table_getters)
                
                # collect data using ids
                c_ids, c_indices = np.unique(cand, return_inverse=True)
                c_data = data.get(c_ids)

                # allocate memory for indices
                nn_indices = np.empty(num_points * query_num, dtype=np.intc)
            
                # calculate distances
                ssdm = ssdist_wrapped(q_data, c_data, c_indices, indptr, 
                                      self.metric, self.margs)
            
                # rank based on distances
                rank(query_num, num_points, ssdm.data, c_indices, indptr)
            
                # build result tuple and append to managed list
                result = (q_ids, c_ids[nn_indices])
                out_queue.put(result)
        
        except Exception as e:
           
            error_queue.put(ExceptionWrapper(os.getpid(), e))
            return
            
    def _store_when_available(self, error_queue, kill_switch, ready_switch, 
                              in_queue, return_pipe, data, storage):
        try:
            while True:
                
                if kill_switch.is_set():
                    return
                
                item = in_queue.get()
                if item == "kill":
                    break
                
                slice_obj, hashlist = item
                storage.store_hashes(data.get_ids(slice_obj), hashlist)
                print('stored')
            
            ready_switch.set()
            return_pipe.send(storage)
        
        except Exception as e:
            error_queue.put(ExceptionWrapper(os.getpid(), e))
            return
    
    def _merge_when_available(self, error_queue, kill_switch, in_queue, 
                              query_num):
        
        try:
            result = []
            
            while True:
                
                if kill_switch.is_set():
                    break
                
                item = in_queue.get()
                if item == "kill":
                    break
                
                q_ids, c_ids = item
                
                j = 0
                for i in range(q_ids.shape[0]):
                    result.append((q_ids[i], tuple(c_ids[j:j + query_num])))
                    j += query_num
                
            return tuple(result)
        
        except Exception as e:
            
            error_queue.put(ExceptionWrapper(os.getpid(), e))
            return
            
    
    def _distribute_and_wait(self, error_queue, error_send, kill_switches, 
                             ready_switches, in_queue, out_queue, data, 
                             num_processes, wait_time=0.1):
        
        try:
            generator = data.partition(num_processes - 2)
            distributed = False
            workers_finished = False
            finished = False
            passed_kills = 0
           
            while not finished:
                
                #time.sleep(wait_time)
                
                try:
                    e = error_queue.get(False)
                    for k in kill_switches:
                        k.set()                   
                    error_send.send(e)
                    break
                except Empty:
                    pass
            
                if not distributed:
                    try:
                        while True:
                            in_queue.put(next(generator), timeout=wait_time)
                            print("distributed")
                    except StopIteration:
                        distributed = True
                    except Full:
                        pass 
                
                elif passed_kills < num_processes - 2:
                    try:
                        while passed_kills < num_processes - 2:
                            in_queue.put("kill", timeout=wait_time)
                            passed_kills += 1
                            print(passed_kills)
                    except Full:
                        pass
            
                elif not workers_finished:
                    
                    workers_finished = True
                    
                    for r in ready_switches[1:]:
                        print(r.is_set())
                        time.sleep(wait_time)
                        if not r.is_set():
                            workers_finished = False
                            break
                    
                if workers_finished:
                    out_queue.put("kill")

        except Exception as e:
            for k in kill_switches:
                k.set()
            error_send.send(ExceptionWrapper(os.getpid(), e))
            return
        
        
    def _hash_all(self, num_processes=4, wait_time=0.1):
        """
        Hash and store references for all objects in data set in parallel.
        """
    
        try:
            hash_func = self._get_hash_func()
            encode_func = self._get_encode_func()
        
            manager = Manager()
            error_queue = manager.Queue(num_processes)
            hash_in_queue = manager.Queue(num_processes - 2)
            hash_out_queue = manager.Queue(num_processes - 2)
            kill_switches = [Event() for _ in range(num_processes - 1)]
            ready_switches = [Event() for _ in range(num_processes - 1)]
            error_recv, error_send = Pipe(False)
            result_revc, result_send = Pipe(False)
            
            # start manager process
            listener_args =  (error_queue, error_send, kill_switches, 
                              ready_switches, hash_in_queue, hash_out_queue,
                              self.data, num_processes, wait_time)
            listener = Process(target=self._distribute_and_wait, 
                               args=listener_args)
            listener.start()
            
            # start storage process
            
            storage_args = (error_queue, kill_switches[0], ready_switches[0], 
                            hash_out_queue, result_send, self.data, self.hashtables)
            storage = Process(target=self._store_when_available, 
                              args=storage_args)
            storage.start()

            # start worker processes
            workers = []
            for k, r in zip(kill_switches[1:], ready_switches[1:]):
                worker_args = (error_queue, k, r, hash_in_queue, hash_out_queue, 
                               hash_func, encode_func, self.data)
                w = Process(target=self._hash_in_parallel, args=worker_args)
                w.start()
                workers.append(w)
            
            for w in workers:
                w.join()
            
            storage.join()
            listener.join()
            print("HERE")
            try:
                e = error_recv.recv()
                e.re_raise()
            except EOFError:
                pass
            
            # save output
            self.hashtables = result_revc.revc()
        
        except Exception as e:
            listener.terminate()
            for w in workers:
                w.terminate()
            raise e
            
    def query(self, query_data, k, MAX_SIZE=1e6):
        
        hash_func = self._get_hash_func()
        encode_func = self._get_encode_func()
        
        if isinstance(query_data[0], object):
            if len(query_data) < MAX_SIZE:
                query_ids = query_data
                query_data = self.data.get(query_ids)
            else:
                query_ids = query_data
                return self._lquery(k, query_ids, hash_func, encode_func) 
        else:
            query_ids = list()
        try:
            n, dims = query_data.shape
        except ValueError:
            n = 1
            dims = query_data.shape[0]
            query_data = query_data.reshape(1,-1)
        if dims != self.dims:
            raise ValueError(
                'Points {} dimensions'.format(
                        'have too many' if query_data.shape[1] > self.dims 
                        else 'are missing' if query_data.shape[0] > 1 
                        else 'has too many' if query_data.shape[1] > self.dims 
                        else 'is missing'))
        return self._squery(k, n, query_ids, query_data, hash_func, 
                            encode_func)
                
    def query_all(self, k):
        hash_func = self._get_hash_func()
        encode_func = self._get_encode_func()
        return self._lquery(k, self.data.all_ids(), hash_func, encode_func)
    
    def _lquery(self, query_num, query_ids, hash_func, encode_func, *, num_processes=4, 
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
    
    def _squery(self, query_num, num_points, query_ids, query_data, hash_func, 
                encode_func, *, num_processes=6, MAX_PER_WORKER=100, 
                wait_time=0.001):
        """
        Hash and store references for all objects in data set in parallel.
        """
        
        manager = Manager()
        error_queue = manager.Queue(num_processes)
        kill_switches = [manager.Event() for _ in range(num_processes)]
        query_in_queue = manager.Queue(num_processes)
        query_out_queue = manager.Queue(num_processes)
        
        # create pool
        pool = Pool(num_processes)
        
        # launch merger process first
        merger_args = (error_queue, kill_switches[0], query_out_queue, 
                       query_num)
        merger = pool.apply_async(self._merge_when_available, merger_args)
        
        # launch workers 
        workers = []
        for kill_switch in kill_switches[1:]:
            worker_args = (error_queue, kill_switch, query_in_queue, 
                           query_out_queue, query_num, hash_func, encode_func, 
                           self.data, self.hashtables)
            res = pool.apply_async(self._query_in_parallel, worker_args)
            workers.append(res)
        
        # distribute work
            
        def mini_blocks(query_ids, query_data, num_points, num_processes,
                        MAX_PER_WORKER=MAX_PER_WORKER):
            
            per_worker = num_points // num_processes
            if per_worker >= MAX_PER_WORKER:
                per_worker = MAX_PER_WORKER
            
            i = 0
            while True:
                if i == num_points:
                    break
                
                #  block to in_queue
                q_ids = query_ids[i:i + per_worker]
                q_data = query_data[i:i + per_worker]
            
                yield (q_ids.shape[0], q_ids, q_data)
                i += per_worker
        
        generator = mini_blocks(query_ids, query_data, num_points, 
                                      num_processes - 1)
        poll_func = lambda x: x.ready()
        
        self._distribute_and_wait(generator, error_queue, kill_switches, 
                                  query_in_queue, workers, poll_func, 
                                  num_processes, wait_time)
        
        # kill merger
        query_out_queue.put("kill")
        
        result = merger.get()
        
        # close
        pool.close()
        
        return result
    
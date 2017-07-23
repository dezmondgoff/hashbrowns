import numpy as np

try:
    from threading import Lock
except ImportError:
    from dummy_threading import Lock

cdef class RandomStateInterface:
    
    def __init__(self, rstate=None):
        if rstate is None:
            rstate = np.random.rand.__self__
            
        if not isinstance(rstate, np.random.RandomState):
            raise ValueError()
        
        self.rstate = rstate
        self.state_copy = <rk_state *> malloc(sizeof(rk_state))
        self.lock = Lock()
     
    def __dealloc__(self):
        if self.state_copy != NULL:
            free(self.state_copy)
            self.state_copy = NULL
       
    cdef rk_state * retreive_state(self):
        cdef np.ndarray key_array
        cdef unsigned long[::1] key 
        cdef long pos
        cdef long has_gauss
        cdef double gauss
        
        _, key_array, pos, has_gauss, gauss = self.rstate.get_state()
        key = key_array.astype(dtype=np.uint)
        memcpy(<void *> (self.state_copy.key), <void *> &key[0], 
               624 * sizeof(unsigned long))
        self.state_copy.pos = pos
        self.state_copy.has_gauss = has_gauss
        self.state_copy.gauss = gauss
  
    cdef void return_state(self):
        cdef unsigned long[::1] key = self.state_copy.key 
        
        self.rstate.set_state(('MT19937', np.asarray(key), 
                               self.state_copy.pos, 
                               self.state_copy.has_gauss,
                               self.state_copy.gauss))

_rand_interface = RandomStateInterface()

def test():
    """Print a random positive integer and a random double from `[0,1)`."""
    cdef unsigned long n
    cdef double x
    cdef rk_state * state
    cdef RandomStateInterface rsi = RandomStateInterface()
    cdef object lock
    
    state = rsi.state_copy
    
    with rsi.lock:
        rsi.retreive_state()
        n = rk_random(state)
        x = rk_double(state)
        rsi.return_state()
    
    print("The integer is {} and the double is {}.".format(n, x))
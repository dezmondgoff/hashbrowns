import numpy as np

class DynamicArray(np.ndarray):

    def __init__(self, *args, **kwargs):
        np.ndarrray.__init__(self, *args, **kwargs)
        self.used = np.ndarray.shape
        self.capacity = self.used

    def __setitem__(self, index, item):
        while True:
            try:
                super(DynamicArray, self).__setitem__(index, item)
                break
            except IndexError:
                new_capacity = tuple(2 * x for x in self.shape)
                self.resize(new_shape, refcheck=False)
                self.capacity = new_capacity

                if isinstance(index, int):
                    self.used = (index + 1,) + self.used[1:]
                elif isintance(index, slice):
                    self.used = (index.stop, ) + self.used[1:s]
                elif isinstance(index, np.ndarray):
                    self.used = (index.max() + 1, ) + self.used[1:]
                else:
                    new_used = []
                    for i, obj in enumerate(index):
                        # figure out new used size
                        if isinstance(obj, int):
                            new_used.append(obj + 1)
                        elif isintance(index, slice):
                            new_used.append(index.stop)
                        elif isinstance(index, np.ndarray):
                            new_used.append(index.max() + 1)
                    self.used = tuple(new_used)

    @property
    def shape(self):
        return self.used

    @property
    def size(self):
        return np.prod(self.used)

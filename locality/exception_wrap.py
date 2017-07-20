# -*- coding: utf-8 -*-
import tblib.pickling_support
tblib.pickling_support.install()

import sys

class ExceptionWrapper(object):

    def __init__(self, pid, ee):
        self.ee = type(ee)("process " + str(pid) + ": " + str(ee))
        __,  __, self.tb = sys.exc_info()

    def re_raise(self):
        raise self.ee.with_traceback(self.tb)
        # for Python 2 replace the previous line by:
        # raise self.ee, None, self.tb
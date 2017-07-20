#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os


def configuration(ext_name, ext_path, package_name, parent_name=None, 
                  package_path=None, top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
    from numpy.distutils.misc_util import get_info as get_misc_info
    
    config = Configuration(package_name, parent_name, package_path, top_path)
    
    top_path = os.path.dirname(os.path.realpath(__file__))
    search_path = os.path.join(top_path, package_path, ext_path)
    
    #config.add_data_dir('tests')
    
    def scandir(d, ext, files=[]):
        for file in os.listdir(d):
            path = os.path.join(d, file)
            if os.path.isfile(path) and path.endswith(ext):
                files.append(path)
            elif os.path.isdir(path):
                scandir(path, ext, files)
        return files
    
    config.add_extension(ext_name,
        sources=scandir(search_path, '.c'),
        depends=scandir(search_path, '.h'),
        include_dirs=[get_numpy_include_dirs()],
        extra_compile_args = ["-O3", "-Wall"],
        extra_link_args = ['-g'],
        extra_info=get_misc_info("npymath"))

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration('_alignment_wrap', '_alignment_wrap', 'distance', 
                          'locality', package_path='locality/distance/',
                          top_path='').todict())
    setup(**configuration('_ssdist_wrap', 'src', 'locality', 
                          package_path='locality/distance/ssdist',
                          top_path='').todict())

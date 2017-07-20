#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 23:04:21 2017

@author: root
"""

# build script for 'locality'

import sys, os
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from numpy.distutils.misc_util import get_numpy_include_dirs

numpyIncludeDir = get_numpy_include_dirs()

# we'd better have Cython installed, or it's a no-go
try:
    from Cython.Distutils import build_ext
except:
    print("You don't seem to have Cython installed. Please get a")
    print("copy from www.cython.org and install it")
    sys.exit(1)


# scan the 'locality' directory for extension files, converting
# them to extension names in dotted notation
def scandir(dir, ext, files=[]):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path) and path.endswith(ext):
            files.append(path)
        elif os.path.isdir(path):
            scandir(path, ext, files)
    return files

def getIncludes(extPaths, crossReference=[], files=[]):
    remove = "<>\""
    if not extPaths:
        return files
    to_visit = []
    for path in extPaths:
        with open(path) as f:
            for line in f:
                if line.startswith("#include"):
                    name = line[9:]
                    for char in remove:
                        name.strip(char)
                    for h in crossReference:
                        h_name = h.split(os.path.sep)[-1]
                        if h_name == name:
                            files.append(h)
                            to_visit.append(h)
    getIncludes(to_visit, crossReference, files)
                
# generate an Extension object from its dotted name
def makeExtension(extName, file_ext):
    extPath = extName.replace(".", os.path.sep) + file_ext
    print(extPath)
    if file_ext == ".c":
        extName = ".".join(extName.split(".")[:-1] + 
                           ["_" + extName.split(".")[-1]])
        topdir = os.path.join(*(extPath.split(os.path.sep)[:-1]))
        headerPaths = scandir(topdir, ".h")
        deps = getIncludes([extPath], headerPaths, [])
    else:
        deps = []
    return Extension(
        extName,
        sources = [extPath],
        depends = deps,
        include_dirs = numpyIncludeDir + ["."],   # adding the '.' to include_dirs is CRUCIAL!!
        extra_compile_args = ["-O3", "-Wall", "-march=native","-mfpmath=sse"],
        extra_link_args = ['-g'],
        )
    
# get the list of extensions
cy_extNames = [path.replace(os.path.sep, ".")[:-4] 
               for path in scandir("locality", ".pyx", [])]
c_extNames = [path.replace(os.path.sep, ".")[:-2]
              for path in scandir("locality/distance", ".c", [])]
# and build up the set of Extension objects
cy_extensions = [makeExtension(name, ".pyx") for name in cy_extNames]
c_extensions = [makeExtension(name, ".c") for name in c_extNames]

# finally, we can pass all this to distutils
setup(
      name="locality",
      version = '1.0',
      description = '''Multiprocessor enabled locality sensitive hashing for 
                       large data sets.''',
      author = 'Dezmond K. Goff',
      author_email = 'goff.dezmond@gmail.com',
      url = '',
      long_description = '''''',
      packages = ["locality", "locality.helpers", "locality.distance"],
      ext_modules = cythonize(cy_extensions) + c_extensions,
      cmdclass = {'build_ext': build_ext},
)

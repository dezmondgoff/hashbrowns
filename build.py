#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 23:04:21 2017

@author: root
"""

# build script for 'hashbrowns'

import sys
import os
import re
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from numpy.distutils.misc_util import get_numpy_include_dirs, get_mathlibs
from distutils.msvccompiler import get_build_version as get_msvc_build_version


numpyIncludeDir = get_numpy_include_dirs()

# we'd better have Cython installed, or it's a no-go
try:
    from Cython.Distutils import build_ext
except:
    print("You don't seem to have Cython installed. Please get a")
    print("copy from www.cython.org and install it")
    sys.exit(1)


# scan the 'hashbrowns' directory for extension files, converting
# them to extension names in dotted notation
def scandir(dir, ext, files=[]):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path) and path.endswith(ext):
            files.append(path)
        elif os.path.isdir(path):
            scandir(path, ext, files)
    return files

def get_depends(topdir, fids, header_names=[], out=set()):
    if not fids:
        return [os.path.join(topdir, x) for x in out]
    to_visit = []
    for filename in fids:
        with open(os.path.join(topdir, filename)) as f:
            for line in f:
                if line.startswith(("#include ", "include ", "from",
                                    "cdef extern from ")):
                    if "<" in line:
                        continue
                    names = re.findall(r"['\"](.*?)['\"]", line)
                    if not names:
                        for x in re.findall(r"from (.*?) cimport", line):
                            x += ".pxd"
                            if os.path.isfile(os.path.join(topdir, x)):
                                to_visit.append(x)
                        continue
                    new_filename = names[0]
                    for header_name in header_names:
                        if header_name == new_filename:
                            out.add(header_name)
                            to_visit.append(header_name)
                            break
    return get_depends(topdir, to_visit, header_names, out)

# generate an Extension object from its dotted name
def makeExtension(ext_name, file_ext):
    ext_path = ext_name.replace(".", os.path.sep) + file_ext
    print(ext_path)
    topdir = os.path.join(*(ext_path.split(os.path.sep)[:-1]))
    header_names = [os.path.split(x)[-1] for x in scandir(topdir, ".h", [])]
    sources = [ext_path]
    libs = []
    deps = []
    defs = []
    if file_ext == ".c":
        ext_name = ".".join(ext_name.split(".")[:-1] +
                           ["_" + ext_name.split(".")[-1]])
        deps = get_depends(topdir, [os.path.split(ext_path)[-1]],
                           header_names, set())
    elif "mtrand" in ext_name:
        sources = []
    else:
        deps = get_depends(topdir, [os.path.split(ext_path)[-1][:-3] + "pxd"],
                           header_names, set())
        sources += [x[:-1] + "c" for x in deps]
    return Extension(
        ext_name,
        sources=sources,
        libraries=libs,
        depends=deps,
        define_macros=defs,
        include_dirs=numpyIncludeDir + ["."],
        extra_compile_args=["-O3", "-Wall", "-march=native", "-mfpmath=sse"],
        extra_link_args=["-g", "-C"],
        )

# get the list of extensions
cy_ext_names = [path.replace(os.path.sep, ".")[:-4]
               for path in scandir("hashbrowns", ".pyx", [])]
c_ext_names = [path.replace(os.path.sep, ".")[:-2]
              for path in scandir("hashbrowns/_distance", ".c", [])]
# and build up the set of Extension objects
cy_extensions = [makeExtension(name, ".pyx") for name in cy_ext_names]
c_extensions = [makeExtension(name, ".c") for name in c_ext_names]

# finally, we can pass all this to distutils
setup(
      name="hashbrowns",
      version='1.0',
      description="Parallel locality-sensitive hashing for large metric "
                  "spaces.",
      author='Dezmond K. Goff',
      author_email='goff.dezmond@gmail.com',
      url='',
      long_description='''''',
      packages=["hashbrowns", "hashbrowns._random",
                "hashbrowns._helpers", "hashbrowns._distance"],
      ext_modules=cythonize(cy_extensions) + c_extensions,
      cmdclass={'build_ext': build_ext},
)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
from setuptools import setup, Extension
from Cython.Build import cythonize


setup(
    ext_modules=cythonize(
        [
            Extension("dragons.munge.regrid", sources=["dragons/munge/regrid.pyx"], include_dirs=[numpy.get_include()]),
            Extension(
                "dragons.munge.tophat_filter",
                sources=["dragons/munge/tophat_filter.pyx"],
                include_dirs=[numpy.get_include()],
            ),
        ]
    ),
)

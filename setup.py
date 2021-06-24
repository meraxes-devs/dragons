#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from distutils.core import Extension

import numpy
from setuptools import setup

setup(
    include_dirs=[numpy.get_include()],
    ext_modules=[
        Extension("dragons.munge.regrid", ["dragons/munge/regrid.c"]),
        Extension("dragons.munge.tophat_filter", ["dragons/munge/tophat_filter.c"]),
    ],
)

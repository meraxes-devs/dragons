#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

try:
    from setuptools import setup, find_packages, Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
    from pkgutil import walk_packages

    def find_packages(path=__path__, prefix=""):
        yield prefix
        prefix = prefix + "."
        for _, name, ispkg in walk_packages(path, prefix):
            if ispkg:
                yield name

try:
    import numpy
    np_inc_dirs = numpy.get_include()
except ImportError:
    np_inc_dirs = ""

readme = open('README.rst').read()
history = open('HISTORY.rst').read().replace('.. :changelog:', '')
requirements = open('requirements.txt').readlines()


setup(
    name='dragons',
    version='0.2.2',
    description='Python tools for dealing with the Meraxes semi-analytic model output and associated processing.',
    long_description=readme + '\n\n' + history,
    author='Simon Mutch',
    author_email='smutch.astro@gmail.com',
    url='https://github.com/smutch/dragons',
    packages=find_packages(),
    package_dir={'dragons': 'dragons'},
    package_data={'dragons': ['stylelib/*']},
    include_package_data=True,
    install_requires=requirements,
    include_dirs = [np_inc_dirs],
    ext_modules = [Extension("dragons.munge.regrid", ["dragons/munge/regrid.c"]),
                   Extension("dragons.munge.tophat_filter", ["dragons/munge/tophat_filter.c"])],
    license="BSD",
    zip_safe=False,
    keywords='dragons',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)

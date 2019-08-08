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

# if sys.argv[-1] == 'publish':
#     os.system('python setup.py sdist upload')
#     sys.exit()

readme = open('README.rst').read()
history = open('HISTORY.rst').read().replace('.. :changelog:', '')
requirements = open('requirements.txt').readlines()


setup(
    name='dragons',
    version='0.2.1',
    description='Python tools for dealing with simulations, semi-analytic models and associated post-processing.',
    long_description=readme + '\n\n' + history,
    author='Simon Mutch',
    author_email='smutch.astro@gmail.com',
    url='https://bitbucket.org/dragons-astro/dragons',
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
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    test_suite='tests',
)

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

# if sys.argv[-1] == 'publish':
#     os.system('python setup.py sdist upload')
#     sys.exit()

readme = open('README.rst').read()
history = open('HISTORY.rst').read().replace('.. :changelog:', '')
requirements = open('requirements.txt').readlines()[1:]

setup(
    name='ssimpl',
    version='0.1.0',
    description='Python tools for dealing with simulations, semi-analytic models and associated post-processing.',
    long_description=readme + '\n\n' + history,
    author='Simon Mutch',
    author_email='smutch.astro@gmail.com',
    url='https://bitbucket.org/smutch/ssimpl',
    packages=find_packages(),
    package_dir={'ssimpl': 'ssimpl'},
    include_package_data=True,
    install_requires=requirements,
    ext_modules = [Extension("ssimpl/munge/regrid", ["ssimpl/munge/regrid.c"])],
    license="BSD",
    zip_safe=False,
    keywords='ssimpl',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        # "Programming Language :: Python :: 2",
        # 'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
    ],
    test_suite='tests',
    use_2to3=True,
)

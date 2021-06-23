#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import Extension, find_packages, setup
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


import numpy

np_inc_dirs = numpy.get_include()

readme = open("README.rst").read()
requirements = open("requirements.txt").readlines()
for ii, req in enumerate(requirements):
    if "egg=" in req:
        requirements[ii] = req.split("egg=")[-1]

setup(
    name="dragons",
    version="0.2.2",
    description="Python tools for dealing with the Meraxes semi-analytic model output and associated processing.",
    long_description=readme,
    author="Simon Mutch",
    author_email="smutch.astro@gmail.com",
    url="https://github.com/smutch/dragons",
    packages=find_packages(),
    package_dir={"dragons": "dragons"},
    install_requires=requirements,
    setup_requires=["Cython", "numpy"],
    include_dirs=[np_inc_dirs],
    ext_modules=[
        Extension("dragons.munge.regrid", ["dragons/munge/regrid.c"]),
        Extension("dragons.munge.tophat_filter", ["dragons/munge/tophat_filter.c"]),
    ],
    license="BSD",
    zip_safe=False,
    keywords="dragons",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)

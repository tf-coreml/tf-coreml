#!/usr/bin/env python

import os
from setuptools import setup, find_packages

README = os.path.join(os.getcwd(), "README.rst")

with open(README) as f:
    _LONG_DESCRIPTION= f.read()

setup(
    name='tfcoreml',
    version='0.2.0',
    description='Tensorflow to Core ML converter',
    long_description=_LONG_DESCRIPTION,
    url='',
    author='tfcoreml',
    author_email='tf-coreml@apple.com',
    license='Apache 2.0',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Developers',
        'Operating System :: MacOS :: MacOS X',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ],
    keywords='converter TF CoreML',
    packages=find_packages(),
    install_requires=[
        'numpy >= 1.6.2',
        'protobuf >= 3.1.0',
        'six >= 1.10.0',
        'tensorflow >= 1.5.0',
        'coremltools >= 0.8'
    ],
    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    package_data={
        '': ['README.md'],
    },

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    data_files=[],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        },
)

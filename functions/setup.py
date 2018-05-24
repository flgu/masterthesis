# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 19:55:15 2018

@author: Rocco

This is a setup script for compilation of a cythonized .pyx module
"""
from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
      ext_modules = cythonize("jacobian.pyx"),
      include_dirs = [numpy.get_include()]
      )

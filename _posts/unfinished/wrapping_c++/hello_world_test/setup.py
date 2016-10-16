from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os

ext = Extension(
    "pyMultiply",               # Extension name
    sources=["pyMultiply.pyx"], # Cython source filename
    )

setup(
  name = 'pyMultiply',
  ext_modules=[ext],
  cmdclass = {'build_ext': build_ext},
)

# Build command
# python setup.py build_ext --inplace


# Sites
# http://stackoverflow.com/questions/16993927/using-cython-to-link-python-to-a-shared-library
# https://studywolf.wordpress.com/2012/09/14/cython-journey-part-1/
# http://docs.cython.org/en/latest/src/userguide/wrapping_CPlusPlus.html?ref=driverlayer.com.html

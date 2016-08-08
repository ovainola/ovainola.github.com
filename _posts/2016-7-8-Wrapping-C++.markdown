---
layout: post
title:  "Wrapping C++"
date:   2016-8-7 21:00:00 +0300
categories: jekyll update
comments: true
identifier: 10003
---
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

It's fun to write new code. You'll get to learn a new theory/algorithm, think
about the architecture, which
language you want to use, debug your mistakes etc. But sometimes, you just don't
have the time to create the whole thing all by yourself and you'll end up hunting for pieces you'll need from the internet. Sometimes you'll get
lucky to find just the right pieces but they're in a different
language then your implementation. Usually my implementation is in Python/Julia and end up finding some C++ code.
Translating the whole code is quite consuming process (at least for me). I've tried copy/paste style like a monkey and just hope that I don't make
any mistakes. And when there are mistakes, that's where the fun begins. Luckily, wrapping
the code makes things a bit more easier.

There are multiple ways how to wrap code. [SWIG](http://swig.org/) is a multitool for number of languages,
python has f.ex. [Cython]((https://pypi.python.org/pypi/Cython/)), [ctypes](https://docs.python.org/2/library/ctypes.html)
& [BoostPython](http://www.boost.org/doc/libs/1_61_0/libs/python/doc/html/index.html), Java has [jni](http://homepage.cs.uiowa.edu/~slonnegr/wpj/JNI.pdf) and
in Julia f.ex. [ccall](http://julia.readthedocs.io/en/latest/manual/calling-c-and-fortran-code/) and [cxx](https://github.com/JuliaLangEs/Cxx.jl). In this post I'll mainly consentrate wrapping C++ for Python
using Cython (since it's my favorite). In future I'll try to add examples, how to do
same tricks f.ex. with Julia or Java.

# and finally scrape a bit julia/Java interface.

## Cython

If you're unfamiliar with Cython, here's a quote from [package's site](https://pypi.python.org/pypi/Cython/):

> The Cython language makes writing C extensions for the Python language as easy as Python itself. Cython is a source code translator based on Pyrex, but supports more cutting edge functionality and optimizations.
The Cython language is a superset of the Python language (almost all Python code is also valid Cython code), but Cython additionally supports optional static typing to natively call C functions, operate with C++ classes and declare fast C types on variables and class attributes. This allows the compiler to generate very efficient C code from Cython code.
This makes Cython the ideal language for writing glue code for external C/C++ libraries, and for fast C modules that speed up the execution of Python code.


##### Simple functions

For this simple excercise we'll need two files: **pyMultiply.pyx** (source file) and **setup.py** (settings for build).
In the source file we'll create following functions:

```python
cdef cMultiply(double a, double b):
    return a * b

cpdef cpSum(double a, double b):
    return a + b

def pyTestMultiply():
    return cMultiply(2,2)

def pyTestSum():
    return cpSum(2,2)
```

The most noticeable difference between these functions that one starts with cdef, one with cpdef and the last ones with def.
cdef defines a C-function, cpdef function, which can be called both in C and python and def is just normal python function.
Now, we'll create a setup.py file, which defines the parameters for build.

```python
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
```

Ok, build time!

```bash
>> python setup.py build_ext --inplace
running build_ext
cythoning pyMultiply.pyx to pyMultiply.c
building 'pyMultiply' extension
creating build
creating build/temp.linux-x86_64-2.7
gcc -pthread -fno-strict-aliasing -g -O2 -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I$HOME/anaconda/include/python2.7 -c pyMultiply.c -o build/temp.linux-x86_64-2.7/pyMultiply.o
gcc -pthread -shared -L$HOME/anaconda/lib -Wl,-rpath=$HOME/anaconda/lib,--no-as-needed build/temp.linux-x86_64-2.7/pyMultiply.o -L$HOME/anaconda/lib -lpython2.7 -o pyMultiply.so
```

There's a lot going on, what just happened? The short version: first Cython read the .pyx file and produced pyMultiply.c file,
a massive 1729 lines of C-code. Then gcc is used to compile pyMultiply.o
object file and finally pyMultiply.so shared library is created. Longer version: I'd suggest
to seek answers from [documentation](http://docs.cython.org/en/latest/). Next stop
is to import newly created library to python:

```python
>> ipython
Python 2.7.12 |Anaconda custom (64-bit)| (default, Jul  2 2016, 17:42:40)
Type "copyright", "credits" or "license" for more information.

IPython 5.0.0 -- An enhanced Interactive Python.
?         -> Introduction and overview of IPython's features.
%quickref -> Quick reference.
help      -> Python's own help system.
object?   -> Details about 'object', use 'object??' for extra details.

In [1]: import pyMultiply

In [2]: pyMultiply.pyTestMultiply()
Out[2]: 4.0

In [3]: pyMultiply.pyTestSum()
Out[3]: 4.0

```

Works like a charm. def and cpdef functions work as promised and use cdef cpdef
functions to perform calculations. In our next example, we'll dig a bit deeper
and try to wrap a C++ class.

##### Wrapping C++ class

For our second example, we'll need four files: **class_example.h** (class header),
**class_example.cpp** (C++ source), **pyCar.pyx** (cython source) and **setup.py** (settings for build).
First task is to compile shared library class_example.so, which we will wrap with
Cython. Let's start by creating a simple C++ class with couple of methods in class_example.h file:

```cpp
#pragma once

namespace vehicles {        
  class Car {               

    private:  
      const char* cBrand;
      const char* cModel;
      int mYear;
      double mSpeed;

    public:
      Car(char* brand, char* model, int yearBought, double currentSpeed);
      ~Car();
      double getSpeed();
      void setSpeed(double newSpeed);
  };
}
```

and the implementation in class_example.cpp file:

```cpp
#include "class_example.h"

using namespace vehicles;

Car::Car(char* brand, char* model, int yearBought, double currentSpeed)
{
    cBrand = brand;
    cModel = model;
    mYear  = yearBought;
    mSpeed = currentSpeed;
}

Car::~Car()
{
}

double Car::getSpeed() {
    return mSpeed;
}

void Car::setSpeed(double newSpeed) {
  mSpeed = newSpeed;
}
```

Compile commands for shared library:

```bash
>> gcc -c class_example.cpp -o class_example.o
>> gcc -shared class_example.o -o class_example.so
```

First part done. Now we'll create the pyCar.pyx file:

```cpp
# distutils: language = c++

cdef extern from "class_example.h" namespace "vehicles":
    cdef cppclass Car:
        Car(char* brand, char* model, int yearBought, double currentSpeed)
        double getSpeed()
        void setSpeed(double newSpeed)


cdef class PyCar:
    cdef Car * pCar

    def __cinit__(self, char* pyBrand, char* pyModel, int pyYear, double pySpeed):
        self.pCar = new Car(pyBrand, pyModel, pyYear, pySpeed)

    def get_speed(self):
        return self.pCar.getSpeed()

    def set_car_speed(self, double new_speed):
        self.pCar.setSpeed(new_speed)
```

Now syntax is a bit more trickier, but the file can be divided into two parts:
C++ name exposure part and python implementation part. In the name exposure part
we'll basically copy the header file and expose C++ class for Cython. In the
Python part we'll create a wrapper class, which holds the C++ instance (which
is allocated in pCar pointer) inside. Note that we'll have to be careful with
types now, since C++ can be picky. The setup.py file is almost the same, but
we'll have to add the shared library into link command with libraries and extra_link_args
flags.

```python
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os

ext = Extension(
    "pyCar",                              # Extension name
    sources=["pyCar.pyx"],                # Cython source filename
    language="c++",                       # Creates C++ source
    libraries=["class_example"],          # .so file, which we want to include in linker
    extra_link_args=["-L" + os.getcwd()], # Add current folder for linker
    )

setup(
  name = 'pyCar',
  ext_modules=[ext],
  cmdclass = {'build_ext': build_ext},
)
```

Build time!

```bash
>> python setup.py build_ext --inplace
running build_ext
cythoning pyCar.pyx to pyCar.cpp
building 'pyCar' extension
gcc -pthread -fno-strict-aliasing -g -O2 -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I$HOME/anaconda/include/python2.7 -c pyCar.cpp -o build/temp.linux-x86_64-2.7/pyCar.o
cc1plus: warning: command line option ‘-Wstrict-prototypes’ is valid for C/ObjC but not for C++
g++ -pthread -shared -L$HOME/anaconda/lib -Wl,-rpath=$HOME/anaconda/lib,--no-as-needed build/temp.linux-x86_64-2.7/pyCar.o -L$HOME/anaconda/lib -lclass_example -lpython2.7 -o pyCar.so -L$HOME/blog
```

As you can see, now Cython reads pyCar.pyx file and produces pyCar.cpp file. Now we can test our
newly created wrapper.

```python
>> ipython
Python 2.7.12 |Anaconda custom (64-bit)| (default, Jul  2 2016, 17:42:40)
Type "copyright", "credits" or "license" for more information.

IPython 5.0.0 -- An enhanced Interactive Python.
?         -> Introduction and overview of IPython's features.
%quickref -> Quick reference.
help      -> Python's own help system.
object?   -> Details about 'object', use 'object??' for extra details.

In [1]: import pyCar

In [2]: car = pyCar.PyCar("Volkwagen", "Passat", 2007, 120)

In [3]: car.get_speed()
Out[3]: 120.0

In [4]: car.set_car_speed(12)

In [5]: car.get_speed()
Out[5]: 12.0
```

Works quite well. If you're having problems with following command:

```python
In [1]: import pyCar
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
<ipython-input-1-90806aedfe9a> in <module>()
----> 1 import pyCar

ImportError: libclass_example.so: cannot open shared object file: No such file or directory
```

do note, that we're working with shared libraries and class_example.so has to be inside a folder,
where python can find it. I created a fast & dirty solution with:

```bash
>> export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/blog
```

#### Real world example?


Now that we have studied couple of academic cube level examples, I figured it could
be nice to have an example, which could be a bit more realistic situation.
For motivation, let's say you're a scientist on some field X. You've done some major
research and used python for your calculations. Now let's say that
some part of your code is really slow, f.ex. Delaunay tree in 2D, which you're
using in some model. After searching for a while for an answer,
you'll find a [website](http://people.sc.fsu.edu/~jburkardt/cpp_src/delaunay_tree_2d/delaunay_tree_2d.html),
where the algorithm you'll need is implemented with C++.

```bash
>> wget http://people.sc.fsu.edu/~jburkardt/cpp_src/delaunay_tree_2d/delaunay_tree_2d.cpp
```

Let's take a look at that file. You'll see that there's couple of classes, functions and
main defined. We want to know how to use these classes, we'll study
the main function. There are three places, where we are interested: tree creation,
adding points and output:

```cpp
  Delaunay_tree DT;
  .
  .
  .
  while ( cin >> x >> y )
  {
    DT += new point ( x, y );
  .
  .
  .
  DT.output ( );    
```

So, we'll need to wrap Delaunay_tree class. Adding new points to
tree is done via overload operator "+=". Cython is capable of wrapping multiple [overload
operators](http://grokbase.com/t/gg/cython-users/126jfb9spq/wrapping-c-operator),
but unfortunately "+=" is not supported. To overcome this, we'll create
a small void function, which wraps this functionality. Next we'll perform following
steps:

 * comment out the main function
 * create void function add_point inside delaunay_tree_2d.cpp file:

```cpp
void add_point(Delaunay_tree *tree, double x, double y) {
 *tree += new point ( x, y );
}
```

  * compile shared library:

```bash
>> gcc -fPIC -c delaunay_tree_2d.cpp -o delaunay_tree_2d.o
>> gcc -shared delaunay_tree_2d.o -o libdelaunay_tree_2d.so
```

Like in last examples, now we'll create two files: **myDelaunay.pyx** (source file)
and **setup.py** (settings for build). First myDelaunay.pyx:

```cpp
# distutils: language = c++

cdef extern from "delaunay_tree_2d.cpp":
    cdef cppclass Delaunay_tree:
        Delaunay_tree ()
        void output ()
    void add_point(Delaunay_tree *DT, double x, double y)

cdef class pyDelaunayTree:
    cdef Delaunay_tree * pDTree

    def __cinit__(self, double [:,:] points):
        """ points is a 2D numpy array. Let's use memory view to specify type.
        """
        self.pDTree = new Delaunay_tree()
        cdef Py_ssize_t *dim = points.shape
        cdef Py_ssize_t ii = dim[0]
        for i in range(ii):
            add_point(self.pDTree, points[i, 0], points[i, 1])

    def output(self):
        self.pDTree.output()
```

Lastly setup.py:

```python
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os

ext = Extension(
    "myDelaunay",                              # Extension name
    sources=["myDelaunay.pyx"],                # Cython source filename
    language="c++",                       # Creates C++ source
    libraries=["delaunay_tree_2d"],          # .so file, which we want to include in linker
    extra_link_args=["-L" + os.getcwd()], # Add current folder for linker
    )


setup(
  name = 'myDelaunay',
  ext_modules=[ext],
  cmdclass = {'build_ext': build_ext},
)
```

and build:

```bash
>> python setup.py build_ext --inplace
```

Final thing to do is to test our code:

```python
>> ipython
Python 2.7.12 |Anaconda custom (64-bit)| (default, Jul  2 2016, 17:42:40)
Type "copyright", "credits" or "license" for more information.

IPython 5.0.0 -- An enhanced Interactive Python.
?         -> Introduction and overview of IPython's features.
%quickref -> Quick reference.
help      -> Python's own help system.
object?   -> Details about 'object', use 'object??' for extra details.

In [1]: import numpy as np

In [2]: import myDelaunay

In [3]: a = np.random.rand(6,2)

In [4]: tree = myDelaunay.pyDelaunayTree(a)

In [5]: tree.output()
%!
0.70547 0.00278302 moveto 0.381816 0.149089 lineto stroke
0.381816 0.149089 moveto 0.256233 0.800743 lineto stroke
0.256233 0.800743 moveto 0.418724 0.904034 lineto stroke
0.256233 0.800743 moveto 0.942192 0.598662 lineto stroke
0.942192 0.598662 moveto 0.418724 0.904034 lineto stroke
0.381816 0.149089 moveto 0.942192 0.598662 lineto stroke
0.70547 0.00278302 moveto 0.887348 0.185637 lineto stroke
0.887348 0.185637 moveto 0.381816 0.149089 lineto stroke
0.887348 0.185637 moveto 0.942192 0.598662 lineto stroke
showpage
```

As a conclusion, it can be said that wrapping C++ is quite tricky, since you'll
have to make sure both C++ and python sides work properly.
In our examples we didn't handle any memory problems (leaks), which
can be quite troublesome sometimes. Using templates and callback functions in C++ also
adds a new layer of complexity for wrapping. But usually tackling these wrapping
problems is worth it. You'll learn new things from both python and C++ and get
a more deeper insight how to combine codes. Also, you don't have to invent the wheel again
but make sure you're using the code within it's license.

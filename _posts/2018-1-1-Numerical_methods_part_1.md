---
layout: post
title:  "Numerical methods in practice, part 1/2"
date:   2018-03-01 21:00:00 +0300
categories: jekyll update
comments: true
identifier: 10013
titlepic: numerical-part1.png
---
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
## TL;DR

In this post we'll cover some basics for a solving root values for linear and non-linear functions.

## Motivation

If you're planning to make some probability calculation/market analysis or just interested in solving equations, you'll most likely need know some math knowledge and  some knowhow to solve math problems on a computer. It's also handy to know, what techniques to use for a given problem. For me, it took a while build this bridge between mathematics and numerical methods. When I began my studies in the school, I thought mathematics was purely about the analytical solutions and the beauty of math itself. But when my studies progressed, solving some problems weren't just possible by hand anymore (mainly finite elemnent problems). But I thought to give a quick review how to solve some common mathematical problems numerically.

## In search for root values

### Linear functions

Let's start with linear functions. One example of linear functions can be described as:

\begin{equation}
f(x) = 2x + 3
\end{equation}

Now, if $$f(x) = 0$$ solve $$x$$. No worries:

\begin{equation}
0 = 2x + 3 \rightarrow x = -\frac{3}{2}
\end{equation}

And one more example, where f.ex. $$f(x) = 2$$, solve $$x$$. Same procedure, but we'll just arrange the equation so that $$x$$ is on the left side of the function:

\begin{equation}
2 = 2x + 3 \rightarrow -2x = 3 - 2
\end{equation}

I guess the answer is quite simple, so we'll stop here. For convenience, let's change the last formula into symbols: $$-2 = A$$, $$3-2 = b$$:

\begin{equation}
Ax = b
\end{equation}

The first example could've also been formulated into this form, so keep in mind that we'll need this formula later. But so far, no special needs for numerical methods here. What about system of linear equations, f.ex. following situations?

<div align="center">
\begin{equation}
\begin{array}{lcl}
    4x_1 + 7x_2 + 2x_3 = 5 \\[0.3em]
    7x_1 + 7x_2 + x_3 = 3 \\[0.3em]
    9x_1 + 3x_2 + x_3 = 1
\end{array}
\end{equation}
</div>

Using the [elimination of variables](https://en.wikipedia.org/wiki/System_of_linear_equations) -method can be quite a workload, but lucky for us, we can take advantage of matrix methods. Let's arrange multipliers and variables into matrices:

<div align="center">
\begin{equation}
\begin{bmatrix}
    4 & 7 & 2 \\[0.3em]
    7 & 7 & 1 \\[0.3em]
    9 & 3 & 1
\end{bmatrix}
\begin{bmatrix}
       x_1 \\[0.3em]
       x_2 \\[0.3em]
       x_3
     \end{bmatrix}
= \begin{bmatrix}
       5 \\[0.3em]
       3 \\[0.3em]
       1
     \end{bmatrix}
\end{equation}
</div>

and let's name the first matrix as $$A$$, second $$x$$ and the last $$b$$:

\begin{equation}
Ax = b
\end{equation}

And look at that, the same formula as before, nice and neat! But no we just can't use division to solve our function, since matrices. We'll apply the inverse of $$A$$:

\begin{equation}
x = A^{-1}b
\end{equation}


There are multiple ways to solve the inverse of matrix (if you're into solving it by hand, check out this [link](http://www.purplemath.com/modules/mtrxinvr.htm)), but we'll just plainly use python and numpy to solve our example (Note: we're using matrices, so we'll use dot product):

```python
import numpy as np

A = np.array([[4,7,2],
              [7,7,1],
              [9,3,1]])
b = np.array([5,3,1])
vals = np.dot(np.linalg.inv(A), b)
```
    [-0.18518519  0.40740741  1.44444444]

### Non-linear functions

As the name suggests, non-linear functions aren't linear and we can't just plainly use matrix inverse here to solve our problems. Here the main tools are iterative solvers and we'll take a look into [Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method) first. If you're not into reading the description for Newton method, what is solves in the nut shell: find $$x$$, so that $$f(x) = 0$$. So first things first: we'll need to formulate our target function into a given shape:

\begin{equation}
f(x) = 0
\end{equation}

in which $$f(x)$$ is our target function. Next thing is to plug it into numerical solver:

\begin{equation}
x_{n+1} = x_n - \frac{f(x_n)}{f^{\'}(x_n)}
\end{equation}

in which $$x_n$$ is the start value in interation, $$x_{n+1}$$ is the new $$x$$ value, $$f$$ is target function and $$f^{'}$$ is the derivative of target function. In order to see function at work, check out this [wikipedia animation](https://en.wikipedia.org/wiki/Newton%27s_method#/media/File:NewtonIteration_Ani.gif). Last thing what we need is a stop condition. If you checked the animation, you can see how function converges towards zero, so let's use that as a stop condition. We'll stop looping, when $$f(x_n) < 0.00001$$.

## Example

Let's solve a familiar function: square root function. Square root is defined as:

\begin{equation}
f(x) = \sqrt x
\end{equation}

Our first task is to change our square root function to look like this:

\begin{equation}
f(x) = 0
\end{equation}

But we don't know, how to solve square root function! We'll need to modify the function a bit, so that we can solve it. So first we could change $$f(x)$$ into $$y$$, just for convenience, and apply square to both sides:

\begin{equation}
y = \sqrt x | ^{2} \rightarrow y^2 = x
\end{equation}

Last thing is to put $$x$$ into left side:

\begin{equation}
y ^2 - x = 0
\end{equation}

And here we have it. Our target function is therefore:

\begin{equation}
f_{target}(y, x) = y ^2 - x
\end{equation}

And what do we try to solve now? Original question was: "if we plug in $$x$$, what do we get for $$y$$?". Now the question could be formed as: "if we plug in $$x$$, what $$y$$ needs to be, so that the function is zero". So that we can to solve $$y$$, we need to take a derivative of our target function:

\begin{equation}
\frac{d f_{target}}{d y} = 2y
\end{equation}

Now we can finally code our solution:

```python
import numpy as np
import time

def newton_solver(x, f, df, max_iter=50, error=1e-5):
    def abs(x):
        return x if x > 0 else -1 * x
    i = 0
    while i < max_iter and abs(f(x)) > error:
        x = x - f(x) / df(x)
        i += 1
    if i > max_iter:
        raise SystemError("Maximum iteration succeeded!")
    return x

def target_function(x, y):
    return y ** 2 - x

def dtarget_function(x, y):
    return 2 * y

def sqrt(x):
    if x < 0:
        raise SystemError("x is negative, no solutions there!")
    x_0 = x / 3.
    f = lambda y: target_function(x, y)
    df = lambda y: dtarget_function(x, y)
    return newton_solver(x_0, f, df)

if __name__ == '__main__':
    value = 2.0
    # Calculate squre root using our own solver
    val = sqrt(value)

    # Calculate square using numpy
    val_2 = np.sqrt(value)

    print("Solution from numpy: {0}\nOur solution: {1}\nDifference: {2:.7f}".format(val,
                                                                                val_2,
                                                                                np.abs(val_2 -
                                                                                       val)))
```

    Solution from numpy: 1.4142137800471977
    Our solution: 1.4142135623730951
    Difference: 0.0000002


Like a charm! But still too much brain work, we could also be a bit more lazy and use [finite difference](https://en.wikipedia.org/wiki/Finite_difference) to calculate the derivative:

\begin{equation}
f^{\'}(x) = \lim_{h \rightarrow 0} \frac{f(x+h)-f(x)}{h}
\end{equation}

As the formula says, we'll get accurate derivate when $$h$$ is zero, but since we live in a finite world, we'll use $$h=1\cdot10^{-6}$$. Let's calculate the same example using the numerical derivative:

```python
import numpy as np
import time

def newton_solver(x, f, df, max_iter=50, error=1e-6):
    def abs(x):
        if x > 0:
            return x
        return -1 * x
    i = 0
    value = f(x)
    while i < max_iter and abs(f(x)) > error:
        x = x - f(x) / df(x)
        i += 1
    if i > max_iter:
        raise SystemError("Maximum iteration succeeded!")
    return x

def target_function(x, y):
    return y ** 2 - x

def derivative(x, f, h=1e-6):
    f_x_h = f(x + h)
    f_x = f(x)
    return (f_x_h - f_x) / h

def sqrt(x):
    x_0 = x / 3.
    f = lambda y: target_function(x, y)
    df = lambda y: derivative(x, f)
    return newton_solver(x_0, f, df)

if __name__ == '__main__':
    value = 2.0
    # Calculate squre root using our own solver
    val = sqrt(value)

    # Calculate square using numpy
    val_2 = np.sqrt(value)

    print("Solution from numpy: {0}\nOur solution: {1}\nDifference: {2:.7f}".format(val,
                                                                                val_2,
                                                                                np.abs(val_2 -
                                                                                       val)))
```
    Solution from numpy: 1.4142137800471977
    Our solution: 1.4142135623730951
    Difference: 0.0000002


Newton is just one of the many root-finding algorithms (bunch of other solves in [wikipedia](https://en.wikipedia.org/wiki/Root-finding_algorithm)), but newton is one of the most simplest and easiest to code.

## Conclusions

Root-searching is quite common in math problems and now you should have at least some knowledge how to use one solver. But do note, that some solvers are better in some problems then others, so don't sink into pride and accept that the one you know is a silver bullet for all problems :). In the next post we'll search for min-max values for some linear and non-linear functions.

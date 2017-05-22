---
layout: post
title:  "Neural Networks journey: Neural networks with softmax classifier"
date:   2017-5-21 21:00:00 +0300
categories: jekyll update
comments: true
identifier: 10010
titlepic: softmax.png
---
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
## TL;DR

Finally to the agenda: Neural networks. In this post we'll cover the categorization problem again using softmax classifier. All the code in this post is written in Python, version 3.6. I've gathered most of the used material from [here](http://cs231n.github.io/).

## Neural networks

Sorry, but this is going to be quite a long post, since there's a lot of stuff we have to go through. In the previous posts we've been doing "linear analysis", or we could say that the functions which we've fitted were linear. Neural networks enables us to fit non-linear data, which can be a huge advantage in some cases (f.ex. speech & image recognition). As before, let's create some data first, so we can visualize the problem.


```python
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from IPython.display import Image
from IPython.core.display import HTML

matplotlib.rcParams['figure.figsize'] = (15.0, 10.0)


num_of_points = 1500

# This is our magic function for classifying points
def magic_function(x, y):
    y_val = np.sin(np.pi*x / 2.) - np.random.rand() * 1.2
    if y_val < y:
        return 0
    if x > -1 - np.random.rand() / 2.:
        return 1
    else:
        return 2

# Let's generate some points
x = np.random.normal(-1, 2, num_of_points)
y = np.random.normal(-1, 1, num_of_points)
n = np.array([magic_function(i, j) for i, j in zip(x, y)])

# Combine data
train_input = np.array([x,y]).T
train_output = np.array(n)

# Shuffle the points
perm = np.random.permutation(train_output.shape[0])
train_input_shuffled = train_input[perm]
train_output_shuffled = train_output[perm]

# plot plot
plt.scatter( x, y, c=n)
plt.grid()
```


![png](/images/machine_learning_softmax/neural_networks_bck_2_0.png)


We'll start to build our neural network from the scratch and we'll begin with "feed forward"...

## Feed forward

As I studied more on neural networks, majority of the websites tended to start with brain analogy and the neurons inside the brain. For me this approach proved to be super confusing, so I'm just gonna skip that and go with the mathematical analogies (and later I try to draw some circles...). Our goal remains the same: we're again just trying to fit our function to some given data and find the optimal multipliers.

Let's first start with the feed forward or forward pass. As the name suggests, we have something, and feed it forward. In this case we have a function, which produces a results $$y$$, which we again feed to another function. This implies that we have some $$x$$ number of functions, instead of one. In the previous [Logistic regression post](http://ovainola.github.io/jekyll/update/2017/02/17/machine_learning_logistic_regression.html) we used sigmoid function, but this function is only capable of presenting binary values: $$1$$ or $$0$$. This is quite undesirable, when there are more than two categories, which is why we'll use the rectified linear unit function. In short, [ReLU]( https://en.wikipedia.org/wiki/Rectifier_(neural_networks) ):

\begin{equation}
f_{ReLU} = max(0, x)
\end{equation}

We're still going to use linear functions, which we will feed to the ReLU function:

\begin{equation}
f(x, \theta, b) = \theta^T x + b
\end{equation}

In this post I'm going to use following notation:

\begin{equation}
(x, \theta, b) \rightarrow f(x, \theta, b) \rightarrow \mathcal{O}
\end{equation}

in which $$\mathcal{O}$$ is the output of the function. In order to approach our problem, let's look at this using an example: we use two (2) functions and create a pipe where one feeds it's result to another function:


```python
linear_function = lambda a, b, x: np.dot(a, x) + b

def relu(x):
    """Sigmoid function https://en.wikipedia.org/wiki/Sigmoid_function
    """
    return np.maximum(0, x)

# functions 1 and 2
f1 = lambda x: relu(linear_function(1.8, 2.4, x))
f2 = lambda x: relu(linear_function(2.4, -2.2, x))

# Our pipe
f3 = lambda x: f2(f1(x))

# x values
x = np.linspace(-10, 10, 200)

# Results for each function
f1_results = f1(x)
f2_results = f2(x)

# Results for feed forward
feed_forward_results = f3(x)

# plotting
matplotlib.rcParams['figure.figsize'] = (15.0, 10.0)
plt.plot(x, feed_forward_results, 'r--', label="$f3=f_2(f_1(x))$")
plt.plot(x, f1_results, 'k', label="$f1$")
plt.plot(x, f2_results, 'b', label="$f2$")
plt.legend()
plt.grid()
```


![png](/images/machine_learning_softmax/neural_networks_bck_5_0.png)


Well, that  doesn't seem to be that useful at all. But wait! Fun part is yet to come. We can notice, how I put both $$f_1$$ and $$f_2$$ their own multipliers. We can like to visualize with some arrows:

\begin{equation}
x \rightarrow f_1(x) \rightarrow f_2(\mathcal{O_1}) \rightarrow y
\end{equation}

Now to the terminology: Our pipe represent our neural network and each functions present a layer. $$f_1$$ is input layer and $$f_2$$ is the output layer. If we would have more functions betweem $$f_1$$ and $$f_2$$, they would be called just a hidden layers. There can be N-number of hidden layers, but just for naming conventions, first is input layer and last output. And don't think it too hard... I tried and got confused.

Ok, easy peasy. Now to the fun part. What about if we increase the number of multipliers in each function? This should pose no problem, since we're using dot function, which will take care of sum and product operations (We just have to make sure all the matrices have proper shapes, because feed forward). We will be using a matrix for storing the multipliers, so we can choose the length and width for the matrix. Let's try with the case, where:
* $$f_1$$ multiplier matrix has length 5 and width 1
* $$f_2$$ multiplier matrix has length 1 and width 5



And if we expand the pipe:

\begin{equation}
x \rightarrow f_1(x) \rightarrow f_2(\mathcal{O_1}) \rightarrow y
\end{equation}

<div align="center">
\begin{equation}
x \rightarrow
\begin{bmatrix}
  \theta_{a, 1} \\
  \theta_{a, 2} \\
  \theta_{a, 3} \\
  \theta_{a, 4} \\
  \theta_{a, 5}
\end{bmatrix} \cdot \begin{bmatrix} x \end{bmatrix} +
\begin{bmatrix}
  b_{a, 1} \\
  b_{a, 2} \\
  b_{a, 3} \\
  b_{a, 4} \\
  b_{a, 5}  
  \end{bmatrix}
  \rightarrow \begin{bmatrix}
  \theta_{b, 1} &
  \theta_{b, 2} &
  \theta_{b, 3} &
  \theta_{b, 4} &
  \theta_{b, 5}
\end{bmatrix} \cdot \begin{bmatrix}
  \mathcal{O_1} \\
  \mathcal{O_2} \\
  \mathcal{O_3} \\
  \mathcal{O_4} \\
  \mathcal{O_5}
 \end{bmatrix} +
 \begin{bmatrix}
 b_b
 \end{bmatrix} \rightarrow y
\end{equation}
</div>

Let's give it a try:


```python
# I'll be using normal distribution for scattering the multiplier values
# I've picked mean and deviation values on random

# First function definitions
f1_matrix_length = 5
f1_matrix_width = 1
theta_a = np.random.normal(0.15, 1.4, (f1_matrix_length, f1_matrix_width))
b_a = np.random.normal(-1.2, 2.3, f1_matrix_length)

# Second function definitions
f2_matrix_length = 1
f2_matrix_width = 5
theta_b = np.random.normal(0.5, 2.2, (f2_matrix_length, f2_matrix_width))
b_b = np.random.normal(-0.12, 2, f2_matrix_length)

# Function wrappers
f1 = lambda x: relu(linear_function(theta_a, b_a, x))
f2 = lambda x: relu(linear_function(theta_b, b_b, x))

# Our pipe
feed_forward = lambda x: f2(f1(x))

# Calculate a result for given x
x = np.linspace(-10, 10, 200)
feed_forward_results = np.zeros(200)
for idx, xi in enumerate(x):
    feed_forward_results[idx] = feed_forward([xi])

# Plotting
matplotlib.rcParams['figure.figsize'] = (15.0, 10.0)
plt.plot(x, feed_forward_results, 'r--', label="$feedforward=f_2(f_1(x))$")
plt.legend()
plt.grid()
```


![png](/images/machine_learning_softmax/neural_networks_bck_7_0.png)


That's more like it. I'd say quite interesting shapes. We only applied one extra dimension to the matrix, but what if we add a second one? And since we're going crazy why not throw an extra function $$f_3$$. Do note, that the first matrix now has shape (5,2) so the input $$x$$ has to have shape (2).

\begin{equation}
x \rightarrow f_1(x) \rightarrow f_2(\mathcal{O_1}) \rightarrow f_3(\mathcal{O_2}) \rightarrow y
\end{equation}

<div align="center">
\begin{equation}
x \rightarrow \begin{bmatrix}
  \theta_{1, 1} & \theta_{1, 2} \\
  \theta_{2, 1} & \theta_{2, 2} \\
  \theta_{3, 1} & \theta_{3, 2} \\
  \theta_{4, 1} & \theta_{4, 2} \\
  \theta_{5, 1} & \theta_{5, 2}
\end{bmatrix} \cdot \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} +
\begin{bmatrix}
  b_{a, 1} \\
  b_{a, 2} \\
  b_{a, 3} \\
  b_{a, 4} \\
  b_{a, 5}  
  \end{bmatrix}
  \rightarrow \begin{bmatrix}
  \theta_{b, 1} & \theta_{b, 2} & \theta_{b, 3} & \theta_{b, 4} & \theta_{b, 5} \\
  \theta_{b, 1} & \theta_{b, 2} & \theta_{b, 3} & \theta_{b, 4} & \theta_{b, 5}
\end{bmatrix} \cdot \begin{bmatrix}
  \mathcal{O_{1, 1}} \\
  \mathcal{O_{1, 2}} \\
  \mathcal{O_{1, 3}} \\
  \mathcal{O_{1, 4}} \\
  \mathcal{O_{1, 5}}
 \end{bmatrix} + \begin{bmatrix}
  b_{a, 1} \\
  b_{a, 2}
  \end{bmatrix} \rightarrow \begin{bmatrix}
  \theta_{c, 1} & \theta_{c, 2}
\end{bmatrix} \cdot \begin{bmatrix}
  \mathcal{O_{2, 1}} \\
  \mathcal{O_{2, 2}}
 \end{bmatrix} + \begin{bmatrix}
  b_{c, 1}
  \end{bmatrix} \rightarrow y
\end{equation}
</div>

```python
# Once again, I'll be using normal distribution for scattering the multiplier values

# First function definitions
f1_num_of_neurons = 5
f1_num_of_multipliers = 2
theta_a = np.random.normal(0.5, 1.4, (f1_num_of_neurons, f1_num_of_multipliers))
b_a = np.random.normal(-1.2, 2.3, f1_num_of_neurons)

# Second function definitions
f2_num_of_neurons = 2
f2_num_of_multipliers = f1_num_of_neurons
theta_b = np.random.normal(1.5, 3.2, (f2_num_of_neurons, f2_num_of_multipliers))
b_b = np.random.normal(-0.12, 2, f2_num_of_neurons)

# Third function definitions
f3_num_of_neurons = 1
f3_num_of_multipliers = f2_num_of_neurons
theta_c = np.random.normal(2.4, 0.19, (f3_num_of_neurons, f3_num_of_multipliers))
b_c = np.random.normal(-2, 2, f3_num_of_neurons)

# Function wrappers
f1 = lambda x: relu(linear_function(theta_a, b_a, x))
f2 = lambda x: relu(linear_function(theta_b, b_b, x))
f3 = lambda x: relu(linear_function(theta_c, b_c, x))

# Pipe
feed_forward = lambda x: f3(f2(f1(x)))

# Calculate points for grid
num_points = 100
x_1 = np.linspace(-10, 10, num_points)
x_2 = np.linspace(-10, 10, num_points)

feed_forward_results = np.zeros((num_points, num_points))
for x_, x_i in enumerate(x_1):
    for y_, x_j in enumerate(x_2):
        value = feed_forward([x_i, x_j])[0]
        feed_forward_results[x_, y_] = value

# Plotting
X, Y = np.meshgrid(x_1, x_2)
matplotlib.rcParams['figure.figsize'] = (15.0, 10.0)
cs = plt.contourf(X, Y, feed_forward_results)
plt.colorbar(cs)
plt.grid()
```


![png](/images/machine_learning_softmax/neural_networks_bck_9_0.png)


Whoa, such matrix joggling... let's stop and go the analogies. They're much more interesting :)

<img src="/images/machine_learning_softmax/neuron_defin.png" alt="Drawing" style="width: 800px;"/>
![](/images/machine_learning_softmax/neuron_defin.png =250x100)

As we talked before, each function represents a layer. In each function, each row in multiplier matrix represents a neuron. Usually (at least the pages where I visited) the multiplier columns aren't drawn. This leaves us with the circles, which represent the neurons. In the picture above we can see that our matrix represents five (5) neurons. With the given analogy, we can draw our last example network as:

![png](/images/machine_learning_softmax/neural_network.png)


and thus the name: neural network. If you're still reading: **Congratulations**! I have to honest: we've only covered the first half here, but do not worry! Following stuff is almost the same as in the previous posts.

## Softmax classifier

### Classification using propabilities

In the last post support vector machine (SVM) and now we're about to use softmax classifier Definitions for the our loss function:

\begin{equation}
L_i = -\log(p_{y_i}), \hspace{1cm} p_k = \frac{e^{f_k}}{\sum_j e^{f_{j}}}
\end{equation}

and if we use logaritmic rules, we can derive:

\begin{equation}
L_i = -\log\bigg( \frac{e^{f_{y_i}}}{\sum_j e^{f_{j}}} \bigg)
\end{equation}

\begin{equation}
L_i = -\bigg( \log\big(e^{f_{y_i}}\big) - log \big(\sum_j e^{f_{j}} \big) \bigg)
\end{equation}

\begin{equation}
L_i =  log \big(\sum_j e^{f_{j}} \big) -f_{y_i}
\end{equation}

SVM produced a score for each class, but softmax produces a propability for each class. Let's calculate the same example as in the last post: input vector, which has six (6) elements and in the output we have four (4) classes.


```python
input_vector = np.array([5.2, 1.0, 6.356, 10.1, -4.45, 8.20]).T
theta = np.random.rand(4,6)
yi = np.array([1])
scores = np.dot(theta, input_vector)
exp_scores = np.exp(scores)
probs = exp_scores / np.sum(exp_scores)
print("Propabilities: {0}".format(probs))
Li = -np.log(probs[yi])[0]
print("Score: {0}".format(Li))
```

    Propabilities: [  8.50640639e-05   7.16811307e-03   3.23025170e-06   9.92743593e-01]
    Score: 4.938112829809168


Again, we apply regularization loss (the same $$L_2$$ as in the last post and since our network has multiple multipliers, let's sum these together) to our loss function which leaves us with:

\begin{equation}
L = \frac{1}{N}\sum_i L_i + \frac{\lambda}{2} \sum_j R(\theta_j)
\end{equation}

in which $$j$$ is the number of $$\theta$$'s. We still need calculate the gradient for the stochastic descent. Luckily we can use calculus for this task.

\begin{equation}
\frac{\partial L}{\partial f_k} = \frac{\partial }{\partial f_k} \Bigg( log \big(\sum_j e^{f_{j}} \big) -f_{y_i} \Bigg)
\end{equation}

Let's use derivate rules of logarithm to derivate our function: $$\frac{d}{dx} \log(x^n) = \frac{nx^{n-1}}{x^n}$$

\begin{equation}
\frac{\partial L}{\partial f_k} = \frac{e^{f_{k}}}{\sum_j e^{f_{j}}} - \mathbb{1}(y_i = k)
\end{equation}

\begin{equation}
\frac{\partial L}{\partial f_k} = p_k - \mathbb{1}(y_i = k)
\end{equation}


Easy peasy. I used the code from [this](http://cs231n.github.io/neural-networks-case-study/) site, since I didn't want to add my mistakes to here.


```python
def softmax_classifier(scores, y):
    """
    Softmax classifier, reference: http://cs231n.github.io/neural-networks-case-study/
    """
    n_values = len(y)
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]

    # compute the loss: average cross-entropy loss and regularization
    corect_logprobs = -np.log(probs[range(n_values),y])
    data_loss = np.sum(corect_logprobs) / n_values

    dscores = probs
    dscores[range(n_values),y] -= 1
    dscores /= n_values
    return data_loss, dscores
```

## Back propagation

In the last post we used only had one function, which we used for fitting (loss function). Updating the multipliers in the optimization loop was quite easy but think a bit our new situation: we have an input, we feed it into a network and get output. We have no idea, what kind of effect each layer had on the result. Solution to this problem is back propagation. In principle is we count backwards how much given layer had effect.

This requires some serious chain rule derivation, but here's a couple of derivates for our upcoming neural network with three layers:

\begin{equation}
\frac{\partial L}{\partial \theta_3} = \frac{\partial L}{\partial  \mathcal{O_3}} \frac{\partial \mathcal{O_3}}{\partial \theta_3} + \lambda \theta_3 \rightarrow \frac{\partial L}{\partial f_k} \mathcal{O_2} + \lambda \theta_3
\end{equation}

\begin{equation}
\frac{\partial L}{\partial b_3} = \frac{\partial L}{\partial \mathcal{O_3}} \frac{\partial \mathcal{O_3}}{\partial b_3} \rightarrow \frac{\partial L}{\partial f_k}\mathbb{1}
\end{equation}


\begin{equation}
\frac{\partial L}{\partial \mathcal{O_2}} = \frac{\partial L}{\partial \mathcal{O_3}} \frac{\partial \mathcal{O_3}}{\partial \mathcal{O_2}} \rightarrow \frac{\partial L}{\partial f_k}\theta_3
\end{equation}

in which $$\mathbb{1}$$ is array with ones. The hidden layers are derivated the same, but with more derivatives:

\begin{equation}
\frac{\partial L}{\partial \theta_2} = \frac{\partial L}{\partial  \mathcal{O_3}} \frac{\partial \mathcal{O_3}}{\partial \mathcal{O_2}} \frac{\partial \mathcal{O_2}}{\partial \theta_2} + \lambda \theta_2 \rightarrow \frac{\partial L}{\partial \mathcal{O_2}} \mathcal{O_1} + \lambda \theta_2
\end{equation}

\begin{equation}
\frac{\partial L}{\partial b_2} = \frac{\partial L}{\partial  \mathcal{O_3}} \frac{\partial \mathcal{O_3}}{\partial \mathcal{O_2}} \frac{\partial \mathcal{O_2}}{\partial b_2} \rightarrow \frac{\partial L}{\partial \mathcal{O_2}} \mathbb{1}
\end{equation}


After calculating all the derivatives we can apply the gradient update, f.ex:

\begin{equation}
\theta_3 = \theta_3 + \Delta \frac{\partial L}{\partial \theta_3}
\end{equation}

in which $$\Delta$$ is the step size. Matrix multiplications need some juggling with the transposes. If you need more information, please check [here](http://cs231n.github.io/optimization-2/) and [here](http://cs231n.github.io/neural-networks-1/) sites for definitions and examples.

## Model

Ok, let's sum things up: we need a pipe/neural network, softmax classifier and we need to back propagate the values for the gradient update. Let's give it a try. I've taken some examples from [here](http://cs231n.github.io/neural-networks-case-study/), which has been a great source for more definitions.


```python
# Number of inputs
D = 2

# Sizes of hidden layers
hidden_size_1 = 100
hidden_size_2 = 25
output_size = 3

θ_1 = 0.01 * np.random.randn(D, hidden_size_1)
b_1 = 0.01 * np.random.randn(1, hidden_size_1)

θ_2 = 0.01 * np.random.randn(hidden_size_1, hidden_size_2)
b_2 = 0.01 * np.random.randn(1, hidden_size_2)

θ_3 = 0.01 * np.random.randn(hidden_size_2, output_size)
b_3 = 0.01 * np.random.randn(1, output_size)

X = train_input_shuffled
y = train_output_shuffled

# some hyperparameters
step_size = 0.5
λ = 1e-3 # regularization strength

b_ones = np.ones(X.shape[0])

iteration_numbers = []
losses = []

for i in range(4000):

    # evaluate class scores, [N x K]
    O_1 = np.maximum(0, np.dot(X, θ_1) + b_1)   # note, ReLU layer
    O_2 = np.maximum(0, np.dot(O_1, θ_2) + b_2) # note, ReLU layer
    scores = np.dot(O_2, θ_3) + b_3

    # Calculate loss and gradient
    L, dL = softmax_classifier(scores, y)

    # Regularization penalty
    reg_penalty = sum([0.5 * λ * np.sum(x**2) for x in [θ_1, θ_2, θ_3]])
    total_loss = L + reg_penalty

    # Backpropagate gradients
    dθ_3 = np.dot(O_2.T, dL)
    db_3 = np.dot(b_ones, dL)

    dO_2 = np.dot(dL, θ_3.T)
    dO_2[O_2 <= 0] = 0            # Relu layer has max function, which requires this step
    dθ_2 = np.dot(O_1.T, dO_2)
    db_2 = np.dot(b_ones, dO_2)

    dO_1 = np.dot(dO_2, θ_2.T)
    dO_1[O_1 <= 0] = 0
    dθ_1 = np.dot(X.T, dO_1)
    db_1 = np.dot(b_ones, dO_1)

    # Add regularization gradient contribution
    dθ_1 += λ * θ_1
    dθ_2 += λ * θ_2
    dθ_3 += λ * θ_3

    # Perform a gradient update
    θ_1 += -step_size * dθ_1
    b_1 += -step_size * db_1

    θ_2 += -step_size * dθ_2
    b_2 += -step_size * db_2

    θ_3 += -step_size * dθ_3
    b_3 += -step_size * db_3

    # Add iteration number and loss for plotting
    iteration_numbers.append(i)
    losses.append(total_loss)

plt.plot(iteration_numbers, losses)
plt.xlabel("Iteration number")
plt.ylabel("Loss")
plt.grid()
```


![png](/images/machine_learning_softmax/neural_networks_bck_17_0.png)


A bit spiky, but converges so looks promising, let's plot the results.


```python
num_points = 300
x_1 = np.linspace(-8, 8, num_points)
x_2 = np.linspace(-5, 3, num_points)

feed_forward_results = np.zeros((num_points, num_points))
X1, Y2 = np.meshgrid(x_1, x_2)
for x_, row in enumerate(zip(X1, Y2)):
    for y_, vals in enumerate(zip(*row)):
        x_j, x_i = vals
        k = [x_j, x_i]
        O_1 = np.maximum(0, np.dot(k, θ_1) + b_1) # note, ReLU
        O_2 = np.maximum(0, np.dot(O_1, θ_2) + b_2) # note, ReLU
        scores = np.dot(O_2, θ_3) + b_3
        value = np.argmax(scores)
        feed_forward_results[x_, y_] = value

matplotlib.rcParams['figure.figsize'] = (15.0, 10.0)
cs = plt.contourf(X1, Y2, feed_forward_results)
plt.colorbar(cs)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.grid()
```


![png](/images/machine_learning_softmax/neural_networks_bck_19_0.png)


### Conclusion

In this post we implemented a neural network using softmax classifier. Next things to improve our model would be to test for over fitting, improving gradient update ([ref](http://cs231n.github.io/neural-networks-3/)), modify the input data for the network ([ref](http://cs231n.github.io/neural-networks-2/)) and overall code review to improve the code quality. Now the model, which we just created is only for educational purposes, so shouldn't use this in real analysis (at least at first). I'd recommend that if you're planning create some serious models, use f.ex. [scikit-learn](http://scikit-learn.org/stable/index.html) or [tensorflow](https://www.tensorflow.org/). But I guess making the best of these tools you need to know what's happening under the hood. Hopefully you liked this post, stay tuned!

---
layout: post
title:  "Deep Reinforced Learning"
date:   2017-12-8 21:00:00 +0300
categories: jekyll update
comments: true
identifier: 10010
titlepic: snake.png
---
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
## TL;DR

In this post we'll cover Deep Reinforced Learning -technique. It's recommended that you've some experience/knowledge how to deploy and use neural networks. Code is written in Python, version 3.6 >= and uses [Keras](https://keras.io/) library. Source codes can be fetched [here](https://github.com/ovainola/SnakeTamer). Inspiration for coding this excercise came from this [blog](https://keon.io/deep-q-learning/), where I also borrowed some code.

## Motivation

This year I've been, again, playing [Halite](https://halite.io/). Using my superior (at least me and my mom think so) machine learning skills I managed to get quite high ranking at first but unfortunately I got quite busy with work/personal life and I didn't have any extra energy to improve my bot. Last I checked my rank was roughty ~100, which is still quite good, at least on my standards. But while playing and gathering some data (all thanks for the Halite team for a great API), I started to think could I teach my bot with unsupervised techniques. Luckily I found this [blog](https://keon.io/deep-q-learning/), where I got some initial code and some explanations. Teaching a halite bot seemed quite a high obstacle, so I picked a bit more easier challenge: teach bot to play [snake](https://en.wikipedia.org/wiki/Snake_(video_game)). But first things first: the theory.

## Reinforced Deep Learning

First we should make a distiction between supervised and unsupervised learning. In supervised learning we have some data at hand: input and the corresponding output. In the previous posts we've been utilizing the supervised learning, since we knew the input and the output. In unsupervised learning we only have the input but not the output. This is a bit more harder task and the common problem solving techniques are clustering and association algorithms.

So what about reinforced learning? It's a type of unsupervised machine learning, where AI learns from the environment by interacting with it. In practical terms: make a prediction from an environment input data, apply prediction to the environment, update state and score the prediction.

## Neural network modifications

Training neural network in reinforced technique is almost the same as in supervised learning, but with some extra steps. First we need to know how to generate data.

As said in the previous chapter, AI needs to interact with the environment. The trick is, that first we'll let AI make random choices and reward (score) each choice. Some nomenclature: state is the frame/position the game has at the given moment. You can think it like chess: when it's your turn in chess, you see the board and positions of all the pieces on the table, which you use as an input to make the decision. As we let the AI play the game, we'll collect the states into input array and calculate the losses based on moves and rewards and add them into output array.

Training is done between the games: we'll collect data f.ex. 30 games, train the network, collect next 30 games and train again. But since we want to use our neural network to make decision at some point, we'll use decay variable to determine when to make actual predictions. As the decay variable suggests, we'll have to decrease the variables value each training cycle. Pseudo python network code would look like this:

```python
import numpy as np

class Network(BaseNeuralNetwork):
    decay_variable = 1.0
    decay_procentage = 0.995
    action_size = 3
    moves = []

    def predict(self, input):
        if np.random.rand() < self.decay_variable:
            return random.randrange(self.action_size)
        else:
            return self.network.predict(self.action_size)

    def add_data(self, state, new_state, reward, move):
        self.move((state, new_state, reward, move))

    def update_decay(self):
        self.decay_variable = self.decay_variable * self.decay_procentage

network = Network()
game = Game()
for i in range(1e3): # Pick some number of games
    state = game.state()
    for round in game:
        move = network.predict(state)
        new_state, reward = game.state(move)
        network.add_data(state, new_state, reward, move)
        state = new_state
    network.train()
    network.update_decay()
```

As you might have guess, after some time we'll only make AI predictions instead of random choices. The real prediction code looks like [this](https://github.com/ovainola/SnakeTamer/blob/master/dqn.py#L54) and the decay variable update can be seen from [here](https://github.com/ovainola/SnakeTamer/blob/master/dqn.py#L70).

In the supervised learning we'll have the input and expected output. In the unsupervised learning we only have the input and we need a reward to score our prediction. Applying the rewards to loss function is done as:

\begin{equation}
loss = (r + \gamma \max_{a^{\'}} \hat{Q}(s^{\'},a^{\'}) - Q(s, a))^2
\end{equation}

in which $$r$$ is the reward, $$\gamma$$ is the decay or discount rate, $$s$$ is the state, $$s^{'}$$ is the next state, $$a$$ is the action, $$Q$$ is the prediction and $$\hat Q$$ is the prediction of the next state. Detailed mathematical descriptions can be found from [here](https://arxiv.org/pdf/1312.5602.pdf). Pseudo code for the loss would look something like this:

```python
import numpy as np

network = Network()
gamma = 0.95
reward = 1
made_move = 0

state = [0.1, 0.4, 0.3]
new_state = [1.1, 0.3, 0.6]

prediction_new_state = network.predict(new_state)

target = (reward + gamma * np.amax(prediction_new_state)[0]))

prediction = network.predict(state)
prediction[made_move] = target - prediction[made_move]
loss = prediction ** 2
```

And what does this mean? Depending on the move, we'll give some feedback what AI did. Loss is something we want to minimize, so if you think about it:

* Loss is always positive, due to power of $$2$$
* We want the summation of loss function to be close to zero, since we want to minimize it
* $$\max_{a^{'}}\hat{Q}$$ produces the maximum value of prediction, most likely a positive value
* When $$r$$ is positive, $$Q$$ is most likely a positive value, since the summation
* When $$r$$ is negative, $$Q$$ is most likely a negative value, since the summation
* When making decision, we'll select the index of the highest positive value from the prediction array as the move

So basically, when $$r$$ is a large positive value we'll encourage AI to make that move again when given input is presented and discourage, if $$r$$ is negative. As you've might have thought about it, not only we have to create a neural network but we also have to create some rules on how to reward the AI. Training the network is done as in the previous post, so need to go there (except the epoch=1, which means how many times the data is shown during the minimization loop). If you'd like to see the actual python code, check out the [github](https://github.com/ovainola/SnakeTamer/tree/master) page.

## Snake game

I guess everybody has played snake at some point, so no explaning here. First task was to code the snake game itself. I didn't want to bother myself by actually coding the game, and ended by modifing the snake game presented [here](https://gist.github.com/sanchitgangwar/2158089). What I had to code myself were the rewards and states.

First the state, what do I want to present my snake as in input. I figured I want to show the snake (coordinates of head + body) and the position of the apple on the board. Only modification was to roll the table, so that the snake's head would be at center coordinates. This was done so that input data would be consistent. State code can be found from [here](https://github.com/ovainola/SnakeTamer/blob/master/snake.py#L63).

Next the rewards. I figured, that if snake collides with itself, I'd give -10. If snake moves closer to the apple, I'll give 1 and if further -1. Lastly if snake eats the apple, I'll reward it with 10. I picked these values, since I felt like it. No magic there :). Example how to give a reward can be seen from [here](https://github.com/ovainola/SnakeTamer/blob/master/snake.py#L123).

## Game and results

Last thing was to let the snake loose! I let it play a few thousands games and the output of game was this:

![png](/images/machine_learning_reinforced_learning/snake.gif)

Not very impressive, but how cool is this!? Without any predefined input data neural network can teach itself to play. I only had to give some rules how it should behave. I wanted to make a bit optimized snake, which can be found from [here](https://github.com/ovainola/SnakeTamer/tree/optimized).

My thoughts about optimizating the snake:

* Snake tends to favour some move combinations (f.ex. up + left), which is direct consequence of discouraging to collides with itself. This is a bit problematic in the late game (no strategy on how to make the snake maze)
* Quality of data: in each round there is only one move, where snake collides with itself, but alot of moves when the snake moves to some direction. This is not very concerning in early game but in the late game snake tends to collide with the tail.
* Changing the network size and width had some effect to the training, but increasing/decreasing these parameters did not dramatically change the results.

## Conclusions

Reinforced deep learning is indeed an interesting subject, which I plan to learn more. Hopefully you also got something from this post and got some idea how to use neural network in a "practical" implementation.

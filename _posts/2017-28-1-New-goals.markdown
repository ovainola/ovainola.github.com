---
layout: post
title:  "Change of the year and new goals"
date:   2017-1-28 21:00:00 +0300
categories: jekyll update
comments: true
identifier: 10003
titlepic: robot_reso.png
---
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

## TL;DR

Started to play halite, ended up studying neural networks.

## Small break

Last November I decided to take a small break from writing and actually set some
goals and learn new stuff. For a couple of weeks I just wondered what I should
do, but then I came a across with [halite](https://halite.io/) on [HackerNews](https://news.ycombinator.com/):
internet competition/game, where you'll have to create
ultimate bot to beat other bots. I didn't have that much
experience on artificial intelligence/bots so I decided to give it a try.
Stepping out of your comfort zone is always a pain in a ass...

The goal of the game is to create a bot, which conquers as much surface area
as possible, thus eliminating other bots in the field. There were couple of
options on what programming language you could use,
but I chose Python since I've used it quite extensively in other projects.

## The game

Game itself has rules somewhat like this: there's  a grid and you'll start with one block
and as you gain more ground, the number of your blocks increases. Each turn
you'll need to decide how to move each block. Waiting increases your blocks
strength, moving to new area diminishes and combining two or more blocks creates one,
with strength of the sum of the ones combined. Also the sides of the map is
connected: left side is connected to the right side and top to bottom.

![png](/images/new_goals/halite_grid.png)



After completing the tutorials I thought what's the simplest bot,
so I end up making this pseudo code:

```
some_irrelevant_initializations()

while True: # Game is running
    for block in all_blocks:
        if block.id == my_id:
            array_all_distances = calculate_hypot_to_each_block(block.location)
            direction = find_shortest_distance_to_block_with_not_my_id(array_all_distances)
            game.add(block.location, direction)    
```

My strategy was to minimize the distance to the nearest "enemy".
This turned out quite efficient at first and I rose to rank ~100 somewhere on December. But the
more bot played, more my rank started to fall. One bug in the algorithm was that
it took the absolute distance from one block to another on the grid,
not relative one. This meant f.ex. let's say block in row 10 and column 1 (let's use python array indeces)
had quite a large distance to block in row 10 and column -1, even if in reality,
on the map they were next to each other. I made some improvement here (used
numpy/scipy for gaining speed) but this clearly wasn't the winning strategy.

Even when I started to play I knew the probability of me winning the game is
quite small, but now I realized I need more elegant tools. At the time I was
talking with my brother about this game and he suggested why not machine learning?
I didn't have that much knowledge on that either so I started from the bottom: from
linear regression to neural networks.

In the couple of following posts, I'll be sharing my learning curve. It
turned out to be quite exciting trip or at least cleared some facts for me on
machine learning. On January I succeeded to make a bot, which was taught
using neural network, with couple of ReLU neural layers. Results weren't
that good (bot couldn't even win my first bot) but there's always room for an
improvement (next stop could be using convolutional neural networks).
And in this case it's about the journey, not the destination.

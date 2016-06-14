---
layout: post
title:  "First post using Github & Jekyll!"
date:   2016-6-10 21:02:14 +0300
categories: jekyll update
comments: true
---
<p><font size="6">S</font>o it begins! Sites are constructed (read edited),
fonts selected and layouts finished. Finally making some progress! The motivation
 to start the blog came from Jukka Aho, who also works in the
  <a href="https://github.com/JuliaFEM/JuliaFEM.jl">JuliaFEM</a> project.
If you're into some hard core finite element analyses, go check out he's
<a href="http://ahojukka5.github.io/">blog</a>. As my first post, I'll share
how did I managed to get the site up.</p>

## Installing requirements

When I started to search for an answer: "how to start a blog", I bumped into
[this site](https://help.github.com/articles/setting-up-your-github-pages-site-locally-with-jekyll/).
The site contains pretty clear instructions how to start building your own blog, but in short, I needed:

 * GitHub account
 * Ruby (version > 2.0.0)
 * Jekyll

 At that point, I already had a GitHub account. If you don't, go check out
  [this page](https://help.github.com/articles/set-up-git/). For me, the first thing
  I had to get was Ruby. And as a note, I have Linux Mint 17 as my operating system.
  In order to minimize my work, I first checked if Ruby was already preinstalled in my system.

```
>> ruby
The program 'ruby' can be found in the following packages:
 * ruby
 * ruby1.8
Try: sudo apt-get install <selected package>
```

So, no Ruby. From the message it's clear that the Ruby version available through
apt-get is 1.8, and that's a no go. Off to the internet to search for the solution.
After Scrolling through the StackOverflow, the most popular solution seemed to
be to build either [rbenv](https://github.com/rbenv/rbenv) or [rvm](https://rvm.io/).
After careful decision making process, I went for the rbenv, purely because I
visually preferred the GitHub pages more. Also rbenv had pretty good installation instructions. Off to building rbenv:

```
>> git clone https://github.com/rbenv/rbenv.git ~/.rbenv
>> cd ~/.rbenv && src/configure && make -C src
>> echo 'export PATH="$HOME/.rbenv/bin:$PATH"' >> ~/.bashrc
>> source ~/.bashrc && which rbenv
~/.rbenv/bin/rbenv
```
I'm still missing Ruby so I also installed [ruby-build](https://github.com/rbenv/ruby-build#readme)
package. The package is used to compile different versions of Ruby.
Since I needed version > 2.0.0, I decided to take the latest: 2.3.1:

```
>> git clone https://github.com/rbenv/ruby-build.git ~/.rbenv/plugins/ruby-build
>> rbenv install -l |grep 2.3
  1.9.1-p243
  2.2.3
  2.3.0-dev
  2.3.0-preview1
  2.3.0-preview2
  2.3.0
  2.3.1
  rbx-2.2.3
  rbx-2.3.0
>> rbenv install 2.3.1
```

Next in line: JeKyll. This was quite smooth sailing since the hard part was over.

```
>> gem install jekyll
```

## Creating repository

First thing to do was to create a new repository in GitHub website. The name was to be
"ovainola.github.io", as it was suggested in [here](https://jekyllrb.com/docs/github-pages/).
After that, I wanted my life to be as easy as possible, so I used Jekyll to create
a test website for me. I also wanted to test it locally immediately and here "jekyll serve" command
proved to be pretty handy.

```
>> cd ~ && mkdir blog && cd blog
>> jekyll new ovainola.github.io
>> cd ovainola.github.io
>> jekyll serve
```

Now I had a first template site working. Next stop was to create initial commit.
Since I already had created a repository for my site, I just initialized
git, added new remote and created the first commit:

```
>> git init
>> git remote add origin git@github.com:ovainola/ovainola.github.io
>> git add .
>> git commit -m "initial commit"
>> git push origin master
```

After a few minutes, [http://ovainola.github.io/](http://ovainola.github.io/) was up and running!
Great success, just like Borat says.

## Making it pretty

Now that site was up, the first thing that bothered me were the styles.
Even if this is supposed to be a technical blog, I still wanted it to look nice and pretty.
Since this is a matter of taste and debate, it took me a while to search for a
nice template. I've been using Bootstrap in the past, but I'm an
engineer, not a designer. I have an eye for technical things, but making something look
nice from the scratch is out of my league.

Luckily, I found [HTML5 up](https://html5up.net/). Very modern looking templates
with [Creative Commons Attribution](https://html5up.net/license) license. There were
multiple candidates, but I selected the Spektral theme. After downloading,
it was more or less copy/paste work. I guess I should have been a bit more original
and at least change the colorsheme... let's add that to TODO list...

And after making a couple of commits, site turned out to look like this. Even though
I said the hard part is over, I guess it's only just about to start since now I can
start to make some real posts. I was planning to make 1 post/month, so don't forget to
check out at least once a month, if something new has been posted ;). Subjects may
vary, but as for the second post I was planning to write something from integrating
element surfaces or how to wrap some numerical C++ library to python/Julia. So, see you
next time!

[jekyll-docs]: http://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/

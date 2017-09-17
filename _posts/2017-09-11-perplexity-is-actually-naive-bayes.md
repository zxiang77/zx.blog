---
layout: post
title:  "Perplexity is Actually Naive Bayes"
date:   2017-09-11 11:34:47 -0700
categories: nlp
description: explain why perplexity in natural language processing language model is a form of linear classifier or naive bayes classifier
---

### Perplexity is Actually Naive Bayes
$$\DeclareMathOperator*{\argmin}{arg\,min}$$
$$\DeclareMathOperator*{\argmax}{arg\,max}$$
$$\DeclareMathOperator*{\sign}{sign}$$

Before everything, it's for everyone's good to define some of the notations at the very beginning:

__Sentence of length $N$__: $W=w_1w_2...w_i...w_N$

__Perplexity__: $PP(W)=p(W)^{-1/N}=p(w_1w_2...w_N)^{-1/N}$

Perplexity is usually used to test how much the test set correlates to a language model. The lower the perplexity, the more correlation the test set has with the language model. So, it is useful in classifying a sentence (negative/ positive reviews, topic classification, etc.) by training language models for different classification and compare their perplexity with test sentence. The test sentence is most likely to be in the class of the language model having the lowest perplexity with it.

Then let's talk about why its doing the same thing as Naive Bayes. I will demonstrate that by discussing a binary classification problem: positive/ negative reviews classification. It will be self-proving to extend it to multinomial classification after this.

The language model way of doing it is to get:

$\argmin\limits_{c\in\{-1,1\}} PP(W|y=c) $

$= \argmin\limits_{c\in\{-1,1\}} p(W|y=c)^{-1/N}$

$=\argmax\limits_{c\in\{-1,1\}} p(W|y=c)$

While in Naive Bayes, we are exactly dealing with this!!

$\argmax\limits_{c\in\{-1,1\}} p(W|y=c)$

Similarly, multinomial classification only replace $c\in\{-1,1\}$ with $c\in C$, with $C$ representing all the possible classes. Then it's getting much more clear now.

$\argmax\limits_{c\in\{-1,1\}} p(W|y=c)$

$=\argmax\limits_{c\in\{-1,1\}} p(w_1|y=c)p(w_1|w_2,y=c)...p(w_i|w_{i+1},y=c)...p(w_{N-1}|w_N,y=c)$
(called introducing Naive Bayes rules for NB, and Ngram for language model)

$=\argmax\limits_{c\in\{-1,1\}} \log p(w_0w_1|y=c)+\log p(w_1|w_2,y=c)+...+\log p(w_i|w_{i+1},y=c)+...+\log p(w_{N-1}|w_N,y=c)$
How do we relate this to n-gram model? If we see the above equation as $y = \vec{w}\vec{x}$, then let's just think both $\vec{w}$ and $\vec{x}$ as vectors of size $V^n$ with $V$ representing vocabulary size. Then these vectors can represent any n-gram in the language model and it is straightforward that each value in the $\vec{w}$ represents the log-likelihood of certain n-gram, while each value in the $\vec{x}$ represents whether that n-gram appears in the sentence, so $\vec{x}[i]\in \{1, -1\}, (1\leq i\leq N)$. And from now on we can represent the correlation of a sentence with a language model by computing $y=\vec{w}\cdot\vec{x}$, the larger the $y$, the higher the probability, the more correlation they have.

And for binary classification, we can have:

$y_P=\vec{w_P}\cdot\vec{x}, y_N=\vec{w_N}\cdot\vec{x}$

$y=\sign y_P\cdot P(y=1)-y_N\cdot P(y=-1) $

$y=\sign (\vec{w_P}\cdot P(y=1)-\vec{w_N}\cdot P(y=-1))\cdot\vec{x}$, don't forget the prior here

$y=\sign \vec{w}\cdot\vec{x}, (\vec{w}=\vec{w_P}\cdot P(y=1)-\vec{w_N}\cdot P(y=-1))$

Finally, we transformed the perplexity problem into a linear classifier form.

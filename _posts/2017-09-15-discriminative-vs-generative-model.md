---
layout: post
title:  "Generative Model vs. Deterministic Model"
date:   2017-09-15 17:57:35 -0700
categories: ml
---

### Generative Model vs. Deterministic Model
In the world of machine learning, we actually deal with probability for the most of the time. To name a few, Naive Bayes, logistic regression. However, there are two main ways to deal with it, given observation $O$, and the sequence of states $S$ and the two ways are related to two different types of models in machine learning:

1)<b>discriminative models</b> maximize the chain probability given the observation (also called MLE - maximum likelihood estimation):

$\DeclareMathOperator*{\argmax}{arg\,max}$
$\argmax\limits\_{S}P(S\|O)$

2)<b>generative models</b> optimize posterior probability given prior probability (also called MAP - maximize a posterior):

$\argmax\limits\_{S}P(S\|O) $

$=\argmax\limits\_{S}\frac{P(O\|S)P(S)}{P(O)}  $ (Bayes Rules)

$=\argmax\limits\_{S} P(O\|S)P(S)$ remove $P(O)$ as that does not affect when $S$ can maximize the term

This is the final posterior term we want to maximize:

$\argmax\limits\_{S} P(O\|S)P(S)$


Here, the $P(S)$ is the prior.


P.S.: the discriminative and generative model's connection with MLE v.s. MAP concepts are just my own words, I didn't find anyone relate them together, though they share the same formula. If anyone knows any difference, please let me know!

It is usually said that generative models are more flexible because it can be re-constructed to the form of $P(S\|O)$, but the fact is, discriminative usually gives better performance than generative model. It is due to the principles behind these two types of models: 1) generative models model how the observations were generated and calculate which class gives the highest probability given the generation assumption ($P(x\|y)P(y)$); 2) discriminative, however, only cares about classifying the data based on data ($P(y\|x)$). These concepts makes generative models have more assumptions than discriminative models, so when its assumption does not stand well, its performance will be worse than discriminative models.

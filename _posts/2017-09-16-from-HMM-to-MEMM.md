---
layout: post
title:  "From HMM to MEMM"
date:   2017-09-16 17:57:35 -0700
categories: NLP
---
$\DeclareMathOperator*{\argmax}{arg\,max}$
Both HMM (Hidden Markov Model) and MEMM(MaxEnt Markov Model) are widely used in the field of NLP. This article is trying to articulate these two models and compare their differences. Most of my knowledge about them comes from [slides of MIT's NLP course](http://www.mit.edu/~6.863/spring2011/jmnew/6.pdf).

### HMM
First thing first, HMM has two inter-dependent sequences: <b>observation</b>, and <b>(hidden) states</b> as we can indicate from the name of HMM (say with me: H-I-D-D-E-N Markov Model), the state changing lies behind what we can observe. e.g. POS (part of speech) tagging (states, as what lies behind) for a sentence (observation, what we see):


observations: I like apple


states: PRP VBP NN


Given <b>observation</b>, and <b> states</b> defined, let's define more important variables for HMM:


$A$: <b>state changing matrix</b>, with $A\_{i,j}$ stands for the probability from state $i$ to state $j$


$B$: <b>emission probability</b>, with $B\_{j,o\_t}$ stands for the probability emission from state $j$ to observation $o\_t$


#### What do we need from HMM?
There are three main things I will cover in this article:

1) estimating probability of a sequence of states

2) decoding, given observations and a HMM model $\lambda(A,B)$

3) learning, given a set of states and observations, learn parameter $A$ and $B$ for HMM.

I'll explain why I need these now, but before that it is useful to know the difference between <b>generative model</b> and <b>discriminative model</b>. You can check this [post]({{ site.url }}/ml/2017/09/15/discriminative-vs-generative-model.html) out.

#### How to Make Use of $A$ and $B$?
First of all, HMM is a generative model. That means it will do:

$\argmax\limits\_{S}P(O\|S)P(S)$

$=\argmax\limits\_{S} \prod\limits\_{s',s\in S, o\in O} P(o\|s)P(s)$

$=\argmax\limits\_{S} \prod\limits\_{s',s\in S, o\in O} B\_{s,o}P(s')A\_{s',s}$

$P(s')$ here stands for probability of previous observation's ending in state $s'$.

Actually, this term solves the first two of our problems: 1) getting probability and 2) getting most likely sequence.

#### Forward Algorithm and Viterbi Algorithm
So we got our objective in deriving probability and decoding, so how could we implement it? If we want to do it with brute force, it will be an $O(n^t)$ solution with $n$ stands for number of states, and $t$ stands for the length of the observations, which is not acceptable. However, it can be solved with dynamic programming with just $O(n\cdot t)$ time complexity, because we can derive the probability for each state in current observation with just all the probability of each state from the observation right before current observation. I am gonna just put the pseudo-code for forward and Viterbi algorithm from MIT slides here.

![Forward]({{ site.baseurl }}/assets/Forward.png "forward algorithm")
![Forward trellis]({{ site.baseurl }}/assets/Forward_trellis.png "Forward trellis")
![Viterbi algorithm]({{ site.baseurl }}/assets/Viterbi.png "Viterbi algorithm")
![Viterbi trellis]({{ site.baseurl }}/assets/Viterbi_trellis.png "Viterbi trellis")
/******************pseudo-code here *****************/
Here we have to note that state 1 and  $N$ here are starting state and ending state, so we only deal with them during initialization and termination step, not in iteration step. Pseudo-code might consider them during iteration step, but it's not actually considering them (there were a little bit inconsistency in the MIT slides when I was reading it, but just be aware of start and end state here)
As we can see, though both use dynamic programming, there are a few differences here:

1) The iteration equation for forward algorithm and Viterbi are respectively:

$forward[t][i] = \sum\limits\_{1 <j <N} forward[t-1][j]\cdot A[j][i]$

$Viterbi[t][i] = \argmax\limits\_{1 <j <N} Viterbi[t-1][j]\cdot A[j][i]\cdot B(i, o\_t)$

$backpointer[t][i] = \argmax\limits\_{1 <j < N} Viterbi[t-1][j]\cdot A[j][i]$ (again not considering $B(i, o\_t)$ here because it does not affect the result)

It is just because we want to get the probability of observations given the model $P(O\|\lambda)$, this probability just sums up probabilities of all the possible sequences of states. While with Viterbi, we are only concerned with the one sequence of states that maximize the probability of the sequence.

#### How to train an HMM?
Here comes the most important part about the model. The quick answer is Expectation Maximization (EM) with forward-backward algorithm.
The long answer goes from here: for a normal Markov model with no hidden state, the state transformation can be easily computed by:
$\hat{A\_{i,j}}=\frac{\text{count of all transformation from i to j}}{\text{count of all transformation from i}}$

But we cannot do it with HMM because of hidden states. Therefore, we will need to derive hyper-parameters with some estimation with probability:

$\hat{A\_{i,j}}=\frac{\text{possibility of all sequences transform from i to j}}{\text{possibility of all sequences from i}}$

How do we get the probability of all sequence from $i$ to $j$? The answer is - forward-backward algorithm!

So as we already know that from forward algorithm, we know $\alpha[t][j]$ stands for the probability so far at $o\_t$ of all the state sequences ending in state $j$. From this probability, we only know the probability of $o\_1o\_2...o\_i...o\_t$, but nothing behind $o\_{t+1}...o\_T$. That's why we need a backward algorithm here to 'foresee' the probability of observations behind current observation.

The backward algorithm is very similar to forward algorithm, except for one going from backward one from forward, so we just need to modify forward algorithm a little bit:

1) Iterate from $T$ to $1$;

2) Initialization: $\beta[T][i] = A\_{iN}, 1<i<N$

3) Recursion: $\beta[t][i] = \sum\limits\_{1<j<N}A[i][j]\cdot B[j][o\_{t+1}]\cdot\beta[t+1][j]$

4) Termination: $\alpha[T][N] = \beta[1][1]= P(O\|\lambda)=\sum\limits\_{j=1}^{N-1}A[1][j]\cdot B[j][o\_1]\cdot\beta(2, j)$

Now we have everything we need for estimating sequence probability:

define $\xi\_t(i, j)$ as probability of being in state $i$ in $t$ and $j$ in $t+1$.

$\xi\_t(i, j)=\frac{\alpha[t][]\cdot A[i][j]\cdot \beta[t+1][j]\cdot B[j][o\_{t+1}]}{\alpha[T][N]}$

Similar for estimating $B$:

$\hat{B}[j][v\_k] = \frac{\text{expected count of observing $v\_k$ on j}}{\text{expected count on state j}}$

To estimate it we want to know the probability of being in state $j$ at time $t$, denoted as $\gamma[t][j]$:

$\gamma[t][j] = \frac{P(q\_t=j, O\|\lambda)}{P(O\|\lambda)}=\frac{\alpha[t][j]\cdot \beta[t][j]}{P(O\|\lambda)}$

and we can estimate $A$ with $xi$ and $B$ from it with $\gamma$. And pseudo-code below.
![forward-backward]({{ site.baseurl }}/assets/forward_backward.png "forward-backward algorithm")

Actually I was first confused by how to train the model when given a dataset of observations, because the code here only takes a singe sequence of observations into account. And I don't know when we a dealing with many sequences of observations, could it still converge? My answer from my friend is "YES", though he doesn't know the proof either. So if anyone knows the proof, feel free to share it with me!

###MEMM
Ok. So we have talked much about HMM and let's briefly talk about MEMM (Max Entropy Markov Model). Before talking about MEMM, we should know MaxEnt classifier first. It is also called multinomial logistic regression. I will explain why later, but now I will focused more on how it works. It also works like logistic, because it takes a few observed features and output the distribution of probabilities among classes. <b>Entropy</b> describes the average amount of information from a distribution. The higher it is, the less information we have. The entropy of a distribution $H(x)$ is computed as:

$H(x)=-\sum\limits\_{x} P(x)\cdot \log\_2 P(x)$

 This sounds too abstract, check out this  [example](https://en.wikipedia.org/wiki/Entropy_(information_theory)#Example) on Wikipedia.

As we can observe from the name, this classifier maximize the entropy of the training data, to achieve this, we have to deal with two cases:

1) When we observed some features that makes the probability distribution among classes unevenly, such as it is more likely be class $c$ given feature $f$, then assign probability accordingly to class $c$;

2) When we have no more features left, we just assign the rest of probability evenly to the rest of the classes.

Actually, it turns out when we are maximizing the entropy of the model given the dataset, we are maximizing the probability of training data as well, that's why MaxEnt is also called multinomial logistic regression.

There is a thing we should note, is that the feature $f$ here, we should see it as a binary function, because it only indicates if the feature exists in the observation or not.

#### Classification
So far the MaxEnt classifier only classify a single observation based on its features, so how do we implement it in  sequence labeling? The answer is MEMM. Actually we can implement Viterbi algorithm from HMM here with a little modification because they just share something in common. Assuming we are classifying observation $o\_t$, the outcome is the probability distribution of all the class in this position. If we think about it in sequence labeling, when we are labeling $o\_t$, we already found the probability distribution of all classes for all previous observations from $o\_1o\_2...o\_{t-1}$. Then we can classify $o\_t$ with previous labels with HMM and more excitingly with other features!

So far, what we are talking about MaxEnt, we are talking about classifying based on given features, this sounds like a discriminative model right? And yes! <b>MaxEnt classifier is a discriminative classifier</b>. Therefore, we will change the iteration equation of Viterbi from HMM form to MaxEnt form, like from:

$Viterbi[t][i] = \argmax\limits\_{1 <j <N} Viterbi[t-1][j]\cdot A[j][i]\cdot B(i, o\_t)$

to:

$Viterbi[t][i] = \argmax\limits\_{1 <j <N} Viterbi[t-1][j]\cdot P(s\_j\|s\_i,o\_t), 1 < j < N, 1 < t < T $

$=\argmax\limits\_{1 <j <N} Viterbi[t-1][j]\cdot \vec{f}\cdot\vec{w}$

And that's it!
#### Why MEMM over HMM?
The reason is quite obvious. Because for HMM, we only derive state for our next sequence given the result of previous sequences. However, this does not work when we have some features only lies behind our current observation. MEMM can introduce these features while training its parameters and that's a huge advantage.

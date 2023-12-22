---
title: Sum-Check
description: Simple, academic implementation of the Sum-Check protocol
date: 2023-12-14
tags:
    - zero-knowledge
---

# Sum-Check

[Github repo](https://github.com/gabrielfior/baby-sumcheck/blob/main/sum-check_sympy.py)

I'm writing this post as an introduction to the Sum-Check protocol.

## Motivation

Although there are many references on Sum-Check (see [original paper](https://dl.acm.org/doi/10.1145/146585.146605), or [link](https://semiotic.ai/articles/sumcheck-tutorial/) or many others), I decided to write this in a hopefully simpler way for people getting started in Zero-Knowledge to understand.

I also highlight the excellent [manuscript](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.html) from Justin Thaler that I used as reference during my learning, as well as the very helpful discussions with [Diego Kingston](https://twitter.com/zkdiegokingston) .


## Math background

>
Consider the polynomial \(g\), defined over a finite field \(\mathbb{F}\) (consider \(\mathbb{F}\) as the Boolean field, containing elements 0 and 1). The idea of the sum-check protocol is for the prover $P$ to convince the verifier $V$ that the sum of the polynomial $g$ over the field is equal to $H$. In mathematical terms:


\begin{align}
H := \sum_{b_1 \in \{0,1\}} \sum_{b_2 \in \{0,1\}} \cdots \sum_{b_v \in \{0,1\}} g(b_1,b_2,...,b_v)  \label{definition}\tag{1} 
\end{align}

We can also represent the equation above as follows:

\begin{align}
H = g(0,0,...,0) + g(0,0,...,1) + \cdots + g(1,1,...,1) \label{definition_expanded}\tag{2} 
\end{align}

As discussed in the manuscript, there are several problems which can be represented as a sum just like equation [1](#mjx-eqn-definition), for example matrix multiplication (see [Thaler13](https://eprint.iacr.org/2013/351.pdf) or this [nice video](https://www.youtube.com/watch?v=tPZDIzrsg-E&ab_channel=ZeroKnowledge) by Modular Labs on how they use sumcheck for proving matrix multiplication in the realm of Machine Learning verification).


A great question at this point is: why should the verifier \(V\) use sumcheck if \(V\) can simply evaluate the polynomial at all points of the finite field? The reason is **speed**. Compare the alternatives below:

- Without sumcheck - evaluate \(g\) at \(2^v\) points
- With sumcheck (as depicted later) - evaluate \(g\) at 2*v points (O(v) runtime complexity)

Hence it's reasonable to use sumcheck.

## Explanation of protocol

I find very useful whenever I see an example of how the protocol can be used with a polynomial I can understand. So let's start with a simple example from the manuscript:

\begin{align}
g(x_1,x_2,x_3) = 2x_1^3 + x_1x_3 + x_2x_3 \label{example_equation}\tag{3} 
\end{align}

First, let's calculate \(H\). We evaluate \(H\) at all points in the finite field, i.e. we use all possible combinations of 0 and 1 for variables \(x_1\), \(x_2\) and \(x_3\).


\begin{gather}
	H = g(0,0,0) + g(0,0,1) + g(0,1,0) + g(0,1,1) + g(1,0,0) + g(1,0,1) + g(1,1,0) + g(1,1,1) \label{eval_g_simple_example}\tag{4}\\
	H = 0 + 0 + 0 + 1 + 2 + 3 + 2 + 4  \notag\\
    H = 12
\end{gather}

Now, we describe how the protocol takes place, so the prover \(P\) convinces the verifier \(V\) that \(H=12\). The key to understand sumcheck is to observe that it's a recursive protocol, i.e. we have multiple rounds (1 per variable of the polynomial), and on each round we will "transform" the polynomial from \(v\) variables (in our example, \(v=3\)) to 1 variable, and evaluate this transformed polynomial in our entire finite field (i.e. at points 0 and 1).

### Start of protocol

- \(P\) sends - \(V\) the claimed value of \(H\), in our case 12.

### First round - \(x_1\)

The prover calculates a "transformed" version of the polynomial (given by \(s_1\)), calculated by:

\begin{gather}
	s_1(x_1) = g(x1,0,0) + g(x1,0,1) + g(x1,1,0) + g(x1,1,1) \label{eval_s1_x1}\tag{5}\\
	s_1(x_1) = (2x_1^3) + (2x_1^3 + x_1) + (2x_1^3) + (2x_1^3 + x_1 + 1) \\
    s_1(x_1) = 8x_1^3 + 2x_1 + 1
\end{gather}

- Prover sends the polynomial \(s_1(x_1)\) to the verifier
- Verifier checks that \(s_1(0) + s_1(1)\) = H
- Verifier draws a random variable \(r_1\) from \(\mathbb{F}\) and sends it to P (let's assume \(r_1 = 2\))


### Second round - \(x_2\)

Again, prover calculates "transformed" version of polynomial \(s_2\), this time using the random \(r_1\) value given by the verifier. Note that again the \(s\) polynomial is univariate.
\begin{gather}
	s_2(x_2) = g(r_1,x_2,0) + g(r_1,x_2,1) = g(2,x_2,0) + g(2,x_2,1) \label{eval_s2_x2}\tag{6}\\
	s_2(x_2) = 16 + (16 + 2 + x_2) \\
    s_2(x_2) = 34 + x_2
\end{gather}

- Prover sends polynomial \(s_2(x_2)\) to the verifier
- Verifier checks that \(s_2(0) + s_2(1)\) = \(s_1(r_1)\)
    
    > Here, observe that the verifier does not **compute** \(s_1(r_1)\). Instead, the verifier gets those values from the prover and simply verifies that the left side and right side match.

- Verifier draws a random variable \(r_2\) from \(\mathbb{F}\) and sends it to P (let's assume \(r_2 = 3\))

### Third round - \(x_3\)

Again, prover calculates "transformed" version of polynomial \(s_3\), this time using the random \(r_1\) and \(r_2\) values given by the verifier. Note that again the \(s\) polynomial is univariate.

\begin{gather}
	s_3(x_3) = g(r_1,r_2,x3) = g(2,3,x_3) \label{eval_s2_x2123}\tag{7}\\
    s_3(x_3) = 16 + 5x_3
\end{gather}


- Prover sends polynomial \(s_3(x_2)\) to the verifier
- Verifier checks that \(s_3(0) + s_3(1)\) = \(s_2(r_2)\)
- Verifier draws a random variable \(r_3\) from \(\mathbb{F}\) and sends it to P (let's assume \(r_3 = 6\))

### Final round

- Verifier has to check that \(s3(r_3) = g(r_1,r_2,r_3) = g(2,3,6)\) 

This is the power of the sumcheck protocol - the verifier is able to evaluate \(g\) at \(r_1,r_2,r_3\) via a single oracle query, much more efficient than calculating sums on every round for \(s_1\),\(s_2\) and so forth.

### Code implementation

I did a [simple implementation](https://github.com/gabrielfior/baby-sumcheck/blob/main/sum-check_sympy.py) in Python of the sumcheck protocol leveraging the [sympy](https://www.sympy.org/en/index.html) library for polynomial implementation. The code is reproduced below.

```python
# Initialization
x1, x2, x3 = symbols("x1 x2 x3")
poly = Poly(2*x1**3 + x1*x3 + x2*x3)
idx_to_vars = {0: x3, 1: x2, 2: x1}
p = Prover(poly, idx_to_vars)
v = Verifier()
num_of_rounds = len(poly.free_symbols)

### Round 1
# Prover (P) sends value of sum to verifier (V)
total_sum = p.calculate_sum(3)
random_values = {}
vars_to_iterate = num_of_rounds-1
prev_s = total_sum
random_values_store = [2,3,6]
random_value = 0
for round_idx in range(3):
    s = p.calculate_sum(vars_to_iterate, random_values)
    # Verifier checks that s1(0) + s1(1) = 12
    v.verify_univariate_poly(s, prev_s, random_value)
    random_value = random_values_store.pop(0)
    # We fetch the random_value idx in desc order
    x_variable = idx_to_vars[num_of_rounds-round_idx-1]
    random_values[x_variable] = random_value
    prev_s = s
    vars_to_iterate -= 1

# Final round, verifies executes oracle query
result_from_oracle = Oracle().evaluate_polynomial(poly, random_values)
# Verifier checks that s3(6) = g(r1,r2,r3)
v.assert_poly_matches_external_query(s, random_value, result_from_oracle)
```


### Conclusion

In this post, I tried to explain the sumcheck protocol with an example from Thaler's [manuscript](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.html), so that one can easier grasp the main concepts.

I also recommend this [article](https://blog.lambdaclass.com/have-you-checked-your-sums/) from Lambdaclass which discussed Sumcheck in greater depth.

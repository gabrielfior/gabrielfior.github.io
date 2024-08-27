---
title: Math for Prediction Markets
description: Simple math for back-of-the-envelope calculations related to Prediction Markets
date: 2024-08-27
tags:
    - prediction-markets
---

# Napkin math for Binary Prediction Markets

We want to prove 2 useful equations for calculating the potential payoff for a binary prediction market powered by an automated market maker:

For a market with equal odds:

(1) $$ t_y = 2 * bet_{amount}$$

For any market:

(2) $$t_y = \frac{bet_{amount}}{1 - p_y}$$


where 

| Variable    | Explanation |
| -------- | ------- |
| $bet_{amount}$ | Amount placed in the bet transaction |
| $t_y$  | Number of YES outcome tokens obtained by wagering $bet_{amount}$     |
| $p_y$ | Probability that the market resolves to YES   |
| $p_n$ | Probability that the market resolves to NO   |

## Introduction

Let's start with a [Prediction Market](https://docs.gnosis.io/conditionaltokens/docs/introduction3/) that has equal odds ($p_y = p_n = 0.5$) and adheres by the Constant Product Market Maker (CPMM) mechanism (similar to [Uniswap v2](https://uniswap.org/whitepaper.pdf)).

We highlight the following parallels between CPMM Prediction markets and constant-product AMMs:

|Concept| AMM| CPMM| Invariant is conserved?
| -------- | ------- | -------- | -------- |
| Add liquidity | The tokens A and B of the AMM are deposited and, in exchange, the depositor receives LP tokens | The market creator makes a single-sided deposit of N collateral tokens and 2N outcome tokens are created, N YES and N NO, and are kept in the market's reserves, while shares are minted that belong to the creator (ERC-1155 pattern)| No
| Trade | User can swap token A for token B and vice-versa. Invariant is conserved. | User can place bets on outcome YES or NO. User makes a single-sided deposit of collateral tokens and a transfer of a corresponding amount of YES or NO tokens is made from the market's contract to the better's address. Invariant is conserved. | Yes
| Remove liquidity | LP tokens are burned by the sender, leading to the tokens A and B stored in the pool to be sent to the sender. | Similarly to the AMM case, shares are burnt and the proportional amount of collateral tokens from the market's reserves is transferred to the sender. Invariant is not conserved. | No

As an example (based on [Gnosis' tutorial](https://docs.gnosis.io/conditionaltokens/docs/introduction3/#an-example-with-cpmm)), we illustrate a situation where a market is created and someone wants to place a bet on said market.

- Digi creates a new CPMM market with 10 xDAI - this creates 10 YES tokens and 10 NO tokens, which are added to the market's reserves
    - Note that n xDAI will always be converted in n YES tokens and n NO tokens on binary markets - this is a feature of prediction markets.
    - Note also that the CPMM invariant ($x * y = k$) is set to $100$ by the creator (product of the 2 outcome tokens in the reserves)
- Alice wants to bet 10 xDAI on the YES outcome, i.e. that it will resolve to YES. The individual steps are:
    - She adds 10 xDAI to the market, which mints 10 Yes tokens and 10 NO tokens, adding this to the market's reserves
    - The new CPMM invariant is $20 * 20 = 400$, which is broken.
    - By sending 15 YES tokens to Alice, the invariant is restored, since $5 * 20 = 100$
    - We will see below how to calculate that the number of YES outcome tokens Alice was entitled to was indeed 15.


## Special case

We consider a special case where the market odds are aligned, i.e. the YES outcome is exactly as likely (from the market's perspective) as the NO outcome.

The deduction starts immediately before Alice's trade and is based on the following observations:
- The market was balanced, i.e. the number of YES- and NO outcome tokens was the same ($s$)
- By placing trades, the CPMM invariant ($k$) must remain constant
- Alice wants to trade $b$ xDAI units in exchange for $t$ YES outcome tokens

By conservation of the invariant $k$:

$$ (s  + b - t)(s + b) = s^2$$
$$ (s + b) - \frac{s^2}{s+b} = t $$
$$ \frac{(s^2 + 2sb + b^2) - s^2}{s+b} = t $$
$$ \frac{2sb + b^2}{s+b} = t $$

Now we observe that $b << s$, i.e. for large enough markets, the market's reserves are much larger than the individual bet amounts. In that case, we can simplify this further:

$$ t = \frac{2sb}{s} $$
$$ t = 2b $$

Thus we have proved (1).

## General case

Now we want to generalize the individual case for the case where the market is not balanced, i.e. where the odds for each outcome are different. To illustrate this difference, we introduce $p_y$ and $p_n$ to reflect the market's odds.

We also note that:

$$ p_y = 1 - p_n$$
$$ p_n = \frac{n_0}{n_0 + y_0}$$

where $n_0$ is the amount of NO outcome tokens in the market's reserves and similarly $y_0$ the amount of YES outcome tokens.

Again we start with the conservation of the CPMM invariant:

$$(y_0 + b - t)(n_0 + b) = y_0  n_0 $$
$$(y_0 + b - t) = \frac{y_0 n_0}{n_0 + b} $$
$$ y_0 + b - \frac{y_0 n_0}{n_0 + b} = t $$
$$ \frac{(y_0 + b)(n_0+b) - y_0 n_0}{n_0 + b} = t $$
$$ \frac{(y_0 + n_0)b}{n_0+b} + \frac{b^2}{n_0+b} = t $$

We use the same trick as above and assume that $b << n_0$. This leads to the 2nd term being deemed negligible.
Furthermore, we use the fact that $y_0 + n_0 = \frac{n_0}{p_n}$, so that the expression becomes:

$$ \frac{(y_0 + n_0)b}{n_0+b} = t $$
$$ \frac{n_0}{p_n}\frac{b}{n_0+b} = t $$

We again make use of the fact that $b << n_0$ and simplify the 2nd term in the denominator, leading us to the final result:

$$ \frac{n_0}{p_n}\frac{b}{n_0} = t $$
$$ \frac{b}{p_n} = t $$
$$ t = \frac{b}{(1 - p_y)} $$

Observe that (2) is the generalized version of (1) for binary prediction markets with arbitrary odds.

## Conclusion

We hope the calculations above, albeit simple, can be helpful to other people looking to play around with Prediction Markets, helping with simple back-of-the-envelope calculations related to outcome tokens received after placing bets.
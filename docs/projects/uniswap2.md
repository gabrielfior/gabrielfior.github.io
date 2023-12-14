---
title: Uniswap LP positions as TradFi instruments - (2/2)
description: 
time: 2022-09-28
tags:
  - defi
---

## Synthetic Options using Uniswap v3

Reference article -> [link](https://lambert-guillaume.medium.com/synthetic-options-and-short-calls-in-uniswap-v3-a3aea5e4e273)

In this article, the main discussion revolves around mathematical finance and the put-call parity. Please check back the article or this [other article](https://www.investopedia.com/terms/p/putcallparity.asp) for more details, but the basic idea is that there is symmetry between options having the same strike price and expiration date, so that one can combine options with shorting/longing stocks to produce different payoffs. See the image below for an interesting way to create an exposure that is equivalent to a long stock (also known as “synthetic long stock”):

![image1](https://bafkreicgf3q7wcee55ji3mhgpeazbi25uat6wbx5rg4qzi46pin527r4wi.ipfs.nftstorage.link/)

Having asserted put-call parity, the next step was to illustrate other interesting options trading strategies, like [strangles](https://www.investopedia.com/terms/s/strangle.asp) and [straddles](https://www.investopedia.com/terms/s/straddle.asp). It’s straightforward to understand those constructs if you simply add the payoffs of each instrument together.

It’s however quite interesting to see that, by simply leveraging Uniswap v3 LP positions (which are actually NFTs) and long/short tokens, one is able to use complex option trading strategies, used extensively in TradFi.
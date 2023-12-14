---
title: Uniswap LP positions as TradFi instruments
description: (1/2)
time: 2022-09-27
tags:
  - defi
---


While preparing for the Kernel DeFi Guild fireside chat with [Guillaume Lambert](https://twitter.com/guil_lambert), I compiled a few notes which might be useful to other people just getting started with understanding LP positions as traditional financial instruments.

My goal is to boil down the excellent description from Guillaume into very simple terms, risking simplifying too much and being at times not precise enough. Because of this, the link to the original post is also available.

For feedback/suggestions/critics, feel free to reach out on [Twitter](https://twitter.com/TheGabrielFior).

## Uniswap V3 liquidity deposits

Reference article -> [link](https://lambert-guillaume.medium.com/uniswap-v3-lp-tokens-as-perpetual-put-and-call-options-5b66219db827)

The Uniswap V3 [whitepaper](https://uniswap.org/whitepaper-v3.pdf) introduces many innovations, none greater than the concept of concentrated liquidity. That means that investors are able to provide liquidity to an AMM pool within a given range instead of the full range.

Some important concepts to highlight:

- Range orders: let’s assume you bought 1 unit of the SPY (ETF that follows the S&P 500) for $100 (current spot price) and immediately wants to sell that same unit for $150 (target price). Your payoff looks roughly like this:

![image1](https://bafkreibhu2lsoriyurmi5hainhhfv2d4benuzmmqjjdxgedqapqak7t4ty.ipfs.nftstorage.link/)


We can see that, by holding the SPY unit, for spot prices lower than the target price, the payoff is linear (going from -$100 if spot price goes to 0 to $50 if the spot price goes to $150). For spot prices higher than the target price, your payoff remains constant and equal to $50, simply because you were “satisfied” with the $150 target price.

This means that, before the target price was reached, your unit of SPY provided a linear payoff, as it was directly proportional to the spot price. After the target price was reached, your unit of SPY was converted to USD, therefore your payoff remains flat.

Exactly the same logic can be applied to liquidity provisiong in Uniswap V3.

![image2](https://bafkreibwsqyq6sj3tzpnrwdktjoo5s2z5w75pdu6jde2qffegla7iyu6sq.ipfs.nftstorage.link/)

From Guillaume's paper.\

Analogous to the SPY range order, when adding liquidity to the ETH-DAI pool, the target price is the upper bound of the liquidity range chosen when depositing. Once the upper bound is crossed, the payoff remains flat as the ETH was converted to DAI.

- The payoff displayed for the range order & liquidity deposit is equivalent to the [covered-call payoff](https://www.investopedia.com/terms/c/coveredcall.asp). The idea is that you hold an asset (e.g. SPY or ETH) and at the same time you sell an option for a given strike price (e.g. $150). If the strike price is never reached, you collect a premium plus any increase/decrease in price experienced by the underlying. However, if the target price was reached, the option is exercised (by the buyer) and you only receive the premium as payoff.

![image3](https://bafkreieyjrlycm4xknsrvc74nbnp5bggmn2q56vkd5ijyj4z7dc7nvblgq.ipfs.nftstorage.link/)

- Specifically in the Uniswap V3 case (as detailled in the article), fees are accrued every time the spot price enters the liquidity range specified when depositing, therefore the premium of the covered call chart is translated into fees accrued whenever a swap involving the AMM pool happens when the spot price lies within the range you provided when making the deposit.
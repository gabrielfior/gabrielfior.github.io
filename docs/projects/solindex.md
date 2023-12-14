---
title: Solindex 
description: Solana Volatility Index
date: 2022-06-07
tags:
    - defi
    - solana
---


[Github link](https://github.com/gabrielfior/solana-volatility-index)

Project submitted to the Solana Summercamp hackathon 2022 and involved options data fetching from DeFi protocol (Zeta.markets), financial data processing (Black-Scholes for calculating implied volatility given option contract characteristics, like expiry and current price) and a dashboard for visualizing the index temporal evolution.

Main conclusions:

- Implied Volatility (IV) is a major factor when pricing options and in general for assessing market conditions.
- The VIX index, which tracks the S&P 500 in TradFi, has very wide usage, specially for hedging purposes.
- The first step is to monitor/calculate the implied volatility. Next steps could potentially involve deploying a vAMM (having the calculated IV as oracle) for providing liquidity, and after that exploring financial derivatives on top of the index, such as options or perps.
- This idea is to my knowledge quite new on Solana. We were inspired by an interesting project in this space, Volmex, currently available for the EVM.
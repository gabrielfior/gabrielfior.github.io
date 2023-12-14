---
title: Sudoswap explorer
description: Uncovering trading activity on sudoswap
date: 2022-08-28
tags:
    - defi
    - nft
---

[Github link](https://github.com/gabrielfior/sudoswap-data-visualization)


I recently submitted a project for the [Macro Hacks](https://macrohacks.notion.site/Macro-Hackathons-fa3eabf382cc4000bd4305f3cb23f88e) NFT Financialization hackathon (github repo, app) and secured a [prize](https://medium.com/macrohacks/macro-hackathons-nft-finance-edition-recap-winner-showcase-8fd3893b206e) from NFTBank.

![image6](https://bafkreigsll5ehzrogwh2wna7c6ek33onaz5gyjzcueyqrxsvhk5pz6nqoe.ipfs.nftstorage.link/)

Main findings:

- [Sudoswap](https://sudoswap.xyz/#/) has been gaining lots of traction since their launch in 2022, reaching 120k NFTs traded as well as 28M USD in total volume.
- Although there is already an excellent [dashboard](https://dune.com/0xRob/sudoamm) available on Dune for sudoswap, a few indicators (like wash trading and additional meta data from NFT collections, such as estimated vs floor prices time dependency and past volumes) were not yet available.
- Having this additional data could help NFT collectors/traders make more informed decisions.
- I gathered sudoswap trading data (minting data was also available but not used for this use-case) from Flipside Crypto using their [SDK](https://docs.flipsidecrypto.com/flipside-api/shroomdk-migration-guide). A data fetching job was packaged as a cronjob to continously fetch data and store in a DB instance in the cloud.
- Additional metadata for each NFT collection was gathered using [NFTBank API](https://docs.nftbank.ai/), including price estimates for a large number of NFT projects.
- I used a rather simple methodology for defining “wash trading” activity: if wallet A sold an item from an NFT collection to wallet B and, within 1 hour (this was set rather arbitrarily) wallet B sells the NFT back to wallet A, then this is considered 1 wash trade. This is of course a simplistic approach, since 1) there might be valid reasons for such a transaction to be occur and 2) more ellaborate approaches, such as having transactions of the form A -> B, B -> C and finally C -> A would not be labelled as “wash trades” in the current methodology.
- Next steps could potentially include gathering additional data from NFTBank for generating more insights of each collection, and also expanding the wash trading methodology (as described in the previous item).
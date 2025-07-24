# Uncovering Market Insights with Correlated Markets

In the world of prediction markets, gaining an edge requires making sense of a vast landscape of questions. Our Prediction Market Agent achieves this by analyzing related markets to gain context and identify profitable opportunities.

Our tooling employs two distinct but complementary approaches to this challenge: the `ThinkThoroughlyAgent` uses [Pinecone's](https://pinecone.io) semantic search to find topically similar questions, while the `DeployableArbitrageAgent` uses a language model to find logically correlated markets and exploit price discrepancies.

## `ThinkThoroughlyAgent`: Finding Semantic Similarity with Pinecone

Often, the answer to one prediction market question can be informed by the outcomes of other, similar questions. To tackle this, we've integrated [Pinecone](https://www.pinecone.io/), a vector database designed for high-performance similarity search, into our `ThinkThoroughlyAgent`. This allows our agent to quickly find and analyze markets with semantically similar questions, leading to more accurate predictions. Also worth noting that other alternatives for vector databases, like [Chroma](https://www.trychroma.com/), could also have been used - we went with Pinecone for simplicity and ease-of-integration with [Langchain](https://python.langchain.com/docs/introduction/).

### Our Integration: A Technical Look

The `ThinkThoroughlyAgent` is designed for deep analysis. 
1. Before it processes new markets, it ensures its knowledge base of past markets is up-to-date. The before_process_markets method in the agent's deployment configuration calls a handler to manage this process (highlighted line).

```py linenums="1" hl_lines="7"
class DeployableThinkThoroughlyAgentBase(DeployableTraderAgent):
    agent_class: type[ThinkThoroughlyBase]
    
    ### hidden code

    def before_process_markets(self, market_type: MarketType) -> None:
        self.agent.pinecone_handler.insert_all_omen_markets_if_not_exists()
        super().before_process_markets(market_type=market_type)
```
[Source code](https://github.com/gnosis/prediction-market-agent/blob/main/prediction_market_agent/agents/think_thoroughly_agent/deploy.py#L32)

This command triggers our custom [`pinecone_handler`](https://github.com/gnosis/prediction-market-agent/blob/main/prediction_market_agent/db/pinecone_handler.py#L28) to fetch markets from [Omen](https://presagio.pages.dev/) (via Subgraph data fetching), a popular prediction market platform, and store them as vector embeddings in our Pinecone index (using the Pinecone Python SDK).

To ensure our data is clean and consistent, we use the `PineconeMetadata` Pydantic model to structure the information before creating the embedding.

``` py linenums="1" hl_lines="2"
class PineconeMetadata(BaseModel):
    question_title: str
    market_address: HexAddress
    close_time_timestamp: int

    @staticmethod
    def from_omen_market(market: OmenMarket) -> "PineconeMetadata":
        return PineconeMetadata(
            question_title=market.question_title,
            market_address=market.id,
            close_time_timestamp=int(market.close_time.timestamp()),
        )
```

[Source code](https://github.com/gnosis/prediction-market-agent/blob/2841708c51514c40a6a1f49fc4ff91e74e374633/prediction_market_agent/agents/think_thoroughly_agent/models.py#L7)

By creating a vector from the `question_title`, we can use Pinecone's similarity search to find markets that are semantically related, even if they don't share exact keywords. Those markets are then finally inserted into the agent's context, expanding its understanding of the current market situation. Additional context could be obtained by including metadata about the market, such as `description`, but unfortunately this metadata is not always available nor consistent.

## `DeployableArbitrageAgent`: Exploiting Price Discrepancies

While the ThinkThoroughlyAgent seeks informational advantages, our `DeployableArbitrageAgent` has a more direct goal: to find and exploit risk-free profit opportunities. This agent specializes in identifying pairs of markets whose outcomes are highly correlated but whose prices have temporarily diverged.

This is classic arbitrage. For example, if two markets are asking the same question in a slightly different way, their probabilities should be nearly identical. If they're not, the agent can buy the underpriced outcome in one market and the underpriced outcome in the other (which would be the opposite bet) to lock in a guaranteed profit, regardless of the final result.

### How It Works

The DeployableArbitrageAgent uses a large language model (LLM) to assess the relationship between two market questions. It uses a specific, structured prompt to determine if the markets are correlated.

```py

PROMPT_TEMPLATE = """Given two markets, MARKET 1 and MARKET 2, provide a boolean
 value that  represents the correlation between these two markets' outcomes. 
 Return True if the outcomes are perfectly or nearly perfectly correlated,
  meaning there is a high probability that both markets resolve to the same
  outcome. Return False if the outcomes are perfectly or nearly perfectly 
  inversely correlated, and finally return None if the correlation is weak or 
  non-existent.

Correlation can also be understood as the conditional probability 
that market 2 resolves to YES, given that market 1 resolved to YES.

In addition to the correlation value, explain the reasoning behind your decision.

[MARKET 1]
{main_market_question}

[MARKET 2]
{related_market_question}

Follow the formatting instructions below for producing an output
 in the correct format.
{format_instructions}"""
```

[Source code](https://github.com/gnosis/prediction-market-agent/blob/main/prediction_market_agent/agents/arbitrage_agent/prompt.py)

If the LLM confirms a high correlation, the agent uses the logic defined in the 
[`CorrelatedMarketPair`](https://github.com/gnosis/prediction-market-agent/blob/main/prediction_market_agent/agents/arbitrage_agent/data_models.py#L24) data model to act. This model includes methods to:

- Calculate the potential profit from the price difference between the two markets.
- Determine the optimal betting directions (e.g., bet YES on market 1 and NO on market 2).
- Calculate the exact bet amounts for each market to guarantee a profit by balancing the positions according to their respective probabilities.

## Conclusion: Two Paths to a Smarter Agent

These two agents demonstrate a powerful dual-strategy approach. The `ThinkThoroughlyAgent` uses Pinecone's semantic search for an informational advantage, enriching its context to make better-educated predictions. In contrast, the `DeployableArbitrageAgent` uses an LLM to find logically correlated markets for a financial advantage, placing calculated bets across a pair of markets to exploit price inefficiencies.

Together, these approaches create a more robust and versatile prediction market platform, showcasing how different AI techniques can be combined to master the complexities of forecasting.

## Explore Our Work

We invite you to explore the open-source repositories behind these agents and share your feedback. You can also view our agents' performance on the Gnosis network.

Repositories:

- [Prediction Market Agent Tooling](https://github.com/gnosis/prediction-market-agent-tooling)
- [Prediction Market Agents](https://github.com/gnosis/prediction-market-agent)

Performance Dashboards:

- [AI Agents Overview on Dune](https://dune.com/gnosischain_team/ai-agents-overview-omen-prediction-markets)
- [Gnosis AI Team Dune Dashboards](https://dune.com/gnosischain_team/gnosis-labs)
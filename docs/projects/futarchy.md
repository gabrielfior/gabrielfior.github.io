---
title: Futarchy for Budget Earmarks
description: How prediction markets could allocate parliamentary budget earmarks by social impact
date: 2026-07-23
tags:
    - prediction-markets
    - futarchy
    - governance
    - research
---

=== "English"

    # Futarchy: Allocating Budget Earmarks by Social Impact

    > **tl;dr:** Brazil's federal government commits roughly R\$25 billion a year through *emendas parlamentares* — discretionary budget earmarks that legislators direct to specific recipients. In practice, much of it is captured or misdirected. This proposes a platform where earmark proposals compete in a prediction market that estimates which one produces the greatest social impact, following Robin Hanson's futarchy principle: **vote on values, bet on beliefs.**

    ---

    ## The problem

    The purpose of parliamentary budget amendments (*emendas parlamentares*) is to let politicians at every level — municipal, state, federal — steer executive-branch funds toward specific destinations (NGOs, hospitals, and so on), on the theory that they understand their local base and its needs better than the center does.

    In practice, this is not what happens. Earmarks have increasingly been used illicitly — directed by third parties, siphoned off, or deployed for electoral gain.

    ## The solution

    A platform where earmark proposals are registered and a prediction market estimates which of them produces the greatest positive impact for society. The idea follows Robin Hanson's principle of *futarchy* — **"vote on values, but bet on beliefs"**: society democratically defines *the metric* to maximize (the value), while the market decides *which proposal* best achieves it (the belief). You don't vote for an earmark — you bet on it.

    As a toy example, assume an administrative body can approve 1 earmark, worth 200,000 reais, and that we have settled on a metric to maximize (a Gini inequality index, the HDI, IDEB — Brazil's basic-education development index — etc.).

    Suppose we have 3 education earmark proposals, all for Hogwarts:

    1. 200,000 reais to buy racing broomsticks
    2. 200,000 reais to buy pumpkin juice for breakfast
    3. 200,000 reais to plant 100 Whomping Willow saplings

    The solution is to create a prediction market for the decision of which earmark to fund (see the figure below) — for example, *"What will the IDEB be at the end of 2027 if proposal X is approved?"*

    <div style="font-family:system-ui,-apple-system,'Segoe UI',sans-serif;padding:12px;color:var(--md-typeset-color)">
    <svg viewBox="0 0 760 458" width="100%" style="max-width:760px;height:auto" role="img" aria-label="Line chart: expected 2027 IDEB for each proposal over time. The Whomping Willow saplings rise from ~5.8 to 6.42; pumpkin juice stays around 5.9; racing broomsticks fall to 5.48.">
      <style>
        .title { fill: var(--md-typeset-color); font-size: 15px; font-weight: 700; }
        .lbl { fill: var(--md-typeset-color); font-size: 11px; }
        .tick { fill: var(--md-default-fg-color--light); font-size: 10px; font-variant-numeric: tabular-nums; }
        .axtitle { fill: var(--md-default-fg-color--light); font-size: 11px; font-weight: 600; }
        .grid { stroke: var(--md-default-fg-color--lightest); stroke-width: 1; }
        .axis { stroke: var(--md-default-fg-color--lightest); stroke-width: 1; }
        .line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
        .l1 { stroke: #3b82f6; } .l2 { stroke: #f59e0b; } .l3 { stroke: #10b981; }
        .m1 { fill: #3b82f6; } .m2 { fill: #f59e0b; } .m3 { fill: #10b981; }
        .dot { stroke: var(--md-default-bg-color); stroke-width: 1.5; }
        .end { stroke: var(--md-default-bg-color); stroke-width: 2; }
      </style>
      <text class="title" x="70" y="28">Decision market: expected 2027 IDEB by proposal</text>
      <g transform="translate(70,46)">
        <line class="l1" x1="0" y1="0" x2="22" y2="0"/><circle class="m1" cx="11" cy="0" r="3.5"/><text class="lbl" x="28" y="4">Racing broomsticks</text>
        <line class="l2" x1="196" y1="0" x2="218" y2="0"/><circle class="m2" cx="207" cy="0" r="3.5"/><text class="lbl" x="224" y="4">Pumpkin juice</text>
        <line class="l3" x1="360" y1="0" x2="382" y2="0"/><circle class="m3" cx="371" cy="0" r="3.5"/><text class="lbl" x="388" y="4">Whomping Willow saplings</text>
      </g>
      <line class="grid" x1="70" y1="396" x2="600" y2="396"/><line class="grid" x1="70" y1="308" x2="600" y2="308"/><line class="grid" x1="70" y1="220" x2="600" y2="220"/><line class="grid" x1="70" y1="132" x2="600" y2="132"/>
      <text class="tick" x="60" y="400" text-anchor="end">5.4</text><text class="tick" x="60" y="312" text-anchor="end">5.7</text><text class="tick" x="60" y="224" text-anchor="end">6.0</text><text class="tick" x="60" y="136" text-anchor="end">6.3</text>
      <line class="axis" x1="70" y1="88" x2="70" y2="396"/><line class="axis" x1="70" y1="396" x2="600" y2="396"/>
      <text class="axtitle" x="24" y="242" text-anchor="middle" transform="rotate(-90 24 242)">Expected 2027 IDEB</text>
      <text class="axtitle" x="335" y="450" text-anchor="middle">Date (2026, dd/mm)</text>
      <text class="tick" x="70" y="412" text-anchor="middle">02/03</text><text class="tick" x="136.25" y="412" text-anchor="middle">09/03</text><text class="tick" x="202.5" y="412" text-anchor="middle">16/03</text><text class="tick" x="268.75" y="412" text-anchor="middle">23/03</text><text class="tick" x="335" y="412" text-anchor="middle">30/03</text><text class="tick" x="401.25" y="412" text-anchor="middle">06/04</text><text class="tick" x="467.5" y="412" text-anchor="middle">13/04</text><text class="tick" x="533.75" y="412" text-anchor="middle">20/04</text><text class="tick" x="600" y="412" text-anchor="middle">27/04</text>
      <polyline class="line l1" points="70,278.7 136.25,284.5 202.5,296.3 268.75,313.9 335,331.5 401.25,346.1 467.5,357.9 533.75,366.7 600,372.5"/>
      <polyline class="line l2" points="70,272.8 136.25,269.9 202.5,264.0 268.75,261.1 335,255.2 401.25,252.3 467.5,252.3 533.75,249.3 600,249.3"/>
      <polyline class="line l3" points="70,281.6 136.25,261.1 202.5,231.7 268.75,199.5 335,170.1 401.25,143.7 467.5,123.2 533.75,108.5 600,96.8"/>
      <g class="dot m1"><circle cx="70" cy="278.7" r="3.5"/><circle cx="136.25" cy="284.5" r="3.5"/><circle cx="202.5" cy="296.3" r="3.5"/><circle cx="268.75" cy="313.9" r="3.5"/><circle cx="335" cy="331.5" r="3.5"/><circle cx="401.25" cy="346.1" r="3.5"/><circle cx="467.5" cy="357.9" r="3.5"/><circle cx="533.75" cy="366.7" r="3.5"/></g>
      <g class="dot m2"><circle cx="70" cy="272.8" r="3.5"/><circle cx="136.25" cy="269.9" r="3.5"/><circle cx="202.5" cy="264.0" r="3.5"/><circle cx="268.75" cy="261.1" r="3.5"/><circle cx="335" cy="255.2" r="3.5"/><circle cx="401.25" cy="252.3" r="3.5"/><circle cx="467.5" cy="252.3" r="3.5"/><circle cx="533.75" cy="249.3" r="3.5"/></g>
      <g class="dot m3"><circle cx="70" cy="281.6" r="3.5"/><circle cx="136.25" cy="261.1" r="3.5"/><circle cx="202.5" cy="231.7" r="3.5"/><circle cx="268.75" cy="199.5" r="3.5"/><circle cx="335" cy="170.1" r="3.5"/><circle cx="401.25" cy="143.7" r="3.5"/><circle cx="467.5" cy="123.2" r="3.5"/><circle cx="533.75" cy="108.5" r="3.5"/></g>
      <circle class="end m1" cx="600" cy="372.5" r="5"/><circle class="end m2" cx="600" cy="249.3" r="5"/><circle class="end m3" cx="600" cy="96.8" r="5"/>
      <text class="lbl" x="612" y="100" font-weight="600">Willow <tspan class="tick">6.42</tspan></text>
      <text class="lbl" x="612" y="253" font-weight="600">Juice <tspan class="tick">5.90</tspan></text>
      <text class="lbl" x="612" y="376" font-weight="600">Brooms <tspan class="tick">5.48</tspan></text>
    </svg>
    </div>

    *Each line is a conditional market — "What will the IDEB be at the end of 2027 if proposal X is approved?". The price of each contract is the IDEB the market expects under that proposal; as trading proceeds the estimates diverge, and the proposal with the highest expected IDEB (here, the Whomping Willow saplings, 6.42) is the one chosen.*

    ## Participants

    There are several ways to let informed people take part in this market. The main goal is to encourage people with deep knowledge of the metric being optimized and of the context (in this imaginary case, Hogwarts) to participate.

    One option would be to allow buying and selling backed by real currency. This makes for real monetary incentives, but it opens regulatory loopholes and lets agents with more capital manipulate the market.

    A more interesting option is to use play money — fictional, non-transferable — the [Manifold Markets](https://manifold.markets) model. There are indications that fictional currency produces forecasts of quality comparable to real money, without the regulatory and capital-manipulation risks.

    Finally, the proposed distribution of capital per participant is:

    - Every Brazilian holding a CPF (taxpayer ID) can trade in the markets
    - Each participant starts with 100 galleons
    - By taking part in the platform and contributing positively to earmark decisions (i.e. betting correctly), a participant grows their galleon balance and, with it, their power to influence decisions

    ## Market mechanics

    ### How each market works (a scalar market)

    Each proposal has its own conditional **scalar market** — "What will the IDEB be in 2027 if this proposal is approved?" The IDEB ranges from 0 to 10, so the market range is \([x_{min}, x_{max}] = [0, 10]\).

    - **Two tokens per market:** `UP` (a bet that the IDEB ends up *high*) and `DOWN` (a bet that it ends up *low*). Minting 1 galleon creates **1 UP + 1 DOWN**; on redemption, `UP + DOWN` always sum to 1 galleon.
    - **Price reflects expectation:** the price of the `UP` token multiplied by the range equals the IDEB the market expects.
    - **UP and DOWN always sum to 1:** since each minted pair costs 1 galleon and redeems for at most 1 galleon in total, the two prices are complementary — \(p_{UP} + p_{DOWN} = 1\). This constraint is what makes the market converge: whoever thinks the IDEB is **underestimated** buys `UP` (pushing \(p_{UP}\) up and therefore \(p_{DOWN}\) down); whoever thinks it is **overestimated** buys `DOWN`. The equilibrium — the price where buying pressure from both sides cancels out — is the market's IDEB estimate. E.g.: `UP` at 0.642 and `DOWN` at 0.358 ⇔ expected IDEB ≈ 6.42.
    - **Proportional redemption:** when the market resolves with the final IDEB \(V\), each token pays

    $$\text{UP} = \frac{V - x_{min}}{x_{max} - x_{min}} \qquad \text{DOWN} = \frac{x_{max} - V}{x_{max} - x_{min}}$$

    - **Decision + cancelled bets:** at the end of trading, only the market of the proposal with the highest expected IDEB (the winner) is *settled against the measured IDEB*. The markets for the other proposals are **cancelled** and the invested amounts are returned to participants.

    ### A worked return — Alice bets on the willows

    Alice believes the willows will beat the market estimate (6.42), so she buys the `UP` token of the "willows" market.

    1. She buys **100 UP** at 0.64 galleon each → she spends **64 galleons**.
    2. The "willows" proposal wins, is implemented, and at the end of 2027 the IDEB is measured (value V).
    3. Each `UP` redeems for \(V/10\) galleon. Two hypothetical scenarios:

    | Scenario | Measured IDEB (\(V\)) | Redemption per UP = \(V/10\) | Total (100 UP) | Return on the 64 galleons |
    |---|---|---|---|---|
    | Positive | 8 | 0.80 | 80 galleons | **+25%** |
    | Negative | 2 | 0.20 | 20 galleons | **−69%** |

    Since `UP + DOWN = 1`, whoever bought `DOWN` gets exactly the complement (0.20 in the positive scenario, 0.80 in the negative one) — the total redeemed never exceeds the collateral deposited.

    As the table shows, Alice's return depends on how close her bet landed to the realized IDEB: forecasting well is profitable, forecasting badly is costly. It is this reward gradient that leads each participant to reveal, through price, their best estimate of the metric — exactly the information the mechanism wants to extract.

    ## Impact

    According to the Tesouro Transparente portal (see references), the federal government commits roughly 25 billion reais a year in earmarks. States and municipalities also have parliamentary-amendment mechanisms — for reference, the state of São Paulo committed 1 billion in earmarks in 2025. So the size of the impact is on the order of tens (possibly hundreds) of billions of reais per year.

    ## References

    - Robin Hanson, *[Futarchy: Vote Values, But Bet Beliefs](https://mason.gmu.edu/~rhanson/futarchy.html)* — the futarchy manifesto: *"vote on values, but bet on beliefs"*. Representatives define the welfare metric; *decision markets* (conditional markets) estimate which policy maximizes it, and the one with the highest expected value becomes law.
    - [Manifold Markets](https://manifold.markets) — play-money prediction market platform.
    - Painel das Emendas Parlamentares, Tesouro Transparente (Federal Government) — <https://www.tesourotransparente.gov.br/consultas/painel-das-emendas-parlamentares-individuais-e-de-bancada>
    - Emendas Parlamentares, Portal da Transparência (Government of São Paulo) — <https://www.transparencia.sp.gov.br/home/emendasparlamentares>

=== "Português"

    # Futarquia: Alocando Emendas Parlamentares por Impacto Social

    > **tl;dr:** O governo federal brasileiro empenha cerca de R\$25 bilhões por ano em *emendas parlamentares* — repasses discricionários que parlamentares direcionam a destinos específicos. Na prática, boa parte é capturada ou desviada. Aqui proponho uma plataforma em que propostas de emenda competem num mercado preditivo que estima qual delas gera o maior impacto social, seguindo o princípio da futarquia de Robin Hanson: **vote nos valores, aposte nas crenças.**

    ---

    ## O problema

    Objetivo das emendas parlamentares é possibilitar a políticos de diversos níveis (municipal, estadual, federal) direcionar recursos do executivo para destinos específicos (ONGs, hospitais, etc.) por conhecerem melhor a ponta e suas bases eleitorais.

    Isso atualmente não ocorre — emendas em geral vêm sendo usadas de maneira ilícita (direcionadas por terceiros, desviadas, usadas por motivos eleitoreiros, etc.).

    ## Solução

    Plataforma onde propostas de emendas são cadastradas e um mercado preditivo estima qual delas gera maior impacto positivo para a sociedade. A ideia segue o princípio da *futarquia* de [Robin Hanson](https://mason.gmu.edu/~rhanson/futarchy.html) — **"vote nos valores, aposte nas crenças"**: a sociedade define democraticamente *a métrica* a maximizar (o valor), enquanto o mercado decide *qual proposta* melhor a atinge (a crença). Não se vota na emenda — aposta-se nela.

    A critério de exemplo, assumimos que um ente administrativo possui capacidade para aprovar 1 emenda, com valor individual de 200 mil reais. Além disso, assumimos uma métrica a ser maximizada (índice Gini de desigualdade, IDH, IDEB, etc.).

    Vamos supor que temos 3 propostas de emendas, todas da área da Educação para a escola de Hogwarts:

    1. 200 mil para comprar vassouras de corrida
    2. 200 mil para comprar suco de abóbora para o café da manhã
    3. 200 mil para plantar 100 mudas de Salgueiro Lutador

    A solução envolve criar um mercado preditivo para essa decisão de qual emenda alocar (vide figura abaixo), por exemplo, *"Qual será o IDEB ao final de 2027 se a proposta X for aprovada?"*.

    <div style="font-family:system-ui,-apple-system,'Segoe UI',sans-serif;padding:12px;color:var(--md-typeset-color)">
    <svg viewBox="0 0 760 458" width="100%" style="max-width:760px;height:auto" role="img" aria-label="Gráfico de linhas: IDEB de 2027 esperado por cada proposta ao longo do tempo. As mudas de Salgueiro Lutador sobem de ~5,8 para 6,42; o suco de abóbora fica em ~5,9; as vassouras de corrida caem para 5,48.">
      <style>
        .title { fill: var(--md-typeset-color); font-size: 15px; font-weight: 700; }
        .lbl { fill: var(--md-typeset-color); font-size: 11px; }
        .tick { fill: var(--md-default-fg-color--light); font-size: 10px; font-variant-numeric: tabular-nums; }
        .axtitle { fill: var(--md-default-fg-color--light); font-size: 11px; font-weight: 600; }
        .grid { stroke: var(--md-default-fg-color--lightest); stroke-width: 1; }
        .axis { stroke: var(--md-default-fg-color--lightest); stroke-width: 1; }
        .line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
        .l1 { stroke: #3b82f6; } .l2 { stroke: #f59e0b; } .l3 { stroke: #10b981; }
        .m1 { fill: #3b82f6; } .m2 { fill: #f59e0b; } .m3 { fill: #10b981; }
        .dot { stroke: var(--md-default-bg-color); stroke-width: 1.5; }
        .end { stroke: var(--md-default-bg-color); stroke-width: 2; }
      </style>
      <text class="title" x="70" y="28">Mercado de decisão: IDEB 2027 esperado por proposta</text>
      <g transform="translate(70,46)">
        <line class="l1" x1="0" y1="0" x2="22" y2="0"/><circle class="m1" cx="11" cy="0" r="3.5"/><text class="lbl" x="28" y="4">Vassouras de corrida</text>
        <line class="l2" x1="196" y1="0" x2="218" y2="0"/><circle class="m2" cx="207" cy="0" r="3.5"/><text class="lbl" x="224" y="4">Suco de abóbora</text>
        <line class="l3" x1="360" y1="0" x2="382" y2="0"/><circle class="m3" cx="371" cy="0" r="3.5"/><text class="lbl" x="388" y="4">Mudas de Salgueiro Lutador</text>
      </g>
      <line class="grid" x1="70" y1="396" x2="600" y2="396"/><line class="grid" x1="70" y1="308" x2="600" y2="308"/><line class="grid" x1="70" y1="220" x2="600" y2="220"/><line class="grid" x1="70" y1="132" x2="600" y2="132"/>
      <text class="tick" x="60" y="400" text-anchor="end">5,4</text><text class="tick" x="60" y="312" text-anchor="end">5,7</text><text class="tick" x="60" y="224" text-anchor="end">6,0</text><text class="tick" x="60" y="136" text-anchor="end">6,3</text>
      <line class="axis" x1="70" y1="88" x2="70" y2="396"/><line class="axis" x1="70" y1="396" x2="600" y2="396"/>
      <text class="axtitle" x="24" y="242" text-anchor="middle" transform="rotate(-90 24 242)">IDEB 2027 esperado</text>
      <text class="axtitle" x="335" y="450" text-anchor="middle">Data (2026)</text>
      <text class="tick" x="70" y="412" text-anchor="middle">02/03</text><text class="tick" x="136.25" y="412" text-anchor="middle">09/03</text><text class="tick" x="202.5" y="412" text-anchor="middle">16/03</text><text class="tick" x="268.75" y="412" text-anchor="middle">23/03</text><text class="tick" x="335" y="412" text-anchor="middle">30/03</text><text class="tick" x="401.25" y="412" text-anchor="middle">06/04</text><text class="tick" x="467.5" y="412" text-anchor="middle">13/04</text><text class="tick" x="533.75" y="412" text-anchor="middle">20/04</text><text class="tick" x="600" y="412" text-anchor="middle">27/04</text>
      <polyline class="line l1" points="70,278.7 136.25,284.5 202.5,296.3 268.75,313.9 335,331.5 401.25,346.1 467.5,357.9 533.75,366.7 600,372.5"/>
      <polyline class="line l2" points="70,272.8 136.25,269.9 202.5,264.0 268.75,261.1 335,255.2 401.25,252.3 467.5,252.3 533.75,249.3 600,249.3"/>
      <polyline class="line l3" points="70,281.6 136.25,261.1 202.5,231.7 268.75,199.5 335,170.1 401.25,143.7 467.5,123.2 533.75,108.5 600,96.8"/>
      <g class="dot m1"><circle cx="70" cy="278.7" r="3.5"/><circle cx="136.25" cy="284.5" r="3.5"/><circle cx="202.5" cy="296.3" r="3.5"/><circle cx="268.75" cy="313.9" r="3.5"/><circle cx="335" cy="331.5" r="3.5"/><circle cx="401.25" cy="346.1" r="3.5"/><circle cx="467.5" cy="357.9" r="3.5"/><circle cx="533.75" cy="366.7" r="3.5"/></g>
      <g class="dot m2"><circle cx="70" cy="272.8" r="3.5"/><circle cx="136.25" cy="269.9" r="3.5"/><circle cx="202.5" cy="264.0" r="3.5"/><circle cx="268.75" cy="261.1" r="3.5"/><circle cx="335" cy="255.2" r="3.5"/><circle cx="401.25" cy="252.3" r="3.5"/><circle cx="467.5" cy="252.3" r="3.5"/><circle cx="533.75" cy="249.3" r="3.5"/></g>
      <g class="dot m3"><circle cx="70" cy="281.6" r="3.5"/><circle cx="136.25" cy="261.1" r="3.5"/><circle cx="202.5" cy="231.7" r="3.5"/><circle cx="268.75" cy="199.5" r="3.5"/><circle cx="335" cy="170.1" r="3.5"/><circle cx="401.25" cy="143.7" r="3.5"/><circle cx="467.5" cy="123.2" r="3.5"/><circle cx="533.75" cy="108.5" r="3.5"/></g>
      <circle class="end m1" cx="600" cy="372.5" r="5"/><circle class="end m2" cx="600" cy="249.3" r="5"/><circle class="end m3" cx="600" cy="96.8" r="5"/>
      <text class="lbl" x="612" y="100" font-weight="600">Salgueiro <tspan class="tick">6,42</tspan></text>
      <text class="lbl" x="612" y="253" font-weight="600">Suco <tspan class="tick">5,90</tspan></text>
      <text class="lbl" x="612" y="376" font-weight="600">Vassouras <tspan class="tick">5,48</tspan></text>
    </svg>
    </div>

    *Cada linha é um mercado condicional — "Qual será o IDEB ao final de 2027 se a proposta X for aprovada?". O preço de cada contrato é o IDEB que o mercado espera sob aquela proposta; ao longo da negociação as estimativas divergem, e a proposta com maior IDEB esperado (aqui, as mudas de Salgueiro Lutador, 6,42) é a escolhida.*

    ## Participantes

    Existem múltiplas formas de permitir que pessoas informadas participem deste mercado. O principal objetivo é incentivar pessoas com conhecimento profundo da métrica a ser otimizada e do contexto (neste caso imaginário, Hogwarts) a participarem deste mercado.

    Uma opção seria permitir operações de compra e venda lastreadas em real. Isso facilita incentivos monetários reais, porém abre brechas regulatórias e permite que agentes com maior disponibilidade de capital manipulem o mercado.

    Uma opção mais interessante seria usar dinheiro fictício, não-transferível — o modelo do [Manifold Markets](https://manifold.markets). Há indícios de que moeda fictícia gera previsões de qualidade comparável à de dinheiro real, sem os riscos regulatórios e de manipulação por capital.

    Finalmente, a proposta de distribuição de capital por participante é:

    - Cada brasileiro portador de CPF pode negociar nos mercados
    - Cada participante começa com 100 galeões
    - Ao participar da plataforma e contribuir positivamente para a tomada de decisões das emendas (i.e. apostar corretamente), o participante aumenta seu saldo de galeões e com isso seu poder de influenciar decisões aumenta

    ## Mecânica do mercado

    ### Como cada mercado funciona (mercado escalar)

    Cada proposta tem seu próprio **mercado escalar** condicional — "Qual será o IDEB em 2027 se esta proposta for aprovada?" O IDEB varia de 0 a 10, então a faixa do mercado é \([x_{min}, x_{max}] = [0, 10]\).

    - **Dois tokens por mercado:** `UP` (aposta que o IDEB fica *alto*) e `DOWN` (aposta que fica *baixo*). Cunhar ("mint") 1 galeão cria **1 UP + 1 DOWN**; no resgate, `UP + DOWN` sempre somam 1 galeão.
    - **O preço reflete a expectativa:** o preço do token `UP` multiplicado pela faixa equivale ao IDEB esperado pelo mercado.
    - **UP e DOWN sempre somam 1:** como cada par cunhado custa 1 galeão e resgata no máximo 1 galeão no total, os dois preços são complementares — \(p_{UP} + p_{DOWN} = 1\). É essa amarra que faz o mercado convergir: quem acha o IDEB **subestimado** compra `UP` (empurrando \(p_{UP}\) para cima e, por consequência, \(p_{DOWN}\) para baixo); quem o acha **superestimado** compra `DOWN`. O equilíbrio — o preço onde a pressão de compra dos dois lados se anula — é a estimativa de IDEB do mercado. Ex.: `UP` a 0,642 e `DOWN` a 0,358 ⇔ IDEB esperado ≈ 6,42.
    - **Resgate proporcional:** ao resolver o mercado com o IDEB final \(V\), cada token paga

    $$\text{UP} = \frac{V - x_{min}}{x_{max} - x_{min}} \qquad \text{DOWN} = \frac{x_{max} - V}{x_{max} - x_{min}}$$

    - **Decisão + apostas canceladas:** ao final da negociação, só o mercado da proposta com maior IDEB esperado (a vencedora) é *liquidado contra o IDEB medido*. Os mercados das demais propostas são **cancelados** e o valor investido é devolvido aos participantes.

    ### Exemplo de retorno — Alice aposta nos salgueiros

    Alice acredita que os salgueiros vão superar a estimativa de mercado (6,42), então compra o token `UP` do mercado "salgueiros".

    1. Compra **100 UP** a 0,64 galeão cada → gasta **64 galeões**.
    2. A proposta "salgueiros" vence, é implementada, e ao final de 2027 o IDEB é medido (valor V).
    3. Cada `UP` resgata \(V/10\) galeão. Dois cenários hipotéticos:

    | Cenário | IDEB medido (\(V\)) | Resgate por UP = \(V/10\) | Total (100 UP) | Retorno sobre os 64 galeões |
    |---|---|---|---|---|
    | Positivo | 8 | 0,80 | 80 galeões | **+25%** |
    | Negativo | 2 | 0,20 | 20 galeões | **−69%** |

    Como `UP + DOWN = 1`, quem comprou `DOWN` recebe exatamente o complemento (0,20 no cenário positivo, 0,80 no negativo) — o total resgatado nunca excede o colateral depositado.

    Como mostra a tabela, o retorno de Alice depende de quão perto sua aposta ficou do IDEB realizado: prever bem é lucrativo, prever mal custa caro. É esse gradiente de recompensa que leva cada participante a revelar, via preço, sua melhor estimativa da métrica — exatamente a informação que o mecanismo quer extrair.

    ## Impacto

    De acordo com o portal Tesouro Transparente (vide referências), o governo federal empenha aproximadamente 25 bilhões de reais anualmente em emendas. Estados e municípios também possuem mecanismos de emendas parlamentares — para referência, o estado de São Paulo em 2025 empenhou 1 bilhão em emendas. Então o tamanho do impacto é da ordem de dezenas (possivelmente centenas) de bilhões de reais anualmente.

    ## Referências

    - Robin Hanson, *[Futarchy: Vote Values, But Bet Beliefs](https://mason.gmu.edu/~rhanson/futarchy.html)* — manifesto da futarquia: *"vote on values, but bet on beliefs"*. Representantes definem a métrica de bem-estar; *decision markets* (mercados condicionais) estimam qual política a maximiza, e a de maior valor esperado vira lei.
    - [Manifold Markets](https://manifold.markets) — plataforma de mercados preditivos com dinheiro fictício.
    - Painel das Emendas Parlamentares, Tesouro Transparente (Governo Federal) — <https://www.tesourotransparente.gov.br/consultas/painel-das-emendas-parlamentares-individuais-e-de-bancada>
    - Emendas Parlamentares, Portal da Transparência (Governo de São Paulo) — <https://www.transparencia.sp.gov.br/home/emendasparlamentares>

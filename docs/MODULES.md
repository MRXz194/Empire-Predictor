# Analytics Modules — Empire-Predictor

The system operates based on the integration of 7 independent analytical modules. Each module utilizes a distinct mathematical or statistical methodology to evaluate market trends.

---

## 1. Sequence Analyzer
Based on long-range sequence analysis with an Attention mechanism to prioritize high-volatility pivot points. This module requires 60 consecutive rounds for optimal reliability.

## 2. State-Space Module
Utilizes dense state-space variables to simulate outcome transitions. Highly effective for identifying complex recurring patterns in large datasets.

## 3. Temporal Fusion
Analyzes the intersection of temporal factors and round outcomes, identifying multi-horizon dependencies and cyclical trends.

## 4. Forecasting Engine (Hierarchical)
Combines forecasts across different frequency levels, from short-term micro-fluctuations to long-term macro-trends, providing a comprehensive view of the sequence state.

## 5. Markov Chain (Order-3)
Calculates probabilities based on the states of the last 3 rounds. This method is highly efficient for detecting basic "streak" or "alternating" patterns.

## 6. Statistical Engine
Calculates pure mathematical metrics:
- **Entropy**: Evaluates the predictability and randomness of recent rounds.
- **Deviation**: Monitors frequency shifts compared to the historical mean.
- **Streak Logic**: Analyzes the length and probability of consecutive outcomes.

## 7. Dynamic Learner
Automatically optimizes internal parameters based on live market outcomes in real-time. This module records encountered states and their performance to refine future estimations.

```mermaid
graph TD
    subgraph Data_In [Synchronized Stream]
        RAW[Last 60 Rounds] -->|Feature Extraction| FEAT[Numerical Feature Vector]
    end

    subgraph Logic_Core [7-Layer Processing]
        FEAT --> SA[Sequence Analyzer]
        FEAT --> SS[State-Space Module]
        FEAT --> TF[Temporal Fusion]
        FEAT --> NF[Neural Forecast]
        FEAT --> MC[Markov Chain]
        FEAT --> SE[Statistical Engine]
        FEAT --> DL[Dynamic Learner]
    end

    subgraph Synthesis [Ensemble Layer]
        SA & SS & TF & NF & MC & SE & DL -->|Probability Vectors| AGG[Weighted Result Aggregator]
        AGG -->|Confidence Calculation| CONF[Final Probability Output]
    end

    subgraph Strategy [Decision Execution]
        CONF -->|High Confidence| BET[Market Entry Action]
        CONF -->|High Volatility| SKIP[Market Wait / Skip]
    end
```

```mermaid
graph TD
    subgraph Signal_Aggregation [Consensus Layer]
        M1[Module 1 Vote] & M2[Module 2 Vote] & M3[Module 3 Vote] --> ACC[Accumulator]
        ACC -->|Unanimous| UC[Ultra-High Confidence]
        ACC -->|Majority| MC[Standard Confidence]
        ACC -->|Divergent| DC[Low Confidence / Skip]
    end

    subgraph Action_Mapping [Final Decision]
        UC -->|85%+| BET_U[High-Stakes Market Entry]
        MC -->|60-85%| BET_M[Standard Market Entry]
        DC -->|<60%| SKIP[Wait for Pattern Maturity]
    end
```

---

## Ensemble Mechanics
Final results are not derived from a single module but are a weighted aggregation from the entire stack. High-complexity sequence modules are given higher precedence during stable data phases, while statistical modules lead during periods of high volatility.

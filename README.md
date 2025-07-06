# Equity Factor Model

A systematic approach to equity factor modeling using S&P 1500 data with advanced dimensionality reduction techniques.

## 📊 Project Overview

This project develops a comprehensive equity factor model that predicts stock returns using fundamental and technical factors. The model employs Singular Value Decomposition (SVD) for dimensionality reduction and implements a long-short hedge portfolio strategy.

### Key Features
- **Universe**: S&P 1500 stocks (filtered to 272 stocks with complete data)
- **Time Period**: 1995-2019 (training: 1995-2014, testing: 2015-2019)
- **Methodology**: Factor selection, SVD optimization, and cross-sectional trading strategy
- **Performance**: 11.06% annualized return with 0.83 Sharpe ratio

## 🎯 Performance Results

### Strategy Performance (2015-2019)
| Metric | Value |
|--------|---------------|
| Annualized Return | **11.06%** |
| Sharpe Ratio | **0.83** |
| Alpha vs CAPM | **+5.36%** |
| Alpha vs FF3 | **+5.08%** |
| Information Ratio | **0.47** |

## 🔬 Methodology

### 1. Data Preparation
- **Universe Selection**: Filtered S&P 1500 to 272 stocks with complete data (1995-2019)
- **Data Cleansing**: Applied winsorization (1st-99th percentiles) and standardization
- **Factor Transformation**: Converted to percentile values (0-1 range)

### 2. Factor Selection Process

We developed a rigorous 4-step factor selection process:

| Selection Criteria | Description |
|-------------------|-------------|
| **Absolute t-statistic** | Cross-sectional regression significance |
| **Time-decayed t-statistic** | Exponential decay weighting (48-month half-life) |
| **Factor Health** | Recent vs. historical performance ratio |
| **Information Coefficient** | Spearman rank correlation with future returns |

### 3. Selected Factors

The top 10 factors based on composite scoring:

| Factor | Description | Expected Relationship |
|--------|-------------|----------------------|
| **Mean Reversion** | Short-term price corrections | Negative (contrarian) |
| **Trend Factor** | Blend of momentum and reversal patterns | Positive |
| **R&D/Market Cap** | Research intensity relative to market value | Mixed (quality dependent) |
| **Short Interest Ratio** | Short selling activity | Positive (squeeze effects) |
| **Price Range (120d)** | Medium-term volatility measure | Positive (momentum) |
| **Price Range (20d)** | Short-term volatility measure | Negative |
| **Accrual** | Earnings vs. cash flow difference | Negative |
| **Free Cash Flow/Price** | Cash generation efficiency | Positive |
| **Industry-Adjusted Volatility** | Peer-relative risk measure | Positive |
| **R&D/Sales** | Operational innovation efficiency | Negative (raw), Positive (SVD) |

### 4. Custom Factor Engineering

We developed four proprietary factors:

#### Mean Reversion Factor
- Captures short-term price reversions to historical averages
- Constructed as negative z-score of 20-60 day returns
- Based on empirical evidence from Poterba & Summers (1988)

#### Macro Uncertainty Factor
- Quantifies economic uncertainty from forecast error volatility
- Uses 1, 3, and 12-month economic indicator horizons
- Strong predictive power during recession periods

#### SEC Filing Sentiment
- Textual analysis of 10-K, 10-Q, and 8-K filings
- Includes negative intensity, fog score, and polarity ratio
- Early signals of fundamental changes and management tone

#### Insider Trading Factor
- Net insider purchases and purchase-to-sale ratios
- Based on SEC Form 4 filings
- Exploits informational asymmetry advantages

### 5. Singular Value Decomposition (SVD)

Applied SVD to create orthogonal factors and reduce noise:

#### Key SVD Insights
| SVD Factor | Beta | Primary Loadings | Interpretation |
|------------|------|------------------|----------------|
| **Factor 9** | 0.1378 | RD_P (-0.70), RD_SALE (+0.71), range_20 (+0.08) | Quality innovation vs. speculative R&D |
| **Factor 2** | 0.1061 | RD_P (+0.65), RD_SALE (+0.66), Accrual (+0.17), FCF_P (+0.13) | Profitable growth and disciplined innovation |
| **Factor 1** | 0.0317 | range_120 (+0.56), range_20 (+0.55), xret_indsize_std120 (+0.53) | Price dispersion as opportunity indicator |
| **Factor 7** | -0.0003 | SIR (-0.98) | Neutralized short squeeze effects |

## 📈 Trading Strategy

### Implementation
1. **Training Phase**: Estimate factor betas using time-series regression (1995-2014)
2. **Prediction**: Calculate expected returns using linear factor model
3. **Portfolio Construction**: 
   - Sort stocks into deciles by predicted returns
   - Long top decile, short bottom decile
   - Monthly rebalancing

### Mathematical Framework
```
Expected Return: r(i,t+1) = α₀ + Σ(βₖ × Fₖ,ᵢ,ₜ)
```
Where:
- r(i,t+1) = predicted return for stock i at time t+1
- α₀ = intercept (alpha)
- βₖ = factor k beta coefficient  
- Fₖ,ᵢ,ₜ = factor k value for stock i at time t


## 📊 Datasets

Email me at m9yang@uwaterloo.ca for the full dataset


## 📚 References

- Poterba, J. M., & Summers, L. H. (1988). Mean reversion in stock prices
- Jurado, K., Ludvigson, S. C., & Ng, S. (2015). Measuring Uncertainty
- Chan, L. K., Lakonishok, J., & Sougiannis, T. (1999). Stock Market Valuation of R&D
- Cohen, L., Malloy, C., & Pomorski, L. (2012). Decoding Inside Information
  
*... [full references in original report]*

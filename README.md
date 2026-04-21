# FinPulse | Financial Intelligence Dashboard

![Live Demo](https://img.shields.io/badge/Demo-Live-brightgreen)

FinPulse is an interactive Streamlit application that provides sophisticated market analytics, anomaly detection, array-level portfolio optimization, and sentiment analysis for retail traders. 

## What it does
- **Market Data Visualisation**: Computes high-fidelity indicator mapping over OHLCV assets natively.
- **Anomaly Detection**: Flags anomalous structural trading behavior accurately using Isolation Forest clustering.
- **Portfolio Optimisation**: Solves SLSQP efficiencies analytically yielding max-sharpe frontier allocations.
- **Sentiment Analysis**: Ingests multi-threaded Google News RSS indices parsing VADER polarity grades dynamically. 
- **Bulk Export Options**: Allows extraction pipelines safely dumping formatted dataset binaries.

## Quick Start
```bash
git clone https://github.com/MrinmoyBANIKYA/FinPulse.git
cd FinPulse
pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack
| Tier | Technology | Purpose |
|------|-----------|---------|
| **Frontend** | Streamlit | UI & Application Routing |
| **Data Access** | yFinance, newspaper3k | Market Data Extraction & HTML News Mining |
| **Data Processing** | Pandas, Numpy, NLTK | Time-Series Transformation & NLP Metrics |
| **Machine Learning**| scikit-learn, SciPy | Anomaly Matrices & Solver Calculations |
| **Visualisation** | Plotly | Dynamic Dark-Mode Interactive Charting |

## Architecture
```
              +--------------------------+
              |      Streamlit UI        |
              +-------------+------------+
                            |
   +------------------------+------------------------+
   |                        |                        |
+--v--+                  +--v--+                  +--v--+
| API |<--yFinance       | ML  |<--Isolation      | NLP |<--newspaper3k
|Fetch|                  |Model|   Forest         |News |
+-----+                  +-----+                  +-----+
                            |
                     +------v-------+
                     | Portfolio    |
                     | Optimization |
                     +--------------+
```

## Key Learnings
1. **Streamlit Component Lifecycle**: Leveraging `@st.cache_data` and session limiters significantly throttles mapping overhead improving iterative UI loading speed.
2. **SciPy Solvers**: Designing constraint environments structurally accelerates `SLSQP` bounding boundaries preventing divergence when scanning covariance returns.
3. **Data Normalization Guards**: Extrapolating non-finite feature metrics intelligently prevents downstream UI crashing handling unknown datasets.

## Streamlit Deploy Instructions
To securely deploy FinPulse out to the web using Streamlit Community Cloud smoothly:
1. **Push to GitHub**: Send up your finalized codebase to your repository.
   ```bash
   git push -u origin main
   ```
2. **Navigate** over to [share.streamlit.io](https://share.streamlit.io) on your browser.
3. **Connect** your authenticated GitHub account and select your FinPulse repository target.
4. **Set** the main execution file explicitly as `app.py`.
5. Click **Deploy**!

---
**Disclaimer**: *For research and educational modeling purposes only. This is not financial advice.*

**Contact**: Mrinmoy Banikya (mrinmoy@example.com)
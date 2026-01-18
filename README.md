# Sentiment-Aware Real Estate Intelligence System for Pune

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A sentiment-aware real estate intelligence platform that combines structured property data with locality-level buyer sentiment to predict demand and inform pricing, marketing, and launch decisions.

## 🎯 Overview

This system helps builders and brokers understand how market emotions influence real estate performance:

- **Builders** can assess locality sentiment before launching new projects
- **Brokers** can prioritize properties in localities with improving sentiment
- **Investors** can identify high-potential areas based on sentiment + fundamentals

## ✨ Key Features

### 📊 Sentiment Analysis
- **Price Perception**: How buyers perceive pricing in each locality
- **Infrastructure Satisfaction**: Sentiment about roads, metro, connectivity
- **Investment Confidence**: Market outlook and growth expectations
- **Buying Urgency**: How urgent buyers feel about purchasing

### 🤖 Demand Prediction
- XGBoost model combining property features with sentiment
- Cross-validated performance metrics
- Feature importance analysis
- Interactive demand predictor

### 📈 Actionable Insights
- Locality investment rankings
- Builder launch recommendations
- Broker priority listings
- Sentiment alerts

### 🗺️ Interactive Dashboard
- 3D geospatial heatmap with demand visualization
- Multi-dimensional sentiment radar charts
- Real-time demand prediction
- Affordability analysis

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd broker_sentiment
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm
```

### 3. Setup Data

**Option A: Kaggle Dataset** (Recommended)
- Download from: https://www.kaggle.com/datasets/rohanchatse/pune-house-prices
- Place `pune_house_prices.csv` in `data/` directory

**Option B: Use Scraped Data**
- See [SETUP.md](SETUP.md) for detailed instructions

### 4. Run Pipeline

```bash
# Full pipeline
python main.py

# With fresh sentiment data
python main.py --generate-data

# Quick test mode
python main.py --test-mode
```

This will generate sentiment data, train the model, and create all outputs.

### 5. Launch Dashboard

```bash
streamlit run dashboard.py
```

Open http://localhost:8501 in your browser.

## 📁 Project Structure

```
broker_sentiment/
├── dashboard.py            # Streamlit dashboard
├── main.py                 # Pipeline orchestrator
├── config.py               # Configuration
├── requirements.txt        # Dependencies
│
├── .streamlit/
│   └── config.toml         # Streamlit configuration
│
├── src/                    # Core modules
│   ├── data_loader.py
│   ├── sentiment_analyzer.py
│   ├── locality_aggregator.py
│   ├── feature_engineer.py
│   ├── demand_predictor.py
│   ├── insights_generator.py
│   └── geospatial.py
│
├── scripts/                # Utility scripts
│   ├── generate_sentiment_data.py
│   ├── enhance_property_data.py
│   └── consolidate_scraped_data.py
│
├── data/                   # Data files
├── outputs/                # Generated outputs
├── models/                 # Trained models
└── scrape/                 # Scraped data
```

## 🌐 Deployment

### Streamlit Cloud (Recommended)

Deploy for free on Streamlit Cloud:

1. Push your code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy!

**See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.**

### Other Platforms

- **Docker**: See Dockerfile example in [DEPLOYMENT.md](DEPLOYMENT.md)
- **Heroku**: Streamlit-compatible deployment guide included
- **AWS/Azure/GCP**: Standard Python app deployment

## 🔧 Configuration

### Environment Variables

Create a `.env` file (optional):

```bash
# Google Gemini API Key (optional - VADER fallback available)
GEMINI_API_KEY=your_api_key_here
```

### Customization

Edit `config.py` to customize:
- Pune localities list
- Sentiment keywords
- Model parameters
- Dashboard settings

See [SETUP.md](SETUP.md) for detailed configuration options.

## 📊 Dashboard Features

### 7 Interactive Tabs

1. **📊 Market Overview** - Sentiment profiles and recommendations
2. **🗺️ Map View** - 3D geospatial demand heatmap
3. **🏆 Investment Rankings** - Locality scoring and rankings
4. **💰 Affordability** - Income-based affordability analysis
5. **🔮 Demand Predictor** - Interactive property demand calculator
6. **⚠️ Alerts & Insights** - Market warnings and opportunities
7. **📈 Model Performance** - ML model metrics and feature importance

## 📈 Example Use Cases

### Builder Planning Project Launch
```
Locality: Hinjewadi
Investment Score: 72.5
Recommendation: Strong Buy
Insight: Despite traffic concerns, strong IT sector presence 
         and metro line sentiment drive positive outlook.
```

### Broker Prioritizing Listings
```
Top Priority Properties:
1. Baner 3BHK - Demand Score: 85.2 (High investment confidence)
2. Koregaon Park 2BHK - Demand Score: 78.4 (Premium locality sentiment)
```

## 🧪 Testing

```bash
# Run tests
pytest

# Test with sample data
python main.py --test-mode
```

## 📚 Documentation

- **[SETUP.md](SETUP.md)** - Data setup and pipeline execution
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Deployment instructions
- **[config.py](config.py)** - Configuration options

## 🔍 Troubleshooting

### Common Issues

**Pipeline errors**: See [SETUP.md](SETUP.md#troubleshooting)
**Deployment issues**: See [DEPLOYMENT.md](DEPLOYMENT.md#troubleshooting)
**Dashboard not loading**: Ensure you've run `python main.py` first

## 🛠️ Technology Stack

- **Python 3.8+**
- **Streamlit** - Dashboard framework
- **XGBoost** - Machine learning
- **NLTK + VADER** - Sentiment analysis
- **spaCy** - NLP processing
- **Plotly** - Visualizations
- **PyDeck** - 3D geospatial maps

## 📝 License

MIT License - See [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 🆘 Support

- **Documentation**: See docs above
- **Issues**: Create a GitHub issue
- **Discussions**: Use GitHub Discussions

## 🙏 Acknowledgments

- Property data from Kaggle: [Pune House Prices Dataset](https://www.kaggle.com/datasets/rohanchatse/pune-house-prices)
- Built with Streamlit
- Sentiment analysis powered by VADER and optionally Google Gemini

---

**Made with ❤️ for the Pune Real Estate Market**

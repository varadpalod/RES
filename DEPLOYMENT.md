# Deployment Guide

## 🚀 Quick Deployment to Streamlit Cloud

Streamlit Cloud is the recommended deployment platform for this dashboard - it's free, easy, and optimized for Streamlit apps.

### Prerequisites

- GitHub account
- Streamlit Cloud account (free at [share.streamlit.io](https://share.streamlit.io))
- Your code pushed to a GitHub repository

### Step-by-Step Deployment

#### 1. Prepare Your Repository

Ensure your repository has all required files:
```bash
✓ dashboard.py           # Main dashboard
✓ requirements.txt       # Dependencies (must include pydeck)
✓ .streamlit/config.toml # Streamlit configuration
✓ config.py              # Application config
✓ src/                   # Source modules
✓ data/                  # Data directory (can be empty for now)
```

#### 2. Push to GitHub

```bash
git add .
git commit -m "Prepare for deployment"
git push origin main
```

#### 3. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Select your repository and branch
4. Set **Main file path**: `dashboard.py`
5. Click **"Deploy"**

#### 4. Configure Secrets (Optional)

If you want to use Gemini AI features:

1. In your app dashboard, click **"⚙️ Settings"**
2. Navigate to **"Secrets"**
3. Add your API key:
   ```toml
   GEMINI_API_KEY = "your_actual_api_key_here"
   ```
4. Save

> **Note**: The dashboard works without Gemini - it will use VADER sentiment analysis as fallback.

#### 5. Prepare Data

Before the dashboard shows data, you need to run the pipeline:

**Option A: Run Locally Then Deploy**
```bash
# Run pipeline locally to generate outputs
python main.py

# Commit the outputs (temporarily remove from .gitignore)
git add outputs/
git commit -m "Add initial pipeline outputs"
git push
```

**Option B: Use Sample Data**
The dashboard will display a helpful message if no data exists, guiding users to run the pipeline.

### Post-Deployment Checklist

- [ ] Dashboard loads without errors
- [ ] All 7 tabs are accessible
- [ ] 3D map renders correctly
- [ ] No missing dependency errors
- [ ] Secrets are configured (if using AI)
- [ ] Mobile view works

---

## 🐳 Alternative: Docker Deployment

### Create Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run

```bash
docker build -t pune-real-estate-dashboard .
docker run -p 8501:8501 pune-real-estate-dashboard
```

---

## ☁️ Alternative: Heroku Deployment

### Create Required Files

**Procfile**:
```
web: streamlit run dashboard.py --server.port=$PORT --server.address=0.0.0.0
```

**setup.sh**:
```bash
mkdir -p ~/.streamlit/
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
```

### Deploy

```bash
heroku login
heroku create your-app-name
git push heroku main
```

---

## 🔧 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'pydeck'"

**Solution**: Ensure `pydeck>=0.8.0` is in `requirements.txt`

### Issue: "No data found" message on dashboard

**Solution**: Run the pipeline first:
```bash
python main.py
```

### Issue: Map not rendering

**Solutions**:
- Check browser console for errors
- Verify pydeck is installed
- Try clearing browser cache
- Check if localhost blocks WebGL

### Issue: Gemini API rate limits

**Solution**: The app automatically falls back to VADER sentiment analysis. No action needed.

### Issue: Large file size preventing git push

**Solution**: Ensure `.gitignore` excludes:
- `outputs/`
- `*.csv`
- `*.joblib`

---

## 🔐 Security Best Practices

1. **Never commit** `.env` or `.streamlit/secrets.toml`
2. **Always use** environment variables for API keys
3. **Rotate API keys** regularly
4. **Limit API key permissions** to minimum required

---

## 📊 Monitoring & Maintenance

### Streamlit Cloud

- View logs: App Settings → Logs
- View metrics: App Settings → Analytics
- Reboot app: App Settings → Reboot

### Local Development

Start dashboard locally:
```bash
streamlit run dashboard.py
```

Access at: http://localhost:8501

---

## 🆘 Getting Help

- **Streamlit Docs**: https://docs.streamlit.io/
- **Community Forum**: https://discuss.streamlit.io/
- **GitHub Issues**: Create an issue in your repository


# Hotel Customer Analytics Dashboard

This repository contains a Streamlit dashboard for hotel customer analytics, now enhanced with an Elbow **and** Silhouette diagnostic in the Clustering tab.

## Quick Start

```bash
git clone https://github.com/<your-username>/hotel_dashboard.git
cd hotel_dashboard
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Features
- **Data Visualization**: Ten insightful charts.
- **Classification**: KNN, DT, RF, GB with full metrics and ROC.
- **Clustering**: Elbow & Silhouette curves, interactive k slider, personas table, 3‑D PCA.
- **Association Rules**: Apriori with adjustable parameters.
- **Regression**: Linear, Ridge, Lasso, Decision Tree regressors.

## Deploy to Streamlit Cloud
1. Push repo to GitHub.
2. Create new Streamlit app, point to `streamlit_app.py`.
3. Add `data/hotel_synthetic_data.csv` as an asset (or use your own).
4. Deploy.

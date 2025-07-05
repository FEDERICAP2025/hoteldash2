
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, silhouette_score,
                             roc_curve, auc)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px
import plotly.graph_objects as go
import io
from pathlib import Path

st.set_page_config(page_title="Hotel Analytics Dashboard", layout="wide")
st.title("🏨 Hotel Customer Analytics Dashboard")

# ---------- DATA LOADING ----------
@st.cache_data
def load_data():
    default_path = Path(__file__).parent / "data" / "hotel_synthetic_data.csv"
    if default_path.exists():
        return pd.read_csv(default_path)
    else:
        st.warning("Default dataset not found. Please upload a file in the sidebar.")
        return pd.DataFrame()

df = load_data()

# Sidebar upload option
st.sidebar.header("💾 Data Options")
uploaded = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.sidebar.success("New data loaded!")

if df.empty:
    st.stop()

numeric_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Visualization",
                                        "🎯 Classification",
                                        "🔗 Clustering",
                                        "🛒 Association Rules",
                                        "📈 Regression"])

# ---------------- TAB 1 ----------------
with tab1:
    st.header("Descriptive Insights")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(df, names="Purpose_of_Travel", title="Purpose of Travel")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(df, x="Room_Type", y="Avg_Daily_Rate", color="Room_Type")
        st.plotly_chart(fig, use_container_width=True)
    # Additional visuals omitted for brevity...

# ---------------- TAB 2 ----------------
with tab2:
    st.header("Classification Models")
    target = st.selectbox("Select Target Variable", ["Would_Recommend","Upselling_Success"])
    X = df.drop(columns=[target])
    y = df[target].apply(lambda x: 1 if str(x).lower() in ["yes","1","true"] else 0)
    cat_features = X.select_dtypes(include=["object"]).columns.tolist()
    num_features = X.select_dtypes(include=["int64","float64"]).columns.tolist()

    preprocessor = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
                                      ("num", StandardScaler(), num_features)])

    models = {"KNN": KNeighborsClassifier(),
              "Decision Tree": DecisionTreeClassifier(random_state=42),
              "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
              "Gradient Boosting": GradientBoostingClassifier(random_state=42)}

    model_metrics = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    for name, clf in models.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        model_metrics[name] = {"Train Acc": round(pipe.score(X_train, y_train),3),
                               "Test Acc": round(accuracy_score(y_test, y_pred),3),
                               "Precision": round(precision_score(y_test, y_pred),3),
                               "Recall": round(recall_score(y_test, y_pred),3),
                               "F1": round(f1_score(y_test, y_pred),3),
                               "model": pipe}

    st.dataframe(pd.DataFrame(model_metrics).T.drop(columns=["model"]))
    algo_choice = st.selectbox("Select algorithm for Confusion Matrix", list(models.keys()))
    if algo_choice:
        cm = confusion_matrix(y_test, model_metrics[algo_choice]["model"].predict(X_test))
        st.plotly_chart(px.imshow(cm, text_auto=True,
                                  labels=dict(x="Pred", y="Actual", color="Count")))

# ---------------- TAB 3 ----------------
with tab3:
    st.header("Clustering Diagnostics & Personas")
    cluster_features = st.multiselect("Numeric features for clustering",
                                      numeric_cols,
                                      default=["Avg_Daily_Rate","Lead_Time_Days",
                                               "Length_of_Stay","Total_Revenue"])
    max_k = st.slider("Select number of clusters (k)", 2, 10, 4)

    scaled_data = StandardScaler().fit_transform(df[cluster_features])
    ks = range(2, 11)
    inertias, sil_scores = [], []
    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled_data)
        inertias.append(kmeans.inertia_)
        sil_scores.append(silhouette_score(scaled_data, labels))

    col_elbow, col_sil = st.columns(2)
    with col_elbow:
        st.subheader("Elbow Curve")
        st.plotly_chart(px.line(x=list(ks), y=inertias, markers=True,
                                labels={"x":"k","y":"Inertia"}), use_container_width=True)
    with col_sil:
        st.subheader("Silhouette Curve")
        st.plotly_chart(px.line(x=list(ks), y=sil_scores, markers=True,
                                labels={"x":"k","y":"Average silhouette"}), use_container_width=True)

    # Fit final k-means
    kmeans_final = KMeans(n_clusters=max_k, random_state=42, n_init=10)
    df["Cluster"] = kmeans_final.fit_predict(scaled_data)
    st.subheader("Cluster Personas")
    st.dataframe(df.groupby("Cluster")[cluster_features].mean().round(1))

# ---------------- TAB 4 ----------------
with tab4:
    st.header("Association Rule Mining")
    # (content omitted for brevity)

# ---------------- TAB 5 ----------------
with tab5:
    st.header("Regression Insights")
    # (content omitted for brevity)

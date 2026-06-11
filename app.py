import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(
page_title="Customer Segmentation Engine",
page_icon="📊",
layout="wide"
)

st.title("📊 Customer Segmentation Engine")

st.markdown("""
Upload a customer dataset and discover meaningful customer segments using machine learning.
""")

uploaded_file = st.file_uploader(
"Upload CSV File",
type=["csv"]
)

if uploaded_file is not None:

```
df = pd.read_csv(uploaded_file)

st.success("Dataset Loaded Successfully")

# -------------------------
# Dataset Preview
# -------------------------

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -------------------------
# Dataset Metrics
# -------------------------

st.subheader("Dataset Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Rows", df.shape[0])

with col2:
    st.metric("Columns", df.shape[1])

with col3:
    st.metric("Missing Values", df.isnull().sum().sum())

st.divider()

# -------------------------
# Feature Selection
# -------------------------

st.subheader("Feature Selection")

numeric_cols = list(
    df.select_dtypes(include=["int64", "float64"]).columns
)

if len(numeric_cols) < 2:
    st.error("Dataset must contain at least two numeric columns.")
    st.stop()

default_features = numeric_cols[:2]

selected_features = st.multiselect(
    "Select Features for Segmentation",
    numeric_cols,
    default=default_features
)

if len(selected_features) < 2:
    st.warning("Please select at least two features.")
    st.stop()

# -------------------------
# Cluster Settings
# -------------------------

st.subheader("Clustering Settings")

k = st.slider(
    "Number of Clusters",
    min_value=2,
    max_value=10,
    value=5
)

# -------------------------
# Run Segmentation
# -------------------------

if st.button("🚀 Generate Customer Segments"):

    X = df[selected_features]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    clusters = model.fit_predict(X_scaled)

    result_df = df.copy()

    result_df["Cluster"] = clusters

    st.success("Segmentation Completed Successfully")

    # -------------------------
    # Cluster Visualization
    # -------------------------

    st.subheader("Customer Segments Visualization")

    fig = px.scatter(
        result_df,
        x=selected_features[0],
        y=selected_features[1],
        color=result_df["Cluster"].astype(str),
        title="Customer Segmentation",
        hover_data=result_df.columns
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -------------------------
    # Segment Summary
    # -------------------------

    st.subheader("Segment Summary")

    summary = (
        result_df["Cluster"]
        .value_counts()
        .reset_index()
    )

    summary.columns = [
        "Cluster",
        "Customer Count"
    ]

    st.dataframe(summary)

    # -------------------------
    # Cluster Details
    # -------------------------

    st.subheader("Cluster Details")

    for cluster in sorted(result_df["Cluster"].unique()):

        cluster_data = result_df[
            result_df["Cluster"] == cluster
        ]

        st.info(
            f"""
```

Cluster {cluster}

Customers: {len(cluster_data)}
"""
)

```
    # -------------------------
    # Download Results
    # -------------------------

    csv = result_df.to_csv(index=False)

    st.download_button(
        label="📥 Download Segmented Dataset",
        data=csv,
        file_name="segmented_customers.csv",
        mime="text/csv"
    )
else:
st.info("Please upload a CSV file to continue.")

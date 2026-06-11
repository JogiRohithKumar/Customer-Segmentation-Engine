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

st.markdown(
    "Upload a customer dataset and discover meaningful customer segments using machine learning."
)

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("✅ Dataset Loaded Successfully")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric("Missing Values", int(df.isnull().sum().sum()))

    st.divider()

    numeric_cols = list(
        df.select_dtypes(include=["int64", "float64"]).columns
    )

    if len(numeric_cols) < 2:
        st.error("Dataset must contain at least two numeric columns.")
        st.stop()

    st.subheader("Feature Selection")

    selected_features = st.multiselect(
        "Select Features for Segmentation",
        options=numeric_cols,
        default=numeric_cols[:2]
    )

    if len(selected_features) < 2:
        st.warning("Please select at least two features.")
        st.stop()

    st.subheader("Clustering Settings")

    k = st.slider(
        "Number of Clusters",
        min_value=2,
        max_value=10,
        value=5
    )

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

        st.success("🎉 Segmentation Completed Successfully")

        st.subheader("Customer Segments Visualization")

        fig = px.scatter(
            result_df,
            x=selected_features[0],
            y=selected_features[1],
            color=result_df["Cluster"].astype(str),
            title="Customer Segmentation"
        )

        st.subheader("Cluster Distribution")

        fig2 = px.pie(
            summary,
            names="Cluster",
            values="Customer Count",
            title="Customer Distribution by Cluster"
        )
        
        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Segment Summary")

        summary = (
            result_df["Cluster"]
            .value_counts()
            .reset_index()
        )

        summary.columns = ["Cluster", "Customer Count"]

        st.dataframe(summary, use_container_width=True)

        # -------------------------
        # Customer Personas
        # -------------------------
        
        st.subheader("Customer Personas")
        
        cluster_profiles = (
            result_df.groupby("Cluster")[selected_features]
            .mean()
            .round(2)
        )
        
        st.dataframe(
            cluster_profiles,
            use_container_width=True
        )
        
        # -------------------------
        # Business Recommendations
        # -------------------------
        
        st.subheader("Business Recommendations")
        
        for cluster in cluster_profiles.index:

            income = cluster_profiles.iloc[cluster][0]
            spending = cluster_profiles.iloc[cluster][1]
        
            if income >= 70 and spending >= 70:
        
                st.success(
                    f"⭐ Cluster {cluster}: Premium Customers → Luxury Product Campaigns"
                )
        
            elif income <= 40 and spending <= 40:
        
                st.warning(
                    f"💰 Cluster {cluster}: Budget Customers → Discount Campaigns"
                )
        
            else:
        
                st.info(
                    f"🎯 Cluster {cluster}: Regular Customers → Loyalty Programs"
                )
        

        st.subheader("Cluster Details")

        for cluster in sorted(result_df["Cluster"].unique()):
            cluster_data = result_df[result_df["Cluster"] == cluster]

            st.info(
                f"Cluster {cluster}\n\nCustomers: {len(cluster_data)}"
            )

        csv = result_df.to_csv(index=False)

        st.download_button(
            label="📥 Download Segmented Dataset",
            data=csv,
            file_name="segmented_customers.csv",
            mime="text/csv"
        )

else:
    st.info("📁 Please upload a CSV file to continue.")

```python
import streamlit as st
import pandas as pd

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
    df = pd.read_csv(uploaded_file)

    st.success("Dataset Loaded Successfully")

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())

else:
    st.info("Please upload a CSV file to continue.")
```


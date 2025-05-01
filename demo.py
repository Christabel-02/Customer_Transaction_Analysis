import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import numpy as np

st.set_page_config(page_title="Customer Transaction Analysis", layout="wide")
st.title("📊 Customer Transaction Analysis App")

# Sidebar
st.sidebar.header("Upload and Settings")

uploaded_file = st.sidebar.file_uploader("📂 Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    # Try reading CSV with fallback encoding
    try:
        data = pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        data = pd.read_csv(uploaded_file, encoding='latin1')

    # Only use numeric columns for clustering
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns

    if len(numeric_cols) < 2:
        st.error("❌ Not enough numeric columns in the uploaded file for clustering.")
    else:
        st.sidebar.subheader("⚙️ Settings")
        feature_x = st.sidebar.selectbox("Select X-axis Feature", numeric_cols)
        feature_y = st.sidebar.selectbox("Select Y-axis Feature", numeric_cols)

        st.sidebar.markdown("---")

        st.subheader("🔍 Raw Data")
        st.dataframe(data)

        st.subheader("📈 Data Overview")
        c1, c2 = st.columns(2)
        with c1:
            st.write("Shape of dataset:", data.shape)
            st.write("Columns:", list(data.columns))
        with c2:
            st.write("Missing values:")
            st.dataframe(data.isnull().sum())

        st.subheader("📊 Feature Distributions")
        c3, c4 = st.columns(2)
        with c3:
            fig1, ax1 = plt.subplots()
            sns.histplot(data[feature_x].dropna(), bins=30, kde=True, ax=ax1, color="skyblue")
            ax1.set_title(f'Distribution of {feature_x}')
            st.pyplot(fig1)

        with c4:
            fig2, ax2 = plt.subplots()
            sns.histplot(data[feature_y].dropna(), bins=30, kde=True, ax=ax2, color="lightgreen")
            ax2.set_title(f'Distribution of {feature_y}')
            st.pyplot(fig2)

        st.subheader("🔵 KMeans Clustering")

        # Drop missing values before clustering
        clustering_data = data[[feature_x, feature_y]].dropna()

        if clustering_data.shape[0] == 0:
            st.error("❌ No valid data points available for clustering. Check for missing values.")
        else:
            inertia = []
            K = range(1, 11)
            for k in K:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(clustering_data)
                inertia.append(kmeans.inertia_)

            fig3, ax3 = plt.subplots()
            ax3.plot(K, inertia, 'bo-')
            ax3.set_xlabel('k')
            ax3.set_ylabel('Inertia')
            ax3.set_title('The Elbow Method')
            st.pyplot(fig3)

            selected_k = st.sidebar.slider("Select number of clusters (k)", 2, 10, 3)

            kmeans = KMeans(n_clusters=selected_k, random_state=42)
            data.loc[clustering_data.index, 'Cluster'] = kmeans.fit_predict(clustering_data)

            st.subheader("🧩 Clustered Data")
            st.dataframe(data)

            st.subheader("🎯 Cluster Visualization")
            fig4, ax4 = plt.subplots()
            palette = sns.color_palette("tab10", selected_k)
            sns.scatterplot(x=feature_x, y=feature_y, hue="Cluster", palette=palette, data=data, ax=ax4)
            ax4.set_title("Customer Segments")
            ax4.legend(title="Cluster")
            st.pyplot(fig4)

            # Download clustered data
            st.subheader("📥 Download Clustered Data")
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name='clustered_customers.csv',
                mime='text/csv'
            )
else:
    st.info("👈 Upload a CSV file from the sidebar to get started!")

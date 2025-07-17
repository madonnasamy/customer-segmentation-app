import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Customer Segmentation using Clustering")
st.markdown("This app uses different unsupervised learning algorithms to segment customers based on their Age, Income, and Spending Score.")

df = pd.read_csv("clustered_customers.csv")
st.subheader("📊 Raw Data")
st.dataframe(df.head())

algo = st.selectbox("Select Clustering Algorithm", ['KMeans', 'HC', 'GMM', 'DBSCAN'])

cluster_col = f"{algo}_Cluster" if algo != 'KMeans' else 'Cluster'

st.subheader("📌 Cluster Summary")
if cluster_col in df.columns:
    summary = df.groupby(cluster_col)[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean().round(1)
    st.dataframe(summary)
else:
    st.warning(f"Clustering column '{cluster_col}' not found!")

st.subheader("📍 Cluster Plot")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)',
                hue=df[cluster_col], palette='tab10', s=80, ax=ax)
st.pyplot(fig)

st.download_button("⬇️ Download Clustered Data as CSV", data=df.to_csv(index=False), file_name="clustered_customers.csv", mime='text/csv')

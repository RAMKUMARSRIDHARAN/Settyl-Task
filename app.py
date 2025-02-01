import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load your dataset
Recom_system = pd.read_csv("D:/DS with Py/Walmart_Dataset.csv")

# Perform your preprocessing steps here (you can copy your existing code from above)
Recom_system_Filtered = Recom_system[['Uniq Id', 'Product Id', 'Product Rating', 'Product Reviews Count', 
                                      'Product Category', 'Product Brand', 'Product Name', 'Product Image Url', 
                                      'Product Description', 'Product Tags']].copy()

# Rename columns for easier use
column_name_mapping = {
    'Uniq Id': 'ID', 'Product Id': 'ProdID', 'Product Rating': 'Rating', 
    'Product Reviews Count': 'ReviewCount', 'Product Category': 'Category', 
    'Product Brand': 'Brand', 'Product Name': 'Name', 'Product Image Url': 'ImageURL', 
    'Product Description': 'Description', 'Product Tags': 'Tags'
}
Recom_system_Filtered = Recom_system_Filtered.rename(columns=column_name_mapping)

# Function for content-based recommendations
def content_based_recommendations(Recom_system_Filtered, item_name, top_n=10):
    # Check if the item_name is in the DataFrame
    if item_name not in Recom_system_Filtered['Name'].values:
        st.error(f"Item '{item_name}' not found in the data.")
        return pd.DataFrame()

    # Proceed with the recommendation logic
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(Recom_system_Filtered['Tags'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Find the index of the item
    item_index = Recom_system_Filtered[Recom_system_Filtered['Name'] == item_name].index[0]

    # Get the most similar items
    similar_items = sorted(enumerate(cosine_sim[item_index]), key=lambda x: x[1], reverse=True)
    top_similar_items = similar_items[1:top_n+1]  # Skip the first item (itself)
    recommended_indices = [x[0] for x in top_similar_items]
    
    return Recom_system_Filtered.iloc[recommended_indices][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]

# Streamlit app layout
st.title("E-Commerce Product Recommendation System")

# User Input for content-based filtering
st.header("Content-Based Recommendations")
item_name = st.text_input("Enter a product name for recommendations:", "")
if item_name:
    # Validate if item_name is not empty and exists in the dataset
    if item_name.strip() == "":
        st.error("Please enter a valid product name.")
    else:
        top_n = st.slider("Select number of recommendations", 1, 20, 10)
        recommendations = content_based_recommendations(Recom_system_Filtered, item_name, top_n)
        if not recommendations.empty:
            st.write(f"Recommendations for {item_name}:")
            st.dataframe(recommendations)
        else:
            st.write("No recommendations found.")

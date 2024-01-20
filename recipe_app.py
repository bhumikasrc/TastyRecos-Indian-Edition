import streamlit as st
import pandas as pd
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from recs import *

# Streamlit app
def main():
    st.title('Recipe Recommendation App')

    # User input
    user_input = st.text_area('Enter your favorite ingredients (comma-separated):')

    if st.button('Get Recommendations'):
        if user_input:
            st.text('Top 5 Recommended Recipes:')
            recommended_recipes = recs(user_input)
            for idx, recipe in enumerate(recommended_recipes, start=1):
                st.write(f"{idx}. {recipe}")
        else:
            st.warning('Please enter ingredients.')

if __name__ == '__main__':
    main()

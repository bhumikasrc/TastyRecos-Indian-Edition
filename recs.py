import re
import sys
import spacy
import pandas as pd
from tqdm import tqdm
from joblib import dump
from joblib import load

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def food_data(food):
    food = food.dropna()
    food = food.drop_duplicates()
    food = food.reset_index(drop=True)
    food = food[['TranslatedRecipeName', 'TranslatedIngredients',
                 'PrepTimeInMins', 'CookTimeInMins', 'TotalTimeInMins',
                 'Servings', 'Cuisine', 'Course', 'Diet',
                 'TranslatedInstructions', 'URL']]
    food.columns = ['Recipe', 'Ingredients', 'PrepTime', 'CookTime',
                    'TotalTime', 'Servings', 'Cuisine', 'Course',
                    'Diet', 'Instructions', 'URL']
    food = food.iloc[:6819, :]
    food.to_csv('data/cleaned_food.csv', index=False)


def train_clean_ingredients(ingredients, nlp=spacy.load('en_core_web_sm')):
    # Remove numbers and unwanted words using regex
    clean = ingredients.split(',')
    clean = [text.strip().lower() for text in clean]
    cleaned_ingredients = [re.sub(
        r'\b\d+(\.\d+)?\s*(\/\s*\d+(\.\d+)?)?\s*(teaspoon|tablespoon|\
            cup|ounce|pound|g|kg)?[s]?\b', '', ingredients)
        for ingredients in clean]

    # Remove special characters and extra whitespaces
    cleaned_ingredients = [re.sub(r'[^a-zA-Z\s]', '', cleaned_ingredients)
                           for cleaned_ingredients in cleaned_ingredients]

    # Remove extra whitespaces
    cleaned_ingredients = [re.sub(r'\s+', ' ', cleaned_ingredients)
                           for cleaned_ingredients in cleaned_ingredients]

    cleaned_ingredients = [text.strip() for text in cleaned_ingredients]

    cleaned_ingredients = ', '.join(cleaned_ingredients)

    # Tokenize the ingredients using spaCy
    doc = nlp(cleaned_ingredients)

    # Filter out tokens that are verbs, adjectives, and prepositions
    cleaned_ingredients_spacy = [token.text.lower() for token in doc
                                 if token.pos_ not in (
        'VERB', 'ADJ', 'ADP') and not token.is_stop]

    cleaned_ingredients_spacy = list(
        filter(lambda x: x.strip() != '', cleaned_ingredients_spacy))

    result_list = []
    current_item = ''

    for item in cleaned_ingredients_spacy:
        if item != ',':
            current_item += item + ' '
        else:
            result_list.append(current_item.strip())
            current_item = ''

    # Append the last item after the last comma
    result_list.append(current_item)

    # Filter out empty strings and remove commas
    result_list = [item.strip() for item in result_list if item]

    return result_list


def ingredients_parser(ingredients):
    clean = ingredients.strip().lower()
    clean = re.sub(r'[^a-zA-Z,\s]', '', clean)
    clean = re.sub(r'\s+', ' ', clean)
    clean = clean.split(', ')

    return clean


def train_vectorization(col):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(col)

    return tfidf_matrix, vectorizer


def vectorization(ingredients, vectorizer):
    return vectorizer.transform([" ".join(ingredients)])


def train_model(tfidf_matrix):
    model = NearestNeighbors(n_neighbors=10,
                             algorithm='brute', metric='cosine')
    model.fit(tfidf_matrix)
    model_filename = 'models/knn_model.joblib'
    dump(model, model_filename)


def recs(input):
    user_input = ingredients_parser(input)
    vectorizer = load('models/tfidf_vectorizer.joblib')
    user_input_tfidf = vectorization(user_input, vectorizer)
    knn_model = load('models/knn_model.joblib')
    _, indices = knn_model.kneighbors(user_input_tfidf, n_neighbors=5)

    food = pd.read_csv('data/cleaned_food.csv')
    recommended_recipes = food.iloc[indices[0]]['Recipe'].tolist()
    recommended_indices = food.iloc[indices[0]]['Ingredients'].tolist()

    return recommended_recipes, recommended_indices


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        food_data(pd.read_csv('data/IndianFoodDatasetCSV.csv'))
        food = pd.read_csv('data/cleaned_food.csv')
        print("Cleaning data...")

        # Use tqdm to display a progress bar during iteration
        tqdm.pandas(desc="Cleaning progress")
        food['Ingredients_Cleaned'] = food['Ingredients']\
            .progress_apply(train_clean_ingredients)

        print("Vectorizing data...")

        # Use tqdm to display a progress bar during vectorization
        tqdm.pandas(desc="Vectorization progress")
        food['Ingredients_str'] = food['Ingredients_Cleaned']\
            .progress_apply(lambda x: ', '.join(x))

        tfidf_matrix, vectorizer = train_vectorization(food['Ingredients_str'])

        vectorizer_filename = 'models/tfidf_vectorizer.joblib'
        dump(vectorizer, vectorizer_filename)

        print("Training model...")
        train_model(tfidf_matrix)
    elif sys.argv[1] == 'recs':
        input = "chicken thigh, risdlfgbviahsddsagv, onion, rice noodle, \
        seaweed nori sheet"
        print(recs(input))

    else:
        print("Please enter a valid command.")
        sys.exit(1)

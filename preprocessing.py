import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib


def preprocess_data():
    # Download NLTK stopwords
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

    def preprocess(text):
        text = text.lower()  # Convert text to lowercase
        text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove punctuation and numbers
        text = " ".join(
            [word for word in text.split() if word not in stop_words]
        )  # Remove stopwords
        return text

    # Load combined dataset generated from combine_datasets.py (update path as needed)
    df = pd.read_csv("data/combined_dataset.csv")

    # Clean text and add to new column 'clean_text'
    df["clean_text"] = df["text"].apply(preprocess)

    # Convert cleaned text to numerical features using TF-IDF
    tfidf = TfidfVectorizer(max_features=5000)  # Limit to 5000 features
    X = tfidf.fit_transform(df["clean_text"])

    # Get labels
    y = df["label"]

    # 60% training, 40% temporary set
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    # Split temporary set into 20% validation and 20% testing
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Training Set: Stored in X_train and y_train
    # Validation Set: Stored in X_val and y_val
    # Testing Set: Stored in X_test and y_test

    # Save data to disk
    joblib.dump(X_train, "data/processed/X_train.pkl")
    joblib.dump(y_train, "data/processed/y_train.pkl")
    joblib.dump(X_val, "data/processed/X_val.pkl")
    joblib.dump(y_val, "data/processed/y_val.pkl")
    joblib.dump(X_test, "data/processed/X_test.pkl")
    joblib.dump(y_test, "data/processed/y_test.pkl")
    print("Data preprocessed!\n")

    # How to load the data in another file
    # X_train = joblib.load('X_train.pkl')
    # y_train = joblib.load('y_train.pkl')
    # X_val = joblib.load('X_val.pkl')
    # y_val = joblib.load('y_val.pkl')
    # X_test = joblib.load('X_test.pkl')
    # y_test = joblib.load('y_test.pkl')


if __name__ == "__main__":
    preprocess_data()

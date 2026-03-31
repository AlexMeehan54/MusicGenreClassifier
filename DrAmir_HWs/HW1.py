import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_data(filepath="diabetes.csv"):
    """
    Loads the diabetes dataset from a CSV file.
    Returns:
        X (pd.DataFrame): feature matrix
        y (pd.Series): target labels
    """
    # TODO: Load csv and separate features from the 'Outcome' column
    # Load the CSV file into a DataFrame
    DataFrame = pd.read_csv(filepath)
    X = DataFrame.drop(columns=['Outcome']) # X contains everything EXCEPT 'Outcome'
    y = DataFrame['Outcome']    # y contains ONLY the 'Outcome' column
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits the data into training and test sets.
    """
    # TODO: Implement train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test 


def train_model(X_train, y_train):
    """
    Trains a Logistic Regression classifier.
    Returns:
        model: trained scikit-learn classifier
    """
    # TODO: Initialize LogisticRegression with max_iter=500 and fit
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model.
    Returns:
        metrics (dict): A dictionary with keys: 
        'accuracy', 'precision', 'recall', 'f1_score'
    """
    # TODO: Calculate all four metrics and return them in a dictionary
    predictions = model.predict(X_test)

    evaluation = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions)
    }
    return evaluation 

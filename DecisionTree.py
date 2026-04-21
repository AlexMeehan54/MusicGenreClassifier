# Uses HW3 Decision Tree and import functions
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.impute import SimpleImputer

# TODO: import required models
from sklearn.tree import DecisionTreeClassifier

def load_data(filepath='train.csv'):
     # Load dataset
    df = pd.read_csv(filepath)

    # Separate features and target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Convert features to numeric (coerce errors → NaN)
    X = X.apply(pd.to_numeric, errors='coerce')

    # Optional: ensure target is clean (especially if classification)
    y = y.astype(str)  # or keep as-is depending on your dataset

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    return X, y.values

def split_data(X, y):
    """
    Split data into training and testing sets.
    Returns:
        X_train, X_test, y_train, y_test
    """
    # TODO: Split data with test_size=0.3 and random_state=42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Don't print anything here - autograder handles printing
    return X_train, X_test, y_train, y_test

def train_decision_tree(X_train, y_train):
    """
    Train a Decision Tree classifier.
    Returns:
        model: Trained Decision Tree model
    """
    # TODO: Train Decision Tree with random_state=42
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    return model

def evaluate_decision_tree(model, X_test, y_test):
    """
    Evaluate the Decision Tree model.
    Returns:
        accuracy: Model accuracy (float)
        predictions: Model predictions on test set (numpy array)
    """
    # TODO: Get predictions and calculate accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    return accuracy, predictions


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Test Decision Tree
    dt_model = train_decision_tree(X_train, y_train)
    dt_acc, dt_preds = evaluate_decision_tree(dt_model, X_test, y_test)
    print("DT_ACCURACY:", round(dt_acc, 4) if dt_acc is not None else None)
    print("DT_PRED_SAMPLE:", dt_preds[:10] if dt_preds is not None else None)


if __name__ == "__main__":
    main()

# src/iris_classifier.py
'''
iris-classification/
├── iris_classifier.py
├── debug_test.py
├── requirements.txt
├── README.md
'''

# import dataset and model
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# import dependencies
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Model has 150 samples, 4 features, 3 classes (ABC)
# X = feature matrix
# y = target
def load_data():
    """
    Returns:
        X (np.ndarray): feature matrix of shape (150, 4)
        y (np.ndarray): target labels of shape (150,)
    """
    # 03 Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    return iris.data, iris.target

# 20% training, 80% learning
def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits the data into training and test sets.
    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """
    Trains a classifier on the training data.
    Returns:
        model: trained scikit-learn classifier
    """
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model.
    Returns:
        accuracy (float): value between 0 and 1
    """
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)    



"""
Homework: K-Means, SVM, Decision Tree on CSV Dataset

Instructions:
- Complete the TODO sections.
- Do NOT modify function names (used for grading).
- Dataset: diabetes.csv
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# TODO: import required models
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def load_data():
    """
    Load the diabetes dataset.
    Returns:
        X: Features array (numpy array)
        y: Target array (numpy array)
    """
    # TODO: Load data from diabetes.csv
    df = pd.read_csv("diabetes.csv")  # Replace with pd.read_csv("diabetes.csv")
    X = df.iloc[:, :-1].values   # Replace with df.iloc[:, :-1].values
    y = df.iloc[:, -1].values   # Replace with df.iloc[:, -1].values
    
    # Don't print anything here - autograder handles printing
    return X, y


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


def kmeans_clustering(X):
    """
    Perform KMeans clustering with 2 clusters.
    Returns:
        centers: Cluster centers (numpy array)
        labels: Cluster labels for all data points (numpy array)
    """
    # TODO: Implement KMeans with n_clusters=2, random_state=42, n_init=10
    
# K-Means Clustering
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(X)

    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    return centers, labels


def train_svm(X_train, y_train):
    """
    Train an SVM classifier.
    Returns:
        model: Trained SVM model
    """
    # TODO: Train SVM with kernel="linear", random_state=42
    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train, y_train)
    
    return model


def evaluate_svm(model, X_test, y_test):
    """
    Evaluate the SVM model.
    Returns:
        accuracy: Model accuracy (float)
        predictions: Model predictions on test set (numpy array)
    """
    # TODO: Get predictions and calculate accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    return accuracy, predictions


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
    
    # Test KMeans
    centers, labels = kmeans_clustering(X)
    print("KMEANS_CENTERS:", centers)
    print("KMEANS_LABELS_SAMPLE:", labels[:10] if labels is not None else None)
    
    # Test SVM
    svm_model = train_svm(X_train, y_train)
    svm_acc, svm_preds = evaluate_svm(svm_model, X_test, y_test)
    print("SVM_ACCURACY:", round(svm_acc, 4) if svm_acc is not None else None)
    print("SVM_PRED_SAMPLE:", svm_preds[:10] if svm_preds is not None else None)
    
    # Test Decision Tree
    dt_model = train_decision_tree(X_train, y_train)
    dt_acc, dt_preds = evaluate_decision_tree(dt_model, X_test, y_test)
    print("DT_ACCURACY:", round(dt_acc, 4) if dt_acc is not None else None)
    print("DT_PRED_SAMPLE:", dt_preds[:10] if dt_preds is not None else None)


if __name__ == "__main__":
    main()

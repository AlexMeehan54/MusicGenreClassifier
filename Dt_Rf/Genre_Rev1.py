# TODO:
# Changes: Test without class 10 in the last column
# Make interchangable for feature subsets
# Make a bar graph of our class variation
# Visualize both models via chart with SVM (Merge commits)

# Uses HW3 Decision Tree and import functions
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.impute import SimpleImputer

# TODO: import required models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def load_data(filepath='csvFiles/train.csv'):
    df = pd.read_csv(filepath)
    # Select features and target
    df = df[['Popularity', 'danceability', 'energy', 'key',
              'loudness', 'mode', 'speechiness', 'acousticness',
                'instrumentalness', 'liveness', 'valence', 'tempo',
                  'duration_in min/ms', 'time_signature',
                    df.columns[-1]]]
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=[df.columns[-1]])
    # Split features and target
    X = df[['Popularity', 'danceability', 'energy', 'key',
              'loudness', 'mode', 'speechiness', 'acousticness',
                'instrumentalness', 'liveness', 'valence', 'tempo',
                  'duration_in min/ms', 'time_signature']]
    y = df.iloc[:, -1]
    counts = df['Class'].value_counts().sort_index()

    # Bar Chart
    plt.bar(counts.index.astype(str), counts.values)
    plt.xlabel('Class')
    plt.ylabel('Records')
    plt.title('Class Summary')
    plt.show()
    return X, y.values

def split_data(X, y):
    # TODO: Split data with test_size=0.3 and random_state=42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    return X_train, X_test, y_train, y_test

def train_decision_tree(X_train, y_train):
    """
    Train a Decision Tree classifier.
    Returns:
        model: Trained Decision Tree model
    """
    # TODO: Train Decision Tree with random_state=42
    model = DecisionTreeClassifier(
        random_state=42,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5
    )
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        random_state=42,
        n_estimators=500,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        max_depth=12,

    )
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting(X_train, y_train):
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test):
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

def plot_genre_confusion_matrix (model, X_test, y_test, class_names):
    """
    Generates and plots a confusion matrix for model evaluation.
    """
    # Confusion
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(12, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
    plt.title("Confusion Matrix: Predicted vs Actual Genres")
    # plt.show()

    # Bar chart
    

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    # Test Decision Tree
    dt_model = train_decision_tree(X_train, y_train)
    dt_acc, dt_preds = evaluate(dt_model, X_test, y_test)
    print("DT_ACCURACY:", round(dt_acc, 4) if dt_acc is not None else None)
    print("DT_PRED_SAMPLE:", dt_preds[:10] if dt_preds is not None else None)
    # Test Random Forest
    rf_model = train_random_forest(X_train, y_train)
    rf_acc, rf_preds = evaluate(rf_model, X_test, y_test)
    print("RF_ACCURACY:", round(rf_acc, 4) if rf_acc is not None else None)
    print("RF_PRED_SAMPLE:", rf_preds[:10] if rf_preds is not None else None)
    # Test Gradient Boosting

    feature_cols = X.columns.tolist()
    unique_classes = [str(c) for c in sorted(set(y))]
    plot_genre_confusion_matrix(dt_model, X_test, y_test, unique_classes)

    """
    gb_model = train_gradient_boosting(X_train, y_train)
    gb_acc, gb_preds = evaluate(gb_model, X_test, y_test)
    print("GB_ACCURACY:", round(gb_acc, 4))
    print("GB_PRED_SAMPLE:", gb_preds[:10])
    """

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# TODO: import required models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import matplotlib.pyplot as plt

def load_data(filepath='csvFiles/train.csv'):
    df = pd.read_csv(filepath)
    # Drop nominal data
    df = df.drop(columns=['Artist Name', 'Track Name'], errors='ignore')
   
    # Drop rows where the Class label is missing
    df = df.dropna(subset=[df.columns[-1]])

    X = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')
    y = df.iloc[:, -1]
    return X, y

def split_data(X, y):
    # Split data with test_size=0.2 and random_state=42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #fills the missing NaN with the average of that column
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    #Scales data (important for SVM)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

#Model Training
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

def train_svm(X_train, y_train):
    """
        params = {
            "C": [0.1, 1, 10, 100],
            "gamma": [0.001, 0.01, 0.1, 1]
        }

        model = SVC(kernel='rbf', gamma = 'scale', random_state=42)
        model.fit(X_train, y_train)
        
        return model
    """
    model = SVC(
        kernel='rbf', 
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

# Model evaluation
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

def evaluate_svm(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy, predictions

def show_tree_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\nFeature Importance (Tree-Based Model):")
    for i in indices:
        print(f"{feature_names[i]}: {importances[i]:.4f}")

def show_svm_importance(model, X_test, y_test, feature_names):
    r = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42)

    indices = r.importances_mean.argsort()[::-1]

    print("\nFeature Importance (SVM Permutation):")
    for i in indices:
        print(f"{feature_names[i]}: {r.importances_mean[i]:.4f}")

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
    
    # Test SVM
    svm_model = train_svm(X_train, y_train)
    svm_acc, svm_preds = evaluate_svm(svm_model, X_test, y_test)
    
    print("SVM_ACCURACY:", round(svm_acc, 4) if svm_acc is not None else None)
    print("SVM_PRED_SAMPLE:", svm_preds[:10] if svm_preds is not None else None)

    
    feature_names = X.columns

    show_tree_importance(dt_model, feature_names)
    show_tree_importance(rf_model, feature_names)
    show_svm_importance(svm_model, X_test, y_test, feature_names)
    

    models = ['Decision Tree', 'Random Forest', 'SVM']
    accuracies = [dt_acc, rf_acc, svm_acc]

    plt.figure(figsize=(8, 5))
    plt.gca().set_facecolor("#FFFFFF")
    plt.bar(models, accuracies, color="#1DB954", alpha=0.2)
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.ylim(0, 1)

    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.02, f"{v:.3f}", ha='center',color = "#000000")

    plt.show()

if __name__ == "__main__":
    main()
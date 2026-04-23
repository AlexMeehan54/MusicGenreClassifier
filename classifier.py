
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier

from pathlib import Path

def load_data():
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "csvFiles" / "train.csv"

    df = pd.read_csv(csv_path)
    df = df.drop(columns=['Artist Name', 'Track Name','time_signature','tempo','key','mode'], errors='ignore')

    """
    df['duration_in min/ms'] = pd.to_numeric(df['duration_in min/ms'], errors='coerce')
    df['duration_in min/ms'] = df['duration_in min/ms'].apply(
        lambda x: x * 60000 if x < 100 else x
    )
    """
   
    # Drop rows where the Class label is missing
    df = df.dropna(subset=[df.columns[-1]])

    df_numeric = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')

    X = df_numeric.values
    y = df.iloc[:, -1].values

    #fills the missing NaN with the average of that column
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)

    return X, y

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #print trainging set here?
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Data loaded and split.")
    print("Train size:", X_train.shape)
    print("Test size:", X_test.shape)

    return X_train, X_test, y_train, y_test


def train_svm(X_train, y_train):
    model = SVC(kernel='rbf', random_state=42)
    model.fit(X_train, y_train)
    
    return model

def evaluate_svm(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    return accuracy, predictions

"""
def train_decision_tree(X_train, y_train):
    # TODO: Train Decision Tree with random_state=42
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_decision_tree(model, X_test, y_test):

    # TODO: Get predictions and calculate accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    return accuracy, predictions

"""
def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    
    svm_model = train_svm(X_train, y_train)
    svm_acc, svm_preds = evaluate_svm(svm_model, X_test, y_test)

    print("SVM_ACCURACY:", round(svm_acc, 4) if svm_acc is not None else None)
    print("SVM_PRED_SAMPLE:", svm_preds[:10] if svm_preds is not None else None)

    """
    dt_model = train_decision_tree(X_train, y_train)
    dt_acc, dt_preds = evaluate_decision_tree(dt_model, X_test, y_test)
    print("DT_ACCURACY:", round(dt_acc, 4) if dt_acc is not None else None)
    print("DT_PRED_SAMPLE:", dt_preds[:10] if dt_preds is not None else None)
    
    """

#plot
"""
    scaler_full = StandardScaler()
    X_scaled = scaler_full.fit_transform(X)

    pca_full = PCA()
    X_pca_full = pca_full.fit_transform(X_scaled)

    var_ratio = pca_full.explained_variance_ratio_
    cumulative = np.cumsum(var_ratio)

    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_scaled)

    # Create DataFrame for plotting
    pca_df = pd.DataFrame(X_pca_2d, columns=['PC1', 'PC2'])
    pca_df['label'] = y

    unique_labels = np.unique(y)
    for label in unique_labels:
        mask = pca_df['label'] == label
        plt.scatter(
            pca_df.loc[mask, 'PC1'],
            pca_df.loc[mask, 'PC2'],
            label=str(label),
            alpha=0.7
        )
    plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
    plt.title('PCA Visulization (2D)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    """
    
if __name__ == "__main__":
    main()

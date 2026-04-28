
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
<<<<<<< Updated upstream
from sklearn.model_selection import GridSearchCV
=======
>>>>>>> Stashed changes

from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier

from pathlib import Path

def load_data():
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "csvFiles" / "train.csv"

    df = pd.read_csv(csv_path)
    df = df.drop(columns=['Artist Name', 'Track Name'], errors='ignore')
   
    # Drop rows where the Class label is missing
    df = df.dropna(subset=[df.columns[-1]])

    X = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')
    y = df.iloc[:, -1]

    #fills the missing NaN with the average of that column
    imputer = SimpleImputer(strategy='median')

    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    return X, y

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #print trainging set here?
    scaler = StandardScaler()

    X_train = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X.columns
    )

    X_test = pd.DataFrame(
        scaler.transform(X_test),
        columns=X.columns
    )

    print("Data loaded and split.")
    print("Train size:", X_train.shape)
    print("Test size:", X_test.shape)

    return X_train, X_test, y_train, y_test


def train_svm(X_train, y_train):
<<<<<<< Updated upstream

    params = {
        "C": [0.1, 1, 10, 100],
        "gamma": [0.001, 0.01, 0.1, 1]
    }

    grid = GridSearchCV(
        SVC(kernel="rbf"),
        params,
        cv=5,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("Best Params:", grid.best_params_)
    print("Best CV Score:", grid.best_score_)

    return grid.best_estimator_

"""
    params = {
        "C": [0.1, 1, 10, 100],
        "gamma": [0.001, 0.01, 0.1, 1]
    }

    model = SVC(kernel='rbf', gamma = 'scale', random_state=42)
    model.fit(X_train, y_train)
    
    return model
"""
=======
    model = SVC(kernel='rbf', random_state=42)
    model.fit(X_train, y_train)
    
    return model
>>>>>>> Stashed changes

def evaluate_svm(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    return accuracy, predictions

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    
    svm_model = train_svm(X_train, y_train)
    svm_acc, svm_preds = evaluate_svm(svm_model, X_test, y_test)
    
    """
    feature_names = X_test.columns
    
    r = permutation_importance(
        svm_model, X_test, y_test,
        n_repeats=2,      # TEMPORARY
        n_jobs=-1         # use all CPU cores
    )

    for i in r.importances_mean.argsort()[::-1]:
        print(f"{feature_names [i]}: {r.importances_mean[i]:.6f}"  
              f" +/- {r.importances_std[i]:.3f}")
    """
    
    print("SVM_ACCURACY:", round(svm_acc, 4) if svm_acc is not None else None)
    print("SVM_PRED_SAMPLE:", svm_preds[:10] if svm_preds is not None else None)
   


    """
    scaler_full = StandardScaler()
    X_scaled = scaler_full.fit_transform(X)

    pca_full = PCA()
    X_pca_full = pca_full.fit_transform(X_scaled)

    var_ratio = pca_full.explained_variance_ratio_
    cumulative = np.cumsum(var_ratio)

    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_scaled)

    for i, (var, cum) in enumerate(zip(var_ratio, cumulative), 1):
        print(f"PC{i}: {var:.3f} ({var*100:.1f}%) - Cumulative: {cum:.3f} ({cum*100:.1f}%)")

    # Scree plot (variance plot) - 5 points
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(var_ratio)+1), var_ratio, alpha=0.7, color='steelblue')
    plt.plot(range(1, len(var_ratio)+1), cumulative, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Scree Plot: Variance by Component')
    plt.xticks(range(1, len(var_ratio)+1))
    plt.legend(['Cumulative', 'Individual'])
    plt.grid(True, alpha=0.3)

    plt.show()
    
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

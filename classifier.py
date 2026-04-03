import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC 
from sklearn.cluster import KMeans
from pathlib import Path

def load_data():

    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "csvFiles" / "trainSample.csv"

    df = pd.read_csv(csv_path)
    #df = pd.read_csv("csvFiles/trainSample.csv")  # Replace with pd.read_csv("diabetes.csv")

    #drops non numerical columns, change later
    X = df.iloc[:, :-1].select_dtypes(include=[float, int]).values
    y = df.iloc[:, -1].values  

    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    return X,y

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    #print trainging set here?

    print("Data loaded and split.")
    print("Train size:", X_train.shape)
    print("Test size:", X_test.shape)

    return X_train, X_test, y_train, y_test


def kmeans_clustering(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
# K-Means Clustering
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    """
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    """

    return kmeans.cluster_centers_, kmeans.labels_



def train_svm(X_train, y_train):
    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train, y_train)
    
    return model

def evaluate_svm(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    return accuracy, predictions

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    svm_model = train_svm(X_train, y_train)
    svm_acc, svm_preds = evaluate_svm(svm_model, X_test, y_test)

    print("SVM_ACCURACY:", round(svm_acc, 4) if svm_acc is not None else None)
    print("SVM_PRED_SAMPLE:", svm_preds[:10] if svm_preds is not None else None)

    print("\nFinding best k with elbow method...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    inertias = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        print(f"k={k}, inertia={km.inertia_:.2f}")


    centers, labels = kmeans_clustering(X)
    print("KMEANS_CENTERS:", centers)
    print("KMEANS_LABELS_SAMPLE:", labels[:10] if labels is not None else None)
    
if __name__ == "__main__":
    main()

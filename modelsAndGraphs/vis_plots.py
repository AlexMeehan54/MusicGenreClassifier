import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    df = pd.read_csv("csvFiles/train.csv")
    return df


def genre_plot(df):
    plt.figure(figsize=(8,5))
    df.iloc[:, -1].value_counts().plot(kind='bar')
    plt.title("Genre Distribution")
    plt.show()

def technical_features(df):
    tech = ['energy', 'danceability', 'tempo', 'loudness']

    clean = df[tech].apply(pd.to_numeric, errors='coerce')

    clean = clean.dropna()

    clean.hist(figsize=(10,6))
    plt.suptitle("Technical Features Distribution")
    plt.show()

def categorical_features(df):
    cat = ['key', 'mode', 'time_signature']

    clean = df[cat].apply(pd.to_numeric, errors='coerce')

    clean = clean.dropna()

    clean.hist(figsize=(10,6))
    plt.suptitle("Catagoroical Features Distribution")
    plt.show()

def main():
    df = load_data()
    
    genre_plot(df)
    technical_features(df)
    categorical_features(df)

if __name__ == "__main__":
    main()
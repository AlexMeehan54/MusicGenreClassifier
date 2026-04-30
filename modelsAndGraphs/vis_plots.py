import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    df = pd.read_csv("csvFiles/train.csv")
    return df

def genre_plot(df):
    genre_col = df.columns[-1]

    counts = df[genre_col].value_counts()

    plt.figure(figsize=(10,5))

    plt.bar(
        counts.index.astype(str),   
        counts.values,              
        color="#1DB954",
        alpha=0.8
    )

    plt.title("Genre Distribution")
    plt.xlabel("Genre Name")
    plt.ylabel("Number of Songs")

    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

def technical_features(df):
    tech = [
    'Popularity','danceability','energy','loudness',
    'speechiness','acousticness','instrumentalness',
    'liveness','valence','tempo','duration_in min/ms'
]

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
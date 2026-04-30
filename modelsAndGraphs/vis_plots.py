import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    df = pd.read_csv("csvFiles/train.csv")
    return df

def genre_plot(df):
    genre_col = df.columns[-1]

    df = df.copy()

    # convert numeric labels → names
    df[genre_col] = df[genre_col].map({
        0: "Acoustic/Folk",
        1: "Alt Music",
        2: "Blues",
        3: "Bollywood",
        4: "Country",
        5: "HipHop",
        6: "Indie Alt",
        7: "Instrumental",
        8: "Metal",
        9: "Pop",
        10: "Rock"
    })

    counts = df[genre_col].value_counts()

    plt.figure(figsize=(10,5))

    plt.bar(counts.index, counts.values)

    plt.title("Genre Distribution")
    plt.xlabel("Genre")
    plt.ylabel("Number of Songs")

    plt.xticks(rotation=70, ha='right')
    plt.tight_layout()

    plt.show()

def discriptive_features(df):
    tech = [
    'Popularity','danceability','energy','loudness',
    'speechiness','acousticness','instrumentalness',
    'liveness','valence'
]

    clean = df[tech].apply(pd.to_numeric, errors='coerce')

    clean = clean.dropna()

    clean.hist(figsize=(10,6))
    plt.suptitle("Descriptive Features Distribution")
    plt.show()

def technical_features(df):
    cat = ['key', 'mode', 'time_signature','tempo','duration_in min/ms']

    clean = df[cat].apply(pd.to_numeric, errors='coerce')

    clean = clean.dropna()

    clean.hist(figsize=(10,6))
    plt.suptitle("Technical Features Distribution")
    plt.show()

def main():
    df = load_data()

    genre_plot(df)
    discriptive_features(df)
    technical_features(df)

if __name__ == "__main__":
    main()
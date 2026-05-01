#Music Genre Classifier

dataset URL: https://www.kaggle.com/datasets/purumalgi/music-genre-classification/data

Reproducibility
=======
```
Files:
MusicGenreClassifier/
│
├── csvFiles/
│   └── train.csv
├── modelsAndGraphs/
│   └── Genre_FeatureEng.py
    └── models.py
    └── vis_plots.py

Environmental setup:
Libraries:
pandas - 3.0.0 - Load CSV files, Clean and filter data,
numpy - 2.4.2 - Computes the math
matplotlib - 3.10.8 - Plot models, Bar charts and histograms
scikit-learn - 1.8.0 - Train models, Split data, preprocessing and feature scaling, evaluates accuracy
```
Genre_FeatureEng.py
=======
```
This is utilized to load our dataset's statistics and verify that our set is clean before modeling.


DATA EXPLORATION notably loads the data set, checks the first 5 rows and produces a summary of the statistics.
DATA EXPLORATION notes the amount of missing values (6819) in the features


DATA CLEANING checked for any duplicate rows (0)


Z-SCORE STANDARDIZATION scales the data for the PCA RESULTS
Means approx. 0 and Std approx. 1 after scaling


PCA RESULTS analysis on scaled data
INTERPRETATION displays the top 7 features for PC1
energy                0.526484
loudness              0.507857
acousticness          0.489495
instrumentalness      0.232769
valence               0.199577
duration_in min/ms    0.189669
tempo                 0.159341


While executing Genre_FeatureEng.py you should see
Expected output summary:
-Printed datashape
-Any Missing Values
-Duplicate rows
-Means approx. 0 and Std approx. 1 after scaling
-Top 7 PC1 features
```

models.py
=======
__________________________________________
```
Music genre classification system using three machine learning models:
Models          Key Parameters
-Decision Tree  max_depth=6, min_samples_split=10, min_samples_leaf=5
-Random Forest  n_estimators=500, max_depth=12, max_features='sqrt'
-SVM (RBF)      kernel='rbf'


Preprocessing
-Drops non-numeric columns (Artist Name, Track Name)
-Removes rows with missing genre labels


Split/Test
Standardization is required for SVM
70% training / 30% testing
random_state = 42


Missing values are replaced with the mean of their column using SimpleImputer
Features are scaled using StandardScaler (Important to SVM)


imports permutation_importance from sklearn for feature importance
separate importance loaders for the tree models and SVM.
-Both of the tree models run through the function show_tree_importance while SVM runs through show_svm_importance
Results of feature importance:


```
```
Feature Importance (Tree-Based Model):
duration_in min/ms: 0.1942
instrumentalness: 0.1824
acousticness: 0.1751
speechiness: 0.1506
energy: 0.0911
danceability: 0.0604
Popularity: 0.0569
valence: 0.0567
loudness: 0.0201
liveness: 0.0108
tempo: 0.0017
time_signature: 0.0000
mode: 0.0000
key: 0.0000

Feature Importance (Tree-Based Model):
duration_in min/ms: 0.1566
speechiness: 0.1204
acousticness: 0.1201
energy: 0.0926
danceability: 0.0908
instrumentalness: 0.0880
Popularity: 0.0781
valence: 0.0719
loudness: 0.0693
tempo: 0.0407
liveness: 0.0395
key: 0.0199
mode: 0.0082
time_signature: 0.0038

Feature Importance (SVM Permutation):
duration_in min/ms: 0.1299
energy: 0.0680
danceability: 0.0668
speechiness: 0.0643
acousticness: 0.0504
valence: 0.0491
instrumentalness: 0.0457
Popularity: 0.0405
loudness: 0.0343
liveness: 0.0125
mode: 0.0092
time_signature: 0.0065
key: 0.0061
tempo: 0.0037
```


```
While executing models.py you should see:
=======
Expected output summary:
-The accuracy scores of each model (Takes a long time to load with SVM)
-Visual comparison graph of each model and their accuracy percentages
-Each feature rated on it importance to each model (Takes a very long time to load with SVM)
```


vis_plots.py
=======
__________________________________________
```
Genre distribution chart
converts numeric genre labels (0-10) into readable genre names
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
Counts how many songs belong to each genre
Displays a bar graph of how many songs are in each genre


Histogram of descriptive features
    'Popularity','danceability','energy','loudness',
    'speechiness','acousticness','instrumentalness',
    'liveness','valence'
shows distribution of feature spread

Histogram of technical features
    'key', 'mode', 'time_signature','tempo','duration_in min/ms'
shows distribution of feature spread


While executing models.py you should see:
-Bar Chart titled "Genre Distribution"
=======
Expexted output summary:
-Bar Chart titled "Genre Distritubtion"
-Histograms titled "Descriptive Features Distribution" and "Technical Features Distribution"
```




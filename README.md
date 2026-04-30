#Music Genre Classifier

dataset URL: https://www.kaggle.com/datasets/purumalgi/music-genre-classification/data

Music genre classification system using three machine learning models:
Models	        Key Parameters
-Decision Tree	max_depth=6, min_samples_split=10, min_samples_leaf=5
-Random Forest	n_estimators=500, max_depth=12, max_features='sqrt'
-SVM (RBF)	    kernel='rbf'

Preprocessing
-Drops non-numeric columns (Artist Name, Track Name)
-Removes rows with missing genre labels

Split/Test
70% training / 30% testing
random_state = 42

Missing values are replaced with the mean of their column using SimpleImputer
Features are scaled using StandardScaler (Important to SVM)

Output:
Models are evaluated using accuracy
A bar graph compares the performance of the three models

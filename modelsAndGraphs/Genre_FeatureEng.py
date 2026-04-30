# Homework: Feature Engineering and PCA
# Student Template with Helpful Hints
# Complete all TODO sections

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the dataset
df = pd.read_csv('csvFiles/train.csv')

# ============================================================
# TASK 1: Data Loading and Exploration (15 points)
# ============================================================
# HINT 1: Use df.describe() to see summary statistics (mean, min, max, etc.)
# HINT 2: Use df.isnull().sum() to count missing values
# HINT 3: Use df.isnull().sum().sum() to get total missing values
# HINT 4: Print with labels so output is clear

print("=== 1. DATA EXPLORATION ===")
print("\n✅ Dataset loaded successfully!")
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Column names: {list(df.columns)}")
print("\nDEBUG: Column names:")
print(df.columns)

# Check first few rows
print("\n✅ First 5 rows:")
print(df.head())

# TODO: Print summary statistics using df.describe()
print("\n✅ Summary Statistics:")
print(df.describe())

# TODO: Print total number of missing values
print("\n✅ Missing Values Check:")
print(df.isnull().sum())
print(f"Missing values: {df.isnull().sum().sum()}")
# Expected output: Missing values: 0


# ============================================================
# TASK 2: Data Cleaning (15 points)
# ============================================================

print("\n=== 2. DATA CLEANING ===")

# TODO: List the columns that need zero replacement
cols = ['Popularity','danceability','energy','key','loudness','mode',
        'speechiness','acousticness','instrumentalness','liveness',
        'valence','tempo','duration_in min/ms','time_signature']

for col in cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

for col in cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

for col in cols:
    if col in df.columns:
        df[col] = df[col].replace(0, df[col].median())

# TODO: Remove duplicate rows
print(f"Before removing duplicates: {len(df)} rows")
df = df.drop_duplicates()
print(f"After removing duplicates: {len(df)} rows")

# TODO: Print confirmation message
print("Zeros replaced with median, duplicates removed")


# ============================================================
# TASK 3: Z-Score Standardization (20 points)
# ============================================================
# HINT 1: Separate features (X) from target (y)
#        X = all columns except 'Outcome'
#        y = only the 'Outcome' column
# HINT 2: Use StandardScaler() from sklearn
# HINT 3: Use fit_transform() on X
# HINT 4: After scaling, mean should be ~0 and std ~1
# HINT 5: Use .mean().round(2) and .std().round(2) to check

print("\n=== 3. Z-SCORE STANDARDIZATION ===")
# TODO: Separate features (X) and target (y)
X = df.drop('Class', axis=1)  # All columns except Outcome
y = df['Class']  # Only Outcome column
X = X.select_dtypes(include=[np.number])

# TODO: Create scaler object
scaler = StandardScaler()

# TODO: Fit and transform the features
X_scaled = scaler.fit_transform(X)

# TODO: Print mean and std after scaling (should be 0 and 1)
print("Mean after scaling:", np.mean(X_scaled, axis=0).round(2))
print("Std after scaling:", np.std(X_scaled, axis=0).round(2))

# TODO: Print advantage of standardization
print("Advantage: All features now on same scale (mean=0, std=1)")

# ============================================================
# TASK 5: PCA (25 points)
# ============================================================
# HINT 1: Apply PCA on SCALED data (X_scaled, NOT original X)
# HINT 2: First use PCA() without n_components to see all components
# HINT 3: Use explained_variance_ratio_ to get variance for each PC
# HINT 4: Print first 5 components with percentages
# HINT 5: Then use PCA(n_components=2) for 2D visualization summary
# HINT 6: Sum the explained variance ratio for PC1 and PC2

print("\n=== 5. PCA RESULTS ===")

# TODO: Apply PCA with all components
pca = PCA()
pca.fit(X_scaled)

# Explained variance ratio
var_ratio = pca.explained_variance_ratio_
cumulative = np.cumsum(var_ratio)

print("\n✅ Explained Variance by each component:")
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

# TODO: Print explained variance ratio for first 5 components
print("Explained variance ratio:")
# Loop through first 5 components and print each
for i, var in enumerate(pca.explained_variance_ratio_[:5], 1):
    print(f"  PC{i}: {var:.3f} ({var*100:.1f}%)")

# TODO: Apply PCA with 2 components
pca_2d = PCA(n_components=2)
X_pca = pca_2d.fit_transform(X_scaled)

# TODO: Print total variance explained by PC1 + PC2
print(f"\nPC1+PC2 explain {pca_2d.explained_variance_ratio_.sum():.3f} ({pca_2d.explained_variance_ratio_.sum()*100:.1f}%) of variance")


# ============================================================
# TASK 6: Interpretation (10 points)
# ============================================================
# HINT 1: PCA components are stored in pca_2d.components_
# HINT 2: Transpose to get feature loadings: .T
# HINT 3: Create DataFrame with columns=['PC1','PC2'] and index=X.columns
# HINT 4: Use .abs() to get absolute values, then .sort_values(ascending=False)
# HINT 5: Use .head(3) to get top 3 features for each PC
# HINT 6: Print your conclusion about important features

print("\n=== 6. INTERPRETATION ===")

# TODO: Create loadings DataFrame
loadings = pd.DataFrame(
    pca_2d.components_.T,
    columns=['PC1', 'PC2'],
    index=X.columns
)

# TODO: Print top 3 features for PC1
print("Top 7 features for PC1:")
print(loadings['PC1'].abs().sort_values(ascending=False).head(7))
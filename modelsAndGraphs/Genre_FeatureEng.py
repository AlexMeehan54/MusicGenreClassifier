# Derived from, Homework: Feature Engineering and PCA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the dataset
df = pd.read_csv('csvFiles/train.csv')

print("=== 1. DATA EXPLORATION ===")
print("\n✅ Dataset loaded successfully!")
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Column names: {list(df.columns)}")
print("\nDEBUG: Column names:")
print(df.columns)

# Check first few rows
print("\n✅ First 5 rows:")
print(df.head())

Print summary statistics using df.describe()
print("\n✅ Summary Statistics:")
print(df.describe())

Print total number of missing values
print("\n✅ Missing Values Check:")
print(df.isnull().sum())
print(f"Missing values: {df.isnull().sum().sum()}")

print("\n=== 2. DATA CLEANING ===")

# List the columns that need zero replacement
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

# Remove duplicate rows
print(f"Before removing duplicates: {len(df)} rows")
df = df.drop_duplicates()
print(f"After removing duplicates: {len(df)} rows")

# Print confirmation message
print("Zeros replaced with median, duplicates removed")

print("\n=== 3. Z-SCORE STANDARDIZATION ===")
# TODO: Separate features (X) and target (y)
X = df.drop('Class', axis=1)  # All columns except Outcome
y = df['Class']  # Only Outcome column
X = X.select_dtypes(include=[np.number])

# Create scaler object
scaler = StandardScaler()

# Fit and transform the features
X_scaled = scaler.fit_transform(X)

# Print mean and std after scaling (should be 0 and 1)
print("Mean after scaling:", np.mean(X_scaled, axis=0).round(2))
print("Std after scaling:", np.std(X_scaled, axis=0).round(2))

# Print advantage of standardization
print("Advantage: All features now on same scale (mean=0, std=1)")

print("\n=== 5. PCA RESULTS ===")

# Apply PCA with all components
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

# Print explained variance ratio for first 5 components
print("Explained variance ratio:")
# Loop through first 5 components and print each
for i, var in enumerate(pca.explained_variance_ratio_[:5], 1):
    print(f"  PC{i}: {var:.3f} ({var*100:.1f}%)")

# Apply PCA with 2 components
pca_2d = PCA(n_components=2)
X_pca = pca_2d.fit_transform(X_scaled)

# Print total variance explained by PC1 + PC2
print(f"\nPC1+PC2 explain {pca_2d.explained_variance_ratio_.sum():.3f} ({pca_2d.explained_variance_ratio_.sum()*100:.1f}%) of variance")

print("\n=== 6. INTERPRETATION ===")

# TODO: Create loadings DataFrame
loadings = pd.DataFrame(
    pca_2d.components_.T,
    columns=['PC1', 'PC2'],
    index=X.columns
)

# Print top features for PC1
print("Top 7 features for PC1:")
print(loadings['PC1'].abs().sort_values(ascending=False).head(7))

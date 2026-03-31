import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv("final_nonredundant_checked.csv")
print("Original shape:", df.shape)

# Remove only 1st column
df_reduced = df.iloc[:, 1:].copy()
print("After removing 1st column:", df_reduced.shape)

# Keep only numeric columns
df_numeric = df_reduced.select_dtypes(include=["int64", "float64"]).copy()
print("Numeric feature shape:", df_numeric.shape)

# Scale data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df_numeric)

# Tanimoto similarity
def tanimoto_similarity(a, b):
    numerator = np.dot(a, b)
    denominator = np.dot(a, a) + np.dot(b, b) - numerator
    if denominator == 0:
        return 0
    return numerator / denominator

# Filter function
def filter_by_tanimoto(X, threshold):
    selected_indices = []
    for i in range(len(X)):
        keep = True
        for j in selected_indices:
            sim = tanimoto_similarity(X[i], X[j])
            if sim > threshold:
                keep = False
                break
        if keep:
            selected_indices.append(i)
    return selected_indices

# Try many thresholds
thresholds = [0.999, 0.995, 0.99, 0.985, 0.98, 0.975, 0.97, 0.965, 0.96, 0.955, 0.95, 0.945, 0.94, 0.935, 0.93, 0.925, 0.92, 0.915, 0.91, 0.905, 0.90]

results = []

for threshold in thresholds:
    selected_indices = filter_by_tanimoto(X_scaled, threshold)
    count = len(selected_indices)
    results.append((threshold, count, selected_indices))
    print(f"Threshold {threshold} -> {count} samples kept")

# Find result closest to 500
best_threshold = None
best_indices = None
best_count = None
best_diff = float("inf")

for threshold, count, selected_indices in results:
    diff = abs(count - 500)
    if diff < best_diff:
        best_diff = diff
        best_threshold = threshold
        best_indices = selected_indices
        best_count = count

df_filtered = df_reduced.iloc[best_indices].copy()

print("\nChosen threshold:", best_threshold)
print("Samples after redundancy removal:", best_count)

# Make exactly 500 if more than 500
if len(df_filtered) > 500:
    df_final = df_filtered.sample(n=500, random_state=42)
else:
    df_final = df_filtered

print("Final dataset shape:", df_final.shape)

# Save files
df_filtered.to_csv("nonredundant_removed_1col.csv", index=False)
df_final.to_csv("final_500samples_removed_1col.csv", index=False)

print("\nSaved files:")
print("1. nonredundant_removed_1col.csv")
print("2. final_500samples_removed_1col.csv")

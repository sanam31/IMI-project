import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

INPUT = "data/processed/final_30_features.csv"
OUTPUT = "data/processed/final_nonredundant_30features.csv"

df = pd.read_csv(INPUT)

# keep only numeric columns for Tanimoto
numeric_df = df.select_dtypes(include=[np.number]).copy()

# remove target from similarity check
X = numeric_df.drop(columns=["target_cap_future"], errors="ignore")

X_scaled = MinMaxScaler().fit_transform(X)

def tanimoto(a, b):
    num = np.dot(a, b)
    den = np.dot(a, a) + np.dot(b, b) - num
    return num / den if den != 0 else 0

def prune(threshold):
    keep = []
    removed = set()

    for i in range(len(X_scaled)):
        if i in removed:
            continue
        keep.append(i)
        for j in range(i + 1, len(X_scaled)):
            if j in removed:
                continue
            if tanimoto(X_scaled[i], X_scaled[j]) > threshold:
                removed.add(j)
    return keep

for th in [0.95, 0.97, 0.98, 0.99]:
    keep = prune(th)
    print(f"Threshold {th}: kept {len(keep)} samples")

chosen = 0.99
keep = prune(chosen)

df_final = df.iloc[keep].reset_index(drop=True)
df_final.to_csv(OUTPUT, index=False)

print("Chosen threshold:", chosen)
print("Final shape:", df_final.shape)
print(df_final.head())

import pandas as pd
import numpy as np
import os
from pymatgen.core import Composition
from sklearn.preprocessing import MinMaxScaler

INPUT_FILE = "data/external/mp_compounds.csv"
OUTPUT_FILE = "data/processed/code1_atomic_10_features.csv"
SIM_THRESHOLD = 0.95

df = pd.read_csv(INPUT_FILE)

def tanimoto_similarity(a, b):
    num = np.dot(a, b)
    den = np.dot(a, a) + np.dot(b, b) - num
    return num / den if den != 0 else 0

def extract_atomic_features(formula):
    comp = Composition(formula)
    elems = comp.elements

    atomic_mass = [float(e.atomic_mass) for e in elems]
    en = [e.X for e in elems if e.X is not None]
    Z = [e.Z for e in elems]

    return pd.Series({
        "avg_atomic_mass": np.mean(atomic_mass),
        "max_atomic_mass": np.max(atomic_mass),
        "min_atomic_mass": np.min(atomic_mass),
        "range_atomic_mass": np.ptp(atomic_mass),
        "std_atomic_mass": np.std(atomic_mass),
        "avg_electronegativity": np.mean(en),
        "max_electronegativity": np.max(en),
        "min_electronegativity": np.min(en),
        "range_electronegativity": np.ptp(en),
        "avg_atomic_number": np.mean(Z)
    })

features = df["formula"].apply(extract_atomic_features)
result = pd.concat([df[["sample_id", "formula"]], features], axis=1)

os.makedirs("data/processed", exist_ok=True)
result.to_csv(OUTPUT_FILE, index=False)

# redundancy check
X = result.drop(columns=["sample_id", "formula"]).dropna()
X_scaled = MinMaxScaler().fit_transform(X)

similar_pairs = []
n = len(X_scaled)

for i in range(n):
    for j in range(i + 1, n):
        sim = tanimoto_similarity(X_scaled[i], X_scaled[j])
        if sim > SIM_THRESHOLD:
            similar_pairs.append((i, j, sim))

print("Saved:", OUTPUT_FILE)
print("Shape:", result.shape)
print("\nExtracted features:")
for col in result.columns[2:]:
    print("-", col)

print("\nRedundancy check using Tanimoto similarity")
print("Threshold:", SIM_THRESHOLD)
print("Highly similar pairs:", len(similar_pairs))
print("First 10 similar pairs:", similar_pairs[:10])

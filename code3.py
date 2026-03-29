import pandas as pd
import numpy as np
import os
from pymatgen.core import Composition
from sklearn.preprocessing import MinMaxScaler

INPUT_FILE = "data/external/mp_compounds.csv"
OUTPUT_FILE = "data/processed/code3_structural_10_features.csv"
SIM_THRESHOLD = 0.95

df = pd.read_csv(INPUT_FILE)

def tanimoto_similarity(a, b):
    num = np.dot(a, b)
    den = np.dot(a, a) + np.dot(b, b) - num
    return num / den if den != 0 else 0

def extract_structural_features(row):
    formula = row["formula"]
    density = row["density"]
    num_atoms = row["num_atoms"]
    space_group_number = row["space_group_number"]

    comp = Composition(formula)
    elems = comp.elements

    atomic_masses = [float(e.atomic_mass) for e in elems]
    atomic_numbers = [e.Z for e in elems]
    electronegativities = [e.X for e in elems if e.X is not None]

    total_mass = sum(atomic_masses)
    mean_Z = np.mean(atomic_numbers)
    mean_en = np.mean(electronegativities)

    return pd.Series({
        "density_feature": density,
        "num_atoms_feature": num_atoms,
        "space_group_number_feature": space_group_number,
        "mass_per_atom": total_mass / comp.num_atoms,
        "atomic_number_per_atom": sum(atomic_numbers) / comp.num_atoms,
        "density_per_atom": density / num_atoms if num_atoms != 0 else 0,
        "mass_density_product": total_mass * density,
        "mass_density_ratio": total_mass / density if density != 0 else 0,
        "spacegroup_per_atom": space_group_number / num_atoms if num_atoms != 0 else 0,
        "mean_en_times_density": mean_en * density
    })

features = df.apply(extract_structural_features, axis=1)
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

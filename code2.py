import pandas as pd
import numpy as np
import os
from pymatgen.core import Composition
from sklearn.preprocessing import MinMaxScaler

INPUT_FILE = "data/external/mp_compounds.csv"
OUTPUT_FILE = "data/processed/code2_composition_10_features.csv"
SIM_THRESHOLD = 0.95

df = pd.read_csv(INPUT_FILE)

transition_metals = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd"
}

def tanimoto_similarity(a, b):
    num = np.dot(a, b)
    den = np.dot(a, a) + np.dot(b, b) - num
    return num / den if den != 0 else 0

def extract_composition_features(formula):
    comp = Composition(formula)
    elems = comp.elements
    fractions = [comp.get_atomic_fraction(el.symbol) for el in elems]

    frac_li = comp.get_atomic_fraction("Li") if "Li" in comp else 0
    frac_o = comp.get_atomic_fraction("O") if "O" in comp else 0
    frac_transition = sum(
        comp.get_atomic_fraction(el.symbol)
        for el in elems if el.symbol in transition_metals
    )

    return pd.Series({
        "num_elements": len(elems),
        "total_atoms_in_formula": comp.num_atoms,
        "fraction_Li": frac_li,
        "fraction_O": frac_o,
        "fraction_transition_metal": frac_transition,
        "max_atomic_fraction": np.max(fractions),
        "min_atomic_fraction": np.min(fractions),
        "range_atomic_fraction": np.ptp(fractions),
        "std_atomic_fraction": np.std(fractions),
        "Li_to_O_ratio": frac_li / frac_o if frac_o != 0 else 0
    })

features = df["formula"].apply(extract_composition_features)
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

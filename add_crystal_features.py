import pandas as pd
import numpy as np
import os

INPUT = "data/processed/nasa_with_material.csv"
OUTPUT = "data/processed/dataset_with_crystal.csv"

df = pd.read_csv(INPUT)

material_map = {
    "LCO": {
        "density": 5.1,
        "space_group_number": 166,
        "avg_atomic_mass": 24.2,
        "avg_electronegativity": 1.88
    },
    "LFP": {
        "density": 3.6,
        "space_group_number": 62,
        "avg_atomic_mass": 23.1,
        "avg_electronegativity": 1.82
    },
    "NMC": {
        "density": 4.8,
        "space_group_number": 166,
        "avg_atomic_mass": 25.5,
        "avg_electronegativity": 1.90
    }
}

for col in ["density", "space_group_number", "avg_atomic_mass", "avg_electronegativity"]:
    df[col] = df["material_label"].map(lambda m: material_map.get(m, {}).get(col, np.nan))

os.makedirs("data/processed", exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("Added crystal features:", df.shape)
print(df.head())

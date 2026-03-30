import pandas as pd
import numpy as np
import os

INPUT = "data/processed/dataset_with_fingerprints.csv"
OUTPUT = "data/processed/final_30_features.csv"

df = pd.read_csv(INPUT)

# Derived features
df["cap_diff_1_3"] = df["cap_1"] - df["cap_3"]
df["cap_ratio_1_3"] = df["cap_3"] / (df["cap_1"] + 1e-9)
df["density_per_atom"] = df["density"] / (df["cap_1"].abs() + 1)
df["mass_density_product"] = df["avg_atomic_mass"] * df["density"]
df["en_density_product"] = df["avg_electronegativity"] * df["density"]
df["temp_scaled"] = df["temp_mean"] / (df["temp_mean"].max() + 1e-9)
df["stability_index"] = df["soh_10"] / (abs(df["slope_cap"]) + 1e-6)
df["energy_like"] = df["avg_atomic_mass"] * df["avg_electronegativity"]
df["structure_factor"] = df["space_group_number"] / (df["cap_1"].abs() + 1)

# Extra statistical features
df["cap_range"] = df["cap_1"] - df["cap_10"]
df["soh_decay"] = df["soh_1"] - df["soh_10"]
df["cap_std"] = df[["cap_1", "cap_2", "cap_3", "cap_5", "cap_10"]].std(axis=1)
df["cap_mean"] = df[["cap_1", "cap_2", "cap_3", "cap_5", "cap_10"]].mean(axis=1)
df["thermal_stability"] = df["temp_mean"] * df["density"]

# 30 core features
features = [
    "cap_1", "cap_2", "cap_3", "cap_5", "cap_10",
    "soh_1", "soh_5", "soh_10",
    "temp_mean", "voltage_mean",
    "slope_cap", "slope_soh",
    "density", "space_group_number",
    "avg_atomic_mass", "avg_electronegativity",
    "cap_diff_1_3", "cap_ratio_1_3",
    "density_per_atom", "mass_density_product",
    "en_density_product", "temp_scaled",
    "stability_index", "energy_like",
    "structure_factor",
    "cap_range", "soh_decay", "cap_std", "cap_mean", "thermal_stability"
]

extra_cols = ["formula", "compound_name"]
fp_cols = [col for col in df.columns if col.startswith("fp_")]

final = df[["sample_id"] + extra_cols + features + fp_cols + ["target_cap_future"]].copy()

os.makedirs("data/processed", exist_ok=True)
final.to_csv(OUTPUT, index=False)

print("Final dataset shape:", final.shape)
print("Core features:", len(features))
print("Fingerprint features:", len(fp_cols))
print(final.head())

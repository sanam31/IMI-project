import pandas as pd
import os

nasa = pd.read_csv("data/processed/nasa_battery_level_features.csv")
mapping = pd.read_csv("nasa_material_map.csv")

df = nasa.merge(mapping, on="sample_id", how="left")

os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/nasa_with_material.csv", index=False)

print("Merged shape:", df.shape)
print(df.head())

import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/dataset_with_crystal.csv")

augmented = []

for _, row in df.iterrows():
    for i in range(30):
        new = row.copy()

        for col in df.columns:
            if col not in ["sample_id", "formula", "target_cap_future"]:
                if pd.api.types.is_numeric_dtype(type(new[col])):
                    noise = np.random.uniform(-0.05, 0.05)
                    new[col] = new[col] * (1 + noise)

        new["sample_id"] = str(row["sample_id"]) + f"_aug{i}"
        augmented.append(new)

df_big = pd.concat([df, pd.DataFrame(augmented)], ignore_index=True)
df_big.to_csv("data/processed/expanded_dataset.csv", index=False)

print("Expanded shape:", df_big.shape)
print(df_big.head())

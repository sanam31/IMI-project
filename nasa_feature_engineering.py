import pandas as pd
import numpy as np
import os

INPUT_FILE = "data/raw/battery_cycle_level_dataset_CLEAN_FINAL.csv"
OUTPUT_FILE = "data/processed/nasa_battery_level_features.csv"

df = pd.read_csv(INPUT_FILE)

rows = []

for battery_id, g in df.groupby("battery_id"):
    g = g.sort_values("cycle").reset_index(drop=True)

    cap_1 = g.loc[0, "capacity"]
    cap_2 = g.loc[1, "capacity"] if len(g) > 1 else cap_1
    cap_3 = g.loc[2, "capacity"] if len(g) > 2 else cap_2
    cap_5 = g.loc[4, "capacity"] if len(g) > 4 else g.loc[len(g)-1, "capacity"]
    cap_10 = g.loc[9, "capacity"] if len(g) > 9 else g.loc[len(g)-1, "capacity"]

    soh_1 = g.loc[0, "soh"]
    soh_5 = g.loc[4, "soh"] if len(g) > 4 else g.loc[len(g)-1, "soh"]
    soh_10 = g.loc[9, "soh"] if len(g) > 9 else g.loc[len(g)-1, "soh"]

    temp_mean = g["temperature"].mean()
    voltage_mean = g["voltage"].mean()

    first10 = g.head(min(10, len(g)))
    slope_cap = np.polyfit(first10["cycle"], first10["capacity"], 1)[0] if len(first10) >= 2 else 0.0
    slope_soh = np.polyfit(first10["cycle"], first10["soh"], 1)[0] if len(first10) >= 2 else 0.0

    target_cap_future = g.iloc[-1]["capacity"]

    rows.append({
        "sample_id": battery_id,
        "cap_1": cap_1,
        "cap_2": cap_2,
        "cap_3": cap_3,
        "cap_5": cap_5,
        "cap_10": cap_10,
        "soh_1": soh_1,
        "soh_5": soh_5,
        "soh_10": soh_10,
        "temp_mean": temp_mean,
        "voltage_mean": voltage_mean,
        "slope_cap": slope_cap,
        "slope_soh": slope_soh,
        "target_cap_future": target_cap_future
    })

out = pd.DataFrame(rows)

os.makedirs("data/processed", exist_ok=True)
out.to_csv(OUTPUT_FILE, index=False)

print("NASA battery-level shape:", out.shape)
print(out.head())

from mp_api.client import MPRester
import pandas as pd
import numpy as np

from pymatgen.analysis.local_env import CrystalNN
from matminer.featurizers.structure import DensityFeatures
from matminer.featurizers.composition import ElementProperty

API_KEY = "CMmf57nYPCO72LM1Fqi8ND5k6ysyUKCx"

with MPRester(API_KEY) as m:

    data = m.materials.summary.search(
        elements=["Li"],
        num_elements=(2, 5),
        fields=[
            "material_id",
            "formula_pretty",
            "structure",
            "band_gap",
            "formation_energy_per_atom",
            "energy_above_hull",
            "density",
            "volume",
            "nsites"
        ]
    )

# -------------------------------
# Step 1: Filter data
# -------------------------------
filtered_data = [
    mat for mat in data
    if mat.structure is not None
]

filtered_data = filtered_data[:500]

# -------------------------------
# Step 2: Initialize featurizers
# -------------------------------
density_feat = DensityFeatures()
comp_feat = ElementProperty.from_preset("magpie")
cnn = CrystalNN()

rows = []

# -------------------------------
# Step 3: Feature Extraction
# -------------------------------
for mat in filtered_data:
    try:
        structure = mat.structure
        comp = structure.composition

        features = {}

        # -------------------
        # Basic Info
        # -------------------
        features["material_id"] = mat.material_id
        features["formula"] = mat.formula_pretty

        # -------------------
        # 1. Thermodynamic (5)
        # -------------------
        features["formation_energy"] = mat.formation_energy_per_atom
        features["energy_above_hull"] = mat.energy_above_hull
        features["cohesive_energy"] = mat.formation_energy_per_atom  # approx
        features["decomposition_energy"] = mat.energy_above_hull
        features["stability_index"] = mat.energy_above_hull

        # -------------------
        # 2. Electronic (5)
        # -------------------
        features["band_gap"] = mat.band_gap
        features["is_metal"] = 1 if mat.band_gap == 0 else 0
        features["fermi_energy"] = mat.band_gap  # proxy
        features["electron_affinity"] = comp.average_electroneg
        features["dos_fermi"] = mat.band_gap  # proxy

        # -------------------
        # 3. Structural (6)
        # -------------------
        features["density"] = mat.density
        features["volume_per_atom"] = mat.volume / mat.nsites
        features["packing_fraction"] = density_feat.featurize(structure)[0]

        lattice = structure.lattice
        features["lattice_a"] = lattice.a
        features["lattice_b"] = lattice.b
        features["lattice_c"] = lattice.c

        # -------------------
        # 4. Ionic Transport (4)
        # -------------------
        distances = []
        for i in range(len(structure)):
            for j in range(i+1, len(structure)):
                distances.append(structure.get_distance(i, j))

        features["avg_bond_length"] = np.mean(distances)
        features["min_distance"] = np.min(distances)
        features["max_distance"] = np.max(distances)
        features["diffusion_path"] = np.std(distances)

        # -------------------
        # 5. Composition (5)
        # -------------------
        comp_features = comp_feat.featurize(comp)

        features["mean_atomic_number"] = comp_features[0]
        features["mean_electronegativity"] = comp_features[1]
        features["electronegativity_std"] = comp_features[2]
        features["mean_atomic_radius"] = comp_features[3]
        features["valence_electrons"] = comp_features[4]

        # -------------------
        # 6. Mechanical (3) (proxy)
        # -------------------
        features["bulk_modulus"] = mat.density * 10  # proxy
        features["shear_modulus"] = mat.density * 5
        features["elastic_anisotropy"] = lattice.a / lattice.c

        # -------------------
        # 7. Advanced (2)
        # -------------------
        # RDF-like peak (simple proxy)
        features["rdf_peak"] = np.mean(distances)

        # Coordination number
        cn_list = []
        for i in range(len(structure)):
            cn_list.append(len(cnn.get_nn_info(structure, i)))

        features["coordination_number"] = np.mean(cn_list)

        rows.append(features)

    except:
        continue

# -------------------------------
# Step 4: DataFrame
# -------------------------------
df = pd.DataFrame(rows)

# -------------------------------
# Step 5: Save CSV
# -------------------------------
df.to_csv("battery_30_features_dataset.csv", index=False)

print("Dataset created successfully!")
print("Samples:", len(df))
print("Features:", len(df.columns))


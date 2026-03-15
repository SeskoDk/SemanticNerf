from pathlib import Path
import numpy as np

# ---------------------------------------------------
# DATASET CONFIGURATION
# ---------------------------------------------------

datasets = [
    {
        "name": "155160",
        "volume_file": "results/wheat_155160/volume/volume_semantic_red.npz",
        "lab_mass_g": 563.16
    },
    {
        "name": "155386",
        "volume_file": "results/wheat_155386/volume/volume_semantic_red.npz",
        "lab_mass_g": 732.04
    },
    {
        "name": "431983",
        "volume_file": "results/wheat_431983/volume/volume_semantic_red.npz",
        "lab_mass_g": 328.75
    },
    {
        "name": "716024",
        "volume_file": "results/wheat_716024/volume/volume_semantic_red.npz",
        "lab_mass_g": 607.80
    },
]

# voxel resolution used during volume extraction
RESOLUTION = 512


# ---------------------------------------------------
# COMPUTE PLANT VOLUME FROM VOXELS
# ---------------------------------------------------

def compute_volume(points, resolution):

    occupied_voxels = len(points)

    total_voxels = resolution ** 3

    voxel_volume = 1.0 / total_voxels  # ROI = 1 m³

    plant_volume = occupied_voxels * voxel_volume

    return plant_volume


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

def main():

    volumes = []
    masses = []

    print("\n--- Computing reconstructed plant volumes ---\n")

    for d in datasets:

        volume = np.load(d["volume_file"])
        points = volume["points"]

        V = compute_volume(points, RESOLUTION)
        M = d["lab_mass_g"] / 1000.0  # convert to kg

        volumes.append(V)
        masses.append(M)

        print(f"{d['name']}")
        print(f"points: {len(points):,}")
        print(f"plant volume: {V:.6f} m³")
        print(f"lab biomass: {M:.4f} kg")
        print()

    volumes = np.array(volumes)
    masses = np.array(masses)

    # ---------------------------------------------------
    # FIT GLOBAL DENSITY
    # ---------------------------------------------------

    rho = np.sum(volumes * masses) / np.sum(volumes ** 2)

    print("\n--- Calibrated density ---")
    print(f"rho = {rho:.2f} kg/m³")

    # ---------------------------------------------------
    # PREDICT BIOMASS
    # ---------------------------------------------------

    print("\n--- Biomass prediction ---\n")

    pred = rho * volumes

    for i, d in enumerate(datasets):

        error = pred[i] - masses[i]
        rel_error = abs(error) / masses[i] * 100

        print(d["name"])
        print(f"predicted biomass: {pred[i]*1000:.2f} g")
        print(f"lab biomass: {masses[i]*1000:.2f} g")
        print(f"error: {error*1000:.2f} g ({rel_error:.2f} %)")
        print()

    mae = np.mean(np.abs(pred - masses))
    rmse = np.sqrt(np.mean((pred - masses) ** 2))

    print("\n--- Overall error ---")
    print(f"MAE:  {mae*1000:.2f} g")
    print(f"RMSE: {rmse*1000:.2f} g")


if __name__ == "__main__":
    main()
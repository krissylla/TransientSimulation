# This script is for the full survey of 91_139_849 sources

from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# Paths
# ============================================================

project_dir = Path(__file__).resolve().parent
input_dir = project_dir / "results_5"
output_file = project_dir / "aggregated_results_with_bands.npz"

# ============================================================
# Configuration
# ============================================================

# Redshift bins
z_edges = np.linspace(0.0, 1.0, 101)
z_mid = 0.5 * (z_edges[:-1] + z_edges[1:])

lsst_bands = ["lsstu", "lsstg", "lsstr", "lssti", "lsstz", "lssty"]

# ============================================================
# Find input files
# ============================================================

files = sorted(input_dir.glob("detected_N*.parquet"))

###
expected_files = 456

if len(files) != expected_files:
    raise RuntimeError(
        f"Expected {expected_files} files, found {len(files)}"
    )
###

if not files:
    raise FileNotFoundError(
        f"No parquet files found in {input_dir}"
    )

print(f"Input directory: {input_dir}")
print(f"Found {len(files)} parquet files.")

# ============================================================
# Accumulators
# ============================================================

# Number of all simulated sources in each redshift bin
n_sources_all = np.zeros(len(z_mid), dtype=np.int64)

# Number of detected sources in each redshift bin
n_sources_detected = np.zeros(len(z_mid), dtype=np.int64)

# Number of 5-sigma detections in each band

n_band_detections = {
    band: np.zeros(len(z_mid), dtype=np.int64)
    for band in lsst_bands
}

# Total number of rows processed
total_rows = 0

# Total number of sources passing detection/selection
total_detected = 0

# ============================================================
# Process one Parquet at a time
# ============================================================

for i, file in enumerate(files, start=1):

    print(f"[{i}/{len(files)}] {file.name}")

    # Only read the columns we actually need
    columns = [
        "z",
        "detected",
        *lsst_bands,
    ]

    df = pd.read_parquet(
        file,
        columns=columns,
    )

    total_rows += len(df)

    # --------------------------------------------------------
    # All simulated sources
    # --------------------------------------------------------

    z_all = df["z"].to_numpy()

    hist_all, _ = np.histogram(
        z_all,
        bins=z_edges,
    )

    n_sources_all += hist_all


    # --------------------------------------------------------
    # Detected sources
    # --------------------------------------------------------

    detected = df["detected"].to_numpy(dtype=bool)

    total_detected += detected.sum()

    z_detected = z_all[detected]

    hist_detected, _ = np.histogram(
        z_detected,
        bins=z_edges,
    )

    n_sources_detected += hist_detected


    # --------------------------------------------------------
    # Detections by band
    # --------------------------------------------------------

    for band in lsst_bands:

        z_band = z_all[detected]
        detections_band = df.loc[detected, band].to_numpy()

        hist_band, _ = np.histogram(z_band,bins=z_edges,weights=detections_band)
        n_band_detections[band] += hist_band.astype(np.int64)

expected_rows = 91_139_849

if total_rows != expected_rows:
    raise RuntimeError(
        f"Expected {expected_rows:,} rows, "
        f"found {total_rows:,}"
    )

# ============================================================
# Print summary
# ============================================================

print("\n========================================")
print("Aggregation complete")
print("========================================")

print(f"Files processed:      {len(files):,}")
print(f"Total sources:        {total_rows:,}")
print(f"Detected sources:     {total_detected:,}")

print("\nDetections by band:")

for band in lsst_bands:
    print(
        f"  {band}: "
        f"{n_band_detections[band].sum():,}"
    )


# ============================================================
# Save results
# ============================================================

np.savez_compressed(
    output_file,

    z_edges=z_edges,
    z_mid=z_mid,

    n_sources_all=n_sources_all,
    n_sources_detected=n_sources_detected,

    total_rows=total_rows,
    total_detected=total_detected,

    n_lsstu=n_band_detections["lsstu"],
    n_lsstg=n_band_detections["lsstg"],
    n_lsstr=n_band_detections["lsstr"],
    n_lssti=n_band_detections["lssti"],
    n_lsstz=n_band_detections["lsstz"],
    n_lssty=n_band_detections["lssty"],
)

print(f"\nSaved:")
print(output_file)
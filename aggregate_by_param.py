# ============================================================
# Aggregate full SNIa simulation
#
# Full dataset:
#     91,139,849 simulated SNIa
#     456 parquet files
#
# Stores:
#     - all-source distributions
#     - detected-source distributions
#     - redshift-dependent detections by LSST band
#
# Parameters:
#     z, x1, c, t0, magabs, magobs, ra, dec
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

project_dir = Path(__file__).resolve().parent

input_dir = project_dir / "results_5"

output_file = project_dir / "aggregated_results_new.npz"


# ============================================================
# Configuration
# ============================================================

lsst_bands = [
    "lsstu",
    "lsstg",
    "lsstr",
    "lssti",
    "lsstz",
    "lssty",
]

parameters = [
    "z",
    "x1",
    "c",
    "t0",
    "magabs",
    "magobs",
    "ra",
    "dec",
]


# ============================================================
# Histogram bin edges
# ============================================================


hist_edges = {

    # Redshift
    "z": np.linspace(0.0, 1.0, 101),

    # SALT2 stretch
    "x1": np.linspace(-5.0, 4.0, 91),

    # SALT2 colour
    "c": np.linspace(-0.4, 1.0, 101),

    # Time of maximum
    "t0": np.linspace(61100.0, 64860.0, 101),

    # Absolute magnitude
    "magabs": np.linspace(-22.0, -15.0, 101),

    # Observed magnitude
    "magobs": np.linspace(0.0, 30.0, 101),

    # Right ascension
    "ra": np.linspace(0.0, 360.0, 101),

    # Declination
    "dec": np.linspace(-90.0, 90.0, 101),
}


# ============================================================
# Find input files
# ============================================================

files = sorted(
    input_dir.glob("detected_N*.parquet")
)

expected_files = 456

if len(files) != expected_files:
    raise RuntimeError(
        f"Expected {expected_files} parquet files, "
        f"found {len(files)}"
    )

if not files:
    raise FileNotFoundError(
        f"No parquet files found in {input_dir}"
    )

print(f"Input directory: {input_dir}")
print(f"Found {len(files)} parquet files.")


# ============================================================
# Accumulators
# ============================================================

# Histogram counts for all simulated sources
n_all = {
    parameter: np.zeros(
        len(hist_edges[parameter]) - 1,
        dtype=np.int64,
    )
    for parameter in parameters
}


# Histogram counts for detected sources
n_detected = {
    parameter: np.zeros(
        len(hist_edges[parameter]) - 1,
        dtype=np.int64,
    )
    for parameter in parameters
}


# ------------------------------------------------------------
# Number of 5-sigma detections in each LSST band
# as a function of redshift
# ------------------------------------------------------------

n_band_detections = {
    band: np.zeros(
        len(hist_edges["z"]) - 1,
        dtype=np.int64,
    )
    for band in lsst_bands
}


# ============================================================
# Global counters
# ============================================================

total_rows = 0

total_detected = 0


# ============================================================
# Process one Parquet file at a time
# ============================================================

for i, file in enumerate(files, start=1):

    print(
        f"[{i}/{len(files)}] {file.name}"
    )

    # --------------------------------------------------------
    # Only read the columns required for the aggregation
    # --------------------------------------------------------

    columns = [
        *parameters,
        "detected",
        *lsst_bands,
    ]

    df = pd.read_parquet(
        file,
        columns=columns,
    )

    total_rows += len(df)


    # ========================================================
    # Detection mask
    # ========================================================

    detected = df["detected"].to_numpy(
        dtype=bool
    )

    total_detected += detected.sum()


    # ========================================================
    # Parameter distributions
    # ========================================================

    for parameter in parameters:

        # ----------------------------------------------------
        # All simulated sources
        # ----------------------------------------------------

        values_all = df[parameter].to_numpy()

        hist_all, _ = np.histogram(
            values_all,
            bins=hist_edges[parameter],
        )

        n_all[parameter] += hist_all


        # ----------------------------------------------------
        # Detected sources
        # ----------------------------------------------------

        values_detected = values_all[detected]

        hist_detected, _ = np.histogram(
            values_detected,
            bins=hist_edges[parameter],
        )

        n_detected[parameter] += hist_detected


    # ========================================================
    # Detections by LSST band as a function of redshift
    # ========================================================

    z_detected = df["z"].to_numpy()[detected]


    for band in lsst_bands:

        # Number of detections in this band for each
        # detected source
        detections_band = df.loc[
            detected,
            band
        ].to_numpy()


        # Histogram weighted by number of detections
        hist_band, _ = np.histogram(
            z_detected,
            bins=hist_edges["z"],
            weights=detections_band,
        )


        n_band_detections[band] += (
            hist_band.astype(np.int64)
        )


# ============================================================
# Check total number of rows
# ============================================================

expected_rows = 91_139_849

if total_rows != expected_rows:

    raise RuntimeError(
        f"Expected {expected_rows:,} rows, "
        f"found {total_rows:,}"
    )


# ============================================================
# Check detected-source total
# ============================================================

detected_hist_total = n_detected["z"].sum()

if detected_hist_total != total_detected:

    raise RuntimeError(
        "Detected-source histogram does not match "
        "total_detected: "
        f"{detected_hist_total:,} != "
        f"{total_detected:,}"
    )


# ============================================================
# Print summary
# ============================================================

print()
print("========================================")
print("Aggregation complete")
print("========================================")

print(
    f"Files processed:      {len(files):,}"
)

print(
    f"Total sources:        {total_rows:,}"
)

print(
    f"Detected sources:     {total_detected:,}"
)

print(
    f"Detection efficiency: "
    f"{total_detected / total_rows:.4%}"
)


# ============================================================
# Print detections by band
# ============================================================

print()
print("5-sigma detections by band:")

for band in lsst_bands:

    print(
        f"  {band}: "
        f"{n_band_detections[band].sum():,}"
    )


# ============================================================
# Save everything
# ============================================================

np.savez_compressed(

    output_file,

    # --------------------------------------------------------
    # Histogram bin edges
    # --------------------------------------------------------

    z_edges=hist_edges["z"],
    x1_edges=hist_edges["x1"],
    c_edges=hist_edges["c"],
    t0_edges=hist_edges["t0"],
    magabs_edges=hist_edges["magabs"],
    magobs_edges=hist_edges["magobs"],
    ra_edges=hist_edges["ra"],
    dec_edges=hist_edges["dec"],


    # --------------------------------------------------------
    # All simulated sources
    # --------------------------------------------------------

    n_z_all=n_all["z"],
    n_x1_all=n_all["x1"],
    n_c_all=n_all["c"],
    n_t0_all=n_all["t0"],
    n_magabs_all=n_all["magabs"],
    n_magobs_all=n_all["magobs"],
    n_ra_all=n_all["ra"],
    n_dec_all=n_all["dec"],


    # --------------------------------------------------------
    # Detected sources
    # --------------------------------------------------------

    n_z_detected=n_detected["z"],
    n_x1_detected=n_detected["x1"],
    n_c_detected=n_detected["c"],
    n_t0_detected=n_detected["t0"],
    n_magabs_detected=n_detected["magabs"],
    n_magobs_detected=n_detected["magobs"],
    n_ra_detected=n_detected["ra"],
    n_dec_detected=n_detected["dec"],


    # --------------------------------------------------------
    # Global totals
    # --------------------------------------------------------

    total_rows=total_rows,
    total_detected=total_detected,


    # --------------------------------------------------------
    # Redshift-dependent LSST band detections
    # --------------------------------------------------------

    n_lsstu=n_band_detections["lsstu"],
    n_lsstg=n_band_detections["lsstg"],
    n_lsstr=n_band_detections["lsstr"],
    n_lssti=n_band_detections["lssti"],
    n_lsstz=n_band_detections["lsstz"],
)


# ============================================================
# Finished
# ============================================================

print()
print("Saved:")
print(output_file)
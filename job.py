import os
import argparse
import pandas as pd

import lightcurve
import lsst_functions
import skysurvey

from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument("--N_tot",type=int,default=100_000)
parser.add_argument("--batch",type=int,required=True)
parser.add_argument("--output_dir",type=str,default="results")

args = parser.parse_args()
N_tot = args.N_tot
batch = args.batch
output_dir = args.output_dir

# Selection criteria
selection_criteria = lsst_functions.set_selection_criteria(total_points=5,n_filters=2,min_points_per_filter=2)

# Load LSST survey opsim
project_dir = os.path.dirname(os.path.abspath(__file__))
opsim_path = os.path.join(project_dir, "baseline_v5.3.5_10yrs.db")
lsst = skysurvey.LSST.from_opsim(opsim_path,sql_where="night < 365")

# Generate SNIa population
snia_dict = lightcurve.generate_snia_dict(
    redshift=None, ra=None, dec=None, x1=None,
    t0=None, magabs=None, c=None)

snia_param_list = lightcurve.generate_ordered_parameter_list(input_dict=True,params=snia_dict,)

detected = lsst_functions.get_total_alerts_from_survey(
    snia_param_list=snia_param_list,
    opsim=lsst,
    N_tot=N_tot,
    selection_criteria=selection_criteria,
)

# Add batch information
detected["batch"] = batch
# Make target IDs globally unique across batches
detected["target_global"] = (detected["target"] + batch * N_tot)

# Save
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, f"detected_{batch:03d}.parquet")

detected.to_parquet(
    output_path,
    engine="pyarrow",
    compression="zstd",
    index=False,
)

print(f"Finished batch {batch}")
print(f"Sources: {len(detected)}")
print(f"Output: {output_path}")
# argparse
# this file will be passed into the script.job file
import lightcurve # any function that directly handles LCs
import lsst_functions # any function that manipulates LSST opsim + data



from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--N_tot', type = int, default = 10)
parser.add_argument('--nights', type = int, default=365)

parser.add_argument("--z", type=float, default=None)
parser.add_argument("--x1", type=float, default=None)
parser.add_argument("--c", type=float, default=None)
parser.add_argument("--t0", type=float, default=None)
parser.add_argument("--magabs", type=float, default=None)
parser.add_argument("--ra", type=float, default=None)
parser.add_argument("--dec", type=float, default=None)

args = parser.parse_args()

selection_criteria = lsst_functions.set_selection_criteria(total_points=5,n_filters=2,min_points_per_filter=2)

detected = lsst_functions.get_total_alerts_from_survey(
    snia_param_list = None, opsim = None, N_tot = None,
    selection_criteria = selection_criteria, by_band = False):

detected.to_parquet("smallparquet.parquet",engine="pyarrow",compression="zstd",index=False)


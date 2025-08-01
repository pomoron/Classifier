from datetime import datetime
from modules.pavement.det_pvmt import det_pvmt
from modules.signs.det_sign import run_sign
from modules.overgrown_veg.overveg import run_overveg
from modules.misc_vlm.det_obstruction import run_obs
from modules.misc_vlm.det_barrier import run_bar
from modules.aggregator.agg_outpvmt import run_agg, agg_outpvmt
import config.paths as paths
import os

start_time = datetime.now()
make_dir = [paths.pavement_outdir, paths.sign_outdir, paths.barrier_outdir, paths.obstruction_outdir, paths.detveg_dir]
for dir in make_dir:
    os.makedirs(dir, exist_ok=True)

# # pavement detection
# det_pvmt()  
# end_pvmt = datetime.now()
# print(f"Pavement detection done. \n Duration: {end_pvmt - start_time}")

# # sign defect detection
# run_sign(num_trial=paths.sign_num_trial)

# # overgrown vegetation detection
# run_overveg()
# end_veg = datetime.now()
# print(f"Overgrown vegetation detection done. \n Duration: {end_veg - end_sign}")

# obstruction detection
run_obs(num_trial=paths.obs_num_trial)
# end_obs = datetime.now()
# print(f"Obstruction detection done. \n Duration: {end_obs - end_veg}")

# # barrier detection
# run_bar(num_trial=paths.bar_num_trial)
# end_bar = datetime.now()
# print(f"Barrier detection done. \n Duration: {end_bar - end_obs}")

# # Aggregate all results
# run_agg()
# end_time = datetime.now()
# print(f"Aggregation duration: {end_time - end_bar}")
agg_outpvmt(
            result_dir = paths.output_dir,
            interested_folders = [paths.obstruction_outdir],
            result_header = 'gemini_result',
            veg_file = paths.detveg_final_fn,
            georef_file = paths.georef_pano,
            )
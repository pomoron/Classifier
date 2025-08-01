import os, glob
import pandas as pd
import config.paths as paths
from .functions.agg_util import Aggregator
from .agg_pvmt import agg_pvmt
from datetime import datetime

def agg_outpvmt(
            result_dir = paths.output_dir,
            interested_folders = [paths.sign_outdir, paths.barrier_outdir, paths.obstruction_outdir],
            result_header = 'gemini_result',
            veg_file = paths.detveg_final_fn,
            georef_file = paths.georef_pano,
            ):

    overall_df = pd.DataFrame()
    agg = Aggregator()
    for dir in interested_folders:
        file_list = sorted(glob.glob(f'{dir}/{result_header}_*.csv'))
        
        # Extract defect types and descriptions
        for read_file in file_list:
            read_df = pd.read_csv(read_file)
            print(f"Processing {os.path.dirname(read_file).split('/')[-1]}/{os.path.basename(read_file)}. Total rows: {len(read_df)}")
            folder = os.path.basename(dir)      # Get the asset type

            overall_df = agg.assemble_df(overall_df, read_df, folder)

    # Judge verdicts over multiple num_trial by simple majority
    overall_df = agg.majority_df(overall_df)

    # Read vegetation data and stick it to the overall_df
    veg_df = pd.read_csv(veg_file)
    veg_keep = agg.extract_veg_defects(veg_df)
    overall_df = pd.concat([overall_df, veg_keep], ignore_index=True)

    # Find locations
    georef_df = pd.read_csv(georef_file)
    overall_df = agg.add_geolocations(overall_df, georef_df)

    overall_df.to_csv(os.path.join(result_dir, 'outpvmt_det.csv'), index=False)
    return overall_df

def run_agg():
    # start = datetime.now()
    _ = agg_outpvmt()
    # end_outpvmt = datetime.now()
    # print(f"Time taken: {end_outpvmt-start}")
    _ = agg_pvmt()
    # end_pvmt = datetime.now()
    # print(f"Time taken: {end_pvmt-end_outpvmt}")





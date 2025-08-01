import os, glob, json
import pandas as pd
from .functions.agg_util import AggPvmt
import modules.tools.plot as myplot
import config.paths as paths

def agg_pvmt(
            result_dir = paths.output_dir,
            interested_folders = [paths.pavement_outdir],
            result_header = 'pvmt',
            georef_file = paths.georef_pvmt,
        ):

    overall_df = pd.DataFrame()
    agg = AggPvmt()

    for folder in interested_folders:
        file_list = sorted(glob.glob(f'{folder}/{result_header}*.json'))
        asset_type = os.path.basename(folder)

        for read_file in file_list:
            print(f"Processing {os.path.dirname(read_file).split('/')[-1]}/{os.path.basename(read_file)}")

            category, images, df = myplot.createDF(read_file)
            
            # Copy the pavement detection result json categories into the aggregated format
            df_keep = df.copy()
            df_keep['asset_type'] = asset_type
            df_keep['defects'] = df_keep['category_id'].apply(lambda x: {category.loc[category['category_id']==x, 'name'].values[0]: "Yes"})
            df_keep['centre'] = df_keep['bbox'].apply(lambda x: ((x[0]+x[2])/2, (x[1]+x[3])/2))
            df_keep['desired_size'] = df_keep['image_id'].apply(lambda x: images.loc[images['id']==x, ['width', 'height']].values[0])
            df_keep = df_keep.drop(columns=['id','image_id','category_id'])
            overall_df = pd.concat([overall_df, df_keep], ignore_index=True)
            # del category, images, df   # clear memory

    # print(overall_df.head())
    # Find locations
    georef_df = agg.readPolygon(georef_file)
    georef_df['rotation'] = georef_df['geometry'].apply(lambda x: agg.rot_angle(x[0]))
    georef_df['file_name'] = georef_df['Img_ID'].apply(lambda x: 'rot_'+x)
    georef_df = pd.merge(georef_df, overall_df, on='file_name', how='right')       # only images with defects will be merged - save time
    georef_df = agg.scale_geom(georef_df)                                           # scale and rotate the centres, area and segmentation

    georef_df.to_csv(os.path.join(result_dir, 'pvmt_det.csv'), index=False)
    return georef_df
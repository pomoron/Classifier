import os, cv2, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import modules.tools.plot as myplot
from config import paths
from modules.overgrown_veg.functions.det_over_util import get_line_array, check_overgrown
from modules.overgrown_veg.functions.det_line_util import draw_lines_on_mask
from modules.tools.initdet import DepthEstimationProcessor

def main(
    input_dir = paths.input_dir,
    seg_list = paths.detveg_fn,                 # segmentation list
    line_list = paths.detveg_line_fn,           # accumulated line list
    output_fn = paths.detveg_final_fn,          # output file
    output_vis = paths.detveg_overgrown_vis,
):

    image_list = sorted(glob.glob(os.path.join(input_dir, 'front*.jpg')))
    depth_model_name = paths.depth_model_name      
    depth_tol = 0.01                        # tolerance for depth comparison
    # Visualisation
    if output_vis:
        output_visdir = paths.detveg_overgrown_visdir
        os.makedirs(output_visdir, exist_ok=True)

    # Load the depth model
    de = DepthEstimationProcessor(depth_model_name=depth_model_name)

    # Load the segmentation list
    category, _, df = myplot.createDF(seg_list)
    veg_cat_id = category.loc[category['name'] == 'vegetation', 'category_id'].values[0]
    accum_line = pd.read_csv(line_list)
    output_df = pd.DataFrame(columns=['image_id', 'file_name', 'start_point', 'end_point', 'overgrown_start', 'overgrown_end'])

    for image_path in image_list:
        image_fn = os.path.basename(image_path)
        print(f"Processing {image_fn}")
        this_df = df[df['file_name'] == image_fn]

        # Detect depth
        predicted_depth = de.get_depth(image_path)
        norm_depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())     # Normalize the depth map

        # Call in the detected line
        line_df = accum_line[(accum_line['file_name'] == image_fn)]
        line_str = line_df['line'].values[0]
        line_array = get_line_array(line_str)
        # Find the normalised depth at the start and end point of the line
        start_point = (line_array[0], line_array[1])
        end_point = (line_array[2], line_array[3])

        # Create segmentation mask of the vegetation
        veg_mask = this_df[this_df['category_id'] == veg_cat_id]['segmentation'].values[0]
        veg_mask = myplot.polygon_to_mask(veg_mask, norm_depth.shape[0], norm_depth.shape[1])
        np_depth = norm_depth.detach().cpu().numpy()
        _, _, overgrown_start_mask = check_overgrown(np_depth, veg_mask, start_point, depth_tol=depth_tol)
        depth_mask, judge_mask, overgrown_end_mask = check_overgrown(np_depth, veg_mask, end_point, depth_tol=depth_tol)

        # Determine if there are any overgrown vegetation at the start and end points
        overgrown_start = np.sum(overgrown_start_mask) > 10
        overgrown_end = np.sum(overgrown_end_mask) > 10
        output_df = pd.concat([output_df, pd.DataFrame({'image_id': line_df['image_id'].values[0],
                                                        'file_name': image_fn,
                                                        'start_point': [start_point],
                                                        'end_point': [end_point],
                                                        'overgrown_start': [overgrown_start],
                                                        'overgrown_end': [overgrown_end]})])

        if output_vis and (overgrown_start or overgrown_end):
            # Visualise the overgrown vegetation
            image = cv2.imread(image_path)
            # Ensure the mask has the same number of channels as the image
            veg_show_mask = veg_mask.astype(np.uint8)*255
            if len(image.shape) == 3 and image.shape[2] == 3:
                veg_show_mask = cv2.merge([veg_show_mask, veg_show_mask, veg_show_mask])
            veg_show = cv2.bitwise_and(veg_show_mask, image)
            judge_show_mask = judge_mask.astype(np.uint8)*255
            judge_mask_with_lines = draw_lines_on_mask(judge_show_mask, [line_array, [end_point[0], end_point[1], end_point[0], 0]], color=[(255, 0, 0), (255, 0, 0)], thickness=5)
            plt.figure(figsize=(28, 10))
            plt.subplot(1,4,1)
            plt.imshow(veg_show)
            plt.title('Vegetation Mask')
            plt.axis('off')  # Hide axis
            plt.subplot(1,4,2)
            plt.imshow(depth_mask)
            plt.title('Depth Mask')
            plt.axis('off')  # Hide axis
            plt.subplot(1,4,3)
            plt.imshow(judge_mask_with_lines)
            plt.title('Judge Mask')
            plt.axis('off')  # Hide axis
            plt.subplot(1,4,4)
            plt.imshow(overgrown_end_mask)
            plt.title('Overgrown Mask')
            plt.axis('off')  # Hide axis
            plt.savefig(os.path.join(output_visdir, image_fn))
            plt.close()
        
    output_df.to_csv(output_fn, index=False)

if __name__ == '__main__':
    main()

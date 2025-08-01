import cv2
import numpy as np
import os, glob
import pandas as pd
import modules.tools.plot as myplot
import config.paths as paths
from modules.overgrown_veg.functions import det_line_util as dlu

def main(
    input_dir=paths.input_dir,
    seg_list=paths.detveg_fn,
    output_fn=paths.detveg_line_fn,
):

    image_list = sorted(glob.glob(os.path.join(input_dir, 'front*.jpg')))
    minLineLength = 100         # minimum length of line segments to consider

    # Load the segmentation list
    category, images, df = myplot.createDF(seg_list)
    accum_line = pd.DataFrame(columns=['image_id', 'file_name', 'line'])

    for image_path in image_list:
        
        image = cv2.imread(image_path)  # Read the image
        width, height = image.shape[:2]
        image_fn = os.path.basename(image_path)
        print(f'Processing {image_fn}')
        
        # Step 1 - Call the seg_list to filter the image
        try:
            image_id = images.loc[images['file_name'] == image_fn, 'id'].values[0]
            category_id = category.loc[category['name'] == 'pavement', 'category_id'].values[0]
            df_image = df[(df['image_id'] == image_id) & (df['category_id'] == category_id)]    # Get the pavement annotations for the image
        except:
            print(f'Image {image_fn} not found in the segmentation list.')
            continue
        polygon = df_image['segmentation'].values[0]
        mask = myplot.polygon_to_mask(polygon, height, width)
        mask = mask.astype(np.uint8)
        pavement_image = cv2.bitwise_and(image, image, mask=mask)

        # Step 2 - Get the white patches
        gray_thres = dlu.get_bright_mask(pavement_image, threshold=100)

        # Step 3 - Filter the noise in the white patches
        filtered_pnt = np.nonzero(gray_thres)
        if len(filtered_pnt[0]) == 0:
            print(f'No white patches found in {image_fn}')
            max_slope_line = dlu.get_leftline_pvmt(pavement_image)  # Use the leftmost edge of the pavement
        else:
            white_patch_points = np.column_stack((filtered_pnt[1], filtered_pnt[0]))        # DBSCAN requires [x, y] format
            db = dlu.clustering(white_patch_points, eps=50, min_samples=10)
            instance_mask = dlu.get_instance_mask(db, height, width, largest_only=False)

            # # Step 4 - Try the cv2.connectedComponentsWithStats to find the largest component (doesn't separate components)
            # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

            # Step 4 - Find lines by Hough Transform
            edges = cv2.Canny(instance_mask * 255, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(
                edges,
                rho=1,                  # distance resolution in pixels
                theta=np.pi / 180,      # angular resolution in radians
                threshold=50,           # minimum number of intersections in Hough space
                minLineLength=minLineLength,       # minimum length of line segments to consider
                maxLineGap=10           # max gap between points on the same line
            )

            # Step 5 - Find the line with the largest slope
            if (isinstance(lines, list) or isinstance(lines,np.ndarray)) and len(lines) > 0:
                # Join collinear lines
                lines = dlu.merge_collinear_lines(lines, slope_threshold=0.05, c_threshold=10)
                lengths = [np.sqrt((x[0][2] - x[0][0])**2 + (x[0][3] - x[0][1])**2) for x in lines]
                slopes = [(x[0][3] - x[0][1]) / (x[0][2] - x[0][0]) for x in lines]
                if len(lines) > 2:
                    # 1. Find the line with the most negative slope (from bottom left to top right)
                    max_2_slopes = np.argpartition(slopes, 2)[:2]   # pick 2 in case there is a strange line, instead of np.argmin that gives 1
                    max_2_lines = [lines[x] for x in max_2_slopes]
                    # 2. Find from the two lines the one that starts from the left most
                    max_slope_line = max_2_lines[np.argmin([x[0][0] for x in max_2_lines])]
                else:
                    max_slope_idx = np.argmax(slopes)
                    max_slope_line = lines[max_slope_idx]
                # print(f"found line: {max_slope_line}")
            else:
                # if no lines, use the leftmost edge of the pavement
                # print(f"Lines not found from white patches. Use the leftmost edge of the pavement.")
                max_slope_line = dlu.get_leftline_pvmt(pavement_image)
        
        # Add the line into the df
        accum_line = pd.concat([accum_line, pd.DataFrame({'image_id': image_id, 'file_name': image_fn, 'line': [max_slope_line]})])

    accum_line.to_csv(output_fn, index=False)

if __name__ == "__main__":
    main()
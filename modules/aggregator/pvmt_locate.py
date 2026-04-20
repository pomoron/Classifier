import modules.tools.plot as myplot
import pandas as pd
import numpy as np
import cv2
import os, glob

# det_file = 'output/pavements/a12-portho2_vallim4.json'
# georef_file = 'input/georef/a12p_reference.csv'
# image_dir = 'output/intermediate/a12-portho2_val_trial'
# fov_h, fov_v = 53.1, 45.3  # camera field of view (degrees)
# h = 1.96696  # camera height (meters)

# image_list = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
# category, images, df = myplot.createDF(det_file)    # load detection results
# ref_df = pd.read_csv(georef_file)                   # load georeference points
# camera_int = {'fov_h': fov_h, 'fov_v': fov_v}

def pos_real_world(camera_ext, camera_int, dim):
    """ 
    Calculate the real world position of the top center pixel of the image
    Input:
        camera_ext: dict with camera extrinsic parameters (pitch, heading, easting, northing, height)
        camera_int: dict with camera intrinsic parameters (fov_h, fov_v)
        dim: tuple with image dimensions (width, height)
    
    p1 = bottom center, p2 = top center, p4 = top left, p3 = top right of the orthorectified image. 
    l(i) is the distance from the camera to point i in metres.
    
    Output:
        pnt_list = [p1, p2, p3, p4], when p_i is a tuples with real world coordinates of point i
        h_scale = pixel to meter scale in height (meters per pixel)
        w_scale = pixel to meter scale in width (meters per pixel)

    """

    # extract parameters
    pitch = camera_ext['pitch']
    heading = camera_ext['heading']
    easting = camera_ext['easting']
    northing = camera_ext['northing']
    h = camera_ext['height']
    fov_h = camera_int['fov_h']
    fov_v = camera_int['fov_v']
    width, height = dim

    # Find p1 and p2
    a = fov_v / 2
    b = fov_h / 2
    l1 = h * np.tan(np.radians(90 - (pitch + a)))
    l2 = h * np.tan(np.radians(90 - (pitch - a)))
    x_p1 = easting + l1 * np.sin(np.radians(heading))
    y_p1 = northing + l1 * np.cos(np.radians(heading))
    x_p2 = easting + l2 * np.sin(np.radians(heading))
    y_p2 = northing + l2 * np.cos(np.radians(heading))
    # # Has the csv corrected the position to be at the bottom center already?
    # x_p1 = easting
    # y_p1 = northing
    # x_p2 = easting + (l2 - l1) * np.sin(np.radians(heading))
    # y_p2 = northing + (l2 - l1) * np.cos(np.radians(heading))
    
    # Find p3 and p4 from l2
    l3 = l4 = l2 / np.cos(np.radians(b))
    x_p3 = easting + l3 * np.sin(np.radians(heading + b))
    y_p3 = northing + l3 * np.cos(np.radians(heading + b))
    x_p4 = easting + l4 * np.sin(np.radians(heading - b))
    y_p4 = northing + l4 * np.cos(np.radians(heading - b))

    pnt_list = [(x_p1, y_p1), (x_p2, y_p2), (x_p3, y_p3), (x_p4, y_p4)]
    
    # calculate pixel to meter scale
    h_scale = (abs(l2 - l1) / height)
    w_scale = ((l4 * np.sin(np.radians(b)) * 2) / width)

    return pnt_list, h_scale, w_scale

def pixel_to_rotated_world(query_point, pivot_pixel, pivot_world, scale_x_per_u, scale_y_per_v, theta_degrees):
    """
    Transforms a pixel coordinate (u_p, v_p) to a new real-world coordinate
    by rotating it around a pivot point (p1) by a given yaw angle.

    Args:
        query_point: Tuple of the pixel coordinates of the point you want to transform. (u_p, v_p)
        pivot_pixel: Tuple of the pixel coordinates of the center of rotation p1. (u_1, v_1)
        pivot_world: Tuple of real-world coordinates of the center of rotation p1. (x_1, y_1)
        scale_u_per_x (float): The x-scale (du/dx) in pixels per meter.
        scale_v_per_y (float): The y-scale (dv/dy) in pixels per meter.
        theta_degrees (float): The yaw angle to rotate by, in degrees.
                               (Counter-clockwise is positive).

    Returns:
        numpy.ndarray: A 2-element array [x_new, y_new] containing the
                       new real-world coordinates.
    """
    u_p, v_p = query_point
    u_1, v_1 = pivot_pixel
    x_1, y_1 = pivot_world

    # --- Step 1: Get the Vector in Pixels (with y-axis flip) ---
    # We find the vector from p1 to p_p in the pixel coordinate system.
    # We flip the y-component (v1 - v_p) so that "up" in the image
    # (negative v) becomes positive "y" in our intermediate space.
    v_pixel_x = u_p - u_1
    v_pixel_y = v_1 - v_p
    
    # --- Step 2: Scale the Vector to the Real World ---
    # Convert the pixel vector to a real-world vector (in meters)
    # by dividing by the respective scales.
    # This is the CRITICAL step to perform *before* rotation.
    v_world_x = v_pixel_x * scale_x_per_u
    v_world_y = v_pixel_y * scale_y_per_v
    v_world = np.array([v_world_x, v_world_y])

    # --- Step 3: Rotate the Real-World Vector ---
    # Convert the yaw angle to radians for trigonometric functions
    theta_rad = np.radians(theta_degrees)
    
    # Create the 2D rotation matrix
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    R = np.array([
        [c , s],
        [-s, c]
    ])
    
    # Apply the rotation
    v_rotated_world = R @ v_world  # Using the @ operator for matrix multiplication

    # --- Step 4: Translate to the Final World Position ---
    # Add the newly rotated real-world vector back to the
    # real-world coordinates of the rotation center (p1).
    p1_world = np.array([x_1, y_1])
    p_new_world = p1_world + v_rotated_world
    
    # print(f"Pixel Vector: {v_pixel_x, v_pixel_y} -> World Vector (pre-rotation): {v_world_x, v_world_y} -> Rotated World Vector: {v_rotated_world}")

    return p_new_world

def pixel_to_rotated_world_vector(query_point, pivot_pixel, pivot_world, scale_x_per_u, scale_y_per_v, theta_degrees):
    """
    Transforms one or more pixel coordinates (u_p, v_p) to new real-world coordinates
    by rotating around a pivot point (p1) by a given yaw angle.

    Args:
        query_point: Tuple (u_p, v_p) or array-like of shape (N, 2) of pixel coordinates.
        pivot_pixel: Tuple (u_1, v_1) of the center of rotation p1.
        pivot_world: Tuple (x_1, y_1) of real-world coordinates of the center of rotation p1.
        scale_x_per_u (float): The x-scale (du/dx) in pixels per meter.
        scale_y_per_v (float): The y-scale (dv/dy) in pixels per meter.
        theta_degrees (float): The yaw angle to rotate by, in degrees.

    Returns:
        numpy.ndarray: Array of shape (N, 2) or (2,) with new real-world coordinates.
    """
    # Convert input to numpy array of shape (N, 2)
    qp = np.atleast_2d(query_point)
    u_1, v_1 = pivot_pixel
    x_1, y_1 = pivot_world

    # Step 1: Get the Vector in Pixels (with y-axis flip)
    v_pixel_x = qp[:, 0] - u_1
    v_pixel_y = v_1 - qp[:, 1]
    v_pixel = np.stack([v_pixel_x, v_pixel_y], axis=1)

    # Step 2: Scale the Vector to the Real World
    v_world = np.empty_like(v_pixel, dtype=float)
    v_world[:, 0] = v_pixel[:, 0] * scale_x_per_u
    v_world[:, 1] = v_pixel[:, 1] * scale_y_per_v

    # Step 3: Rotate the Real-World Vector
    theta_rad = np.radians(theta_degrees)
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    R = np.array([[c, s], [-s, c]])
    v_rotated_world = v_world @ R.T  # (N,2) @ (2,2) -> (N,2)

    # Step 4: Translate to the Final World Position
    p1_world = np.array([x_1, y_1])
    p_new_world = v_rotated_world + p1_world

    # Return shape (N, 2) or (2,) if input was a single point
    if np.asarray(query_point).ndim == 1:
        return p_new_world[0]
    return p_new_world

def pitch_correction(ref_df, index):
    """
    Calculate pitch correction based on altitude difference between consecutive georeference points.
    """
    image_georef_ind = index
    if image_georef_ind < len(ref_df) - 1:
        next_image_georef = ref_df.iloc[image_georef_ind + 2]
        delta_dist = np.sqrt((next_image_georef['projectedX[m]'] - ref_df.iloc[image_georef_ind]['projectedX[m]']) ** 2 +
                    (next_image_georef['projectedY[m]'] - ref_df.iloc[image_georef_ind]['projectedY[m]']) ** 2)
        if delta_dist < 10:         # only apply correction if next point is within 10m
            delta_alt = next_image_georef['projectedZ[m]'] - ref_df.iloc[image_georef_ind]['projectedZ[m]'] 
            pitch_correction = np.degrees(np.arctan2(delta_alt, delta_dist))
            return pitch_correction
    return 0

def pvmt_locate(category, images, df, ref_df, camera_param, asset_type='pavement'):
    """ 
    Locate pavement defects in real world coordinates based on detection results and georeference points.
    Input:
        images: dataframe with image metadata from detection json (get from myplot.createDF)
        df: dataframe with detection results from detection json (get from myplot.createDF)
        ref_df: dataframe with georeference points (from georef csv)
        camera_param: dict with camera intrinsic parameters {'fov_h': fov_h, 'fov_v': fov_v, 'h': h}
    """
    image_list = sorted(images['file_name'].tolist())
    these_results = []
    for image_fn in image_list:
        image_bn = os.path.basename(image_fn)
        image_bn_noext = os.path.splitext(image_bn)[0]
        print(f"\nProcessing image: {image_bn}")

        if image_bn not in df['file_name'].values:
            continue    # skip images without detections
        image_georef = ref_df[ref_df['file_name'] == image_bn_noext]    # georef csv doesn't have extension
        image_id = images.loc[images['file_name'] == image_bn, 'id'].values[0]
        image_detections = df[df['image_id'] == image_id]

        pitch, heading = image_georef['pitch[deg]'].values[0], image_georef['heading[deg]'].values[0]
        easting, northing = image_georef['projectedX[m]'].values[0], image_georef['projectedY[m]'].values[0]

        # Pitch correction using georef point altitude
        image_georef_ind = image_georef.index[0]
        pitch_corr = pitch_correction(ref_df, image_georef_ind)
        pitch -= pitch_corr
        # print(f"csv pitch {pitch + pitch_corr:.2f} deg, apply pitch correction {pitch_corr:.2f} deg -> corrected pitch {pitch:.2f} deg")
        # if image_georef_ind < len(ref_df) - 1:
        #     next_image_georef = ref_df.iloc[image_georef_ind + 2]
        #     delta_dist = np.sqrt((next_image_georef['projectedX[m]'] - easting) ** 2 +
        #                 (next_image_georef['projectedY[m]'] - northing) ** 2)
        #     if delta_dist < 10:         # only apply correction if next point is within 10m
        #         delta_alt = next_image_georef['projectedZ[m]'] - image_georef['projectedZ[m]'].values[0] 
        #         pitch_correction = np.degrees(np.arctan2(delta_alt, delta_dist))
        #         print(f"csv pitch {pitch:.2f} deg, add pitch correction {pitch_correction:.2f} deg")
        #         pitch -= pitch_correction
        #     else:
        #         print(f"csv pitch {pitch:.2f} deg, no pitch correction applied")
        camera_int = {'fov_h': camera_param['fov_h'], 'fov_v': camera_param['fov_v']}
        camera_ext = {'pitch': abs(pitch), 'heading': heading, 'easting': easting, 'northing': northing, 'height': camera_param['h']}   # assume pitch is always negative (down from horizontal)
        dim = (images.loc[images['file_name'] == image_bn, 'width'].values[0], images.loc[images['file_name'] == image_bn, 'height'].values[0])

        # work out the positions of the georeference points in the real world
        # p1 = bottom center, p2 = top center, p4 = top left, p3 = top right. l(i) is distance from camera to point i
        points, h_scale, w_scale = pos_real_world(camera_ext, camera_int, dim)
        p1_img = np.array([dim[0] / 2, dim[1]])      # bottom center
        p1 = np.array(points[0])
        # print(f"Camera Position (Easting, Northing): ({easting}, {northing})\nHeading: {heading} degrees")
        # print(f"p1 (bottom center) World Coordinates: {p1}")
        # print(f"Pixel to Meter Scale - Width: {w_scale:.4f} m/pixel, Height: {h_scale:.4f} m/pixel")

        # Find the centres of the defects in real world coordinates
        for idx, det in image_detections.iterrows():
            bbox = det['bbox'] # [x_min, y_min, w, h]
            x_center_pix = bbox[0] + bbox[2] / 2
            y_center_pix = bbox[1] + bbox[3] / 2
            defect_world_pos = pixel_to_rotated_world(
                query_point=(x_center_pix, y_center_pix),
                pivot_pixel=(p1_img[0], p1_img[1]),
                pivot_world=(p1[0], p1[1]),
                scale_x_per_u=w_scale,
                scale_y_per_v=h_scale,
                theta_degrees=heading
            )
            # print(f"Defect ID: {det['id']}\nPixel Position: {bbox}\nWorld Position: {defect_world_pos}")

            area_img = det['area']
            area_real = area_img * w_scale * h_scale
            # print(f"Defect Area: {area_img} pixels, {area_real:.2f} m^2")

            # Compile defect information
            defect_name = category.loc[category['category_id'] == det['category_id'],'name'].values[0]
            # real_seg_mask = [
            #     [
            #         pixel_to_rotated_world(
            #             query_point=(mask[i], mask[i+1]),
            #             pivot_pixel=(p1_img[0], p1_img[1]),
            #             pivot_world=(p1[0], p1[1]),
            #             scale_x_per_u=w_scale,
            #             scale_y_per_v=h_scale,
            #             theta_degrees=heading
            #         )
            #         for i in range(0, len(mask), 2)
            #     ]
            #     for mask in det['segmentation']
            # ]
            real_seg_mask = [
                pixel_to_rotated_world_vector(
                    query_point=mask,
                    pivot_pixel=(p1_img[0], p1_img[1]),
                    pivot_world=(p1[0], p1[1]),
                    scale_x_per_u=w_scale,
                    scale_y_per_v=h_scale,
                    theta_degrees=heading
                )
                for mask in det['segmentation']
            ]
            bbox_x1, bbox_y1 = pixel_to_rotated_world(
                query_point=(bbox[0], bbox[1]),
                pivot_pixel=(p1_img[0], p1_img[1]),
                pivot_world=(p1[0], p1[1]),
                scale_x_per_u=w_scale,
                scale_y_per_v=h_scale,
                theta_degrees=heading
            )
            bbox_x2, bbox_y2 = pixel_to_rotated_world(
                query_point=(bbox[0] + bbox[2], bbox[1] + bbox[3]),
                pivot_pixel=(p1_img[0], p1_img[1]),
                pivot_world=(p1[0], p1[1]),
                scale_x_per_u=w_scale,
                scale_y_per_v=h_scale,
                theta_degrees=heading
            )
            bbox_xyxy = [bbox_x1, bbox_y1, bbox_x2, bbox_y2]
            desc = {
                'area': area_real,
                'bbox': bbox_xyxy,  # in x1, y1, x2, y2 format - because the box can be rotated
                'segmentation': real_seg_mask
            }
            defect_dict = {
                'file_name': image_bn,
                'asset_type': asset_type,
                'defects': {defect_name: "Yes"},
                'descriptions': desc,
                'projectedX[m]': defect_world_pos[0],
                'projectedY[m]': defect_world_pos[1],
            }
            these_results.append(defect_dict)
    results_df = pd.DataFrame(these_results)
    return results_df

if __name__ == '__main__':
    pvmt_locate()

    

    

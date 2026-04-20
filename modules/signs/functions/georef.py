import numpy as np
import pandas as pd
import re
from scipy.spatial.transform import Rotation
from PIL import Image
import math
import torch

def get_coord_rot(camera_poses, georef_pd, filelist, height=0):
    # Get SfM and GPS coordinates -> (N, 3) numpy array
    # SfM: pos = camera_poses[0] @ np.array([0,0,0,1]) - as all cameras are measured relative to the origin, that's basically the last column of the c2w matrix
    sfm_centers = np.array([camera_poses[i][0:3,3] for i in range(len(camera_poses))])
    sfm_rot = np.array([camera_poses[i][0:3,0:3] for i in range(len(camera_poses))])
    # GPS coordinates and rotations
    filelist_clean = [re.search(r'pano_\d+_\d+', filename).group() for filename in filelist if re.search(r'pano_\d+_\d+', filename)]
    georef_need = georef_pd.loc[georef_pd['file_name'].isin(filelist_clean), ['file_name', 'roll[deg]', 'pitch[deg]', 'heading[deg]', 'projectedX[m]', 'projectedY[m]', 'projectedZ[m]']]
    gps_coords = georef_need[['projectedX[m]', 'projectedY[m]', 'projectedZ[m]']].values    # output is already a numpy array of shape (N, 3)
    gps_coords = gps_coords - np.array([0,0,height])  # Adjust for height of the camera above ground
    gps_rot = georef_need[['roll[deg]', 'pitch[deg]', 'heading[deg]']].values
    # # Basically the same implementation as the scipy package
    # real_world_rotations = []
    # for rot in gps_rot:
    #     real_world_rotations.append(rotMat(rot[0], rot[1], rot[2]))
    r = Rotation.from_euler('zyx', gps_rot, degrees=True)
    real_world_rotations = r.as_matrix() # Shape (N, 3, 3) in radians
    return sfm_centers, gps_coords, sfm_rot, real_world_rotations

# from Ellie's code
def rotMat(roll, pitch, heading, mode='norm'):
    
    alpha = math.radians(heading)
    cosa = math.cos(alpha)
    sina = math.sin(alpha)
    
    beta  = math.radians(pitch)
    cosb = math.cos(beta)
    sinb = math.sin(beta)
    
    gamma = math.radians(roll)
    cosg = math.cos(gamma)
    sing = math.sin(gamma)
        
    yaw_mat = np.array([[cosa , -sina , 0],
                        [sina, cosa, 0],
                        [0, 0, 1]])
    
    pitch_mat = np.array([[cosb, 0, sinb],
                          [0, 1, 0],
                          [-sinb, 0, cosb]])
    
    roll_mat = np.array([[1, 0, 0],
                         [0, cosg, -sing],
                         [0, sing, cosg]])
    if mode =='norm':
        rotmat = yaw_mat @ pitch_mat @ roll_mat
    elif mode == 'rev':
        rotmat = roll_mat @ pitch_mat @ yaw_mat
    else:
        print("error in mode")
    
    return rotmat

# -------------------------------------------------
# (251022 - Non-uniform scaling implementation - Filt5 and filt7)
# From filt4, the SfM reconstruction may have disproportionally distorted the z-axis because of not having enough vertical constraints
# Keep using Rotation.align_vectors to find the best rotation, but calculate independent scaling factors for each axis

def align_with_non_uniform_scaling(sfm_centers, gps_coords):
    """
    Calculates the transformation from SfM to GPS coordinates, allowing for
    non-uniform scaling on each axis. This is the ideal method when the SfM
    reconstruction might distort one axis (e.g., Z) differently than others.

    Args:
        sfm_centers (np.ndarray): (N, 3) array of SfM camera positions.
        gps_coords (np.ndarray): (N, 3) array of corresponding GPS positions.

    Returns:
        A (np.ndarray): The 3x3 affine transformation matrix (combining rotation, reflection, and non-uniform scale).
        t (np.ndarray): The 3x1 translation vector.
    """
    # print("--- Aligning with Non-Uniform Scaling ---")

    # 1. Initial setup: Apply standard convention and center the point clouds
    R_convention = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    sfm_centers_conv = sfm_centers @ R_convention.T
    
    sfm_centroid = sfm_centers_conv.mean(axis=0)
    gps_centroid = gps_coords.mean(axis=0)
    
    sfm_centered = sfm_centers_conv - sfm_centroid
    gps_centered = gps_coords - gps_centroid

    # 2. Find the optimal rotation that best aligns the shapes
    R_align, _ = Rotation.align_vectors(gps_centered, sfm_centered)
    R_candidate_1 = R_align.as_matrix()
    # R_align_matrix = R_align.as_matrix()

    # # 3. Correct for reflections (mirror image) by checking the determinant
    # # Check if the alignment matrix is a reflection (determinant = -1).
    # if np.linalg.det(R_align_matrix) < 0:
    #     R_align_matrix[2, :] *= -1      # Flip one of the axes to turn it into a proper rotation.
    
    # sfm_rotated = sfm_centered @ R_align_matrix

    # --- NEW: Optimization Step ---
    # Create the flipped candidate
    R_candidate_2 = R_candidate_1.copy()
    R_candidate_2[2, :] *= -1 # Flip its determinant
    # R_roll_180 = np.diag([-1.0, 1.0, -1.0])
    
    # Create the "90-degree yawed" candidate
    R_yaw_90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    R_candidate_3 = R_candidate_1 @ R_yaw_90
    # R_yaw_180 = np.diag([-1.0, -1.0, 1.0])

    # Create the "90-degree yawed" candidate
    R_yaw_minus90 = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    R_candidate_4 = R_candidate_1 @ R_yaw_minus90

    # Create the "180-degree yawed" candidate
    R_yaw_180 = np.diag([-1.0, -1.0, 1.0])
    R_candidate_5 = R_candidate_1 @ R_yaw_180
    
    # Test which candidate minimizes the L2 error
    cand_dict = {}
    for ind, R_cand in enumerate([R_candidate_1, R_candidate_2, R_candidate_3, R_candidate_4, R_candidate_5], start=1):
        sfm_rotated = sfm_centered @ R_cand
        sign_cand , sv_cand = compute_sign_scale(sfm_rotated, gps_centered)
        if ind == 5:
            sfm_rotated_test = sfm_rotated @ R_yaw_180 @ np.diag(sign_cand * sv_cand)    # Apply signs and scales
            sfm_rotated_test[:] = sfm_rotated_test[::-1]
        else:
            sfm_rotated_test = sfm_rotated @ np.diag(sign_cand * sv_cand)   # Apply signs and scales
        error_test = np.sum(np.linalg.norm((sfm_rotated_test - sfm_rotated_test.mean(axis=0)) - gps_centered, axis=1)**2)
        cand_dict[ind] = {'R': R_cand, 'sfm_center': sfm_rotated, 'error': error_test, 'signs': sign_cand, 'scales': sv_cand}
        
    # Check if minimum error improves from candidate 1 by at least 4%
    # Extract all errors
    errors = {ind: cand_dict[ind]['error'] for ind in cand_dict.keys()}
    min_error_key = min(errors.keys(), key=lambda k: errors[k])
    min_error_value = errors[min_error_key]
    improvement_threshold = 0.2
    candidate_1_error = errors[1]
    if (candidate_1_error - min_error_value) / candidate_1_error < improvement_threshold:
        # Improvement is less than 20%, use candidate 1
        chosen_key = 1
    else:
        # Improvement is 20% or more, use the best candidate
        chosen_key = min_error_key
    min_error_cand = cand_dict[chosen_key]
    
    # Extract the values
    R_align_matrix = min_error_cand['R']
    sfm_rotated = min_error_cand['sfm_center']
    min_error = min_error_cand['error']
    signs = min_error_cand['signs']
    scale_vector = min_error_cand['scales']
    # sfm_centroid_chosen = sfm_rotated.mean(axis=0)

    print(f"Align errors: Original={cand_dict[1]['error']:.4f}, Flipped={cand_dict[2]['error']:.4f}, +90-Yawed={cand_dict[3]['error']:.4f}, -90-Yawed={cand_dict[4]['error']:.4f}, 180-Yawed={cand_dict[5]['error']:.4f}")
    
    # --- End of NEW Optimization Step ---

    # # Past work with only 1 candidate rotation
    # # First, calculate the simple, per-axis signs
    # signs = np.sign(np.sum(sfm_rotated * gps_centered, axis=0))
    # signs[signs == 0] = 1.0  # Avoid zero-scaling if an axis has no variation

    # # b. DISTINGUISH 180-YAW from SIMPLE ALIGNMENT
    # # # Count how many planar (X, Y) axes are inverted
    # # num_planar_inversions = np.sum(signs[:2] < 0)

    # # if num_planar_inversions == 2:
    # #     # This is the 180-degree yaw case. Both X and Y are inverted.
    # #     print(f"Detected 180-degree yaw (num_planar_inversions=2). Forcing [-1, -1, 1] signs.")
    # #     signs = np.array([-1.0, -1.0, 1.0])
    # # else:
    # #     # This is the simple alignment case (Model 0_0).
    # #     # num_planar_inversions is 0 or 1. We trust the per-axis signs.
    # #     print(f"Detected simple alignment (num_planar_inversions={num_planar_inversions}). Using per-axis signs: {signs}")
    # #     pass 

    # sfm_corrected_signs = sfm_rotated * signs # Element-wise correction
    
    # # 4. Calculate non-uniform scaling factors for each axis independently
    # # We use standard deviation as a robust measure of the spread on each axis.
    # with np.errstate(divide='ignore', invalid='ignore'):
    #     sfm_std = np.std(sfm_corrected_signs, axis=0)
    #     gps_std = np.std(gps_centered, axis=0)
    #     # Handle cases where an axis has zero variance to avoid division by zero
    #     scale_vector = np.divide(gps_std, sfm_std, out=np.ones_like(gps_std), where=sfm_std!=0)
    
    # print(f"Computed Reflection Signs (X, Y, Z): {signs}")
    # print(f"Computed Non-Uniform Scale Vector (X, Y, Z): {scale_vector}")

    # 5. Combine rotation, reflection, and scaling into a single affine matrix 'A'
    # A = R_align @ diag(signs) @ diag(scale_vector)
    if chosen_key == 5:
        A = R_align_matrix @ np.diag(signs * scale_vector) @ R_yaw_180
    else:
        A = R_align_matrix @ np.diag(signs * scale_vector)
    # print(f"SfM rotated 1: \n {cand_dict[1]['sfm_center'] @ np.diag(cand_dict[1]['signs'] * cand_dict[1]['scales'])}")
    # print(f"SfM rotated 2: \n {cand_dict[2]['sfm_center'] @ np.diag(cand_dict[2]['signs'] * cand_dict[2]['scales'])}")
    # print(f"SfM rotated 3: \n {cand_dict[3]['sfm_center'] @ np.diag(cand_dict[3]['signs'] * cand_dict[3]['scales'])}")
    # print(f"SfM rotated 4: \n {cand_dict[4]['sfm_center'] @ np.diag(cand_dict[4]['signs'] * cand_dict[4]['scales'])}")
    # print(f"SfM rotated 5: \n {cand_dict[5]['sfm_center'] @ R_yaw_180 @ np.diag(cand_dict[5]['signs'] * cand_dict[5]['scales']) }")
    # print(f"GPS centered: \n {gps_centered}")

    # 6. Calculate the final translation vector 't'
    translation = gps_centroid - (sfm_centroid @ A)
    # translation = gps_centroid - (sfm_centroid_chosen @ A)

    return A, translation, scale_vector

def compute_sign_scale(sfm_rotated, gps_centered):
    """
    Computes the reflection signs and non-uniform scaling factors.
    Allows computation for each of the candidate rotations during optimization.
    """
    # First, calculate the simple, per-axis signs
    signs = np.sign(np.sum(sfm_rotated * gps_centered, axis=0))
    signs[signs == 0] = 1.0  # Avoid zero-scaling if an axis has no variation
    sfm_corrected_signs = sfm_rotated * signs # Element-wise correction

    # Calculate non-uniform scaling factors for each axis independently
    # We use standard deviation as a robust measure of the spread on each axis.
    with np.errstate(divide='ignore', invalid='ignore'):
        sfm_std = np.std(sfm_corrected_signs, axis=0)
        gps_std = np.std(gps_centered, axis=0)
        # Handle cases where an axis has zero variance to avoid division by zero
        scale_vector = np.divide(gps_std, sfm_std, out=np.ones_like(gps_std), where=sfm_std!=0)
    
    return signs, scale_vector

def compute_non_uniform_transform(sfm_points, gps_points, final=False):
    """
    Helper function for RANSAC. Calculates the non-uniform affine transform
    for a given (small) sample of points.
    Assumes points are already centered.

    Returns:
        A (np.ndarray): The 3x3 affine transformation matrix for the sample.
    """
    # 1. Find the optimal rotation that best aligns the shapes of the sample
    R_align, _ = Rotation.align_vectors(gps_points, sfm_points)
    R_align_matrix = R_align.as_matrix()
    
    sfm_rotated = sfm_points @ R_align_matrix
    
    # 2. Correct for any reflections (inverted axes) by checking data trends
    signs = np.sign(np.sum(sfm_rotated * gps_points, axis=0))
    signs[signs == 0] = 1.0
    
    sfm_corrected_signs = sfm_rotated * signs
    
    # 3. Calculate non-uniform scaling factors for each axis independently
    with np.errstate(divide='ignore', invalid='ignore'):
        sfm_std = np.std(sfm_corrected_signs, axis=0)
        gps_std = np.std(gps_points, axis=0)
        scale_vector = np.divide(gps_std, sfm_std, out=np.ones_like(gps_std), where=sfm_std!=0)
    
    if final:
        print(f"Sample Non-Uniform Scale Vector (X, Y, Z): \n {scale_vector}")

    # 4. Combine rotation, reflection, and scaling into a single affine matrix 'A'
    A = R_align_matrix @ np.diag(signs * scale_vector)
    
    return A

def ransac_align_non_uniform(sfm_centers, gps_coords, num_iterations=500, inlier_threshold=1.0):
    """
    Finds the best non-uniform affine transform using RANSAC to reject outliers.
    This is the primary function to use for robust alignment.
    
    Returns:
        A (np.ndarray): The final 3x3 affine transformation matrix.
        t (np.ndarray): The final 3x1 translation vector.
        inliers_mask (np.ndarray): Boolean mask of the inlier points.
    """
    print("--- Starting RANSAC with Non-Uniform Scaling ---")

    # 1. Initial setup: Apply standard convention and center the full point clouds
    R_convention = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    sfm_centers_conv = sfm_centers @ R_convention.T
    
    sfm_centroid = sfm_centers_conv.mean(axis=0)
    gps_centroid = gps_coords.mean(axis=0)
    
    sfm_centered = sfm_centers_conv - sfm_centroid
    gps_centered = gps_coords - gps_centroid
    
    best_inlier_count = 0
    best_A = np.eye(3)
    best_inliers_mask = np.zeros(len(sfm_centers), dtype=bool)
    num_points = len(sfm_centers)

    for i in range(num_iterations):
        # a. Select a minimal random sample (4 points for an affine transform in 3D)
        sample_size = min(4, num_points)
        sample_indices = np.random.choice(num_points, sample_size, replace=False)
        
        sfm_sample = sfm_centered[sample_indices]
        gps_sample = gps_centered[sample_indices]

        # b. Compute a trial model from the sample
        try:
            A_trial = compute_non_uniform_transform(sfm_sample, gps_sample)
        except (np.linalg.LinAlgError, ValueError):
            continue # Skip if sample is degenerate

        # c. Find inliers by checking positional error
        sfm_transformed = sfm_centered @ A_trial
        errors = np.linalg.norm(sfm_transformed - gps_centered, axis=1)
        inliers_mask = errors < inlier_threshold
        inlier_count = np.sum(inliers_mask)

        # d. Keep the best model
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inliers_mask = inliers_mask

    print(f"RANSAC found {best_inlier_count}/{num_points} inliers.")
    
    if best_inlier_count < 4:
        print("Warning: RANSAC failed to find a reliable model. Returning identity transform.")
        return np.eye(3), np.zeros(3), best_inliers_mask

    # 2. Recompute the final model using all identified inliers for better accuracy
    sfm_inliers = sfm_centered[best_inliers_mask]
    gps_inliers = gps_centered[best_inliers_mask]
    
    A_final = compute_non_uniform_transform(sfm_inliers, gps_inliers, final=True)
    
    # 3. Calculate the final translation vector using original centroids
    t_final = gps_centroid - (sfm_centroid @ A_final)

    return A_final, t_final, best_inliers_mask

def get_primary_vector(points):
    """
    (251024) Finds the "vector of best fit" (first principal component) for a set of points.
    This is more robust than just using the start and end points.
    """
    # Center the points
    centered_points = points - np.mean(points, axis=0)
    
    try:
        # Use SVD to find principal components. Vh[0] is the first principal component.
        _, _, Vh = np.linalg.svd(centered_points, full_matrices=False)
    except np.linalg.LinAlgError:
        # Fallback for degenerate data (e.g., all points are the same)
        return np.array([1.0, 0.0, 0.0])
        
    primary_vector = Vh[0, :]
    
    # SVD gives a direction, but the sign is arbitrary.
    # We ensure it points in the same general direction as the simple (start -> end) vector.
    simple_vector = points[-1] - points[0]
    if np.dot(primary_vector, simple_vector) < 0:
        primary_vector = -primary_vector # Flip it
    
    return primary_vector / np.linalg.norm(primary_vector)

# (251110 - Function to decide whether rolling 180 degrees are needed, improved version)
def need_180_roll(pred, 
                  trans_tensor, 
                  rotate_vector, 
                  device,
                  camera = None,
                  transform_key = ["pts3d_global_transformed"]):
    """
    This function now takes in project points from all cameras in the point cloud.

    Checks if the point cloud is "upside-down" by comparing the average
    height of points above the road to the average depth of points below it.
    If the scene is upside-down, it applies a 180-degree roll.
    """
    
    # Roll - roll 180 degrees (pi radians) around the camera direction axis
    roll_axis = rotate_vector  # a normalised vector around which to roll
    roll_angle_rad = np.pi
    roll_rot_vec = roll_axis * roll_angle_rad
    R_roll_180 = Rotation.from_rotvec(roll_rot_vec).as_matrix()
    R_roll_180_tensor_T = torch.tensor(R_roll_180.T, dtype=torch.float32).to(device)

    # Create broadcastable (1, 1, 3) tensors for translation
    trans_tensor_expanded = trans_tensor.unsqueeze(0).unsqueeze(0)
    all_mean_heights, all_mean_depths = [], []
    
    # Decide whether to roll based on points from all cameras
    for i in range(len(pred)):
        p_pitched_only = pred[i][transform_key[0]]  # (N, P, 3) tensor of points already pitched correctly
        
        # Determine the road Z-height
        if camera is None:
            # Fallback to translation Z if no per-camera GPS is provided
            road_z = trans_tensor[2]
        else:
            # Use the specific Z-height of this camera's GPS coordinate
            road_z = camera[i][2]  # Anchor Z-height (scalar)
    
        # Calculate Z-position of all points relative to the road
        relative_z = p_pitched_only[..., 2] - road_z

        # Create masks for points above and below the road
        # We add a small buffer (e.g., 10cm) to ignore points *on* the road
        above_road_mask = relative_z > 0.1
        below_road_mask = relative_z < -0.1

        # Find the average height and depths 
        height_above = torch.mean(relative_z[above_road_mask]) # "height" of points above the road
        depth_below = torch.mean(relative_z[below_road_mask])  # "depth" (negative height) of points below the road
        all_mean_heights.append(height_above)
        all_mean_depths.append(depth_below)
    
    # Filter out any NaN values that might occur from empty masks
    valid_heights = [h for h in all_mean_heights if not torch.isnan(h)]
    valid_depths = [d for d in all_mean_depths if not torch.isnan(d)]

    # Check if we have enough data to make a decision
    if not valid_heights or not valid_depths:
        # Cannot make a confident decision, so don't roll
        return pred  # Return the prediction as is

    # Convert lists to tensors and calculate means
    all_heights_tensor = torch.stack(valid_heights)
    all_depths_tensor = torch.stack(valid_depths)

    mean_height_above = torch.mean(all_heights_tensor)
    mean_depth_below = torch.mean(all_depths_tensor)

    # --- The New Criterion ---
    # If the average depth (as a positive number) is greater than the average height,
    # it means the "trees" are below and the "road" is above.
    # (e.g., mean_height = 0.2, mean_depth = -8.0)
    if torch.abs(mean_depth_below) > torch.abs(mean_height_above):
        # Apply the roll rotation to all cameras
        print(f"Points below road are 'deeper' (avg {mean_depth_below:.2f}m) than points above are 'tall' (avg {mean_height_above:.2f}m). Applying 180-degree roll.")
        
        for i in range(len(pred)):
            for key in transform_key:
                p_pitched_only = pred[i][key]
                p_shifted_again = p_pitched_only - trans_tensor_expanded
                p_rolled = p_shifted_again @ R_roll_180_tensor_T
                p_corrected_rotation = p_rolled + trans_tensor_expanded
                pred[i][key] = p_corrected_rotation.cpu()
        
        return pred
    else:
        # This is the correct orientation. No roll needed.
        for i in range(len(pred)):
            for key in transform_key:
                p_pitched_only = pred[i][key]
                pred[i][key] = p_pitched_only.cpu()
        return pred

# # (251105 - Function to decide whether rolling 180 degrees are needed)
# def need_180_roll(p_pitched_only, 
#                   trans_tensor, 
#                   rotate_vector, 
#                   device,
#                   camera_i = None):
    
#     # Roll - roll 180 degrees (pi radians) around the camera direction axis
#     roll_axis = rotate_vector   # a normalised vector around which to roll
#     roll_angle_rad = np.pi
#     roll_rot_vec = roll_axis * roll_angle_rad
#     R_roll_180 = Rotation.from_rotvec(roll_rot_vec).as_matrix()
#     R_roll_180_tensor_T = torch.tensor(R_roll_180.T, dtype=torch.float32).to(device)

#     # Create broadcastable (1, 1, 3) tensors for translation
#     trans_tensor_expanded = trans_tensor.unsqueeze(0).unsqueeze(0)
    
#     if camera_i is not None:
#         road_z = camera_i[2]  # Anchor Z-height (scalar)
#     else:
#         # # c. NEW: Check if a 180-degree roll is needed for this cloud
#         road_z = trans_tensor[2] # Anchor Z-height (scalar)
    
#     points_below_road = p_pitched_only[..., 2] < road_z
    
#     # Check if a significant portion (e.g., > 50%) of points are below the road
#     if torch.mean((points_below_road).float()) > 0.5:
#         # Apply the roll rotation
#         print("Points are below the road. Applying 180-degree roll correction.")
#         p_shifted_again = p_pitched_only - trans_tensor_expanded
#         p_rolled = p_shifted_again @ R_roll_180_tensor_T
#         p_corrected_rotation = p_rolled + trans_tensor_expanded
#     else:
#         # print(f"Cloud {i}: Points are above road. No roll needed.")
#         p_corrected_rotation = p_pitched_only # No roll needed

#     return p_corrected_rotation


# ----------------------------------------------------------
# DEPRECATED CODE BELOW
# ----------------------------------------------------------
# (251016 - initial attempts to compute R, s, t using SVD on rotations and positions)
def rst(sfm_centers, gps_coords, sfm_rotations, real_world_rotations, lookback=False):
    
    # (251021) Original implementation to compute R, s, t using SVD on rotations and positions - Filt
    # --- Add functions to align the SfM coordinate system to the GPS coordinate system ---
    # Define the convention change matrix
    R_convention = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ])

    # Pre-rotate the SfM data
    sfm_centers_aligned = sfm_centers @ R_convention.T  # to all SfM camera positions
    sfm_rotations_aligned = R_convention @ sfm_rotations    # multiply with the SfM camera rotations
    
    # 1. Center the point clouds on the origin
    sfm_centroid = sfm_centers_aligned.mean(axis=0)
    gps_centroid = gps_coords.mean(axis=0)
    sfm_centered = sfm_centers_aligned - sfm_centroid
    gps_centered = gps_coords - gps_centroid

    # # Did not consider rotations of the image in GPS coordinates
    # # 1. Find the optimal rotation matrix using SciPy
    # R, _ = Rotation.align_vectors(sfm_centered, gps_centered)
    # rot_matrix = R.as_matrix()

    # 2. Find the optimal rotation matrix using SVD
    # Create the correlation matrix $$M = \sum_{i=1}^{N} R_{real, i} \cdot R_{sfm, i}^T$$
    M = np.zeros((3, 3))
    R_z_180 = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ])
    for i in range(len(sfm_centers)):
        if lookback:
            # real_world_rot_corrected = R_z_180 @ real_world_rotations[i]    # The real-world reads rotations look at the back of the car
            real_world_rot_corrected = real_world_rotations[i] @ R_z_180    # The real-world reads rotations look at the back of the car
            M += real_world_rot_corrected @ sfm_rotations_aligned[i].T      # Accumulate the product of the real-world rotation and the transpose of the SfM rotation
        else:
            M += real_world_rotations[i] @ sfm_rotations_aligned[i].T
        # # Accumulate the product of the real-world rotation and the transpose of the SfM rotation
        # M += real_world_rotations[i] @ sfm_rotations_aligned[i].T

    # Find the optimal rotation using SVD
    U, S, Vt = np.linalg.svd(M)
    R_global = U @ Vt

    # --- Sanity Check for Reflections ---
    # The SVD can sometimes produce a reflection matrix (determinant = -1).
    # We must correct for this to ensure it's a valid rotation matrix.
    if np.linalg.det(R_global) < 0:
        print("Reflection detected, correcting for it.")
        Vt[-1, :] *= -1
        R_global = U @ Vt
    # print(f"U: {U} \n and scale: {S} \n and Vt: {Vt} ")

    # 3. Find the optimal scale factor
    # Apply rotation to the centered SfM points
    # sfm_rotated = sfm_centered @ rot_matrix
    sfm_rotated = sfm_centered @ R_global.T
    # Calculate scale as the ratio of standard deviations
    # scale = np.sum(gps_centered * sfm_rotated) / np.sum(sfm_rotated**2)

    # Use my method to calculate scale instead
    scale = scale_only(sfm_rotated, gps_centered)

    # 4. Find the optimal translation
    # t = gps_centroid - s * R @ sfm_centroid
    # translation = gps_centroid - scale * (rot_matrix @ sfm_centroid)
    translation = gps_centroid - scale * (R_global @ sfm_centroid)

    return R_global, scale, translation

def scale_only(sfm_rotated, gps_centered):
    # 1. Calculate the distance of each point to the origin
    sfm_distances = np.linalg.norm(sfm_rotated, axis=1)  # Distances for rotated SfM points
    gps_distances = np.linalg.norm(gps_centered, axis=1)   # Distances for GPS points
    
    # # 2. Sum all distances
    # sfm_sum_d = np.sum(sfm_distances)
    # gps_sum_d = np.sum(gps_distances)
    
    # # 3. Divide GPS sum by SfM sum to get the scale
    # scale = gps_sum_d / sfm_sum_d

    # Use d_gps * d_sfm / d_sfm * d_sfm to find scale
    num = np.sum(gps_distances * sfm_distances)
    den = np.sum(sfm_distances**2)
    scale = num / den

    return scale


def rst_new(sfm_centers, gps_coords, sfm_rotations, real_world_rotations, lookback=False):
    
    # (251021) Written to try rotating per camera - doesn't vary much from the SVD approach - Filt2/filt4

    # --- Add functions to align the SfM coordinate system to the GPS coordinate system ---
    # Define the convention change matrix
    R_convention = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ])

    # Pre-rotate the SfM data
    sfm_centers_aligned = sfm_centers @ R_convention.T  # to all SfM camera positions
    sfm_rotations_aligned = R_convention @ sfm_rotations    # multiply with the SfM camera rotations
    
    # 1. Center the point clouds on the origin
    sfm_centroid = sfm_centers_aligned.mean(axis=0)
    gps_centroid = gps_coords.mean(axis=0)
    sfm_centered = sfm_centers_aligned - sfm_centroid
    gps_centered = gps_coords - gps_centroid
    
    # # Now do rotations directly per camera
    # R_global, sfm_rotated = [], []
    # # R_global = np.zeros((3,3))
    # # sfm_rotated = np.zeros((sfm_centered.shape))
    # for i in range(len(sfm_centers)):
    #     # 2. Rotate each camera by A = R_real_world * inv(R_sfm)
    #     this_cam_rot = real_world_rotations[i] @ np.linalg.inv(sfm_rotations_aligned[i])
    #     R_global.append(this_cam_rot)

    #     # Apply rotation of each camera to the centered SfM points
    #     sfm_rotated.append(this_cam_rot @ sfm_centered[i])    
    # # 3. Find the optimal scale factor
    # # Calculate scale as the ratio of standard deviations
    # sfm_rotated_np = np.array(sfm_rotated)
    # scale = np.sum(gps_centered * sfm_rotated_np) / np.sum(sfm_rotated_np**2)
    print(f"Before alignment and axis permutation: sfm camera positions \n {sfm_centered} \n GPS centered positions \n {gps_centered}")

    # 2. Find the optimal rotation matrix using SciPy
    # Forget about SVD or inverse - just use scipy's align_vectors per camera
    R_align_shape, _ = Rotation.align_vectors(gps_centered, sfm_centered)
    R_align_shape_matrix = R_align_shape.as_matrix()
    
    # Apply this initial alignment to the centered SfM points
    sfm_centered_aligned = sfm_centered @ R_align_shape_matrix
    print(f"After alignment, before axis permutation: sfm camera positions \n {sfm_centered_aligned} \n GPS centered positions \n {gps_centered}")
    
    # Find the axis permutation/reflection correction matrix
    # Previously it was a 180-degree Y-axis correction. Later experiments found it may vary
    R_correct_axes = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    # R_correct_axes = find_axis_correction_matrix(sfm_centered_aligned, gps_centered)
    R_global = R_align_shape_matrix @ R_correct_axes
    
    # 3. Find the optimal scale factor
    sfm_rotated = sfm_centered_aligned @ R_global
    scale = scale_only(sfm_rotated, gps_centered)
    print(f"After axis permutation: sfm rotated camera positions \n {sfm_rotated} \n GPS centered positions \n {gps_centered}")

    # 4. Find the optimal translation
    # t = gps_centroid - s * R @ sfm_centroid
    # translation = gps_centroid - scale * (rot_matrix @ sfm_centroid)
    translation = gps_centroid - scale * (sfm_rotated @ R_global)

    return R_global, scale, translation

# (251021 - RANSAC implementation to reject outliers)

import numpy as np
from scipy.spatial.transform import Rotation

# Helper function to compute the transform using BOTH rotations and positions
def compute_transform_from_poses(sfm_points, gps_points, sfm_rotations, gps_rotations):
    """
    Computes s, R, t by finding R from rotations, then s & t from positions.
    """
    # 1. Find the best rotation 'A' by aligning the orientation matrices
    M = np.zeros((3, 3))
    for i in range(len(sfm_rotations)):
        M += gps_rotations[i] @ sfm_rotations[i].T
    
    U, S, Vt = np.linalg.svd(M)
    A = U @ Vt
    
    if np.linalg.det(A) < 0:
        Vt[-1, :] *= -1
        A = U @ Vt
        
    # 2. With 'A' fixed, find the optimal scale 's' and translation 't' from positions
    sfm_rotated = sfm_points @ A.T
    
    sfm_centroid = sfm_rotated.mean(axis=0)
    gps_centroid = gps_points.mean(axis=0)
    
    sfm_centered = sfm_rotated - sfm_centroid
    gps_centered = gps_points - gps_centroid
    
    # s_num = np.sum(gps_centered * sfm_centered)
    # s_den = np.sum(sfm_centered**2)
    # s = s_num / s_den
    s = scale_only(sfm_centered, gps_centered)
    
    t = gps_centroid - s * sfm_centroid
    
    return s, A, t

def ransac_align_with_poses(sfm_points, gps_points, sfm_rotations, gps_rotations, 
                             num_iterations=200, inlier_threshold=0.5):
    """
    Finds the best similarity transform using RANSAC on full poses.
    """
    best_inlier_count = 0
    best_s, best_R, best_t = 1.0, np.eye(3), np.zeros(3)
    best_inliers_mask = np.zeros(len(sfm_points), dtype=bool)
    
    num_points = len(sfm_points)
    
    for i in range(num_iterations):
        # 1. Select a minimal random sample of poses
        nos_sample = min(4, num_points)  # Ensure we don't sample more points than available
        sample_indices = np.random.choice(num_points, nos_sample, replace=False) # 4 is more stable
        sfm_p_sample = sfm_points[sample_indices]
        gps_p_sample = gps_points[sample_indices]
        sfm_R_sample = sfm_rotations[sample_indices]
        gps_R_sample = gps_rotations[sample_indices]
        
        # 2. Compute a trial model from the pose sample
        try:
            s_trial, R_trial, t_trial = compute_transform_from_poses(
                sfm_p_sample, gps_p_sample, sfm_R_sample, gps_R_sample)
        except np.linalg.LinAlgError:
            continue
            
        # 3. Find inliers by checking positional error
        sfm_transformed = s_trial * (sfm_points @ R_trial.T) + t_trial
        errors = np.linalg.norm(sfm_transformed - gps_points, axis=1)
        inliers_mask = errors < inlier_threshold
        inlier_count = np.sum(inliers_mask)
        
        # 4. Keep the best model
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inliers_mask = inliers_mask
            
            # Recompute model using all inliers for better accuracy
            best_s, best_R, best_t = compute_transform_from_poses(
                sfm_points[best_inliers_mask],
                gps_points[best_inliers_mask],
                sfm_rotations[best_inliers_mask],
                gps_rotations[best_inliers_mask]
            )

    print(f"RANSAC found {best_inlier_count}/{num_points} inliers.")
    return best_R, best_s, best_t, best_inliers_mask

# --------------------------------------
# (251022 - Uniform scaling implementation - Filt6)
# From filt5, the planar axes were found to scale and translate well, but the Z-axis was squished because of using a different scaling factor
# This implementation uses a single uniform scale factor based only on the planar axes (X and Y).

def align_uniform_no_ransac(sfm_centers, gps_coords):
    """
    Finds the best UNIFORM similarity transform (s, R, t) using the
    ENTIRE point cloud (no RANSAC).

    This is a fast, non-robust method.

    Returns:
        A (np_ndarray): The final 3x3 transformation matrix (s * R).
        t (np_ndarray): The final 3x1 translation vector.
    """
    print("--- Starting Alignment with Uniform Scaling (No RANSAC) ---")

    # 1. Initial setup: Apply standard convention and center the full point clouds
    R_convention = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    sfm_centers_conv = sfm_centers @ R_convention.T
    
    sfm_centroid = sfm_centers_conv.mean(axis=0)
    gps_centroid = gps_coords.mean(axis=0)
    
    sfm_centered = sfm_centers_conv - sfm_centroid
    gps_centered = gps_coords - gps_centroid

    # 2. Compute the final model using all points
    A_final = compute_uniform_transform(sfm_centered, gps_centered)
    
    # 3. Calculate the final translation vector using original centroids
    t_final = gps_centroid - (sfm_centroid @ A_final)

    print("--- Non-RANSAC Alignment Complete ---")
    return A_final, t_final

def compute_uniform_transform(sfm_points, gps_points):
    """
    Helper function for RANSAC. Calculates a UNIFORM similarity transform (s, R)
    for a given sample of points. Assumes points are already centered.
    This solves the Z-axis "squishing" problem by using a single scale factor.

    Returns:
        A (np.ndarray): The 3x3 transformation matrix (s * R).
    """
    # 1. Find the optimal rotation that best aligns the shapes of the sample
    R_align, _ = Rotation.align_vectors(gps_points, sfm_points)
    R_align_matrix = R_align.as_matrix()

    # 2. Fix 180-degree reflection problem:
    # Check if the alignment matrix is a reflection (determinant = -1).
    if np.linalg.det(R_align_matrix) < 0:
        # print("Correcting for reflection in alignment matrix.")
        # Flip one of the axes to turn it into a proper rotation.
        R_align_matrix[2, :] *= -1
    
    sfm_rotated = sfm_points @ R_align_matrix
    
    # 3. Fix Z-axis squishing:
    # Calculate a SINGLE, uniform scale factor based ONLY on the reliable
    # planar (X and Y) axes.
    with np.errstate(divide='ignore', invalid='ignore'):
        # Calculate standard deviation (a robust measure of spread)
        sfm_std_planar = np.std(sfm_rotated[:, :2], axis=0)
        gps_std_planar = np.std(gps_points[:, :2], axis=0)
        
        # Get scale for each planar axis
        planar_scales = np.divide(gps_std_planar, sfm_std_planar, 
                                  out=np.ones(2), 
                                  where=sfm_std_planar > 1e-6) # Avoid division by zero

    # Use the mean of the valid planar scales
    valid_scales = planar_scales[planar_scales > 1e-6]
    if len(valid_scales) == 0:
        # print("Warning: Could not determine planar scale. Defaulting to 1.0")
        scale_uniform = 1.0
    else:
        scale_uniform = np.mean(valid_scales)
    
    # 4. Combine rotation and UNIFORM scaling into a single matrix 'A'
    A = R_align_matrix * scale_uniform  # s * R
    
    return A

# def ransac_align_uniform(sfm_centers, gps_coords, num_iterations=500, inlier_threshold=1.0):
#     """
#     Finds the best UNIFORM similarity transform (s, R, t) using RANSAC.
#     This is the primary function to use for robust alignment.
    
#     Returns:
#         A (np.ndarray): The final 3x3 transformation matrix (s * R).
#         t (np.ndarray): The final 3x1 translation vector.
#         inliers_mask (np.ndarray): Boolean mask of the inlier points.
#     """
#     print("--- Starting RANSAC with Uniform Scaling ---")

#     # 1. Initial setup: Apply standard convention and center the full point clouds
#     R_convention = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
#     sfm_centers_conv = sfm_centers @ R_convention.T
    
#     sfm_centroid = sfm_centers_conv.mean(axis=0)
#     gps_centroid = gps_coords.mean(axis=0)
    
#     sfm_centered = sfm_centers_conv - sfm_centroid
#     gps_centered = gps_coords - gps_centroid
    
#     best_inlier_count = 0
#     best_A = np.eye(3)
#     best_inliers_mask = np.zeros(len(sfm_centers), dtype=bool)
#     num_points = len(sfm_centers)

#     for i in range(num_iterations):
#         # a. Select a minimal random sample (3 points for a similarity transform in 3D)
#         sample_size = min(3, num_points)
#         sample_indices = np.random.choice(num_points, sample_size, replace=False)
        
#         sfm_sample = sfm_centered[sample_indices]
#         gps_sample = gps_centered[sample_indices]

#         # b. Compute a trial model from the sample
#         try:
#             A_trial = compute_uniform_transform(sfm_sample, gps_sample)
#         except (np.linalg.LinAlgError, ValueError):
#             continue # Skip if sample is degenerate

#         # c. Find inliers by checking positional error
#         # A_trial already includes scale, so we apply it directly
#         sfm_transformed = sfm_centered @ A_trial
#         errors = np.linalg.norm(sfm_transformed - gps_centered, axis=1)
#         inliers_mask = errors < inlier_threshold
#         inlier_count = np.sum(inliers_mask)

#         # d. Keep the best model
#         if inlier_count > best_inlier_count:
#             best_inlier_count = inlier_count
#             best_inliers_mask = inliers_mask

#     print(f"RANSAC found {best_inlier_count}/{num_points} inliers.")
    
#     if best_inlier_count < 3:
#         print("Warning: RANSAC failed to find a reliable model. Returning identity transform.")
#         return np.eye(3), np.zeros(3), best_inliers_mask

#     # 2. Recompute the final model using all identified inliers for better accuracy
#     sfm_inliers = sfm_centered[best_inliers_mask]
#     gps_inliers = gps_centered[best_inliers_mask]
    
#     A_final = compute_uniform_transform(sfm_inliers, gps_inliers)
    
#     # 3. Calculate the final translation vector using original centroids
#     t_final = gps_centroid - (sfm_centroid @ A_final)

#     return A_final, t_final, best_inliers_mask

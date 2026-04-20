import sys, cv2, os, glob, torch
from PIL import Image
import numpy as np
import pandas as pd
import shutil
import re
from types import SimpleNamespace
from scipy.spatial.transform import Rotation
from time import time
import open3d as o3d
from scipy.spatial.distance import cdist

# Make paths absolute relative to this script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))  # ~/Classifier/modules/signs
parent_dir = os.path.dirname(script_dir)  # ~/Classifier/modules
grandparent_dir = os.path.dirname(parent_dir)  # ~/Classifier

sys.path.insert(0, grandparent_dir) # Add Classifier root to path for modules.tools, etc.
sail_recon_dir = os.path.join(grandparent_dir, 'sail_recon')    # Add sail_recon to path
sys.path.insert(0, sail_recon_dir)

# run from the Classifier root from now on...
from modules.tools import initdet, detcrop, predmask
from modules.signs.functions import cluster_sign, georef
import sail_recon.inference as inference

# -------------
# (260420) Change slice_signs() and cluster_signs() functions to use SAM3 instead...
# -------------

def slice_signs(gemini_client, image_dir, intermediate_dir, mask_dir, model_cfg, checkpoint):
    
    # Step 1 - Slice full signs
    detector = detcrop.DetAndCrop(
        categories = ['sign with metal poles and supports, the whole sign'],
        input_dir = image_dir,
        output_dir = intermediate_dir,
        gemini_client = gemini_client,
        task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>",
    )

    # Output the list of dictionaries {'name': fn, 'bbox': [x1, y1, x2, y2], 'category': cat} for all detected images
    det_bbox = detector.det(bool_output_image = False)   
    bbox_dict = {item['name']: item['bbox'] for item in det_bbox}

    # Step 1a - use the bounding boxes to create masks for final projection
    mask_gen = predmask.predMask(model_cfg=model_cfg, checkpoint=checkpoint)
    image_list = mask_gen.create_imagelist(image_dir)
    for image_fn in image_list:
        image = cv2.imread(image_fn)
        base_fn = os.path.basename(image_fn)
        bbox_xyxy = bbox_dict.get(base_fn)
        df_bbox = [[bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2]-bbox_xyxy[0], bbox_xyxy[3]-bbox_xyxy[1]]]    # convert to x,y,w,h format
        masks_accum, scores_accum, logits_accum = mask_gen.sam_pred_mask(image, 
                                                                        df_bbox = df_bbox,)
        # print(f"Processed {base_fn}, masks {masks_accum} of shape {len(masks_accum)}")
        if len(masks_accum) > 1:
            combined_mask = np.logical_or.reduce(masks_accum, axis=0)    # I only need 1 mask per image
        elif len(masks_accum) == 1:
            combined_mask = np.array(masks_accum[0])
        else:
            continue

        combined_mask = (combined_mask * 255).astype(np.uint8)
        os.makedirs(mask_dir, exist_ok=True)
        full_path = os.path.join(mask_dir, base_fn)
        try:
            pil_mask = Image.fromarray(combined_mask[0], mode='L') 
            pil_mask.save(full_path)
        except Exception as e:
            print(f"Save failed for {full_path}: {e}")

    del detector
    torch.cuda.empty_cache()    # remove the detector from GPU memory

# time_1 = time()



def cluster_signs(gemini_client, image_dir, face_dir, sorted_dir, ref_df, device):
    # Step 2 - Allocate images to groups based on sign similarity and geolocation
    # https://pytorch.org/vision/stable/models.html
    batch_size = 16
    # Detect the assets again so that only the salient parts are used for embedding extraction
    detector_face = detcrop.DetAndCrop(
        categories = ['sign'],
        input_dir = image_dir,
        output_dir = face_dir,
        gemini_client = gemini_client,
        task_prompt = "<OPEN_VOCABULARY_DETECTION>",
        # skip_verification = True  # Skip verification step
    )
    _ = detector_face.det()
    image_list = sorted(glob.glob(os.path.join(face_dir, '*.jpg')))     # Use the list of images with faces for grouping (in case detection goes wrong)
    all_embeddings = cluster_sign.get_embedding(image_list, batch_size, device, model_id='vit_large_patch16_dinov3.lvd1689m')

    # Perform DBSCAN clustering on the embeddings
    em_np = all_embeddings.cpu().numpy()
    em_labels, em_unique_labels = cluster_sign.cluster(em_np, eps=0.6, min_samples=3)    # eps 0.5 gives more stringent clustering but 0.6 returns more signs

    for i in em_unique_labels:
        indices = np.where(em_labels == i)[0]  # Find indices of elements with label i
        selected_images = [os.path.basename(image_list[idx]) for idx in indices]  # Extract corresponding images from image_list
        # Search for the pattern in the filename
        pattern = r'pano_\d{6}_\d{6}'
        sel_original = [re.search(pattern, img_name).group(0) for img_name in selected_images]

        # cluster with the geolocation
        related_df = ref_df[ref_df['file_name'].isin(sel_original)].sort_values(by='file_name')
        geo_loc = related_df[['projectedX[m]', 'projectedY[m]']].values.tolist()
        geo_labels, geo_unique_labels = cluster_sign.cluster(np.array(geo_loc), eps=20, min_samples=3)
        for j in geo_unique_labels:
            geo_indices = np.where(geo_labels == j)[0]  # Find indices of elements with label i
            geo_images = [selected_images[idx] for idx in geo_indices]  # Extract corresponding images from image_list
            folder_name = os.path.join(sorted_dir, f'{i}_{j}')
            os.makedirs(folder_name, exist_ok=True)
            for img in geo_images:
                shutil.copy(os.path.join(image_dir, img), os.path.join(folder_name, img))   # After grouping, copy the original images (without cropping) to a new folder

    del detector_face
    torch.cuda.empty_cache()    # remove the detector from GPU memory

def sign_recon_and_georef(sorted_dir, mask_dir, pc_dir, ref_df, sail_ckpt, device, bool_output_all, bool_output_mask, bool_debug):
    # Step 3 - 3D Reconstruction with sail-recon
    entries = os.listdir(sorted_dir)
    cluster_list = sorted([entry for entry in entries if os.path.isdir(os.path.join(sorted_dir, entry))])

    for dir in cluster_list:       # Iterate through every clustered sign
        entry_path = os.path.join(sorted_dir, dir)

        print(f"Reconstructing sign {dir}")
        # Find the largest images in the folder to use for reconstruction
        filelist = sorted(glob.glob(os.path.join(entry_path, "*.jpg")))
        filelist = cluster_sign.largest_images(filelist, n=10, move_file=True)
        recon_image_dir = os.path.dirname(filelist[0])  # find the folder that contains the images for reconstruction

        args = SimpleNamespace(
                img_dir=recon_image_dir,      # Set the path to your image directory
                # vid_dir = None,
                out_dir=pc_dir,         # Define your output folder
                ckpt=sail_ckpt,         # Path to a specific checkpoint file, or None to auto-download
                pc_name = "pred",       # name for the point cloud. Give name only, will add the extension in the code.
                # filter_img = True,
            )
            
        os.makedirs(args.out_dir, exist_ok=True)    # Ensure the output directory exists
        
        # --- Reconstruction with retries against distorted SfM predictions ---
        MAX_RETRIES = 3
        SCALE_THRESHOLD = 0.75   # s_x and s_y should not differ by more than 100%
        success = False

        # (Initialize variables to None in case loop fails)
        A, t, scale_vector, gps_centroid = None, None, None, None
        preds, camera_poses = None, None
        sfm_centers, gps_coords = None, None

        for attempt in range(MAX_RETRIES):
            preds, camera_poses = inference.inference(args) # Call sail-recon and estimate camera poses

            # Step 4 - Georeferencing the reconstruction
            # Get SfM and GPS coordinates -> (N, 3) numpy array
            sfm_centers, gps_coords, sfm_rot, real_world_rotations = georef.get_coord_rot(camera_poses, ref_df, filelist, height=0) # camera height = 1.96696
            A, translation, scale_vector = georef.align_with_non_uniform_scaling(sfm_centers, gps_coords)
            
            s_x, s_y, s_z = scale_vector
            avg_planar_scale = (s_x + s_y) / 2.0
            
            if avg_planar_scale < 1e-6: # Check for zero or near-zero scale
                print(f"Attempt {attempt + 1} failed: Average planar scale is near zero. Retrying...")
                continue
                
            scale_diff_ratio = np.abs(s_x - s_y) / avg_planar_scale
            
            if attempt > 0:
                SCALE_THRESHOLD *= 1.25  # Relax the threshold on the last attempt
                print(f"Attempt {attempt + 1}: Relaxing scale threshold to {SCALE_THRESHOLD*100:.1f}%.")

            if scale_diff_ratio <= SCALE_THRESHOLD:
                print(f"Attempt {attempt + 1} successful: Planar scales {s_x:.2f} and {s_y:.2f} are consistent (Difference: {scale_diff_ratio*100:.1f}%).")
                success = True
                break # Success! Exit the loop.
            else:
                print(f"Attempt {attempt + 1} failed: Planar scales {s_x:.2f} and {s_y:.2f} differ by {scale_diff_ratio*100:.1f}% (Threshold: {SCALE_THRESHOLD*100:.1f}%). Retrying...")
            
        if not success:
            print(f"CRITICAL: Failed to get a stable SfM solution after {MAX_RETRIES} attempts. Aborting.")
            # Continue to next cluster instead of exiting
            continue
        
        # preds, camera_poses = inference.inference(args) # Call sail-recon and estimate camera poses

        # Step 4 - Georeferencing the reconstruction
        # Get SfM and GPS coordinates -> (N, 3) numpy array

        R_convention = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        z_scale_np = np.eye(3) * np.array([1, 1, np.mean(scale_vector[0:2]) / scale_vector[2]])

        # Post-processing pitch and roll correction
        sfm_conv = sfm_centers @ R_convention.T
        georef_scaled_centers = sfm_conv @ A @ z_scale_np + translation
        
        # Correct pitch - by dot product the georef and gps vector
        # a. Find the georef sfm camera center vector
        vec_actual_norm = georef.get_primary_vector(georef_scaled_centers)
        # b. Find the "supposed" correct vector
        vec_supposed_norm = georef.get_primary_vector(gps_coords)
        # c. Find the pitch correction (axis and angle of rotation)
        # The axis of rotation is the horizontal vector perpendicular to the GPS path ("right" vector)
        vec_supposed_planar = vec_supposed_norm.copy()
        vec_supposed_planar[2] = 0.0 # Make it purely horizontal
        axis_of_rotation = np.cross(vec_supposed_planar, np.array([0.0, 0.0, 1.0]))
        axis_norm = axis_of_rotation / np.linalg.norm(axis_of_rotation)
        # The angle is the difference between the 'actual' pitch and the 'supposed' pitch
        angle_actual_rad = np.arcsin(np.clip(vec_actual_norm[2], -1.0, 1.0))
        angle_supposed_rad = np.arcsin(np.clip(vec_supposed_norm[2], -1.0, 1.0))
        angle_rad = angle_supposed_rad - angle_actual_rad # The correction angle
        # d. Create the correction rotation matrix
        rot_vec = axis_norm * angle_rad
        R_pitch = Rotation.from_rotvec(rot_vec).as_matrix()
        

        # Convert matrices to tensors
        R_convention_tensor = torch.tensor(R_convention, dtype=torch.float32).to(device)
        A_tensor = torch.tensor(A, dtype=torch.float32).to(device)
        trans_tensor = torch.tensor(translation, dtype=torch.float32).to(device)
        z_scale_tensor = torch.tensor(np.eye(3) * np.array([1, 1, np.mean(scale_vector[0:2]) / scale_vector[2]]), dtype=torch.float32).to(device)
        R_pitch_tensor_T = torch.tensor(R_pitch.T, dtype=torch.float32).to(device)
        # R_correction_tensor_T = torch.tensor(R_correction.T, dtype=torch.float32).to(device)

        # For debugging
        if bool_debug:
            print("Computed Affine Matrix A:\n", A)
            print("Computed Translation Vector t:\n", translation)
            print("Computed Scale Vector s:\n", scale_vector)
            print(f"Pitch correction:\n Actual pitch: {np.degrees(angle_actual_rad):.2f}, Supposed pitch: {np.degrees(angle_supposed_rad):.2f}")
            print(f"Applying pitch correction of {np.degrees(angle_rad):.2f} degrees around {axis_norm}.")
            # print(f"Found pitch error of {np.degrees(angle_rad):.2f} degrees.")     # Post-processing step c
            # print(f"Pitch matrix \n {R_pitch}")                                     # Post-processing step d
            print("Georeferenced SfM Centers:\n", georef_scaled_centers)
            print("GPS Coordinates:\n", gps_coords)

        # Retrieve the masks to only project the desired objects
        filelist_bn = [os.path.join(mask_dir, os.path.basename(f)) for f in filelist]
        for i in range(len(preds)):
            pts3d = preds[i]['point_map_by_unprojection'][0]  # (H, W, 3)
            # georef_cam_center = gps_coords[i]
            
            pts3d_tensor = torch.tensor(pts3d, dtype=torch.float32).to('cuda')  # ensure points are in tensor
            rotated_pts3d = pts3d_tensor @ R_convention_tensor.T @ A_tensor         # Apply rotation first
            georef_pts3d_tensor = rotated_pts3d @ z_scale_tensor @ R_pitch_tensor_T + trans_tensor
            preds[i]['pts3d_global_transformed'] = georef_pts3d_tensor.cpu()
                
            if bool_output_mask:
                # Apply mask here
                mask = cv2.imread(filelist_bn[i], cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (pts3d.shape[1], pts3d.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask_bool = mask.astype(bool)
                mask_expanded = mask_bool[:, :, None]  # Expand to (H, W, 1) for broadcasting
                georef_pts3d_masked = np.where(mask_expanded, preds[i]['pts3d_global_transformed'], np.nan)    # mask the full output point cloud directly
                preds[i]['pts3d_masked_transformed'] = torch.tensor(georef_pts3d_masked, dtype=torch.float32)
            
            # Check if the transformed points are in device
            preds[i]['pts3d_global_transformed'] = preds[i]['pts3d_global_transformed'].to(device)  # ensure points are in device
            if bool_output_mask:
                preds[i]['pts3d_masked_transformed'] = preds[i]['pts3d_masked_transformed'].to(device)  # ensure points are in device


        # Post-processing: decide whether a 180-degree roll is needed. If so, apply to all cameras
        if bool_output_mask:
            trsfm_key = ["pts3d_global_transformed", "pts3d_masked_transformed"]
        else:
            trsfm_key = ["pts3d_global_transformed"]
        preds = georef.need_180_roll(preds, trans_tensor, vec_supposed_norm, device, camera = gps_coords, transform_key = trsfm_key)

        if bool_output_all:
            georef_pc_fn = os.path.join(pc_dir, os.path.split(recon_image_dir)[-1], "all.ply")
            inference.save_pointcloud_with_plyfile(preds, mapkey="pts3d_global_transformed", filename=georef_pc_fn, downsample_ratio=1)
        if bool_output_mask:
            georef_masked_fn = os.path.join(pc_dir, os.path.split(recon_image_dir)[-1], "filt.ply")
            inference.save_pointcloud_with_plyfile(preds, mapkey="pts3d_masked_transformed", filename=georef_masked_fn, downsample_ratio=1)

    torch.cuda.empty_cache()

def find_sign_centre(pc_dir):
    # Utility function to find the centre of a sign point cloud
    entries = os.listdir(pc_dir)
    cluster_list = sorted([entry for entry in entries if os.path.isdir(os.path.join(pc_dir, entry))])
    asset_centre = []
    
    for dir_name in cluster_list:
        ply_path = os.path.join(pc_dir, dir_name, "filt.ply")
        if os.path.exists(ply_path):
            # Read the PLY file using Open3D
            pcd = o3d.io.read_point_cloud(ply_path)
            points = np.asarray(pcd.points)
            
            if len(points) == 0:
                print(f"Sign {dir_name}: No points in PLY file.")
                continue
            
            # Filter outliers: remove points with fewer than 2 neighbors within 0.1m
            if len(points) > 1:
                dist_matrix = cdist(points, points)
                neighbor_counts = np.sum(dist_matrix < 0.1, axis=1) - 1  # exclude self
                mask = neighbor_counts >= 2
                filtered_points = points[mask]
            else:
                filtered_points = points
            
            if len(filtered_points) == 0:
                print(f"Sign {dir_name}: No points after outlier filtering.")
                continue
            
            # Compute the bounding box on filtered points
            min_coords = np.min(filtered_points, axis=0)
            max_coords = np.max(filtered_points, axis=0)
            
            # Bottom centre: center of X and Y, at min Z
            bottom_centre = np.array([
                (min_coords[0] + max_coords[0]) / 2,
                (min_coords[1] + max_coords[1]) / 2,
                min_coords[2]
            ])
            
            # print(f"Sign {dir_name}: Bottom centre at {bottom_centre}")
            # Extract {i}_{j} from dir_name (assuming format like "max_{i}_{j}")
            parts = dir_name.split('_')
            i_j = f"{parts[-2]}_{parts[-1]}"
            _asset_dict = {'asset': i_j,
                           'assetX': bottom_centre[0],
                           'assetY': bottom_centre[1],
                           'assetZ': bottom_centre[2],}
            asset_centre.append(_asset_dict)
            del pcd
        else:
            print(f"Sign {dir_name}: filt.ply not found.")
    
    return asset_centre

def where_sign_clustered(sorted_dir):
    # Utility function to find where a sign image is clustered
    entries = os.listdir(sorted_dir)
    cluster_list = sorted([entry for entry in entries if os.path.isdir(os.path.join(sorted_dir, entry))])
    sign_location = []
    
    for dir_name in cluster_list:
        entry_path = os.path.join(sorted_dir, dir_name)
        filelist = sorted(glob.glob(os.path.join(entry_path, "*.jpg")))
        for file in filelist:
            base_fn = os.path.basename(file)
            sign_location.append({'file_name': base_fn, 'asset': dir_name})
    
    return sign_location

def sign_recon_wrapper(
        result_dir,
        img_comb,
        georef_file,
        sail_ckpt,
        pc_bool = {'bool_output_all': True, 'bool_output_mask': True, 'bool_debug': False},
):

    # wrapper function with some paths imported
    mask_dir = os.path.join(result_dir, f'sign_recon/mask_{img_comb}')
    sorted_dir = os.path.join(result_dir, f'sign_recon/grouped_{img_comb}')
    pc_dir = os.path.join(result_dir, f'sign_recon/pointclouds_{img_comb}')

    ref_df = pd.read_csv(georef_file)  # import ref_file for geolocation clustering
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # # Would imagine sign slicing and clustering are done in pre-processing
    # slice_signs(gemini_client, image_dir, intermediate_dir, mask_dir, model_cfg, checkpoint)
    # cluster_signs(gemini_client, image_dir, face_dir, sorted_dir, ref_df, device)

    # Run reconstruction and georeferencing
    # sign_recon_and_georef(sorted_dir, mask_dir, pc_dir, ref_df, sail_ckpt, device, pc_bool['bool_output_all'], pc_bool['bool_output_mask'], pc_bool['bool_debug'])

    # Get sign centres and which sign each image is clustered to
    print("Finding sign centres from reconstructed point clouds...")
    asset_centres = find_sign_centre(pc_dir)
    print("Finding which images belong to which signs...")
    sign_locations = where_sign_clustered(sorted_dir)

    # Merge and save to CSV
    sign_loc_df = pd.DataFrame(sign_locations)
    asset_centre_df = pd.DataFrame(asset_centres)
    output_df = sign_loc_df.merge(asset_centre_df, on='asset', how='left')
    output_df.to_csv(os.path.join(result_dir, f'sign_recon/sign_locations_{img_comb}.csv'), index=False)

# print(f"Processed {len(image_list)} images clustered into {len(cluster_list)} signs.")
# print(f"Time for Step 1a (Bbox detection): {time_1a - start_time:.2f} seconds")
# print(f"Time for Step 1 (Mask generation): {time_1 - time_1a:.2f} seconds")
# print(f"Time for Step 2 (Clustering): {time_2 - time_1:.2f} seconds")
# print(f"Time for Step 3 and 4 (Reconstruction and Georeferencing): {end_time - time_2:.2f} seconds")
  
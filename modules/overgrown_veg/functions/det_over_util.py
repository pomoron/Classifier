from PIL import Image
import torch
import numpy as np
from transformers import pipeline, AutoImageProcessor, AutoModelForDepthEstimation

def get_line_array(str):
    # Example line_str = '[[  26 1725  583 1348]]'
    
    # Remove the outer brackets and split the string by spaces
    line_list = str.strip('[]').split()
    # Convert the elements to integers
    line_list = list(map(int, line_list))
    
    return np.array(line_list)

def get_judge_mask(mask, point):
    new_mask = np.zeros_like(mask)
    pnt_x = point[0]
    new_mask[:,pnt_x:] = mask[:,pnt_x:]
    return new_mask
    
def check_overgrown(np_depth, veg_mask, end_point, depth_tol=0.01):
    end_depth = np_depth[end_point[1], end_point[0]]
    depth_mask = (np_depth >= (end_depth - depth_tol)) & (np_depth <= (end_depth + depth_tol))
    judge_mask = get_judge_mask(depth_mask, end_point)
    overgrown_mask = np.logical_and(veg_mask, judge_mask)

    return depth_mask, judge_mask, overgrown_mask

# # Create image
    # depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
    # depth = depth.detach().cpu().numpy() * 255
    # depth = Image.fromarray(depth.astype("uint8"))
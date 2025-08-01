import numpy as np
from . import plot as myplot

def merge_masks(mask1, mask2):
    """
    Merge two masks.
    Args:
        mask1 (np.array): First mask of shape (H, W).
        mask2 (np.array): Second mask of shape (H, W).
    Returns:
        np.array: Merged mask if IoU is above the threshold, otherwise None.
    """
    merged_mask = np.logical_or(mask1, mask2)
    return merged_mask

def mask_iou(masks):
    """
    Calculate the IoU of a list of masks.
    Args:
        masks (list): a list of masks of shape (N, H, W).
    Returns:
        np.array: outputs of a matrix of shape (len(masks), len(masks)).
    """
    num_masks = len(masks)
    mask_ious = np.zeros((num_masks, num_masks), dtype=np.float32)

    # Calculate the intersection and union of masks using logical AND and OR operation
    for i in range(num_masks):
        for j in range(i+1, num_masks):
            intersection = np.logical_and(masks[i], masks[j])
            union = np.logical_or(masks[i], masks[j])
            iou = np.sum(intersection) / np.sum(union)
            mask_ious[i, j] = iou
            mask_ious[j, i] = iou

    return mask_ious

# also need to fix areas
def find_remasked_bbox(mask):
    """
    Find the bounding box that fits the resized mask.

    Args:
        mask (np.array): Resized mask of shape (H', W').
        original_shape (tuple): Shape of the original image or mask (H, W).

    Returns:
        list: Bounding box coordinates in COCO format [x_min, y_min, delta_x, delta_y].
    """
    # Find the minimum and maximum coordinates that enclose the mask region
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return [x_min, y_min, x_max-x_min, y_max-y_min]

def df_combine_masks(df_check, height_i, width_i, iou_merge_thres=0.5):
    df_check = df_check.reset_index(drop=True)
    seg_polygons = df_check['segmentation'].to_list()
    seg_catid = df_check['category_id'].to_list()

    # # IoU now calculated per iteration because it should be re-calculated after merging each polygon
    # merge masks
    for i in range(len(seg_polygons)):
        for j in range(i + 1, len(seg_polygons)):
            # create the mask after merging
            concerned_polygons = [df_check.loc[i, 'segmentation'], df_check.loc[j, 'segmentation']]
            if all(concerned_polygons):
                seg_masks = []
                for idplygn, x in enumerate(concerned_polygons):
                    mask = myplot.polygon_to_mask(x, height_i, width_i)
                    seg_masks.append(mask)
                the_mask_iou = mask_iou(seg_masks)[0, 1]
                # the_mask_iou = mask_iou(seg_masks)[i,j]
                # dataframe fixing
                if the_mask_iou > iou_merge_thres and seg_catid[i]==seg_catid[j]:
                    # merged_mask = merge_masks(seg_masks[i],seg_masks[j])
                    merged_mask = merge_masks(seg_masks[0],seg_masks[1])
                    merged_polygon = myplot.mask_to_polygon(merged_mask)
                    plygn_rd = [[round(x,2) for x in merged_polygon[0][0]]]
                    # seg_concat = []
                    # seg_concat[:] = merged_polygon[0]
                    # add the new merged polygon to the i-th entry
                    df_check.at[i, 'segmentation'] = plygn_rd           # new segmentation for the new mask
                    df_check.at[i,'area'] = merged_polygon[1]                     # new area for the new mask
                    df_check.at[i,'bbox'] = find_remasked_bbox(merged_mask)       # new bbox for the new mask
                    # clear j-th entry 
                    df_check.at[j,'segmentation'] = []
                    df_check.at[j,'area'] = np.nan
                    df_check.at[j,'bbox'] = []
            else:
                continue
    
    df_check = df_check.dropna(subset=['area'])
    return df_check

def bbox_iou(bbox1, bbox2, bool_xywh=True):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    
    Parameters:
        bbox1: list or tuple [x1, y1, w, h] (top-left and bottom-right corners)
        bbox2: list or tuple [x1, y1, w, h] (top-left and bottom-right corners)
    
    Returns:
        IoU: float (Intersection over Union value)
    """
    # Extract coordinates
    if bool_xywh:
        x1_1, y1_1, w1, h1 = bbox1
        x2_1 = x1_1 + w1
        y2_1 = y1_1 + h1
        x1_2, y1_2, w2, h2 = bbox2
        x2_2 = x1_2 + w2
        y2_2 = y1_2 + h2
    else:
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

    # Calculate intersection coordinates
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    # Calculate merged bbox coordiantes
    x1_merged = min(x1_1, x1_2)
    y1_merged = min(y1_1, y1_2)
    x2_merged = max(x2_1, x2_2)
    y2_merged = max(y2_1, y2_2)

    # Compute intersection area
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    intersection_area = inter_width * inter_height

    # Compute union area
    bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = bbox1_area + bbox2_area - intersection_area

    # Compute IoU
    if union_area == 0:
        return 0.0, None  # Avoid division by zero
    iou = intersection_area / union_area
    if bool_xywh:
        merged_bbox = [x1_merged, y1_merged, x2_merged-x1_merged, y2_merged-y1_merged]
    else:
        merged_bbox = [x1_merged, y1_merged, x2_merged, y2_merged]
    merged_bbox = [round(x, 2) for x in merged_bbox]
    return iou, merged_bbox
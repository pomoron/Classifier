import cv2
import numpy as np
from sklearn.cluster import DBSCAN

def get_bright_mask(image, threshold=200):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)      # Convert the image to grayscale
    gray_blur = cv2.GaussianBlur(gray, (25, 25), 0)     # Apply Gaussian blur
    _, gray_thres = cv2.threshold(gray_blur, threshold, 255, cv2.THRESH_BINARY)   # Threshold the image to get white patches
    return gray_thres

def clustering(points, eps=50, min_samples=5):
    """
    Perform DBSCAN clustering on a set of points.
    
    Parameters:
    - points: List or np.array of [x, y] coordinates.
    - eps: The maximum distance between two points to be considered in the same neighborhood (DBSCAN parameter).
    - min_samples: The minimum number of points to form a cluster (DBSCAN parameter).

    Returns:
    - centroids: np.array of [x, y] coordinates representing the centroids of the clusters.
    """
    # Convert the points to a numpy array
    if isinstance(points, list):
        points_array = np.array(points)
    else:
        points_array = points

    # clean the number of points
    downsample_to = 30000
    if len(points_array) > downsample_to:
        points_array = points_array[np.random.choice(len(points_array), downsample_to, replace=False)]
    else:
        points_array = points_array

    # Perform DBSCAN clustering
    cluster = DBSCAN(eps=eps, min_samples=min_samples).fit(points_array)
    return cluster
    # mask = get_instance_mask(cluster, height, width)
    # return mask

def get_instance_mask(cluster, height, width, largest_only=True):
    labels = cluster.labels_
    points = cluster.components_

    if largest_only:
        unique_labels, counts = np.unique(labels, return_counts=True)
        unique_labels = unique_labels[counts > 0]
        if len(unique_labels) > 1:
            largest_label = unique_labels[np.argmax(counts)]
            labels[labels != largest_label] = -1
    
    mask = np.zeros((height, width), dtype=np.uint8)
    # Assign different labels to the clustered points in the mask
    for point, label in zip(points, labels):
        if label != -1:  # Ignore noise points (label == -1)
            cv2.circle(mask, tuple(point), radius=1, color=int(label + 1), thickness=-1)
    return mask

def merge_collinear_lines(lines, slope_threshold=0.05, c_threshold=10):
    # Calculate the slope of each line
    slopes = [(x[0][3] - x[0][1]) / (x[0][2] - x[0][0]) for x in lines]
    # Join collinear lines
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            line_constant_i = -slopes[i] * lines[i][0][0] + lines[i][0][1]    # find the c of y = mx + c
            line_constant_j = -slopes[j] * lines[j][0][0] + lines[j][0][1]
            # if m and c are similar
            if abs(slopes[i] - slopes[j]) < slope_threshold and abs(line_constant_i - line_constant_j) < c_threshold:   
                # Determine the new start and end points
                points = np.array([lines[i][0][:2], lines[i][0][2:], lines[j][0][:2], lines[j][0][2:]])
                start_point = points[np.argmin(points[:, 0] + points[:, 1])]
                end_point = points[np.argmax(points[:, 0] - points[:, 1])]
                new_line = [[start_point[0], start_point[1], end_point[0], end_point[1]]]

                # check if the slopes and constants are still the same
                new_slope = (new_line[0][3] - new_line[0][1]) / (new_line[0][2] - new_line[0][0])
                new_line_constant = -new_slope * new_line[0][0] + new_line[0][1]
                # if the m and c are still the same, replace the old line with the new line
                if abs(new_slope - slopes[i]) < slope_threshold and abs(new_line_constant - line_constant_i) < c_threshold:
                    lines[i] = new_line
                    lines[j] = [[0, 0, 0, 0]]
    
    # Remove the zero lines
    lines = [x for x in lines if not np.array_equal(np.squeeze(x), np.array([0, 0, 0, 0]))]
    slopes = [(x[0][3] - x[0][1]) / (x[0][2] - x[0][0]) for x in lines]
    return lines

def get_leftline_pvmt(mask):
    # 1. Find the outer boundary of the pavement mask
    if len(mask.shape) == 2:
        unique_values = np.unique(mask)
        if np.array_equal(unique_values, [0, 1]) or np.array_equal(unique_values, [0]) or np.array_equal(unique_values, [1]):
            # It is a "Binary mask"
            pvmt_contours, hierarchy = cv2.findContours(mask*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            # It is a "Grayscale image" with intensities [0,255]
            pvmt_contours, hierarchy = cv2.findContours(mask*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        # It is a "Color image" with intensities [0,255]
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)      # Convert the image to grayscale
        pvmt_contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)       # no *255 if the raw pavement image is used
    
    largest_pvmt_contour = max(pvmt_contours, key=cv2.contourArea)
    pvmt_contour_points = largest_pvmt_contour.squeeze()                     # shape (N,2)
    pcp_sorted_by_x = np.array(sorted(pvmt_contour_points, key=lambda p: (p[0], p[1]))) # Sort by x, then by y to find the leftmost portion

    # 2. Extract the portion with straight line (in case the boundary is not straight at a turn)
    pcp_x = pcp_sorted_by_x[:, 0]           # Extract x and y coordinates
    pcp_y = pcp_sorted_by_x[:, 1]
    old_settings = np.seterr(divide='ignore', invalid='ignore')     # Temporarily set NumPy to ignore divide by zero warnings
    dy_dx = np.gradient(pcp_y, pcp_x)       # Calculate first and second derivatives
    d2y_dx2 = np.gradient(dy_dx, pcp_x)
    np.seterr(**old_settings)               # Reset NumPy settings
    # Identify points of segments where the second derivative is close to zero
    curvature_threshold = 0.02 
    straight_line_indices = np.where(np.abs(d2y_dx2) < curvature_threshold)[0]
    straight_line_points = pcp_sorted_by_x[straight_line_indices]
    
    # 3. Find the leftmost segment that grows upwards in the image
    if straight_line_points.size > 0:       # in case when the second derivative method can't find straight lines
        start_point, end_point = line_counter(straight_line_points)
    else:
        start_point, end_point = line_counter(pcp_sorted_by_x)

    # The resulting straight line segment can be represented as:
    max_slope_line = np.array([[int(start_point[0]), int(start_point[1]),
                            int(end_point[0]), int(end_point[1])]])
    return max_slope_line

def line_counter(point_list):
    start_point = point_list[0]
    end_point = start_point  # will update as we move along the list
    straight_count = 1      # Initialize a counter (number of consecutive points satisfying our condition)
    # Iterate over the points from the second one onward
    for i in range(1, len(point_list)):
        prev_point = point_list[i - 1]
        curr_point = point_list[i]
        # For a valid upward (top of image) movement: 
        # x should increase and y should decrease (since y=0 is top).
        if curr_point[0] >= prev_point[0] and curr_point[1] <= prev_point[1]:
            straight_count += 1
            end_point = curr_point
        else:
            break
    return start_point, end_point

# Function to draw lines on the mask
def draw_lines_on_mask(mask, lines, color=[(255,0,0)], thickness=2):
    for i, line in enumerate(lines):
        if len(line) == 1:  # probably nested
            draw = line[0]
        else:               # probably not nested
            draw = line
        x1, y1, x2, y2 = draw
        cv2.line(mask, (x1, y1), (x2, y2), color[i], thickness)
    return mask
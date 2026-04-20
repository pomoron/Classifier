import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from GetInfo import GetInfo
from GetVanishingPoint import GetVanishingPoint
from TransformImage2Ground import TransformImage2Ground
from TransformGround2Image import TransformGround2Image
import cv2
import numpy as np
# For GetIPM_pytorch
import torch
import torch.nn.functional as F
from math import cos, sin, pi


class Info(object):
    def __init__(self, dct):
        self.dct = dct

    def __getattr__(self, name):
        return self.dct[name]

def GetIPM(image, camera_info, ipm_info):
    
    '''
    Inputs:
    images - cv2.imread(image_fn)
    camera_info - class Info, with items:
    Info({
        "focalLengthX": int(width / 100) * 100, # 1200.6831,         # focal length x
        "focalLengthY": int(height / 100) * 100, # 1200.6831,         # focal length y
        "opticalCenterX": int(width / 2), # 638.1608,        # optical center x
        "opticalCenterY": int(height / 2), # 738.8648,       # optical center y
        "cameraHeight": 1500, # 1879.8,  # camera height in `mm`
        "pitch": 2.5,           # rotation degree around x
        "yaw": 0,              # rotation degree around y
        "roll": 0              # rotation degree around z
    })
    ipmInfo - class Info, with items:
    Info({
        "inputWidth": width,
        "inputHeight": height,
        "left": 50,
        "right": width-50,
        "top": 2600,
        "bottom": height
    })

    Outputs:
    outImage - cv2.imwrite(out_fn, outImage)
    '''

    I = image

    R = I[:, :, :]
    height = int(I.shape[0]) # row y
    width = int(I.shape[1]) # col x

    cameraInfo = camera_info
    ipmInfo = ipm_info
    # # Template

    # IPM
    vpp = GetVanishingPoint(cameraInfo)
    vp_x = vpp[0][0]
    vp_y = vpp[1][0]
    ipmInfo.top = float(max(int(vp_y), ipmInfo.top))
    uvLimitsp = np.array([[vp_x, ipmInfo.right, ipmInfo.left, vp_x],
                [ipmInfo.top, ipmInfo.top, ipmInfo.top, ipmInfo.bottom]], np.float32)

    xyLimits = TransformImage2Ground(uvLimitsp, cameraInfo)
    row1 = xyLimits[0, :]
    row2 = xyLimits[1, :]
    xfMin = min(row1)
    xfMax = max(row1)
    yfMin = min(row2)
    yfMax = max(row2)
    xyRatio = (xfMax - xfMin)/(yfMax - yfMin)
    outImage = np.zeros((height,width,4), np.float32)
    outImage[:,:,3] = 255
    outRow = int(outImage.shape[0])
    outCol = int(outImage.shape[1])
    stepRow = (yfMax - yfMin)/outRow
    stepCol = (xfMax - xfMin)/outCol
    xyGrid = np.zeros((2, outRow*outCol), np.float32)
    y = yfMax-0.5*stepRow

    for i in range(0, outRow):
        x = xfMin+0.5*stepCol
        for j in range(0, outCol):
            xyGrid[0, (i-1)*outCol+j] = x
            xyGrid[1, (i-1)*outCol+j] = y
            x = x + stepCol
        y = y - stepRow

    # TransformGround2Image
    uvGrid = TransformGround2Image(xyGrid, cameraInfo)
    # mean value of the image
    means = np.mean(R)/255
    RR = R.astype(float)/255
    for i in range(0, outRow):
        for j in range(0, outCol):
            ui = uvGrid[0, i*outCol+j]
            vi = uvGrid[1, i*outCol+j]
            #print(ui, vi)
            if ui < ipmInfo.left or ui > ipmInfo.right or vi < ipmInfo.top or vi > ipmInfo.bottom:
                outImage[i, j] = 0.0
            else:
                x1 = np.int32(ui)
                x2 = np.int32(ui+0.5)
                y1 = np.int32(vi)
                y2 = np.int32(vi+0.5)
                x = ui-float(x1)
                y = vi-float(y1)
                try:
                    outImage[i, j, 0] = float(RR[y1, x1, 0])*(1-x)*(1-y)+float(RR[y1, x2, 0])*x*(1-y)+float(RR[y2, x1, 0])*(1-x)*y+float(RR[y2, x2, 0])*x*y
                    outImage[i, j, 1] = float(RR[y1, x1, 1])*(1-x)*(1-y)+float(RR[y1, x2, 1])*x*(1-y)+float(RR[y2, x1, 1])*(1-x)*y+float(RR[y2, x2, 1])*x*y
                    outImage[i, j, 2] = float(RR[y1, x1, 2])*(1-x)*(1-y)+float(RR[y1, x2, 2])*x*(1-y)+float(RR[y2, x1, 2])*(1-x)*y+float(RR[y2, x2, 2])*x*y
                except:
                    # print(f'point({i}, {j}) has errors. (x1, x2, y1, y2) = ({x1}, {x2}, {y1}, {y2})')
                    outImage[i, j] = 0.0

    outImage[-1,:] = 0.0 
    # show the result
    outImage = outImage * 255
    
    #print("finished");
    return outImage

def get_ipm_transform_matrix(camera_info, device='cpu'):
    """
    Computes the transformation matrix from ground coordinates (X, Y, Z=0) 
    to image coordinates (u, v). This is a vectorized version of the logic
    in TransformGround2Image.py.
    """
    c1 = cos(camera_info.pitch * pi / 180)
    s1 = sin(camera_info.pitch * pi / 180)
    c2 = cos(camera_info.yaw * pi / 180)
    s2 = sin(camera_info.yaw * pi / 180)

    fx = camera_info.focalLengthX
    fy = camera_info.focalLengthY
    cx = camera_info.opticalCenterX
    cy = camera_info.opticalCenterY

    # This is the matrix from TransformGround2Image.py
    matp = torch.tensor([
        [fx * c2 + c1 * s2 * cx, -fx * s2 + c1 * c2 * cx, -s1 * cx],
        [s2 * (-fy * s1 + c1 * cy), c2 * (-fy * s1 + c1 * cy), -fy * c1 - s1 * cy],
        [c1 * s2, c1 * c2, -s1]
    ], dtype=torch.float32, device=device)
    
    return matp

def GetIPM_pytorch(image_cv, camera_info, ipm_info):
    """
    PyTorch-based implementation of Inverse Perspective Mapping.
    
    Inputs:
    image_cv - Input image in OpenCV format (H, W, C) -> BGR, uint8
    camera_info - The camera parameters class
    ipm_info - The IPM parameters class
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")

    img_height, img_width = image_cv.shape[:2]
    
    # 1. Convert image to PyTorch tensor
    # OpenCV is HWC, PyTorch expects CHW or NCHW. Also, normalize to [0, 1].
    image_tensor = torch.from_numpy(image_cv).to(device).float() / 255.0
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # HWC to NCHW

    # 2. Define the output grid in the ground plane
    # This replaces the xyGrid creation loops
    # Using the same logic to find the ground plane boundaries
    # Note: GetVanishingPoint and TransformImage2Ground are only needed to define the
    # bounds of our IPM view, so we can keep them in numpy for simplicity as they
    # are run only once.
    # cameraInfo = camera_info
    # ipmInfo = ipm_info

    vpp = GetVanishingPoint(camera_info)
    vp_y = vpp[1][0]
    
    # Ensure ipm_info.top is a float
    ipm_info.top = float(max(int(vp_y), ipm_info.top))

    uvLimitsp = np.array([
        [vpp[0][0], ipm_info.right, ipm_info.left, vpp[0][0]],
        [ipm_info.top, ipm_info.top, ipm_info.top, ipm_info.bottom]
    ], dtype=np.float32)

    xyLimits = TransformImage2Ground(uvLimitsp, camera_info)
    xfMin, xfMax = min(xyLimits[0, :]), max(xyLimits[0, :])
    yfMin, yfMax = min(xyLimits[1, :]), max(xyLimits[1, :])

    # Create a grid of (X, Y) coordinates in the ground plane
    out_height, out_width = img_height, img_width # Or define custom output size
    y_ground = torch.linspace(yfMax, yfMin, out_height, device=device)
    x_ground = torch.linspace(xfMin, xfMax, out_width, device=device)
    
    # Use meshgrid to create a 2D grid
    grid_x, grid_y = torch.meshgrid(x_ground, y_ground, indexing='xy')

    # Add the Z coordinate (height) and a homogeneous coordinate
    # Shape of grid_x/y: (out_height, out_width)
    # We want a (3, out_height * out_width) tensor for matrix multiplication
    Z_coord = torch.full_like(grid_x, -camera_info.cameraHeight)
    
    # Stack to create ground points: shape (3, H, W)
    ground_points = torch.stack([grid_x, grid_y, Z_coord], dim=0)
    # Reshape for matrix multiplication: shape (3, H*W)
    ground_points = ground_points.view(3, -1)

    # 3. Get the transformation matrix and apply it
    transform_matrix = get_ipm_transform_matrix(camera_info, device)
    
    # Project ground points to image plane: (3, 3) @ (3, N) -> (3, N)
    image_points_homogeneous = torch.matmul(transform_matrix, ground_points)

    # 4. Homogeneous to Cartesian coordinates (perspective divide)
    u_homogeneous = image_points_homogeneous[0, :]
    v_homogeneous = image_points_homogeneous[1, :]
    w_homogeneous = image_points_homogeneous[2, :]
    
    u_cartesian = u_homogeneous / w_homogeneous
    v_cartesian = v_homogeneous / w_homogeneous

    # 5. Normalize coordinates for grid_sample
    # grid_sample expects coordinates in the range [-1, 1]
    # (u,v) -> (x_norm, y_norm)
    # x_norm = 2 * u / (W-1) - 1
    # y_norm = 2 * v / (H-1) - 1
    
    u_normalized = (2 * u_cartesian / (img_width - 1)) - 1
    v_normalized = (2 * v_cartesian / (img_height - 1)) - 1
    
    # Stack and reshape for grid_sample
    # grid_sample expects a grid of shape (N, H_out, W_out, 2)
    sampling_grid = torch.stack([u_normalized, v_normalized], dim=-1)
    sampling_grid = sampling_grid.view(1, out_height, out_width, 2)

    # 6. Perform the sampling
    # The `grid_sample` function handles bilinear interpolation automatically
    ipm_tensor = F.grid_sample(
        image_tensor,
        sampling_grid,
        mode='bilinear',
        padding_mode='zeros', # or 'border', 'reflection'
        align_corners=False
    )

    # 7. Convert back to OpenCV format
    # NCHW -> HWC
    ipm_image_np = ipm_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out_image = (ipm_image_np * 255).astype(np.uint8)
    
    return out_image


# if __name__ == '__main__':
#     # This is a dummy example of how to run it.
#     # You would need the helper functions GetVanishingPoint and TransformImage2Ground
#     # from your original files for the setup part.
#     from GetVanishingPoint import GetVanishingPoint
#     from TransformImage2Ground import TransformImage2Ground
    
#     # Create a dummy image
#     width, height = 2048, 2048
#     dummy_image = np.zeros((height, width, 3), dtype=np.uint8)
#     # Draw some lines to visualize the transformation
#     cv2.line(dummy_image, (width//2, height//2), (width//2, height-1), (255, 0, 0), 20)
#     cv2.line(dummy_image, (100, height-100), (width-100, height-100), (0, 255, 0), 20)
    
#     cameraInfo = Info({
#         "focalLengthX": 1600,
#         "focalLengthY": 1600,
#         "opticalCenterX": width / 2,
#         "opticalCenterY": height / 2,
#         "cameraHeight": 1500,  # mm
#         "pitch": 5,           # degrees
#         "yaw": 0,
#         "roll": 0
#     })

#     ipmInfo = Info({
#         "inputWidth": width,
#         "inputHeight": height,
#         "left": 50,
#         "right": width - 50,
#         "top": height // 2 + 100, # Start below horizon
#         "bottom": height
#     })

#     print("Running PyTorch IPM conversion...")
#     import time
#     start_time = time.time()
    
#     outImage = GetIPM_pytorch(dummy_image, cameraInfo, ipmInfo)
    
#     end_time = time.time()
#     print(f"Finished in {end_time - start_time:.4f} seconds.")

#     # Save or display the result
#     cv2.imwrite("original_image.png", dummy_image)
#     cv2.imwrite("ipm_output_pytorch.png", outImage)
#     print("Output image saved as ipm_output_pytorch.png")
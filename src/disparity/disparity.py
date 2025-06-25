import os
from PIL import Image
from ..pipeline import utils
import cv2
from tqdm.notebook import tqdm
import numpy as np
import plotly.graph_objs as go

def compute_disparity(left_gray, right_gray):
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=64,  # must be divisible by 16
        blockSize=5,
        P1=8 * 3 * 5 ** 2,
        P2=32 * 3 * 5 ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )
    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    return disparity

def disparity_depth_estimation(rectified_images_source, disparity_output, path_to_calibration_data, stop=None):
    calib_data = np.load(path_to_calibration_data)
    Q = calib_data["Q"]

    pairs = utils.get_image_paths_paired(rectified_images_source)
    if stop is None:
        stop = len(pairs)
    
    disparity_imgs = {}

    for base_id, (left_path, right_path) in tqdm([*pairs.items()][:stop], desc="Computing disparity maps", unit="pair"):
        label = utils.get_info(left_path)['label']
        # Load grayscale images
        imgL = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
        imgR = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

        # Compute disparity
        disparity = compute_disparity(imgL, imgR)
        print("Disparity stats:")
        print("\tmin:", np.min(disparity))
        print("\tmax:", np.max(disparity))
        print("\tmean:", np.mean(disparity))
        print(f"")
        print(f"Estimated distance:", np.mean(disparity[disparity > 0]))

        print(f"Label: \t{label}")
        

        # Normalize disparity to 0–255 for display
        disp_vis = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        disp_vis = disp_vis.astype(np.uint8)

        points3D = cv2.reprojectImageTo3D(disparity.astype(np.float32), Q)
        depth_map = points3D[:, :, 2]   # Z axis = depth


        # Plot point cloud of estimated depth
        plot_3d_point_cloud(disparity=disparity, Q=Q)

        # maximum depth estimation
        max_profile_depth = get_max_profile_depth(depth_map)
        print(max_profile_depth)

        # Convert arrays to PIL in mode "L"
        pilL  = Image.fromarray(imgL,  mode="L")
        pilD  = Image.fromarray(disp_vis, mode="L")
        pilR  = Image.fromarray(imgR,  mode="L")

        # Create a new grayscale canvas and paste side by side
        w, h = pilL.width, pilL.height
        combined = Image.new("L", (w * 3, h))
        combined.paste(pilL, (0,   0))
        combined.paste(pilD, (w,   0))
        combined.paste(pilR, (w*2, 0))

        # Save result
        save_path = os.path.join(disparity_output, f"{base_id}_disparity.png")
        combined.save(save_path)

        disparity_imgs[save_path] = pilD

    return disparity_imgs

def get_max_profile_depth(depth_map, mask=None):
    """
    Computes max-min depth within mask (or entire image if mask None).
    """
    if mask is None:
        valid = np.isfinite(depth_map) & (depth_map > 0)
    else:
        valid = mask & (depth_map > 0)
    if not np.any(valid):
        return None  # no valid points

    d = depth_map[valid]
    return float(np.max(d) - np.min(d))  # in same units as baseline/focal length (e.g. cm or mm)



def plot_3d_point_cloud(disparity, Q, max_points=100000, mask=None):
    # Reproject disparity to 3D
    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    depth_map = points_3D[:, :, 2]

    # Create mask for valid disparities (> 0)
    valid_mask = disparity > 0
    if mask is not None:
        valid_mask &= mask

    # Flatten all arrays
    x = points_3D[:, :, 0][valid_mask]
    y = points_3D[:, :, 1][valid_mask]
    z = points_3D[:, :, 2][valid_mask]

    # Subsample if necessary
    if len(x) > max_points:
        idx = np.random.choice(len(x), max_points, replace=False)
        x, y, z = x[idx], y[idx], z[idx]

    # Create 3D scatter plot
    scatter = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=1.5,
            color=z,             # Color by depth
            colorscale='Viridis',
            opacity=0.8,
        )
    )

    layout = go.Layout(
        title="3D Point Cloud from Disparity",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Depth (Z)',
            aspectmode='data'
        )
    )

    fig = go.Figure(data=[scatter], layout=layout)
    fig.show()
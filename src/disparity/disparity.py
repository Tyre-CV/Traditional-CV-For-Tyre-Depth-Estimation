import os
from PIL import Image
from ..pipeline import utils
import cv2
from tqdm.notebook import tqdm
import itertools
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt

def compute_disparity(left_gray, right_gray):
    window_size = 5
    min_disp = 0
    nDispFactor = 8
    num_disp = 16*nDispFactor - min_disp

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,  # must be divisible by 16
        blockSize=window_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=5,
        speckleWindowSize=0,
        speckleRange=1,
        preFilterCap = 63,
        mode=cv2.STEREO_SGBM_MODE_HH
    )
    
    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    return disparity

def disparity_depth_estimation(rectified_images_source, disparity_output, path_to_calibration_data, stop=1, visualise_hist=True):
    # calib_data = np.load(path_to_calibration_data)
    # Q = calib_data["Q"]

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
        # Normalize disparity to 0–255 for display
        disp_vis = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        disp_vis = disp_vis.astype(np.uint8)
        # points3D = cv2.reprojectImageTo3D(disparity.astype(np.float32), Q)
        # depth_map = points3D[:, :, 2]   # Z axis = depth
        # Plot point cloud of estimated depth
        #plot_3d_point_cloud(disparity=disparity, Q=Q)
        # maximum depth estimation
        #max_profile_depth = get_max_profile_depth(depth_map)
        #print(max_profile_depth)
        # Convert arrays to PIL in mode "L"

        # Optional: Histogram
        if visualise_hist:
            plt.figure(figsize=(4, 3))
            plt.hist(disparity.ravel(), bins=50, range=(np.min(disparity), np.max(disparity)))
            plt.title(f"Disparity Histogram - {base_id}")
            plt.xlabel("Disparity")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.tight_layout()
            hist_path = os.path.join(disparity_output, f"{base_id}_hist.png")
            plt.savefig(hist_path)
            plt.close()
        
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

def compute_left_right_consistency_error(disp_left, disp_right):
    h, w = disp_left.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x_r = (x - disp_left).astype(np.int32)
    x_r = np.clip(x_r, 0, w - 1)

    disp_right_sampled = disp_right[y, x_r]
    valid = (disp_left > 0) & (disp_right_sampled > 0)
    if not np.any(valid):
        return np.inf
    return np.mean(np.abs(disp_left[valid] - disp_right_sampled[valid]))

def optimize_sgbm_params_from_dir(rectified_images_source):
    pairs = utils.get_image_paths_paired(rectified_images_source)

    # Only use the first pair
    first_key = next(iter(pairs))
    left_path, right_path = pairs[first_key]

    # Load grayscale images
    imgL = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

    # Now run optimization on just this pair
    return optimize_sgbm_params(imgL, imgR)


def optimize_sgbm_params(imgL, imgR):
    best_score = np.inf
    best_params = None

    nDispFactors        = [4, 5, 6, 7]
    window_sizes        = [3, 5, 7]
    uniquenessRatios    = [5, 10, 15]
    speckleWindowSizes  = [0, 50]
    speckleRanges       = [1, 2]
    preFilterCaps       = [31, 63]
    minDisparities      = [0]

    param_grid = list(itertools.product(
        nDispFactors,
        window_sizes,
        uniquenessRatios,
        speckleWindowSizes,
        speckleRanges,
        preFilterCaps,
        minDisparities
    ))

    print(f"Total combinations to try: {len(param_grid)}")

    for combo in tqdm(param_grid, desc="Optimizing SGBM params", unit="combo"):
        nDispFactor, window_size, uniquenessRatio, speckleWindowSize, speckleRange, preFilterCap, minDisparity = combo
        num_disp = 16 * nDispFactor

        try:
            stereoL = cv2.StereoSGBM_create(
                minDisparity=minDisparity,
                numDisparities=num_disp,
                blockSize=window_size,
                P1=8 * 3 * window_size ** 2,
                P2=32 * 3 * window_size ** 2,
                disp12MaxDiff=1,
                uniquenessRatio=uniquenessRatio,
                speckleWindowSize=speckleWindowSize,
                speckleRange=speckleRange,
                preFilterCap=preFilterCap,
                mode=cv2.STEREO_SGBM_MODE_HH
            )

            stereoR = cv2.StereoSGBM_create(
                minDisparity=minDisparity,
                numDisparities=num_disp,
                blockSize=window_size,
                P1=8 * 3 * window_size ** 2,
                P2=32 * 3 * window_size ** 2,
                disp12MaxDiff=1,
                uniquenessRatio=uniquenessRatio,
                speckleWindowSize=speckleWindowSize,
                speckleRange=speckleRange,
                preFilterCap=preFilterCap,
                mode=cv2.STEREO_SGBM_MODE_HH
            )

            dispL = stereoL.compute(imgL, imgR).astype(np.float32) / 16.0
            dispR = stereoR.compute(imgR, imgL).astype(np.float32) / 16.0

            score = compute_left_right_consistency_error(dispL, dispR)

            if score < best_score:
                best_score = score
                best_params = {
                    "nDispFactor": nDispFactor,
                    "window_size": window_size,
                    "uniquenessRatio": uniquenessRatio,
                    "speckleWindowSize": speckleWindowSize,
                    "speckleRange": speckleRange,
                    "preFilterCap": preFilterCap,
                    "minDisparity": minDisparity,
                    "score": score
                }

        except Exception as e:
            print(f"Error in combo {combo}: {e}")
            continue

    print("Best parameters found:")
    print(best_params)
    return best_params
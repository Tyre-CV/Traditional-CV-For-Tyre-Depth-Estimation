from collections import defaultdict
import multiprocessing
import os
from PIL import Image
from ..utils import get_file_names, get_image_paths_paired, get_info
import cv2
from tqdm.notebook import tqdm
import itertools
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def compute_disparity(left_gray, right_gray):
    window_size = 5
    min_disp = 0
    nDispFactor = 8
    num_disp = 16*(nDispFactor - min_disp)

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,  # must be divisible by 16
        blockSize=window_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=5,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap = 63,
        mode=cv2.STEREO_SGBM_MODE_HH
    )
    
    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    return disparity

def _compute_one(args):
    left_path, right_path = args
    info = get_info(left_path)
    label = info['label']
    imgL = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
    disparity = compute_disparity(imgL, imgR)
    return label, disparity

def compute_disparities_per_label(rectified_images_source, output_path, stop=None, n_workers=None, normalise=True):
    pairs = get_image_paths_paired(rectified_images_source)

    if stop is None:
        stop = len(pairs)

    # shuffle & cut down to `stop`
    items = list(pairs.values())
    np.random.shuffle(items)
    items = items[:stop]

    disparities_per_label = defaultdict(list)

    # choose pool size
    if n_workers is None:
        n_workers = multiprocessing.cpu_count() - 1 # leave one core free for other tasks

    with multiprocessing.Pool(processes=n_workers) as pool:
        # imap_unordered yields (label, disparity) as they complete
        for label, disp in tqdm(
                pool.imap_unordered(_compute_one, items),
                total=len(items),
                desc="Computing disparity maps",
                unit="pair"
            ):
            disparities_per_label[label].append(disp)

    normalised_disparities_per_label = defaultdict(list)
    if normalise:
        for label, disp_list in disparities_per_label.items():
            n_images = len(disp_list)

            # if no images for this label, zero‐length or all zeros
            if n_images == 0:
                normalised_disparities_per_label[label] = np.array([], dtype=float)
                continue

            # concatenate & filter out the sentinel
            all_vals = np.concatenate([d.ravel() for d in disp_list])
            valid   = all_vals[all_vals != -1].astype(int)

            # build one bin per observed disparity value
            if valid.size == 0:
                # no valid pixels at all
                hist = np.zeros(0, dtype=float)
            else:
                counts = np.bincount(valid)           # counts[i] = total pixels==i across all images
                # divide by number of images, not number of pixels
                hist = counts / n_images              # hist[i] = avg. pixels per image with value i

            normalised_disparities_per_label[label] = hist
    
    # Save the disparities to a file
    np.savez(os.path.join(output_path, "disparities_per_label.npz"), **disparities_per_label)
    np.savez(os.path.join(output_path, "normalised_disparities_per_label.npz"), **normalised_disparities_per_label)


    return disparities_per_label, normalised_disparities_per_label
    

def disparity_depth_estimation(rectified_images_source, stop=None, mask=None):
    pairs = get_image_paths_paired(rectified_images_source)
    if stop is None:
        stop = len(pairs)

    for base_id, (left_path, right_path) in tqdm(list(pairs.items())[:stop],
                                               desc="Computing disparity maps",
                                               unit="pair"):
        label = get_info(left_path)['label']

        imgL = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
        imgR = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
        disparity = compute_disparity(imgL, imgR)

        # Apply mask
        if mask is None:
            disp_masked = np.where(disparity != -1, disparity, np.nan)
        else:
            disp_masked = np.where(((disparity >= mask[1]) | (disparity < mask[0])), np.nan, disparity)
            disp_masked = np.where(disparity != -1, disp_masked, np.nan)

        # Convert grayscale images to RGB for Plotly
        imgL_rgb = np.stack([imgL]*3, axis=-1)
        imgR_rgb = np.stack([imgR]*3, axis=-1)

        # Plot with Plotly
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=['Left', 'Disparity', 'Right'],
            horizontal_spacing=0.02,
            specs=[[{"type": "image"}, {"type": "heatmap"}, {"type": "image"}]]
        )

        fig.add_trace(go.Image(z=imgL_rgb), row=1, col=1)
        fig.add_trace(go.Heatmap(
            z=disp_masked,
            colorscale='Gray',
            colorbar=dict(title='Disparity (px)', lenmode='fraction', len=0.8),
            showscale=True
        ), row=1, col=2)
        fig.add_trace(go.Image(z=imgR_rgb), row=1, col=3)

        fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)

        fig.update_yaxes(autorange='reversed', row=1, col=2)

        fig.update_layout(
            title_text=f"{base_id}  —  label: {label}",
            width=900, height=300,
            margin=dict(t=50, l=10, r=10, b=10)
        )

        fig.show()


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




## Parameter optimisation stuff

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
    pairs = get_image_paths_paired(rectified_images_source)

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


###################################

def canny_edge_detection(image_dir, threshholds=(100, 200), stop=None):
    img_paths = get_file_names(image_dir, stop=stop)

    for img_path in tqdm(img_paths, desc="Canny Edge Detection", unit=" image"):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        edges = cv2.Canny(img, threshholds[0], threshholds[1])

        # Convert edges to RGB so Plotly can render it
        edges_rgb = np.stack([edges]*3, axis=-1)  # shape: H x W x 3

        # Plot with Plotly
        fig = go.Figure()
        fig.add_trace(go.Image(z=edges_rgb))
        fig.update_layout(
            title=f"Canny Edges for {os.path.basename(img_path)}",
            width=800, height=600,
            margin=dict(t=50, l=10, r=10, b=10)
        )
        fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
        fig.show()

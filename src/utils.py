import os
from collections import defaultdict
from PIL import Image
import numpy as np
from tqdm.notebook import tqdm
import cv2
# Utility

# Function to retrieve all file names in a directory
def get_file_names(directory, stop=None):
    file_names = []
    for root, dirs, files in os.walk(directory):
        files.sort()
        for file in tqdm(files[:stop],
                         desc="Scanning file (names)",
                         unit=" file"):
            if file.lower().endswith('.png'):
                file_names.append(os.path.join(root, file))
    return file_names

def unify_label_type(directory):
    file_names = get_file_names(directory)
    for file_path in tqdm(file_names,
                          desc="Unifying labels",
                          unit=" file"):
        file_info = get_info(file_path)
        file_dir = get_file_dir(file_path)
        new_file_name = f"{file_info['id']}_{file_info['label']}_{file_info['side'].upper()}.png"
        new_file_path = os.path.join(file_dir, new_file_name)
        if file_path != new_file_path:
            os.rename(file_path, new_file_path)

def get_image_paths_paired(directory, stop=None):
    file_pairs_paths = {} # Will store tuples of (left_image_path, right_image_path) with key = base_id
    file_names = get_file_names(directory, stop)
    for file_path in tqdm(file_names,
                          desc="Creating File-Pairs",
                          unit=" pair"):
        file_info = get_info(file_path)
        if file_info['side'].upper() == 'L':
            # If it's a left image, check for the corresponding right image
            base_id = file_info['id']
            right_image_path = os.path.join(get_file_dir(file_path), f"{base_id}_{file_info['label']}_R.png")
            if os.path.exists(right_image_path):
                file_pairs_paths[base_id] = (file_path, right_image_path)
            
        # Else case not needed, as we only want pairs of left and right images
    return file_pairs_paths

def relabel_images(image_dict, new_label):
    images = {}
    for file_path, img in tqdm(image_dict.items(),
                               desc="Relabeling images",
                               unit=" img"):
        file_info = get_info(file_path)
        new_file_name = f"{file_info['id']}_{new_label}_{file_info['side'].upper()}.png"
        new_file_path = os.path.join(get_file_dir(file_path), new_file_name)
        images[new_file_path] = img

    return images

# Function to get the filename without the directory and wihthout the extension
def get_file_name(file_path):
    base_name = os.path.basename(file_path)
    name, _ = os.path.splitext(base_name)
    return name

# Function to get the directory path of the passed file (i.e. the path without the file name)
def get_file_dir(file_path):
    return os.path.dirname(file_path)

# Function to get the id, side (l/r), and label of the passed file name
def get_info(file_path):
    file_name = get_file_name(file_path)
    parts = file_name.split('_')
    if len(parts) < 3:
        raise ValueError(f"File name '{file_name}' does not contain enough parts to extract id, side, and label.")
    
    id_part = parts[0]
    label_part = parts[1]
    side_part = parts[2].upper()
    # label needs to be passed from string to float
    # try:
    #     label_part = float(label_part)
    # except ValueError:
    #     raise ValueError(f"Label '{label_part}' in file name '{file_name}' is not a valid float.")
    
    return dict(
        id=id_part,
        side=side_part,
        label=label_part
    )

# Function to load all images from passed directory
def load_images_from_dir(directory, stop=None, batch_size=None):
    if batch_size is not None:
        assert isinstance(batch_size, int) and batch_size > 0 and batch_size % 2 == 0, \
            "batch_size must be a positive even integer."
    file_names = get_file_names(directory, stop)
    images = {}
    for idx, file_path in enumerate(tqdm(file_names,
                                         desc="Loading images",
                                         unit=" img")):
        images[file_path] = Image.open(file_path).convert('RGB')
        if batch_size is not None and (idx + 1) % batch_size == 0:
            yield images
            images = {}
    # Yield any remaining images
    if images and batch_size is not None:
        yield images

    if batch_size is None:
        return images
    
def load_images(file_paths, stop=None, batch_size=None):
    if batch_size is not None:
        assert isinstance(batch_size, int) and batch_size > 0 and batch_size % 2 == 0, \
            "batch_size must be a positive even integer."
    images = {}
    for idx, file_path in enumerate(tqdm(file_paths[:stop],
                                         desc="Loading images",
                                         unit=" img")):
        if os.path.exists(file_path):
            images[file_path] = Image.open(file_path).convert('RGB')
        else:
            print(f"File {file_path} does not exist.")
        if batch_size is not None and (idx + 1) % batch_size == 0:
            yield images
            images = {}
    # Yield any remaining images
    if images and batch_size is not None:
        yield images
    
    if batch_size is None:
        return images

# Function to save images to the output directory
def save_images(images_dict, output_path):
    os.makedirs(output_path, exist_ok=True)
    
    for file_path, img in tqdm(images_dict.items(),
                                  desc="Saving images",
                                  unit=" img"):
        out_fp = os.path.join(output_path, os.path.basename(file_path))
        img.save(out_fp, format='PNG', optimize=True) # PNG is lossless

# Function to get the size of a directory in bytes, kilobytes, megabytes, or gigabytes
def get_dir_size(directory, unit='MB'):
    units = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    unit = unit.upper()
    if unit not in units:
        raise ValueError(f"Unit '{unit}' not supported. Choose from {list(units.keys())}.")
    return total_size / units[unit]

# Function to get the size of a list of files in bytes, kilobytes, megabytes, or gigabytes
def get_size_of_files(file_paths, unit='MB'):
    units = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
    total_size = 0
    for file_path in file_paths:
        if os.path.exists(file_path):
            total_size += os.path.getsize(file_path)
        else:
            print(f"File {file_path} does not exist.")
    unit = unit.upper()
    if unit not in units:
        raise ValueError(f"Unit '{unit}' not supported. Choose from {list(units.keys())}.")
    return total_size / units[unit]

# Function to group stereo images by base id
def group_stereo(images_dict):
    # Returns a dict: base_id -> {'L': (file_path, image), 'R': (file_path, image)}.
    groups = defaultdict(dict)
    for fp, img in images_dict.items():
        info = get_info(fp)
        base_id = info['id']
        side = info['side'].upper()
        if side in ('L', 'R'):
            groups[base_id][side] = (fp, img)
    return groups

# Function to match histogram of one image to another
def match_histogram(src_arr, ref_arr):
    # remap src_arr histogram to match ref_arr
    hist_src, _ = np.histogram(src_arr.ravel(), bins=256, range=(0,255))
    hist_ref, _ = np.histogram(ref_arr.ravel(), bins=256, range=(0,255))
    # CDFs
    cdf_src = np.cumsum(hist_src).astype(np.float64)
    cdf_src /= cdf_src[-1]
    cdf_ref = np.cumsum(hist_ref).astype(np.float64)
    cdf_ref /= cdf_ref[-1]
    # mapping
    mapping = np.zeros(256, dtype=np.uint8)
    j = 0
    for i in range(256):
        while j < 255 and cdf_ref[j] < cdf_src[i]:
            j += 1
        mapping[i] = j
    return mapping[src_arr]

# Function to apply erosion or dilation to a single grayscale PIL Image
def apply_morphology_single(img, op='erode', kernel_size=3, shape='rect', iterations=1):
    if img.mode != 'L':
        img_l = img.convert('L')
    else:
        img_l = img
    arr = np.array(img_l, dtype=np.uint8)
    # Choose structuring element shape
    k = kernel_size
    if shape == 'rect':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    elif shape == 'ellipse':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    elif shape == 'cross':
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (k, k))
    else:
        raise ValueError(f"Unknown shape '{shape}'. Use 'rect', 'ellipse', or 'cross'.")
    # Apply op
    if op == 'erode':
        arr2 = cv2.erode(arr, kernel, iterations=iterations)
    elif op == 'dilate':
        arr2 = cv2.dilate(arr, kernel, iterations=iterations)
    else:
        raise ValueError(f"Unknown op '{op}'. Use 'erode' or 'dilate'.")
    # Convert back to PIL Image
    return Image.fromarray(arr2, mode='L')
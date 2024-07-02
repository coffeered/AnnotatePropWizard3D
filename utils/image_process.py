import math

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode, rotate
from gui.interactive_utils import (
    image_to_torch,
    index_numpy_to_one_hot_torch,
    overlay_davis,
    torch_prob_to_numpy_mask,
)

__all__ = [
    "cubic_interpolation",
    "normalize_slice",
    "normalize_volume",
    "get_side_pred",
    "rescale_centroid",
    "determine_degree",
    "rotate_predict",
]


def normalize_slice(slice_tensor, dtype=np.uint8):
    """
    Normalize and convert a 2D tensor or ndarray to a 3-channel uint8 image.

    Args:
        slice_tensor (torch.Tensor or np.ndarray): Input 2D slice to be normalized.
        dtype (np.dtype, optional): Desired data type for the output. Default is np.uint8.

    Returns:
        np.ndarray: Normalized 3-channel image of the specified dtype.
    """
    if torch.is_tensor(slice_tensor):
        slice_tensor = slice_tensor.cpu().numpy()

    slice_tensor = slice_tensor.astype(np.float32)
    slice_tensor -= slice_tensor.min()

    if slice_tensor.max() != 0:
        slice_tensor /= slice_tensor.max()

    slice_tensor *= 255
    slice_tensor = slice_tensor.astype(dtype)
    slice_tensor = np.stack([slice_tensor] * 3, axis=2)

    return slice_tensor


def normalize_volume(img):
    """
    Normalize a 3D tensor volume to the range [0, 255].

    Args:
        img (torch.Tensor): Input 3D tensor volume.

    Returns:
        torch.Tensor: Normalized tensor volume.
    """
    img -= img.min()
    if img.max() != 0:
        img /= img.max()
    img *= 255
    return img


def get_side_pred(
    pred, img, gt, rot_point, offset, i, processor, size=480, device="cuda"
):
    TRACE_NUM = 4
    # Transform ground truth, no background
    gt_transformed = torch.where(gt == (i + 1), 1, 0)

    target_rotation = int(rot_point[1])  # which is centroid z
    the_slice = normalize_slice(img[:, target_rotation])
    gt_slice = gt[:, target_rotation].clone()  # [cubix_size, cubix_size]

    # Convert slice to torch tensor and interpolate
    frame_torch = image_to_torch(the_slice, device=device)
    frame_torch = F.interpolate(frame_torch.unsqueeze(0), (size, size)).squeeze(0)

    # Interpolate ground truth slice
    mask_torch = cubic_interpolation(gt_slice, size=size)
    with torch.inference_mode():
        if offset == 0:
            init_mask_torch = torch.zeros(TRACE_NUM, size, size).to(device)
            init_mask_torch[0] = mask_torch

            processor.clear_memory()
            mask, _ = processor.step(frame_torch, init_mask_torch, idx_mask=False)

        else:
            mask, _ = processor.step(frame_torch)

    mask = F.interpolate(mask.unsqueeze(0), *(gt_slice.shape)).squeeze()
    mask = torch_prob_to_numpy_mask(mask)

    if mask.sum() == 0:
        return pred

    # Update prediction tensor
    slice_indices = [target_rotation - 1, target_rotation, target_rotation + 1]
    pred[:, slice_indices] = (
        torch.tensor(mask, dtype=torch.float32, device=device)
        .unsqueeze(1)
        .repeat(1, 3, 1)
    )

    return pred


def rotate_point(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    Args:
        origin (tuple): The origin point (ox, oy) around which to rotate.
        point (tuple): The point (px, py) to be rotated.
        angle (float): The rotation angle in radians.

    Returns:
        tuple: The rotated point (qx, qy).
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def cubic_interpolation(input_tensor, size, mode="trilinear", dtype=torch.float32):
    """
    Perform cubic interpolation on a 3D tensor to resize it to the specified size.

    Args:
        input_tensor (torch.Tensor): The input tensor with shape (z, x, y).
        size (int): The target size for all dimensions (will result in a tensor of shape (size, size, size)).
        dtype (torch.dtype): The desired data type for the output tensor. Default is torch.float32.

    Returns:
        torch.Tensor: The resized tensor with shape (size, size, size).
    """
    # Ensure the input tensor is in the correct dtype
    input_tensor = input_tensor.to(dtype)

    # Add batch and channel dimensions
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, z, y, x)

    # Perform interpolation
    output_tensor = F.interpolate(
        input_tensor, size=(size, size, size), mode=mode, align_corners=True
    )

    # Remove batch and channel dimensions
    output_tensor = output_tensor.squeeze(0).squeeze(0)  # Shape: (size, size, size)

    return output_tensor


def rescale_centroid(centroid, size_z, size_y, size_x, max_length):
    """
    Rescale the centroid coordinates to the target size.

    Args:
        centroid (np.ndarray): The centroid coordinates as a NumPy array.
        size_z (int): The original size of the z-dimension.
        size_x (int): The original size of the x-dimension.
        size_y (int): The original size of the y-dimension.
        max_length (int): The target size for rescaling.

    Returns:
        np.ndarray: The rescaled centroid coordinates.
    """
    centroid = np.array(centroid, dtype=np.float32)
    scale_factors = np.array(
        [max_length / size_z, max_length / size_y, max_length / size_x],
        dtype=np.float32,
    )
    rescaled_centroid = centroid * scale_factors
    return rescaled_centroid


def determine_degree(size_x, size_y, size_z):
    """
    Determine the degree based on the ratio of the maximum of size_x and size_y to size_z.

    Args:
        size_x (int): The size of the x-dimension.
        size_y (int): The size of the y-dimension.
        size_z (int): The size of the z-dimension.

    Returns:
        int: The determined degree.
    """
    img_volume_ratio = max(size_x, size_y) / size_z

    if img_volume_ratio < 6:
        degree = 5
    elif img_volume_ratio < 10:
        degree = 3
    else:
        degree = 2

    return degree


def rotate_predict(rot_dict, offset, degree, processor, device):
    """
    Rotate the elements of the rotation dictionary and update predictions.

    Args:
        rot_dict (dict): Dictionary containing 'img', 'pred', 'mask', 'point', and 'prop_index'.
        offset (int): The current rotation offset.
        degree (int): The degree of rotation.

    Returns:
        tuple: Updated rotation dictionary and new offset.
    """
    VOS_SIZE = 480
    if offset != 0:
        center = [rot_dict["img"].shape[1] // 2, rot_dict["img"].shape[2] // 2]
        angle_rad = -degree * np.pi / 180

        rot_dict["img"] = rotate(
            rot_dict["img"], degree, interpolation=InterpolationMode.BILINEAR
        )
        rot_dict["pred"] = rotate(rot_dict["pred"], degree)
        rot_dict["mask"] = rotate(rot_dict["mask"], degree)
        rot_dict["point"] = rotate_point(center, rot_dict["point"], angle_rad)

    rot_dict["pred"] = get_side_pred(
        pred=rot_dict["pred"],
        img=rot_dict["img"],
        gt=rot_dict["mask"],
        rot_point=rot_dict["point"],
        offset=offset,
        i=rot_dict["prop_index"],
        processor=processor,
        size=VOS_SIZE,
        device=device,
    )

    offset += degree

    return rot_dict, offset

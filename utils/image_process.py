import math
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode, rotate

from gui.interactive_utils import image_to_torch, torch_prob_to_numpy_mask

__all__ = [
    "determine_degree",
    "normalize_volume",
    "rescale_centroid",
    "reset_rotate",
    "rotate_predict",
    "interpolate_tensor",
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


def get_slice(img_array, mask_array, idx):
    """
    Get a specific slice from the input image and mask arrays.

    Args:
        img_array (np.ndarray): Input image array.
        mask_array (np.ndarray): Input mask array.
        idx (int): Index of the slice to retrieve.

    Returns:
        tuple: A tuple containing the normalized slice data and the corresponding mask data.
    """
    slice_data = np.stack([img_array[idx]] * 3, axis=2)
    mask_data = mask_array[idx]

    return slice_data, mask_data


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
    gt_transformed = torch.where(gt == (i + 1), 1, 0).float()

    target_rotation = int(rot_point[1])  # which is centroid z
    the_slice = normalize_slice(img[:, target_rotation])
    gt_slice = gt_transformed[:, target_rotation].clone()  # [cubic_size, cubic_size]

    # Convert slice to torch tensor and interpolate
    frame_torch = image_to_torch(the_slice, device=device)  # [3, x, y]
    # frame_torch = F.interpolate(frame_torch.unsqueeze(0), (size, size)).squeeze(0)
    frame_torch = interpolate_tensor(
        input_tensor=frame_torch, size=size, mode="bilinear"
    )  # [3, size, size]

    mask_torch = interpolate_tensor(
        input_tensor=gt_slice,
        size=size,
        mode="nearest",
    )
    with torch.inference_mode():
        if offset == 0:
            init_mask_torch = torch.zeros(TRACE_NUM, size, size).to(device)
            init_mask_torch[0] = mask_torch

            processor.clear_memory()
            mask = processor.step(
                frame_torch, init_mask_torch, idx_mask=False
            )  # [1, TRACE_NUM, size, size]

        else:
            mask = processor.step(frame_torch)  # [1, TRACE_NUM, size, size]

    mask = interpolate_tensor(
        input_tensor=mask,
        size=gt_slice.size(),
        mode="nearest",
    )
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


def interpolate_tensor(
    input_tensor: torch.Tensor,
    size,
    mode: str = "trilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Interpolates a tensor to a target size using a specified mode.

    Args:
    input_tensor (torch.Tensor): The tensor to interpolate.
    size (tuple or int): Target size. For "trilinear" and "bilinear",
                         should be a tuple of three and two integers respectively.
    mode (str): Interpolation mode: "trilinear", "bilinear", or "nearest".
                Default is "trilinear".
    align_corners (bool): Align corners. Default is True. Ignored for "nearest".

    Returns:
    torch.Tensor: The interpolated tensor.
    """
    # Determine the number of dimensions to unsqueeze based on the tensor shape
    match mode:
        case "trilinear":
            num_unsqueezes = 5 - input_tensor.dim()
        case "nearest" | "bilinear":
            if isinstance(size, int) or len(size) != 3:
                num_unsqueezes = 4 - input_tensor.dim()
            else:
                num_unsqueezes = 5 - input_tensor.dim()
            if mode == "nearest":
                align_corners = None
        case _:
            raise ValueError(f"Invalid mode: {mode}")

    for _ in range(num_unsqueezes):
        input_tensor = input_tensor.unsqueeze(0)

    # Interpolate the tensor
    output_tensor = F.interpolate(
        input_tensor, size=size, mode=mode, align_corners=align_corners
    )

    # Squeeze the tensor back to the original number of dimensions
    for _ in range(num_unsqueezes):
        output_tensor = output_tensor.squeeze(0)

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


def reset_rotate(rot_dict, centroid, offset):
    rot_dict["point"] = centroid[[2, 1]].astype(float)
    rot_dict["img"] = rotate(
        rot_dict["img"], -offset, interpolation=InterpolationMode.BILINEAR
    )
    rot_dict["pred"] = rotate(rot_dict["pred"], -offset)
    rot_dict["mask"] = rotate(rot_dict["mask"], -offset)

    return rot_dict


def crop_and_pad(
    input_tensor: torch.Tensor,
    y_min: int,
    y_max: int,
    x_min: int,
    x_max: int,
    padding: Tuple[int, int, int, int],
    device: torch.device,
) -> np.ndarray:
    """
    Crop and pad a tensor.

    Args:
        input_tensor (torch.Tensor): The input tensor to be cropped and padded.
        y_min (int): The minimum y-coordinate for cropping.
        y_max (int): The maximum y-coordinate for cropping.
        x_min (int): The minimum x-coordinate for cropping.
        x_max (int): The maximum x-coordinate for cropping.
        padding (Tuple[int, int, int, int]): A tuple of four integers (pad_left, pad_right, pad_top, pad_bottom) for padding.
        device (torch.device): The device to move the tensor to.
        dtype (torch.dtype, optional): The data type of the tensor (default is torch.float).

    Returns:
        np.ndarray: The cropped and padded tensor as a NumPy array.
    """
    crop = input_tensor[:, y_min:y_max, x_min:x_max].to(
        dtype=torch.float, device=device
    )
    pad_left, pad_right, pad_top, pad_bottom = padding
    padded = torch.nn.functional.pad(
        crop, (pad_left, pad_right, pad_top, pad_bottom, 0, 0), mode="constant", value=0
    )
    return padded.cpu().numpy()


def vos_step(processor, input_slice, size, device):
    frame_torch = image_to_torch(input_slice, device=device)
    frame_torch = F.interpolate(frame_torch.unsqueeze(0), (480, 480), mode="bilinear")
    frame_torch = frame_torch[0]

    with torch.inference_mode():
        prediction, logits = processor.step(frame_torch, end=True)

    prediction = torch_prob_to_numpy_mask(prediction)

    # prediction = torch.tensor(prediction).float().unsqueeze(0).unsqueeze(0)
    # prediction = F.interpolate(prediction, (gt.shape[0], gt.shape[1]))[0][0]
    prediction = interpolate_tensor(prediction, size=size)

    return frame_torch, prediction, logits


def sam_step(
    predictor, input_slice, logits, box, feature_size=256, multimask_output=False
):
    mask_logits = F.interpolate(
        logits[:, [1]], (feature_size, feature_size), mode="bilinear"
    )

    predictor.set_image(input_slice.astype(float))
    masks, scores, logits = predictor.predict(
        box=box,
        multimask_output=multimask_output,
        mask_input=mask_logits[:, 0].cpu().numpy(),
    )

    return masks

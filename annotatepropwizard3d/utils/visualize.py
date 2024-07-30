import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import morphology
from skimage.measure import label, regionprops
from torchvision.transforms.functional import rotate

from utils.image_process import interpolate_tensor, normalize_volume, rotate_point

__all__ = [
    "visualize",
    "show_mask",
    "show_points",
    "show_box",
]


def visualize(slce, predict_mask, gt, box, filename, mode="point"):
    if mode == "box":
        plt.figure(figsize=(5, 5))
        plt.imshow(slce, cmap="gray")

        new_gt = morphology.dilation(np.array(gt), morphology.square(4))
        new_gt = new_gt - gt
        show_mask(new_gt, plt.gca(), random_color=True)

        new_predict_mask = morphology.dilation(
            np.array(predict_mask), morphology.square(4)
        ).astype(float)
        new_predict_mask = new_predict_mask - predict_mask
        show_mask(new_predict_mask, plt.gca())

        output_folder = "output_box"
    elif mode == "point":
        show_mask(gt, plt.gca(), random_color=True)
        show_mask(predict_mask, plt.gca())

        output_folder = "output_point"
    else:
        raise ValueError("mode should be either 'box' or 'point'")

    plt.axis("off")
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, filename))


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.array([0, 1, 0, 0.5])
    else:
        color = np.array([1, 0, 0, 0.5])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=100):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker=".",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker=".",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def mask_propagation_visualize(img, predict_masks, output_folder):
    norm_img = normalize_volume(img)

    for z, predict_mask in predict_masks.items():
        the_slice, the_mask = norm_img[z], predict_masks[z]

        ch3_slice = cv2.cvtColor(the_slice.astype(np.uint8), cv2.COLOR_GRAY2RGB)

        dilated_mask = morphology.dilation(np.array(the_mask), morphology.square(4))
        dilated_mask = dilated_mask - the_mask

        ch3_slice[dilated_mask.astype(bool)] = np.array([0, 0, 255])

        cv2.imwrite(os.path.join(output_folder, f"{z}.png"), ch3_slice)


def visualize_zaxis_rotation(
    img_volume, mask_volume, spacing_yxz, output_folder, device, rotate_degree=3
):
    img_volume = img_volume.astype(float)
    norm_volume = normalize_volume(img_volume)
    labeled_mask_volume = label(mask_volume)

    spacing_zyx = np.array(spacing_yxz)[[2, 0, 1]]
    size_z, size_y, size_x = img_volume.shape

    resample_size = (norm_volume.shape * spacing_zyx).astype(int).tolist()

    tensor_img = interpolate_tensor(
        input_tensor=torch.tensor(norm_volume, dtype=float, device=device),
        size=resample_size,
        mode="nearest",
    )
    tensor_mask = interpolate_tensor(
        input_tensor=torch.tensor(
            labeled_mask_volume.astype(float), dtype=float, device=device
        ),
        size=resample_size,
        mode="nearest",
    )

    for i, prop in enumerate(regionprops(labeled_mask_volume)):
        resample_centroid = prop.centroid * np.array(
            [
                resample_size[0] / size_z,
                resample_size[1] / size_y,
                resample_size[2] / size_x,
            ]
        )

        current_degree = 0
        while current_degree < 360:
            print(current_degree)
            current_volume = rotate(tensor_img, current_degree)
            current_mask = rotate(tensor_mask, current_degree)
            current_centroid = rotate_point(
                [tensor_img.shape[1] // 2, tensor_img.shape[2] // 2],
                resample_centroid[[2, 1]],  # [z, y, x] -> [x, y]
                -current_degree * np.pi / 180,
            )
            current_img_plane = (
                current_volume[:, int(current_centroid[1])].cpu().numpy()
            )
            current_img_plane_RGB = cv2.cvtColor(
                current_img_plane.astype(np.uint8), cv2.COLOR_GRAY2RGB
            )
            current_mask_plane = current_mask[:, int(current_centroid[1])].cpu().numpy()
            dilated_mask_plane = morphology.dilation(
                morphology.convex_hull_image(current_mask_plane), morphology.square(4)
            )
            # print(dilated_mask_plane.dtype, current_mask_plane.dtype)
            dilated_mask_plane = dilated_mask_plane ^ current_mask_plane.astype(bool)
            current_img_plane_RGB[dilated_mask_plane] = np.array([0, 0, 255])
            cv2.imwrite(
                os.path.join(output_folder, f"{i}_{current_degree:03d}.png"),
                current_img_plane_RGB,
            )
            current_degree += rotate_degree

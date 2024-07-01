import matplotlib.pyplot as plt
from skimage import morphology
import os


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

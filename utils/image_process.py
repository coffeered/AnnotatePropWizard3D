import math

import torch
from torchvision.transforms.functional import InterpolationMode, rotate


def norm_slce(slce):
    if torch.is_tensor(slce):
        slce = slce.cpu().numpy()
    slce -= slce.min()
    slce /= slce.max()
    slce *= 255
    slce = slce.astype(np.uint8)
    slce = np.stack([slce, slce, slce], axis=2)
    return slce


def get_side_pred(predictor, pred, img, gt, rot_point, centroid, offset):

    slce = norm_slce(rot_img[:, int(rot_point[1])])

    predictor.set_image(slce)
    input_point = [[int(rot_point[0]), centroid[0]]]
    input_label = [1]

    proj = torch.nonzero(gt[centroid[0], int(rot_point[1])])
    if proj.shape[0] > 0:

        proj_min, proj_max = proj.min().cpu().numpy(), proj.max().cpu().numpy()
        input_point += [[proj_min + 5, centroid[0]], [proj_max - 5, centroid[0]]]
        input_label += [1, 1]

    input_point = np.array(input_point)
    input_label = np.array(input_label)

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    # visualize(slce, masks[0], gt[:, int(rot_point[1])].cpu().numpy(), input_point, input_label, f"{offset}.png")

    z = int(rot_point[1])

    pred[:, [z - 1, z, z + 1]] = (
        torch.tensor(masks[0]).unsqueeze(1).repeat(1, 3, 1).cuda()
    )

    return pred


def rotate_(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

import os
from glob import glob

import click
import numpy as np
import SimpleITK as sitk
import torch
from skimage.measure import label, regionprops
from tqdm.auto import tqdm

from cutie.inference.inference_core import InferenceCore
from cutie.model.cutie import CUTIE
from utils.checkpoint import download_ckpt
from utils.image_process import (
    determine_degree,
    interpolate_tensor,
    normalize_volume,
    rescale_centroid,
    reset_rotate,
    rotate_predict,
)
from utils.yaml_loader import yaml_to_dotdict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 512


def predict_case(folder, processor):
    case_tps, case_fns, case_fps = list(), list(), list()
    case_dices, case_size_zs = list(), list()

    try:
        img = sitk.ReadImage(os.path.join(folder, "axc.nii.gz"))
        img = sitk.GetArrayFromImage(img)

        mask = sitk.ReadImage(os.path.join(folder, "seg.nii.gz"))
        mask = sitk.GetArrayFromImage(mask)
    except Exception:
        return case_tps, case_fns, case_fps, case_dices, case_size_zs

    img = normalize_volume(img.astype(float))
    mask = label(mask)
    size_z, size_y, size_x = img.shape

    tensor_img = interpolate_tensor(
        input_tensor=torch.tensor(img, device=DEVICE),
        size=(MAX_LENGTH, MAX_LENGTH, MAX_LENGTH),
        mode="nearest",
    )
    tensor_mask = interpolate_tensor(
        input_tensor=torch.tensor(mask, device=DEVICE),
        size=(MAX_LENGTH, MAX_LENGTH, MAX_LENGTH),
        mode="nearest",
    )
    tensor_img = tensor_img.permute(1, 0, 2).contiguous()  # [z, y, x] -> [y, z, x]
    tensor_mask = tensor_mask.permute(1, 0, 2).contiguous()  # [z, y, x] -> [y, z, x]

    for i, prop in enumerate(regionprops(mask)):
        if prop.area == 0:
            continue

        centroid = rescale_centroid(
            prop.centroid,
            size_z=size_z,
            size_y=size_y,
            size_x=size_x,
            max_length=MAX_LENGTH,
        )
        centroid = centroid[[1, 0, 2]]  # [z, y, x] -> [y, z, x]

        rot_dict = {
            "img": tensor_img,
            "mask": tensor_mask,
            "point": centroid[[2, 1]],  # take (x, z)
            "prop_index": i,
            "prop": prop,
        }
        rot_dict["pred"] = torch.zeros_like(rot_dict["img"], device=DEVICE).float()

        degree = determine_degree(size_x=size_x, size_y=size_y, size_z=size_z)
        offset = 0
        while offset <= 90:
            rot_dict, offset = rotate_predict(
                rot_dict, offset, degree, processor, device=DEVICE
            )
        rot_dict = reset_rotate(rot_dict, centroid=centroid, offset=offset)

        rot_dict["pred"] = rot_dict["pred"].permute(1, 0, 2).contiguous()

        centroid = centroid[[1, 0, 2]]  # [y, z, x] -> [z, y, x]

        rot_dict["pred"] = interpolate_tensor(
            input_tensor=rot_dict["pred"],
            size=(size_z, size_y, size_x),
            mode="trilinear",
        )
        z_pred = (rot_dict["pred"].sum((1, 2)) > 0).int().to(DEVICE)
        z_gt = torch.tensor((mask.sum((1, 2)) > 0), device=DEVICE, dtype=int)

        # Calculate true positives, false positives, false negatives and dice score
        tp = (z_pred * z_gt).sum().item()
        fp = ((z_gt == 0) * z_pred).sum().item()
        fn = ((z_pred == 0) * z_gt).sum().item()
        dice = 2 * tp / (2 * tp + fp + fn)

        case_tps.append(tp)
        case_fps.append(fp)
        case_fns.append(fn)
        case_dices.append(dice)
        case_size_zs.append(size_z)
    return case_tps, case_fns, case_fps, case_dices, case_size_zs


@click.command()
@click.option(
    "--dataset",
    "-D",
    default="/volume/open-dataset-ssd/ai99/gen_data/meningioma",
    help="The medical dataset",
    type=click.Path(exists=True),
)
def run(dataset: str):
    with torch.inference_mode():
        config = yaml_to_dotdict("yaml/eval_config.yaml")

        # Load the network weights
        cutie = CUTIE(config).to(DEVICE).eval()
        if not os.path.isfile(config.weights):
            download_ckpt(
                ckpt_path=config.weights,
            )
        model_weights = torch.load(config.weights)
        cutie.load_weights(model_weights)
    processor = InferenceCore(cutie, cfg=config)

    cases_folders = glob(os.path.join(dataset, "*"))

    tps, fps, fns = list(), list(), list()
    dices, size_zs = list(), list()

    for case_folder in tqdm(cases_folders):
        case_tps, case_fns, case_fps, case_dices, case_size_zs = predict_case(
            folder=case_folder, processor=processor
        )
        tps.extend(case_tps)
        fps.extend(case_fps)
        fns.extend(case_fns)
        dices.extend(case_dices)
        size_zs.extend(case_size_zs)

    result_folder = "results"
    os.makedirs(result_folder, exist_ok=True)
    np.save(os.path.join(result_folder, "tps.npy"), tps)
    np.save(os.path.join(result_folder, "fps.npy"), fps)
    np.save(os.path.join(result_folder, "fns.npy"), fns)
    np.save(os.path.join(result_folder, "z_dices.npy"), dices)
    np.save(os.path.join(result_folder, "zs.npy"), size_zs)


if __name__ == "__main__":
    run()

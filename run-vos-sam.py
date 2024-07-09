import os
from glob import glob
from types import SimpleNamespace

import click
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from skimage.measure import label, regionprops
from tqdm.auto import tqdm

# from cutie.inference.inference_core import InferenceCore
from utils.inference_core_with_logits import InferenceCoreWithLogits
from cutie.model.cutie import CUTIE
from segment_anything import sam_model_registry
from segment_anything.predictor_sammed import SammedPredictor
from utils.checkpoint import download_ckpt
from utils.image_process import (
    normalize_volume,
    crop_and_pad,
    get_slice,
    vos_step,
    sam_step,
    interpolate_tensor,
    pad_box,
)
from utils.yaml_loader import yaml_to_dotdict
from gui.interactive_utils import (
    image_to_torch,
    index_numpy_to_one_hot_torch,
)


def predict_case(
    folder,
    sam_predictor,
    vos_processor,
):
    TRACE_NUM = 4
    case_dices, case_dice_3Ds = list(), list()
    try:
        sitk_img = sitk.ReadImage(os.path.join(folder, "data.nii.gz"))
        img = sitk.GetArrayFromImage(sitk_img)
        sitk_mask = sitk.ReadImage(os.path.join(folder, "seg.nii.gz"))
        mask = sitk.GetArrayFromImage(sitk_mask)
    except RuntimeError:
        return case_dices, case_dice_3Ds

    img = normalize_volume(img.astype(float))
    mask = label(mask)
    size_z, size_y, size_x = img.shape

    for i, prop in enumerate(regionprops(mask)):
        if prop.area == 0:
            continue

        tp_4_dice, pred_4_dice = 0, 0
        init_center = np.array(prop.centroid, dtype=int)

        if size_y > MODEL_SIZE or size_x > MODEL_SIZE:
            y_min = max(init_center[1] - MODEL_SIZE // 2, 0)
            y_max = min(init_center[1] + MODEL_SIZE // 2, size_y)
            x_min = max(init_center[2] - MODEL_SIZE // 2, 0)
            x_max = min(init_center[2] + MODEL_SIZE // 2, size_x)

            # new_center = (init_center[0], MODEL_SIZE // 2, MODEL_SIZE // 2)
            padding_top = (MODEL_SIZE - size_y) // 2
            padding_bottom = MODEL_SIZE - size_y - padding_top
            padding_left = (MODEL_SIZE - size_x) // 2
            padding_right = MODEL_SIZE - size_x - padding_left

            crop_img = crop_and_pad(
                img,
                y_min,
                y_max,
                x_min,
                x_max,
                (padding_left, padding_right, padding_top, padding_bottom),
                DEVICE,
            )
            crop_mask = crop_and_pad(
                mask,
                y_min,
                y_max,
                x_min,
                x_max,
                (padding_left, padding_right, padding_top, padding_bottom),
                DEVICE,
            )
        else:
            crop_img = np.array(img, dtype=float, copy=True)
            crop_mask = np.array(mask, dtype=int, copy=True)

        index_z = init_center[0]
        slice_data, mask_data = get_slice(crop_img, crop_mask, index_z)

        mask_transformed = np.where(mask_data == (i + 1), 1.0, 0.0)

        # Convert slice to torch tensor and interpolate
        frame_torch = image_to_torch(slice_data, device=DEVICE)  # [3, x, y]
        # init_frame_torch = frame_torch.clone()
        init_frame_torch = interpolate_tensor(
            input_tensor=frame_torch, size=MODEL_SIZE, mode="bilinear"
        )  # [3, size, size]

        # mask_transformed = torch.where(mask_data == (i + 1), 1, 0).float()
        mask_torch = index_numpy_to_one_hot_torch(mask_transformed, 2).to(DEVICE)
        mask_torch = interpolate_tensor(
            input_tensor=mask_torch, size=MODEL_SIZE, mode="nearest"
        )

        init_mask_torch = torch.zeros(TRACE_NUM, MODEL_SIZE, MODEL_SIZE).to(DEVICE)
        init_mask_torch[0] = mask_torch[1]

        vos_processor.clear_memory()
        vos_processor.step(init_frame_torch, init_mask_torch, idx_mask=False)

        temp_center_area = mask_transformed.sum()

        index_z += 1
        while index_z < size_z:
            slice_data, mask_data = get_slice(crop_img, crop_mask, index_z)
            mask_transformed = np.where(mask_data == (i + 1), 1.0, 0.0)
            if mask_transformed.sum() == 0:
                break

            frame_torch, prediction, logits = vos_step(
                processor=vos_processor,
                input_slice=slice_data,
                model_size=MODEL_SIZE,
                size=mask_transformed.shape,
                device=DEVICE,
            )

            regionprop = regionprops(prediction.int().cpu().numpy())
            if len(regionprop) == 0:
                break

            box = pad_box(regionprop[0].bbox, 1)
            masks = sam_step(
                sam_predictor,
                slice_data,
                logits,
                box,
            )

            # reset VOS
            mask_torch = torch.zeros(TRACE_NUM, MODEL_SIZE, MODEL_SIZE).to(DEVICE)
            resized_masks = F.interpolate(
                torch.tensor(masks, device=DEVICE).unsqueeze(0),
                (MODEL_SIZE, MODEL_SIZE),
            )[0]
            mask_torch[0] = resized_masks[0]
            vos_processor.step(frame_torch, mask_torch, idx_mask=False)

            current_tp = (mask_transformed * masks[0]).sum()
            current_pred = masks[0].sum()
            tp_4_dice += current_tp
            pred_4_dice += current_pred
            case_dice = 2 * current_tp / (mask_transformed.sum() + current_pred + 1e-8)
            case_dices.append(case_dice)

            index_z += 1

        vos_processor.clear_memory()
        vos_processor.step(init_frame_torch, init_mask_torch, idx_mask=False)

        index_z = init_center[0] - 1
        while index_z >= 0:
            slice_data, mask_data = get_slice(crop_img, crop_mask, index_z)
            mask_transformed = np.where(mask_data == (i + 1), 1.0, 0.0)
            if mask_transformed.sum() == 0:
                break

            frame_torch, prediction, logits = vos_step(
                processor=vos_processor,
                input_slice=slice_data,
                model_size=MODEL_SIZE,
                size=mask_transformed.shape,
                device=DEVICE,
            )

            regionprop = regionprops(prediction.int().cpu().numpy())
            if len(regionprop) == 0:
                break

            box = pad_box(regionprop[0].bbox, 1)
            masks = sam_step(
                sam_predictor,
                slice_data,
                logits,
                box,
            )

            # reset VOS
            mask_torch = torch.zeros(TRACE_NUM, MODEL_SIZE, MODEL_SIZE).to(DEVICE)
            resized_masks = F.interpolate(
                torch.tensor(masks, device=DEVICE).unsqueeze(0),
                (MODEL_SIZE, MODEL_SIZE),
            )[0]
            mask_torch[0] = resized_masks[0]
            vos_processor.step(frame_torch, mask_torch, idx_mask=False)

            current_tp = (mask_transformed * masks[0]).sum()
            current_pred = masks[0].sum()
            tp_4_dice += current_tp
            pred_4_dice += current_pred
            case_dice = 2 * current_tp / (mask_transformed.sum() + current_pred + 1e-8)
            case_dices.append(case_dice)

            index_z -= 1

        if prop.area - temp_center_area > 0:
            case_dice_3D = (
                2 * tp_4_dice / (prop.area + pred_4_dice - temp_center_area + 1e-8)
            )
            case_dice_3Ds.append(case_dice_3D)
            tqdm.write(
                f"case: {os.path.basename(folder)}, label_idx: {i}, {case_dice_3D} {np.mean(case_dices)}"
            )
    return case_dices, case_dice_3Ds


@click.command()
@click.option(
    "--sam_checkpoint",
    "-ckpt",
    default="weights/sam_vit_b_01ec64.pth",
    help="The checkpoint path that SAM core is using",
)
@click.option(
    "--dataset",
    "-D",
    default="/volume/open-dataset-ssd/ai99/gen_data/meningioma",
    help="The medical dataset",
    type=click.Path(exists=True),
)
def run(sam_checkpoint: str, dataset: str):
    # defiune SAM components
    global MODEL_SIZE, DEVICE

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "sam-med2d" in sam_checkpoint:
        MODEL_SIZE = 256
        encoder_adapter = True
        image_size = 256
    else:
        MODEL_SIZE = 480
        encoder_adapter = False
        image_size = 1024
    args = dict(
        image_size=image_size,
        encoder_adapter=encoder_adapter,
        sam_checkpoint=sam_checkpoint,
    )
    args = SimpleNamespace(**args)
    print(args)

    if not os.path.isfile(args.sam_checkpoint):
        download_ckpt(ckpt_path=args.sam_checkpoint)

    model = sam_model_registry["vit_b"](args).to(DEVICE)
    sam_predictor = SammedPredictor(model)

    # define VOS components
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
    vos_processor = InferenceCoreWithLogits(cutie, cfg=config)

    # tps, fps, fns = list(), list(), list()
    dices, dices_3D = list(), list()
    cases_folders = sorted(glob(os.path.join(dataset, "*")))
    for case_folder in tqdm(cases_folders):
        with torch.inference_mode():
            case_dices, case_dices_3D = predict_case(
                folder=case_folder,
                sam_predictor=sam_predictor,
                vos_processor=vos_processor,
            )
        dices.extend(case_dices)
        dices_3D.extend(case_dices_3D)


if __name__ == "__main__":
    run()

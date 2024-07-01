import os
from types import SimpleNamespace

import click
import sitk
import torch
from glob2 import glob
from skimage.measure import label, regionprops
from tqdm.auto import tqdm

from segment_anything import sam_model_registry
from segment_anything.predictor_sammed import SammedPredictor
from utils.checkpoint import download_ckpt
from utils.constant import MIN_MASK_AREA
from utils.image_process import get_side_pred, norm_slce, rotate


def predict_case(predictor, folder: str, degree: int = 30, offset: int = 0):
    case_dices, case_dicesb, case_dicesc = list(), list(), list()
    case_mms, case_mmms = list(), list()

    try:
        img = sitk.ReadImage(os.path.join(folder, "axc.nii.gz"))
        img = sitk.GetArrayFromImage(img)

        mask = sitk.ReadImage(os.path.join(folder, "seg.nii.gz"))
        mask = sitk.GetArrayFromImage(mask)
    except:
        return case_dices, case_dicesb, case_dicesc, case_mms, case_mmms

    mask = label(mask)

    for i, prop in enumerate(regionprops(mask)):
        if prop.area == 0:
            continue

        centroid = ((np.array(prop.bbox[:3]) + np.array(prop.bbox[3:])) / 2).astype(int)
        rot_img = torch.tensor(img.astype(float)).cuda()
        rot_mask = torch.tensor(mask.astype(float)).cuda()
        rot_pred = torch.zeros_like(rot_img).float().cuda()
        rot_point = centroid[[2, 1]].astype(float)

        slce = norm_slce(rot_img[:, int(rot_point[1])])
        rot_pred = get_side_pred(
            predictor, rot_pred, slce, rot_mask, rot_point, centroid, offset
        )

        while offset < 180:
            rot_img = rotate(rot_img, degree)
            rot_pred = rotate(rot_pred, degree)
            rot_mask = rotate(rot_mask, degree)
            rot_point = rotate(
                [rot_img.shape[1] // 2, rot_img.shape[2] // 2],
                rot_point,
                -degree * np.pi / 180,
            )
            rot_pred = get_side_pred(
                predictor, rot_pred, rot_img, rot_mask, rot_point, centroid, offset
            )
            offset += degree

        rot_pred = rotate(rot_pred, -offset)
        z_nonzero = torch.nonzero(rot_pred.sum((1, 2))).flatten()

        dices_tp, dices_pred = 0, 0
        for z in z_nonzero:
            slce = img[z].astype(float)

            input_point, input_label = list(), list()

            regionprop = regionprops(rot_pred[z].int().cpu().numpy())

            if len(regionprop) == 0:
                continue

            box, area = regionprop[0].bbox, regionprop[0].area

            gt = np.copy(mask[z])
            gt[gt != (i + 1)] = 0
            gt[gt == (i + 1)] = 1

            if area <= MIN_MASK_AREA:
                continue
            pad = 1.0
            h, w = box[3] - box[1], box[2] - box[0]
            y, x = (box[3] + box[1]) / 2, (box[2] + box[0]) / 2
            box = np.array(
                [y - h / 2 * pad, x - w / 2 * pad, y + h / 2 * pad, x + w / 2 * pad]
            )

            predictor.set_image(norm_slce(slce))
            masks, scores, logits = predictor.predict(
                box=box,
                multimask_output=False,
            )

            dices_tp += (gt * masks[0]).sum()
            dices_pred += masks[0].sum()

            # visualize(slce, masks[0], gt, input_point, input_label, f"{os.path.basename(di)}_{z}.jpg")

            dice = (
                2 * ((gt * masks[0]).sum() + 1e-5) / (gt.sum() + masks[0].sum() + 1e-5)
            )
            dicesc.append(dice)
            mms.append(area)

        torch.cuda.empty_cache()
        dicesb_ = 2 * dices_tp / (dices_pred + prop.area)
        dicesb.append(dicesb_)
        mmms.append(prop.area)

    return case_dices, case_dicesb, case_dicesc, case_mms, case_mmms


@click.command()
@click.option("--image_size", default=256, help="Size of image.")
@click.option("--encoder_adapter/--no-encoder_adapter", default=True)
@click.option(
    "--sam_checkpoint",
    "-ckpt",
    default="sam-med2d_b.pth",
    help="The checkpoint that SAM core is using",
)
@click.option(
    "--dataset",
    "-D",
    default="/volume/open-dataset-ssd/ai99/gen_data/neuroma",
    help="The medical dataset",
    type=click.Path(exists=True),
)
def run(image_size: int, encoder_adapter: bool, sam_checkpoint: str, dataset: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = dict(
        image_size=image_size,
        encoder_adapter=encoder_adapter,
        sam_checkpoint=sam_checkpoint,
    )
    args = SimpleNamespace(**args)
    print(args)

    if not os.path.isfile(args.sam_checkpoint):
        download_ckpt(ckpt_name=args.sam_checkpoint, root=os.getcwd())

    model = sam_model_registry["vit_b"](args).to(device)
    predictor = SammedPredictor(model)

    cases_folders = glob(os.path.join(dataset, "*"))

    dices, dicesb, dicesc = list(), list(), list()
    mms, mmms = list(), list()

    for case_folder in cases_folders:
        case_dices, case_dicesb, case_dicesc, case_mms, case_mmms = predict_case(
            predictor, folder=case_folder, degree=15
        )
        dices.extend(cur_dices)
        dicesb.extend(cur_dicesb)
        dicesc.extend(cur_dicesc)
        mms.extend(cur_mms)
        mmms.extend(cur_mmms)


if __name__ == "__main__":
    run()

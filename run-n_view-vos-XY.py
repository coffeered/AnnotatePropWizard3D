import os

import click
import torch

from cutie.inference.inference_core import InferenceCore
from cutie.inference.utils.args_utils import get_dataset_cfg
from cutie.model.cutie import CUTIE
from utils.checkpoint import download_ckpt
from utils.yaml_loader import yaml_to_dotdict
from utils.image_process import norm_volume

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 512


def predict_case(folder):
    case_dices, case_dicesb, case_dicesc = list(), list(), list()
    case_mms, case_mmms = list(), list()

    try:
        img = sitk.ReadImage(os.path.join(folder, "axc.nii.gz"))
        img = sitk.GetArrayFromImage(img)

        mask = sitk.ReadImage(os.path.join(folder, "seg.nii.gz"))
        mask = sitk.GetArrayFromImage(mask)
    except:
        return case_dices, case_dicesb, case_dicesc, case_mms, case_mmms

    img = norm_volume(img)
    mask = label(mask)

    for i, prop in enumerate(regionprops(mask)):
        if prop.area == 0:
            continue
        centroid = np.array(prop.centroid, dtype=np.float32)
        tensor_img = torch.tensor(img, dtype=torch.float32, device=DEVICE)
        tensor_mask = torch.tensor(mask, dtype=torch.float32, device=DEVICE)

        tensor_img = F.interpolate(
            tensor_img.unsqueeze(0).unsqueeze(0),
            [MAX_LENGTH, MAX_LENGTH, MAX_LENGTH],
        )[0][0]
        tensor_mask = F.interpolate(
            tensor_mask.unsqueeze(0).unsqueeze(0),
            [MAX_LENGTH, MAX_LENGTH, MAX_LENGTH],
        )[0][0]


@click.command()
@click.option(
    "--dataset",
    "-D",
    default="/volume/open-dataset-ssd/ai99/gen_data/neuroma",
    help="The medical dataset",
    type=click.Path(exists=True),
)
def run(dataset: str):
    with torch.inference_mode():
        config = yaml_to_dotdict("yaml/eval_config.yaml")

        # Load the network weights
        cutie = CUTIE(config).cuda().eval()
        if not os.path.isfile(config.weights):
            download_ckpt(os.path.basename(model_weights))
        model_weights = torch.load(config.weights)
        cutie.load_weights(model_weights)
    processor = InferenceCore(cutie, cfg=config)

    cases_folders = glob(os.path.join(dataset, "*"))

    dices, dicesb, dicesc = list(), list(), list()
    mms, mmms = list(), list()

    for case_folder in cases_folders:
        predict_case(folder=case_folder)


if __name__ == "__main__":
    run()

import os
from glob import glob

import numpy as np
import SimpleITK as sitk
import torch
from skimage.measure import regionprops
from tqdm import tqdm

from .xyroll_prediction import XYrollPrediction

model = XYrollPrediction(
    "/volume/willy-dev/sota/3DAnnotatePropWizard/yaml/eval_config.yaml"
)

cases_folders = glob(
    os.path.join("/volume/open-dataset-ssd/ai99/gen_data/meningioma", "*")
)
for case_folder in tqdm(cases_folders):
    sitk_img = sitk.ReadImage(os.path.join(case_folder, "axc.nii.gz"))
    img = sitk.GetArrayFromImage(sitk_img)

    sitk_mask = sitk.ReadImage(os.path.join(case_folder, "seg.nii.gz"))
    mask = sitk.GetArrayFromImage(sitk_mask)

    mask = mask.astype(np.uint8)

    for prop in regionprops(mask):
        init_center = np.array(prop.centroid, dtype=int)
        index_z = init_center[0]

        with torch.inference_mode():
            result = model.predict(img, mask[index_z], index_z)

        result = result.sum((1, 2)) > 0
        mask = mask.sum((1, 2)) > 0

        f1 = 2 * (result * mask).sum() / (result.sum() + mask.sum())

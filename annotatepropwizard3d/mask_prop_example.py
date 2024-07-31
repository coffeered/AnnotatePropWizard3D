import os
import SimpleITK as sitk
from tqdm import tqdm
from glob import glob
import numpy as np
from annotatepropwizard3d.mask_propagation import MaskPropagation
from skimage.measure import regionprops
import torch

model = MaskPropagation(
    "/volume/willy-dev/sota/3DAnnotatePropWizard/weights/sam_vit_b_01ec64.pth",
    "/volume/willy-dev/sota/3DAnnotatePropWizard/yaml/eval_config.yaml",
)

cases_folders = sorted(
    glob(os.path.join("/volume/open-dataset-ssd/ai99/gen_data/meningioma", "*"))
)
for case_folder in tqdm(cases_folders):
    sitk_img = sitk.ReadImage(os.path.join(case_folder, "axc.nii.gz"))
    img = sitk.GetArrayFromImage(sitk_img)

    sitk_mask = sitk.ReadImage(os.path.join(case_folder, "seg.nii.gz"))
    mask = sitk.GetArrayFromImage(sitk_mask)

    mask = mask.astype(np.uint8)
    labeled_mask = np.zeros_like(mask)

    for prop in regionprops(mask):
        init_center = np.array(prop.centroid, dtype=int)
        index_z = init_center[0]
        labeled_mask[index_z] = mask[index_z]

    with torch.inference_mode():
        result = model.predict_by_volume(img, labeled_mask)

    f1 = 2 * (result * mask).sum() / (result.sum() + mask.sum())
    print(f1)

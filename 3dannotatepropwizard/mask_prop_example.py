import os
import gdown
import SimpleITK as sitk
from tqdm import tqdm
from glob import glob
import numpy as np
from mask_propagation import MaskPropagation
from skimage.measure import label, regionprops
import torch

url = 'https://drive.google.com/file/d/1TtQyLI0X6nq90n0dw3zJFFZZzgrZj8br/view?usp=sharing'
output = 'amos_0001.data.nii.gz'
gdown.download(url, output, quiet=False)

url = 'https://drive.google.com/file/d/1MgfbufE3802ZsNwQO8Xloce1ezXMszpM/view?usp=sharing'
output = 'amos_0001.label.nii.gz'
gdown.download(url, output, quiet=False)

model = MaskPropagation('../weights/sam_vit_b_01ec64.pth', '../yaml/eval_config.yaml')

sitk_img = sitk.ReadImage('amos_0001.data.nii.gz')
img = sitk.GetArrayFromImage(sitk_img)

sitk_mask = sitk.ReadImage('amos_0001.label.nii.gz')
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
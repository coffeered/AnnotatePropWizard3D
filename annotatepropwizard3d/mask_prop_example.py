import os
import SimpleITK as sitk
from tqdm import tqdm
from glob import glob
import numpy as np
from mask_propagation import MaskPropagation
from skimage.measure import label, regionprops
import torch
import gdown

output = 'amos_0001_data.nii.gz'
if os.path.exists(output):
    file_id = '1TtQyLI0X6nq90n0dw3zJFFZZzgrZj8br'
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output)

output = 'amos_0001_label.nii.gz'
if os.path.exists(output):
    file_id = '1MgfbufE3802ZsNwQO8Xloce1ezXMszpM'
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output)

model = MaskPropagation('../weights/sam_vit_b_01ec64.pth', '../yaml/eval_config.yaml')

sitk_img = sitk.ReadImage('amos_0001_data.nii.gz')
img = sitk.GetArrayFromImage(sitk_img)

sitk_mask = sitk.ReadImage('amos_0001_label.nii.gz')
mask = sitk.GetArrayFromImage(sitk_mask)

mask = mask.astype(int)
mask[mask != 1] = 0
mask[mask == 1] = 1

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
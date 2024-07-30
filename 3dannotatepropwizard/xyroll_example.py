import os
import SimpleITK as sitk
from tqdm import tqdm
from glob import glob
import numpy as np
from xyroll_prediction import XYrollPrediction
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

model = XYrollPrediction('../yaml/eval_config.yaml')

sitk_img = sitk.ReadImage('amos_0001_data.nii.gz')
img = sitk.GetArrayFromImage(sitk_img)

sitk_mask = sitk.ReadImage('amos_0001_label.nii.gz')
mask = sitk.GetArrayFromImage(sitk_mask)

mask = mask.astype(int)
mask[mask != 2] = 0
mask[mask == 2] = 1

mask = mask.astype(np.uint8)

for prop in regionprops(mask):

    init_center = np.array(prop.centroid, dtype=int)
    index_z = init_center[0]

    with torch.inference_mode():
        result = model.predict(img, mask[index_z], index_z)

    result = result.sum((1,2)) > 0
    mask = mask.sum((1,2)) > 0

    f1 = 2 * (result * mask).sum() / (result.sum() + mask.sum())
    print(f1)
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

from utils.inference_core_with_logits import InferenceCoreWithLogits
from cutie.model.cutie import CUTIE
from segment_anything import sam_model_registry
from segment_anything.predictor_sammed import SammedPredictor
from utils.checkpoint import download_ckpt
from utils.image_process import (
    normalize_volume,
    crop_and_pad,
    crop_and_pad_reverse,
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

__all__ = [
    "MaskPropagation",
]

class MaskPropagation:

    def __init__(self, sam_checkpoint:str, cutie_yaml:str):

        """
        Initialize the MaskPropagation class.

        Args:
            sam_checkpoint (str): Path to the SAM checkpoint file.
            cutie_yaml (str): Path to the CUTIE configuration file.

        Attributes:
            model_size (int): Size of the input image.
            encoder_adapter (bool): Whether to use the encoder adapter.
            image_size (int): Size of the input image after resizing.
            device (torch.device): Device to use for tensor operations.
            sam_predictor (SammedPredictor): SAM predictor object.
            vos_processor (InferenceCoreWithLogits): VOS processor object.
            trace_num (int): Number of traces to use for mask propagation.

        Methods:
            predict(img, initial_mask, input_z): Predict the mask propagation.
        """

        if "sam-med2d" in sam_checkpoint:
            self.model_size = 256
            self.encoder_adapter = True
            self.image_size = 256
        else:
            self.model_size = 480
            self.encoder_adapter = False
            self.image_size = 1024
        args = dict(
            image_size=self.image_size,
            encoder_adapter=self.encoder_adapter,
            sam_checkpoint=sam_checkpoint,
        )
        args = SimpleNamespace(**args)
        
        if not os.path.isfile(sam_checkpoint):
            download_ckpt(ckpt_path=sam_checkpoint)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = sam_model_registry["vit_b"](args).to(self.device)
        self.sam_predictor = SammedPredictor(model)

        with torch.inference_mode():
            config = yaml_to_dotdict(cutie_yaml)
    
            # Load the network weights
            cutie = CUTIE(config).to(self.device).eval()
            if not os.path.isfile(config.weights):
                download_ckpt(
                    ckpt_path=config.weights,
                )
            model_weights = torch.load(config.weights)
            cutie.load_weights(model_weights)
        self.vos_processor = InferenceCoreWithLogits(cutie, cfg=config)
        self.trace_num = 4

    def predict(self, img:np.ndarray, initial_mask:np.ndarray, input_z:int):

        """
        Predict the mask propagation.

        Args:
            img (numpy.ndarray): Input image.
            initial_mask (numpy.ndarray): Initial mask.
            input_z (int): Index of the slice to process.

        Returns:
            dict: Dictionary containing the predicted masks for each slice.
        """

        img = normalize_volume(img.astype(float))
        size_z, size_y, size_x = img.shape

        prop = regionprops(initial_mask)[0]
        init_center = np.array(prop.centroid, dtype=int)

        if size_y > self.model_size or size_x > self.model_size:
            y_min = max(init_center[0] - self.model_size // 2, 0)
            y_max = min(init_center[0] + self.model_size // 2, size_y)
            x_min = max(init_center[1] - self.model_size // 2, 0)
            x_max = min(init_center[1] + self.model_size // 2, size_x)

            padding_top = (self.model_size - (y_max - y_min)) // 2
            padding_bottom = self.model_size - (y_max - y_min) - padding_top
            padding_left = (self.model_size - (x_max - x_min)) // 2
            padding_right = self.model_size - (x_max - x_min) - padding_left

            crop_img = crop_and_pad(
                img,
                y_min,
                y_max,
                x_min,
                x_max,
                (padding_left, padding_right, padding_top, padding_bottom),
                self.device,
            )
            crop_mask = crop_and_pad(
                initial_mask,
                y_min,
                y_max,
                x_min,
                x_max,
                (padding_left, padding_right, padding_top, padding_bottom),
                self.device,
            )
        else:
            crop_img = np.array(img, dtype=float, copy=True)
            crop_mask = np.array(initial_mask, dtype=int, copy=True)

        predict_masks_dict = {}
        
        index_z = input_z
        mask_data = crop_mask
        predict_masks_dict[index_z] = initial_mask
        slice_data = get_slice(crop_img, None, index_z)

        # Convert slice to torch tensor and interpolate
        frame_torch = image_to_torch(slice_data, device=self.device)  # [3, x, y]
        init_frame_torch = interpolate_tensor(
            input_tensor=frame_torch, size=self.model_size, mode="bilinear"
        )  # [3, size, size]

        mask_torch = index_numpy_to_one_hot_torch(mask_data, 2).to(self.device)
        mask_torch = interpolate_tensor(
            input_tensor=mask_torch, size=self.model_size, mode="nearest"
        )

        init_mask_torch = torch.zeros(self.trace_num, self.model_size, self.model_size).to(self.device)
        init_mask_torch[0] = mask_torch[1]

        self.vos_processor.clear_memory()
        self.vos_processor.step(init_frame_torch, init_mask_torch, idx_mask=False)

        index_z += 1
        while index_z < size_z:
            slice_data = get_slice(crop_img, None, index_z)

            frame_torch, prediction, logits = vos_step(
                processor=self.vos_processor,
                input_slice=slice_data,
                model_size=self.model_size,
                size=mask_data.shape,
                device=self.device,
            )

            regionprop = regionprops(prediction.int().cpu().numpy())
            if len(regionprop) == 0:
                break

            box = pad_box(regionprop[0].bbox, 1)
            masks = sam_step(
                self.sam_predictor,
                slice_data,
                logits,
                box,
            )

            predict_masks_dict[index_z] = crop_and_pad_reverse(masks[0],
                                                                y_min,
                                                                y_max,
                                                                x_min,
                                                                x_max,
                                                                (padding_left, padding_right, padding_top, padding_bottom), 
                                                                (size_y, size_x))

            # reset VOS
            mask_torch = torch.zeros(self.trace_num, self.model_size, self.model_size).to(self.device)
            resized_masks = F.interpolate(
                torch.tensor(masks, device=self.device).unsqueeze(0),
                (self.model_size, self.model_size),
            )[0]
            mask_torch[0] = resized_masks[0]
            self.vos_processor.step(frame_torch, mask_torch, idx_mask=False)

            index_z += 1

        self.vos_processor.clear_memory()
        self.vos_processor.step(init_frame_torch, init_mask_torch, idx_mask=False)

        index_z = input_z - 1
        while index_z >= 0:
            slice_data = get_slice(crop_img, None, index_z)

            frame_torch, prediction, logits = vos_step(
                processor=self.vos_processor,
                input_slice=slice_data,
                model_size=self.model_size,
                size=mask_data.shape,
                device=self.device,
            )

            regionprop = regionprops(prediction.int().cpu().numpy())
            if len(regionprop) == 0:
                break

            box = pad_box(regionprop[0].bbox, 1)
            masks = sam_step(
                self.sam_predictor,
                slice_data,
                logits,
                box,
            )

            predict_masks_dict[index_z] = crop_and_pad_reverse(masks[0],
                                                                y_min,
                                                                y_max,
                                                                x_min,
                                                                x_max,
                                                                (padding_left, padding_right, padding_top, padding_bottom), 
                                                                (size_y, size_x))

            # reset VOS
            mask_torch = torch.zeros(self.trace_num, self.model_size, self.model_size).to(self.device)
            resized_masks = F.interpolate(
                torch.tensor(masks, device=self.device).unsqueeze(0),
                (self.model_size, self.model_size),
            )[0]
            mask_torch[0] = resized_masks[0]
            self.vos_processor.step(frame_torch, mask_torch, idx_mask=False)

            index_z -= 1

        return predict_masks_dict

    def predict_by_volume(self, img:np.ndarray, labeled_mask:np.ndarray):

        """
        Predict the mask propagation for a whole volume.

        Args:
            img (numpy.ndarray): Input image.
            labeled_mask (numpy.ndarray): Labeled mask containing some labeled slices.

        Returns:
            numpy.ndarray: Predicted mask for the whole volume.
        """
        
        predict_mask = np.zeros_like(img)

        for z in np.nonzero(labeled_mask.sum((1,2)))[0]:
            result = self.predict(img, labeled_mask[z], z)
            for k, v in result.items():
                predict_mask[k] = v
        
        return predict_mask




        
        
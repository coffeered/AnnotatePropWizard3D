import os

import numpy as np
import torch
from skimage.measure import regionprops

from annotatepropwizard3d.cutie.inference.inference_core import InferenceCore
from annotatepropwizard3d.cutie.model.cutie import CUTIE
from annotatepropwizard3d.utils.checkpoint import download_ckpt
from annotatepropwizard3d.utils.image_process import (
    determine_degree,
    interpolate_tensor,
    normalize_volume,
    rescale_centroid,
    reset_rotate,
    rotate_predict,
)
from annotatepropwizard3d.utils.yaml_loader import yaml_to_dotdict

MAX_LENGTH = 480


class XYrollPrediction:
    def __init__(self, cutie_yaml: str):
        """
        Initialize the XYrollPrediction class.

        Args:
            cutie_yaml (str): Path to the CUTIE configuration file.
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.vos_processor = InferenceCore(cutie, cfg=config)
        self.trace_num = 4

    def predict(self, img, inital_mask, input_z, dynamic_degree=False):
        """
        Predict the XYroll prediction for a given input image and initial mask.

        Args:
            img (np.ndarray): Input image as a NumPy array.
            inital_mask (np.ndarray): Initial mask as a NumPy array.
            input_z (int): The z-coordinate of the initial mask.
            dynamic_degree (bool): According to the zx/zy ratio determine the rotation degree dynamicly

        Returns:
            np.ndarray: The predicted XYroll prediction as a NumPy array.
        """

        img = normalize_volume(img.astype(float))
        size_z, size_y, size_x = img.shape
        mask = np.zeros_like(img, dtype=np.uint8)
        mask[input_z] = inital_mask

        tensor_img = interpolate_tensor(
            input_tensor=torch.tensor(img, dtype=float, device=self.device),
            size=(MAX_LENGTH, MAX_LENGTH, MAX_LENGTH),
            mode="nearest",
        )
        tensor_mask = interpolate_tensor(
            input_tensor=torch.tensor(mask, dtype=float, device=self.device),
            size=(MAX_LENGTH, MAX_LENGTH, MAX_LENGTH),
            mode="nearest",
        )
        tensor_img = tensor_img.permute(1, 0, 2).contiguous()  # [z, y, x] -> [y, z, x]
        tensor_mask = tensor_mask.permute(
            1, 0, 2
        ).contiguous()  # [z, y, x] -> [y, z, x]

        prop = regionprops(mask)[0]

        centroid = rescale_centroid(
            prop.centroid,
            size_z=size_z,
            size_y=size_y,
            size_x=size_x,
            max_length=MAX_LENGTH,
        )

        centroid = centroid[[1, 0, 2]]

        rot_dict = {
            "img": tensor_img,
            "mask": tensor_mask,
            "point": centroid[[2, 1]],  # take (x, z)
            "prop_index": 0,
            "prop": prop,
        }

        rot_dict["pred"] = torch.zeros_like(rot_dict["img"], device=self.device).float()

        degree = determine_degree(size_x=size_x, size_y=size_y, size_z=size_z) if dynamic_degree else 2
        offset = 0
        while offset <= 90:
            rot_dict, offset = rotate_predict(
                rot_dict, offset, degree, self.vos_processor, device=self.device
            )
        rot_dict = reset_rotate(rot_dict, centroid=centroid, offset=offset)
        offset = 0
        while offset > -90:
            rot_dict, offset = rotate_predict(
                rot_dict, offset, -degree, self.vos_processor, device=self.device
            )
        rot_dict = reset_rotate(rot_dict, centroid=centroid, offset=offset)

        rot_dict["pred"] = rot_dict["pred"].permute(1, 0, 2).contiguous()

        centroid = centroid[[1, 0, 2]]  # [y, z, x] -> [z, y, x]

        rot_dict["pred"] = interpolate_tensor(
            input_tensor=rot_dict["pred"],
            size=(size_z, size_y, size_x),
            mode="trilinear",
        )

        return rot_dict["pred"].cpu().numpy()

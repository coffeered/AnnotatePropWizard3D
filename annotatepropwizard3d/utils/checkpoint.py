import os
from typing import Dict

import gdown
import requests
from tqdm.auto import tqdm

__all__ = ["download_ckpt"]

CKPT_INFO_HUBS: Dict[str, Dict[str, str]] = {
    "sam-med2d_b.pth": {
        "mode": "gdrive",
        "url": "https://drive.google.com/uc?id=1ARiB5RkSsWmAB_8mqWnwDF8ZKTtFwsjl",
    },
    "sam_vit_b_01ec64.pth": {
        "mode": "huggingface",
        "url": "https://huggingface.co/segments-arnaud/sam_vit_b/resolve/main/sam_vit_b_01ec64.pth?download=true",
    },
    "cutie-base-mega.pth": {
        "mode": "github",
        "url": "https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-mega.pth",
    },
}


def download_ckpt(ckpt_path: str) -> None:
    """
    Download a checkpoint file.

    Args:
        ckpt_path (str): The path of the checkpoint file.

    Returns:
        None
    """
    root, ckpt_name = os.path.split(ckpt_path)
    os.makedirs(root, exist_ok=True)
    output_file = os.path.join(root, ckpt_name)
    ckpt_info = CKPT_INFO_HUBS[ckpt_name]

    match ckpt_info["mode"]:
        case "gdrive":
            gdown.download(ckpt_info["url"], output_file, quiet=False)
        case "github" | "huggingface":
            r = requests.get(ckpt_info["url"], stream=True)
            total_size = int(r.headers.get("content-length", 0))
            BLOCK_SIZE = 1024
            t = tqdm(total=total_size, unit="iB", unit_scale=True)
            with open(output_file, "wb") as op:
                for data in r.iter_content(BLOCK_SIZE):
                    t.update(len(data))
                    op.write(data)
            t.close()
            if total_size != 0 and t.n != total_size:
                raise RuntimeError(f"Error while downloading {ckpt_name}")
        case _:
            raise ValueError(f"Invalid mode: {ckpt_info['mode']}")

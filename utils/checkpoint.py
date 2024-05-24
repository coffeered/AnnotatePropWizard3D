import os

import gdown

__all__ = ["download_med2d_b"]


def download_ckpt(ckpt_name: str, root: str):
    url = CKPT_URL_HUBS[ckpt_name]
    ouput_file = os.path.join(root, ckpt_name)
    gdown.download(url, ouput_file, quiet=False)


CKPT_URL_HUBS = {
    "sam-med2d_b.pth": "https://drive.google.com/uc?id=1ARiB5RkSsWmAB_8mqWnwDF8ZKTtFwsjl",
}

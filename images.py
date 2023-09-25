import numpy as np
import torch
from PIL import Image
from torch import Tensor


def tensor2pil(image: Tensor) -> Image.Image:
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil2tensor(image: Image.Image) -> Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

SAMPLER_MAP = {
    "nearest": Image.NEAREST,
    "lanczos": Image.LANCZOS,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "box": Image.BOX,
    "hamming": Image.HAMMING,
}

class RotateImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "rotation": ("INT", {"default": 0, "min": 0, "max": 360, "step": 1}),
                "sampler": (["nearest", "lanczos", "bilinear", "bicubic", "box", "hamming"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "rotate"

    CATEGORY = "KepKitchenSink/images"

    def rotate(self, image, rotation, sampler):
        pil_image = tensor2pil(image)
        
        # Rotate the image using PIL's rotate function
        angle = rotation  # Change this to the desired angle
        center_x = pil_image.width / 2
        center_y = pil_image.height / 2
        rotated_image = pil_image.rotate(
            angle, resample=SAMPLER_MAP[sampler], center=(center_x, center_y)
        )
        
        # Convert the rotated image back to a tensor
        return (pil2tensor(rotated_image),)


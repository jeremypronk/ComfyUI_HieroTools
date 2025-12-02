import torch
import math

class DepthWranglerNode:
    """
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth_image": ("IMAGE",),
                "min_depth": ("FLOAT", {
                    "default": 0.0,
                    "step": 0.01,
                    "description": "Depth value to set to 0"
                }),
                "max_depth": ("FLOAT", {
                    "default": 1.0,
                    "step": 0.01,
                    "description": "Depth value to set to 1"
                }),
                "invert": ("BOOLEAN", {
                    "default": True,
                    "description": "Invert depth if required (depth map output should have closest depth at 1, farthest depth at 0)"
                }),
                "bias": ("FLOAT", {
                    "default": 0.0,
                    "step": 0.1,
                    "min": -10.0,
                    "max": 10.0,
                    "description": "Bias depth closer to camera with > 0.0 or to bias away set < 0.0"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output_depth_image",)
    FUNCTION = "depth_wrangle"
    CATEGORY = "HieroTools"

    def __init__(self, ):
        pass

    def depth_wrangle(self, depth_image, min_depth, max_depth, invert, bias):

        depth_output = depth_image.clone()

        # normalise
        depth_output = torch.clamp((depth_output - min_depth) / (max_depth - min_depth), 0.0, 1.0)

        # invert
        if invert:
            depth_output = 1.0 - depth_output

        # bias
        if not math.isclose(bias, 0.0):
            if bias>0.0:
                depth_output.pow_(1.0 + bias)
            else:
                depth_output.pow_(1.0 / (1.0 - bias))

        return (depth_output,)
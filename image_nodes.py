import torch

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
                    "description": "Depth value to reset to 0"
                }),
                "max_depth": ("FLOAT", {
                    "default": 1.0,
                    "step": 0.01,
                    "description": "Depth value to reset to 1"
                }),
                "gamma": ("FLOAT", {
                    "default": 1.0,
                    "step": 0.01,
                    "description": "Bias depth using gamma"
                }),
                "invert": ("BOOLEAN", {
                    "default": True,
                    "description": "Invert depth if required (closest depth should be 1, farthest depth should be 0)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output_depth_image")
    FUNCTION = "depth_wrangle"
    CATEGORY = "HieroTools"

    def __init__(self, ):
        pass

    def depth_wrangle(self, image, min_depth, max_depth):

        depth_output = image.clone()
        depth_output = torch.clamp((depth_output - min_depth) / (max_depth - min_depth), 0.0, 1.0)

        return (depth_output,)
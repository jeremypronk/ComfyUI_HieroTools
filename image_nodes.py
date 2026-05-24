import torch
import torch.nn.functional as F

import comfy.utils

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



class ResizeToWidth:
    # High-level node description visible in the UI
    DESCRIPTION = """Resizes an image (and optionally a mask) to match a precise Target Width and Target Height without stretching or distortion, with width being the master dimension.

Logic Flow:
1. Scale: The image is resized so its Width exactly matches the Target Width. The aspect ratio is preserved.
2. Crop (Too Tall): If the new Height exceeds the Target Height, the excess pixels are cropped from the center.
3. Pad (Too Short): If the new Height falls short, missing pixels are padded equally to the top and bottom."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "The input image tensor(s) to be processed."
                }),
                "target_width": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 8,
                    "tooltip": "The master dimension. The image is strictly scaled to this width."
                }),
                "target_height": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 8,
                    "tooltip": "The absolute vertical constraint. The image is cropped/padded to match this height."
                }),
                "padding_mode": (["edge", "black", "white"], {
                    "tooltip": "'edge': Stretches outermost row of pixels to fill the void.\n'black': Fills the void with solid black pixels (0.0).\n'white': Fills the void with solid white pixels (1.0)."
                }),
                "interpolation": (["lanczos", "bilinear", "bicubic", "area", "nearest", "nearest-exact"], {
                    "tooltip": "lanczos/bicubic/bilinear: Standard photos and general scaling.\narea: Best for downscaling massive images.\nnearest: Preserves hard edges (pixel art/masks)."
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Optional mask to resize alongside the image. Will be padded/cropped identically."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "resize_image"
    CATEGORY = "HieroTools"

    def resize_image(self, image, target_width, target_height, padding_mode, interpolation, mask=None):
        # --- 1. PROCESS IMAGE ---
        # ComfyUI images are shape [Batch, Height, Width, Channels]
        B, H, W, C = image.shape

        # Calculate the proportional new height based on the master target_width
        new_height = int(round(H * (target_width / W)))

        # Permute the tensor to [Batch, Channels, Height, Width] for PyTorch/Comfy functions
        img_t = image.permute(0, 3, 1, 2)

        # Scale the image using ComfyUI's internal upscale utility
        img_resized = comfy.utils.common_upscale(
            img_t,
            target_width,
            new_height,
            interpolation,
            "disabled"
        )

        # Conditional logic: crop or pad the height
        if new_height > target_height:
            # Image is too tall -> crop the center
            crop_top = (new_height - target_height) // 2
            crop_bottom = crop_top + target_height
            # img_resized is [B, C, H, W], so we slice the 3rd dimension (Height)
            img_final = img_resized[:, :, crop_top:crop_bottom, :]

        elif new_height < target_height:
            # Image is too short -> pad top and bottom
            pad_total = target_height - new_height
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top

            # F.pad format: (pad_left, pad_right, pad_top, pad_bottom) -> pads the W and H dims
            if padding_mode == "edge":
                img_final = F.pad(img_resized, (0, 0, pad_top, pad_bottom), mode='replicate')
            elif padding_mode == "white":
                img_final = F.pad(img_resized, (0, 0, pad_top, pad_bottom), mode='constant', value=1.0)
            else:  # black
                img_final = F.pad(img_resized, (0, 0, pad_top, pad_bottom), mode='constant', value=0.0)

        else:
            # Image height matches perfectly
            img_final = img_resized

        # Permute back to ComfyUI's expected format [Batch, Height, Width, Channels]
        img_out = img_final.permute(0, 2, 3, 1)

        # Clamp values to valid range [0.0, 1.0] to fix lanczos/bicubic overshooting artifacts
        if interpolation in ["bicubic", "lanczos"]:
            img_out = torch.clamp(img_out, 0.0, 1.0)

        # --- 2. PROCESS MASK ---
        if mask is not None:
            # Masks are [Batch, Height, Width]. We extract shape dynamically in case it differs from the image.
            B_m, H_m, W_m = mask.shape
            mask_new_height = int(round(H_m * (target_width / W_m)))

            # Add a dummy channel dimension: [Batch, 1, Height, Width]
            mask_t = mask.unsqueeze(1)

            # Scale mask
            mask_resized = comfy.utils.common_upscale(
                mask_t,
                target_width,
                mask_new_height,
                interpolation,
                "disabled"
            )

            # Crop or pad mask
            if mask_new_height > target_height:
                m_crop_top = (mask_new_height - target_height) // 2
                m_crop_bottom = m_crop_top + target_height
                mask_final = mask_resized[:, :, m_crop_top:m_crop_bottom, :]

            elif mask_new_height < target_height:
                m_pad_total = target_height - mask_new_height
                m_pad_top = m_pad_total // 2
                m_pad_bottom = m_pad_total - m_pad_top

                if padding_mode == "edge":
                    mask_final = F.pad(mask_resized, (0, 0, m_pad_top, m_pad_bottom), mode='replicate')
                elif padding_mode == "white":
                    mask_final = F.pad(mask_resized, (0, 0, m_pad_top, m_pad_bottom), mode='constant', value=1.0)
                else:  # black
                    mask_final = F.pad(mask_resized, (0, 0, m_pad_top, m_pad_bottom), mode='constant', value=0.0)
            else:
                mask_final = mask_resized

            # Remove the dummy channel dimension: [Batch, Height, Width]
            mask_out = mask_final.squeeze(1)

            # Masks must strictly remain between 0.0 and 1.0, regardless of interpolation mode
            mask_out = torch.clamp(mask_out, 0.0, 1.0)

        else:
            # If no mask is connected, output a dummy black mask matching the new dimensions
            mask_out = torch.zeros((B, target_height, target_width), dtype=torch.float32, device=image.device)

        return (img_out, mask_out)
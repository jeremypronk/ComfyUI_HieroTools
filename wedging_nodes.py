import torch

from nodes import EmptyLatentImage

class ParameterWedgingLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "wedges": ("INT", {"default": 4, "min": 1, "max": 64, "step": 1}),
                "width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 16}),
                "start_value": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "end_value": ("FLOAT", {"default": 8.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT", "FLOAT", "INT", "INT", "INT" )
    RETURN_NAMES = ("latent_list", "wedge_floats", "wedge_ints", "width", "height")
    OUTPUT_IS_LIST = (True, True, True, False, False)

    FUNCTION = "generate_wedge"
    CATEGORY = "HieroTools/wedging"

    def generate_wedge(self, wedges, width, height, start_value, end_value):
        empty = EmptyLatentImage()
        latent_list = [empty.generate(width, height, 1)[0] for _ in range(wedges)]
    
        if wedges <= 1:
            float_list = [float(start_value)]
        else:
            step = (end_value - start_value) / (wedges - 1)
            float_list = [float(start_value + i * step) for i in range(wedges)]
        int_list = [int(round(v)) for v in float_list]

        return (latent_list, float_list, int_list, width, height)


class ParameterWedgingGridLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 16}),
                "wedges_1": ("INT", {"default": 3, "min": 1, "max": 64, "step": 1}),
                "start_value_1": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "end_value_1": ("FLOAT", {"default": 8.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "wedges_2": ("INT", {"default": 3, "min": 1, "max": 64, "step": 1}),
                "start_value_2": ("FLOAT", {"default": 20.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "end_value_2": ("FLOAT", {"default": 40.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT", "FLOAT", "INT", "FLOAT", "INT", "INT", "INT" )
    RETURN_NAMES = ("latent_list", "wedge_1_floats", "wedge_1_ints", "wedge_2_floats", "wedge_2_ints", "width", "height")
    OUTPUT_IS_LIST = (True, True, True, True, True, False, False)

    FUNCTION = "generate_grid"
    CATEGORY = "HieroTools/wedging"

    def generate_grid(self, width, height, wedges_1, start_value_1, end_value_1, wedges_2, start_value_2, end_value_2):
        total_batch = wedges_1 * wedges_2

        # 1. Latent list (one reference-identical latent per grid cell)
        empty = EmptyLatentImage()
        latent_list = [empty.generate(width, height, 1)[0] for _ in range(total_batch)]

        # 2. Axis values
        def linspace(start, end, n):
            if n <= 1:
                return [float(start)]
            step = (end - start) / (n - 1)
            return [float(start + i * step) for i in range(n)]

        axis_1_vals = linspace(start_value_1, end_value_1, wedges_1)
        axis_2_vals = linspace(start_value_2, end_value_2, wedges_2)

        # 3. Cross product (axis 1 is the slow axis, axis 2 varies fastest)
        float_list_1, int_list_1 = [], []
        float_list_2, int_list_2 = [], []

        for v1 in axis_1_vals:
            for v2 in axis_2_vals:
                float_list_1.append(v1)
                int_list_1.append(int(round(v1)))
                float_list_2.append(v2)
                int_list_2.append(int(round(v2)))

        return (latent_list, float_list_1, int_list_1, float_list_2, int_list_2, width, height)
import os.path

class PathAndFilenameNode:
    """
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene_name": ("STRING",),
                "scene_version": ("INT", {"default": 0, "min": 0, "max": 999, "step":1 }),
                "suffix": ("STRING",),
                "frame_number": ("INT", {"default": 0, "min": 0,}),
                "path_output_fstring": ("STRING", {"default": "{scene_name}_v{scene_version:03}"}),
                "filename_output_fstring": ("STRING", {"default": "{path_output}.{frame_number:05}"}),
                "filename_suffix_output_fstring": ("STRING", {"default": "{path_output}_{suffix}.{frame_number:05}"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING",)
    RETURN_NAMES = ("path_output", "filename_output", "filename_suffix_output", "path_and_filename_output", "path_and_filename_suffix_output")
    FUNCTION = "path_and_filename"
    CATEGORY = "HieroTools"

    def __init__(self, ):
        pass

    def path_and_filename(self, scene_name, scene_version, suffix, frame_number, path_output_fstring, filename_output_fstring, filename_suffix_output_fstring):

        scene_name_clean = scene_name.replace(" ", "_")

        path_output = path_output_fstring.format(scene_name=scene_name_clean, scene_version=scene_version, frame_number=frame_number)
        filename_output = filename_output_fstring.format(scene_name=scene_name_clean, scene_version=scene_version, frame_number=frame_number, path_output=path_output)
        path_and_filename_output = os.path.join(path_output, filename_output)

        filename_suffix_output = filename_suffix_output_fstring.format(scene_name=scene_name_clean, scene_version=scene_version, suffix=suffix, frame_number=frame_number, path_output=path_output)
        path_and_filename_suffix_output = os.path.join(path_output, filename_suffix_output)

        return path_output, filename_output, filename_suffix_output, path_and_filename_output, path_and_filename_suffix_output
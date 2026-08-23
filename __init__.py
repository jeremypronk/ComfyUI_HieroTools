
from .filename_nodes import PathAndFilenameNode
from .image_nodes import DepthWranglerNode
from .image_nodes import DepthNormaliseNode
from .image_nodes import ResizeToWidth
from .image_nodes import NumpyToImage
from .image_nodes import NumpyToMask
# from .helper_nodes import DevReloaderNode

NODE_CLASS_MAPPINGS = {
    "PathAndFilenameNode": PathAndFilenameNode,
    "DepthWranglerNode": DepthWranglerNode,
    "DepthNormalise": DepthNormaliseNode,
    "ResizeToWidth": ResizeToWidth,
    "NumpyToImage": NumpyToImage,
    "NumpyToMask": NumpyToMask,
    # "DevReloaderNode": DevReloaderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PathAndFilenameNode": "Path and Filename",
    "DepthWranglerNode": "Depth Wrangler",
    "DepthNormalise": "Depth Normalise",
    "ResizeToWidth": "Resize To Width",
    "NumpyToImage": "Numpy to Image",
    "NumpyToMask": "Numpy to Mask",
    # "DevReloaderNode": "Dev Reloader",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

from .filename_nodes import PathAndFilenameNode
from .image_nodes import DepthWranglerNode
from .image_nodes import DepthNormaliseNode
from .image_nodes import ResizeToWidth
from .image_nodes import NumpyToImage
from .image_nodes import NumpyToMask
from .image_nodes import ImageTextOverlay
# from .helper_nodes import DevReloaderNode
from .wedging_nodes import ParameterWedgingLatent, ParameterWedgingGridLatent

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
NODE_CLASS_MAPPINGS = {
    "PathAndFilenameNode": PathAndFilenameNode,
    "DepthWranglerNode": DepthWranglerNode,
    "DepthNormalise": DepthNormaliseNode,
    "ResizeToWidth": ResizeToWidth,
    "NumpyToImage": NumpyToImage,
    "NumpyToMask": NumpyToMask,
    "ImageTextOverlay": ImageTextOverlay,
    # "DevReloaderNode": DevReloaderNode,
    "ParameterWedgingLatent": ParameterWedgingLatent,
    "ParameterWedgingGridLatent": ParameterWedgingGridLatent
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PathAndFilenameNode": "Path and Filename",
    "DepthWranglerNode": "Depth Wrangler",
    "DepthNormalise": "Depth Normalise",
    "ResizeToWidth": "Resize To Width",
    "NumpyToImage": "Numpy to Image",
    "NumpyToMask": "Numpy to Mask",
    "ImageTextOverlay": "Image Text Overlay",
    # "DevReloaderNode": "Dev Reloader",
    "ParameterWedgingLatent": "Parameter Wedge (1D Linear)",
    "ParameterWedgingGridLatent": "Parameter Wedge (2D Grid)"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
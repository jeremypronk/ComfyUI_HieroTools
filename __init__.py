
from .filename_nodes import PathAndFilenameNode
from .image_nodes import DepthWranglerNode
from .image_nodes import ResizeToWidth

NODE_CLASS_MAPPINGS = {
    "PathAndFilenameNode": PathAndFilenameNode,
    "DepthWranglerNode": DepthWranglerNode,
    "ResizeToWidth": ResizeToWidth,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PathAndFilenameNode": "Path and Filename",
    "DepthWranglerNode": "Depth Wrangler",
    "ResizeToWidth": "Resize To Width",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
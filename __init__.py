
from .filename_nodes import PathAndFilenameNode
from .image_nodes import DepthWranglerNode

NODE_CLASS_MAPPINGS = {
    "PathAndFilenameNode": PathAndFilenameNode,
    "DepthWranglerNode": DepthWranglerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PathAndFilenameNode": "Path and Filename",
    "DepthWranglerNode": "Depth Wrangler"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
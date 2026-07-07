import importlib
import sys

class DevReloaderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "module_name": ("STRING", {"default": "my_dev_node"}),
            },
            "optional": {
                "trigger": ("*",),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("trigger",)
    FUNCTION = "reload"
    CATEGORY = "HieroTools"

    def reload(self, module_name, trigger=None):
        if module_name in sys.modules:
            module = sys.modules[module_name]
            importlib.reload(module)

            try:
                from nodes import NODE_CLASS_MAPPINGS as GLOBAL_MAP
                if hasattr(module, 'NODE_CLASS_MAPPINGS'):
                    GLOBAL_MAP.update(module.NODE_CLASS_MAPPINGS)
                    print(f"[DevReloader] Reloaded and re-registered: {module_name}")
            except Exception as e:
                print(f"[DevReloader] Reloaded but re-registration failed: {e}")
        else:
            print(f"[DevReloader] Module '{module_name}' not found in sys.modules")

        return (trigger,)
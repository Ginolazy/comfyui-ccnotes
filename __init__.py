## ComfyUI/custom_nodes/CCNotes/__init__.py
import os
import importlib.util
import inspect
import sys
import pkg_resources
from collections import defaultdict
from .py.utility import Pytorch_Fix_IntelMac

plugin_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(plugin_dir)

NODE_CLASS_MAPPINGS = {} 
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./js"
__version__ = "1.2.0"

def check_dependencies():
    required_packages = {
        "torch": "2.0.0",
        "numpy": "1.24.0",
        "Pillow": "9.0.0",
        "kornia": "0.7.0"
    }
    
    missing_packages = []
    outdated_packages = []
    
    for package, min_version in required_packages.items():
        try:
            installed_version = pkg_resources.get_distribution(package).version
            if pkg_resources.parse_version(installed_version) < pkg_resources.parse_version(min_version):
                outdated_packages.append(package)
        except pkg_resources.DistributionNotFound:
            missing_packages.append(package)
    
    if missing_packages or outdated_packages:
        for package in missing_packages:
            print(f"Cannot import CCNotes module for custom nodes: No module named '{package}'")
        if outdated_packages:
            print(f"CCNotes requires newer versions of: {', '.join(outdated_packages)}")
        return False
    return True

check_dependencies()

py_dir = os.path.join(plugin_dir, "py")
if os.path.isdir(py_dir):
    for filename in filter(lambda f: f.endswith(".py") and f != "__init__.py", os.listdir(py_dir)):
        module_path = os.path.join(py_dir, filename)
        module_name = filename[:-3]
        
        try:
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec and (module := importlib.util.module_from_spec(spec)):
                module.__package__ = "py"
                spec.loader.exec_module(module)
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if hasattr(obj, "INPUT_TYPES") and hasattr(obj, "FUNCTION"):
                        NODE_CLASS_MAPPINGS[name] = obj
                        NODE_DISPLAY_NAME_MAPPINGS[name] = getattr(obj, "DISPLAY_NAME", name)
        except Exception as e:
            print(f"[CCNotes] Failed to load module {module_name}: {str(e)}")
            import traceback
            traceback.print_exc()
else:
    print(f"[CCNotes] Directory not found: {py_dir}")

category_map = defaultdict(list)
for name, cls in NODE_CLASS_MAPPINGS.items():
    category = getattr(cls, "CATEGORY", "Other")
    display_name = getattr(cls, "DISPLAY_NAME", name)
    category_map[category].append((name, cls, display_name))

NODE_CLASS_MAPPINGS.clear()
NODE_DISPLAY_NAME_MAPPINGS.clear()
for cat in sorted(category_map.keys(), key=lambda x: x.lower()):
    for name, cls, display_name in category_map[cat]:
        NODE_CLASS_MAPPINGS[name] = cls
        NODE_DISPLAY_NAME_MAPPINGS[name] = display_name

js_dir = os.path.join(plugin_dir, "js")
if not os.path.exists(js_dir):
    print(f"[CCNotes] Directory not found: {js_dir}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# comfyui-azul-scripts/__init__.py

# Import the node classes and their mappings from other files in this directory
# The leading '.' indicates a relative import within the same package
from . import image_save_jpg
from . import weighted_conditioning_average

# Aggregate the mappings from all imported modules
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Update the package-level dictionaries with mappings from each module
NODE_CLASS_MAPPINGS.update(image_save_jpg.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(weighted_conditioning_average.NODE_CLASS_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS.update(image_save_jpg.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(weighted_conditioning_average.NODE_DISPLAY_NAME_MAPPINGS)

# Optional: Specify what gets imported when someone does 'from comfyui-azul-scripts import *'
# This is good practice but not strictly necessary for ComfyUI loading
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("✅ Azul Scripts: Loaded SaveImageAsJPG and WeightedConditioningAverage nodes.")
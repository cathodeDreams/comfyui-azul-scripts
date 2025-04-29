# comfyui-azul-scripts/image_save_jpg.py

import torch
import numpy as np
from PIL import Image
import os
import json
import folder_paths
from comfy.cli_args import args
from comfy.comfy_types import IO, InputTypeDict

# Dictionary initialization for ComfyUI custom nodes
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

class SaveImageAsJPG:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = "_jpg" # Append suffix to differentiate

    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        return {
            "required": {
                "images": (IO.IMAGE, {"tooltip": "The images to save as JPG."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the files to save."}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1, "tooltip": "JPEG quality (1-100). Higher is better quality, larger file size."}),
                "subsampling": (["default", "4:4:4", "4:2:2", "4:2:0"], {"tooltip": "Chroma subsampling. 4:4:4 preserves the most color detail (good for graphics, larger file). 4:2:0 is common for photos (smaller file)."})
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_jpgs"

    OUTPUT_NODE = True

    CATEGORY = "Azul's Scripts" # <--- Changed Category
    DESCRIPTION = "Saves the input images as JPG files to your ComfyUI output directory with adjustable quality."

    def save_jpgs(self, images, filename_prefix="ComfyUI", quality=95, subsampling="default", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()

        subsampling_map = {
            "default": -1, # Pillow's default based on quality
            "4:4:4": 0,
            "4:2:2": 1,
            "4:2:0": 2,
        }
        subsampling_val = subsampling_map.get(subsampling, -1)

        # Metadata handling for JPG is complex and often ignored, skipping for simplicity
        # If needed, use libraries like piexif and map prompt/extra_pnginfo to EXIF tags

        for (batch_number, image) in enumerate(images):
            # Convert tensor to PIL Image
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            # Ensure image is in RGB mode (JPG doesn't support alpha)
            if img.mode != 'RGB':
                 img = img.convert('RGB')

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.jpg" # Change extension
            file_path = os.path.join(full_output_folder, file)

            try:
                # Save as JPG with specified quality and subsampling
                save_opts = {
                    "format": 'JPEG',
                    "quality": quality,
                    "optimize": True
                }
                # Only add subsampling if not default (-1)
                if subsampling_val != -1:
                    save_opts["subsampling"] = subsampling_val

                img.save(file_path, **save_opts)

            except Exception as e:
                 print(f"Error saving JPG file {file_path}: {e}")
                 # Optional: Basic fallback without optimize/subsampling
                 try:
                      print(f"Attempting basic JPG save for {file_path}...")
                      img.save(file_path, format='JPEG', quality=quality)
                 except Exception as fallback_e:
                      print(f"Fallback JPG save also failed: {fallback_e}")
                      continue # Skip adding this file to results if save fails

            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }

# --- Add Node to Mappings ---
NODE_CLASS_MAPPINGS["SaveImageAsJPG"] = SaveImageAsJPG
NODE_DISPLAY_NAME_MAPPINGS["SaveImageAsJPG"] = "Save Image (JPG)"
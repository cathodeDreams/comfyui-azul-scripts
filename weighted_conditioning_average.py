# comfyui-azul-scripts/weighted_conditioning_average.py

import torch
import logging
from comfy.comfy_types import IO, InputTypeDict
# Optional imports, usually available implicitly:
# import comfy.model_management
# import node_helpers

# Dictionary initialization for ComfyUI custom nodes
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

class WeightedConditioningAverage:
    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        return {
            "required": {
                "conditioning_to": (IO.CONDITIONING, {"tooltip": "The primary conditioning. Its strength is controlled by 'conditioning_to_strength'."}),
                "conditioning_from": (IO.CONDITIONING, {"tooltip": "The secondary conditioning being averaged into the primary."}),
                "conditioning_to_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "The weight of the 'conditioning_to' input. 'conditioning_from' will have a weight of (1.0 - this value)."
                }),
                "overall_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,  # Allow boosting significantly
                    "step": 0.01,
                    "tooltip": "An additional multiplier applied to the strength of the final averaged conditioning."
                })
            }
        }
    RETURN_TYPES = (IO.CONDITIONING,)
    OUTPUT_TOOLTIPS = ("The averaged conditioning, with an optional overall strength multiplier applied.",)
    FUNCTION = "addWeighted"

    CATEGORY = "Azul's Scripts" # <--- Changed Category
    DESCRIPTION = "Averages two conditionings based on 'conditioning_to_strength', then applies an 'overall_strength' multiplier to the result's influence."

    def addWeighted(self, conditioning_to, conditioning_from, conditioning_to_strength, overall_strength):
        out = []

        if not conditioning_to:
             logging.error("WeightedConditioningAverage: conditioning_to is empty or invalid.")
             return ([],) # Return empty conditioning if input is invalid

        if not conditioning_from:
             logging.warning("WeightedConditioningAverage: conditioning_from is empty or invalid. Passing through conditioning_to.")
             # Apply overall_strength directly to conditioning_to if conditioning_from is missing
             for t_item in conditioning_to:
                 t1, d1 = t_item
                 d1_copy = d1.copy()
                 original_strength = d1.get("strength", 1.0)
                 d1_copy["strength"] = original_strength * overall_strength
                 out.append([t1, d1_copy])
             return (out,)


        if len(conditioning_from) > 1:
            logging.warning("Warning: WeightedConditioningAverage conditioning_from contains more than 1 cond item, only the first one will be used for averaging.")

        # Use the first conditioning item [tensor, dict] from conditioning_from
        cond_from_tensor, cond_from_dict = conditioning_from[0]
        pooled_output_from = cond_from_dict.get("pooled_output", None)
        strength_from_info = cond_from_dict.get("strength", 1.0)

        for i in range(len(conditioning_to)):
            t1, d1 = conditioning_to[i] # tensor and dict for the current item in conditioning_to list
            pooled_output_to = d1.get("pooled_output", None)
            strength_to_info = d1.get("strength", 1.0)

            # Ensure tensor shapes are compatible for broadcasting or pad/slice
            target_len = t1.shape[1]
            source_tensor = cond_from_tensor

            if source_tensor.shape[1] < target_len:
                # Pad source tensor (t0)
                padding_size = target_len - source_tensor.shape[1]
                padding = torch.zeros((source_tensor.shape[0], padding_size, source_tensor.shape[2]), device=source_tensor.device, dtype=source_tensor.dtype)
                t0 = torch.cat([source_tensor, padding], dim=1)
            elif source_tensor.shape[1] > target_len:
                 # Truncate source tensor (t0)
                t0 = source_tensor[:, :target_len, :]
            else:
                t0 = source_tensor # Shapes match

            # --- Averaging Logic ---
            # Ensure conditioning_to_strength is within [0, 1] although input widget might enforce this
            clamped_to_strength = max(0.0, min(1.0, conditioning_to_strength))
            tw = torch.lerp(t0, t1, clamped_to_strength) # Linear interpolation is equivalent

            # Copy the dictionary from conditioning_to item to preserve its metadata
            output_dict = d1.copy()

            # Average pooled outputs if both exist
            if pooled_output_from is not None and pooled_output_to is not None:
                # Add check for shape compatibility before lerp if necessary
                if pooled_output_from.shape == pooled_output_to.shape:
                    output_dict["pooled_output"] = torch.lerp(pooled_output_from, pooled_output_to, clamped_to_strength)
                else:
                    logging.warning(f"WeightedConditioningAverage: Pooled output shapes mismatch ({pooled_output_from.shape} vs {pooled_output_to.shape}). Using conditioning_to's pooled output.")
                    # Decide on fallback: keep pooled_output_to, or discard? Keep 'to' seems safer.
                    if pooled_output_to is None: # if 'to' was None, remove key
                         if "pooled_output" in output_dict: del output_dict["pooled_output"]
                    # else pooled_output_to remains in output_dict
            elif pooled_output_from is not None and pooled_output_to is None:
                 # If only 'from' has pooled output, lerp towards zero-equivalent or just take 'from'?
                 # Original code just took 'from'. Let's blend 'from' towards nothing based on strength.
                 # Or simpler: Take 'from' but scale its strength?
                 # Let's try scaling 'from' based on its weight (1.0 - clamped_to_strength)
                 output_dict["pooled_output"] = pooled_output_from * (1.0 - clamped_to_strength) # Scaled pool from 'from'
            # else: pooled_output_to remains if it existed, or remains None

            # --- Apply Overall Strength Multiplier ---
            # Calculate effective base strength after averaging
            avg_original_strength = (strength_to_info * clamped_to_strength) + (strength_from_info * (1.0 - clamped_to_strength))
            final_strength = avg_original_strength * overall_strength
            output_dict["strength"] = final_strength

            # Create the new conditioning item [tensor, dict]
            new_cond_item = [tw, output_dict]
            out.append(new_cond_item)

        return (out,)

# --- Add Node to Mappings ---
NODE_CLASS_MAPPINGS["WeightedConditioningAverage"] = WeightedConditioningAverage
NODE_DISPLAY_NAME_MAPPINGS["WeightedConditioningAverage"] = "Conditioning Average (Weighted Output)"
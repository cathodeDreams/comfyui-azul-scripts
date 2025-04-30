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
                "conditioning_to": (IO.CONDITIONING, {"tooltip": "The primary conditioning. The blend amount is controlled by 'conditioning_to_strength'."}),
                "conditioning_from": (IO.CONDITIONING, {"tooltip": "The secondary conditioning being averaged into the primary."}),
                "conditioning_to_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "The blend factor towards 'conditioning_to'. 1.0 means pure 'conditioning_to', 0.0 means pure 'conditioning_from'."
                }),
                "overall_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,  # Allow boosting significantly
                    "step": 0.01,
                    "tooltip": "Sets the final strength multiplier for the averaged conditioning's influence. Similar to (averaged_concept:overall_strength)."
                })
            }
        }
    RETURN_TYPES = (IO.CONDITIONING,)
    OUTPUT_TOOLTIPS = ("The averaged conditioning, with its influence scaled by 'overall_strength'.",) # <-- Updated tooltip
    FUNCTION = "addWeighted"

    CATEGORY = "Azul's Scripts"
    # <-- Updated Description
    DESCRIPTION = "Averages two conditionings based on 'conditioning_to_strength', then sets the resulting conditioning's influence using 'overall_strength'."

    def addWeighted(self, conditioning_to, conditioning_from, conditioning_to_strength, overall_strength):
        out = []

        if not conditioning_to:
             logging.error("WeightedConditioningAverage: conditioning_to is empty or invalid.")
             return ([],) # Return empty conditioning if input is invalid

        if not conditioning_from:
             logging.warning("WeightedConditioningAverage: conditioning_from is empty or invalid. Passing through conditioning_to and applying overall_strength.")
             # Apply overall_strength directly to conditioning_to if conditioning_from is missing
             for t_item in conditioning_to:
                 t1, d1 = t_item
                 d1_copy = d1.copy()
                 # --- MODIFIED ---
                 # Directly set the strength based on the overall_strength parameter
                 d1_copy["strength"] = overall_strength
                 # --- END MODIFIED ---
                 out.append([t1, d1_copy])
             return (out,)


        if len(conditioning_from) > 1:
            logging.warning("Warning: WeightedConditioningAverage conditioning_from contains more than 1 cond item, only the first one will be used for averaging.")

        # Use the first conditioning item [tensor, dict] from conditioning_from
        cond_from_tensor, cond_from_dict = conditioning_from[0]
        pooled_output_from = cond_from_dict.get("pooled_output", None)
        # We no longer need strength_from_info for the main calculation

        for i in range(len(conditioning_to)):
            t1, d1 = conditioning_to[i] # tensor and dict for the current item in conditioning_to list
            pooled_output_to = d1.get("pooled_output", None)
            # We no longer need strength_to_info for the main calculation

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

            # --- Averaging Logic (Blending Tensors) ---
            # Ensure conditioning_to_strength is within [0, 1]
            clamped_to_strength = max(0.0, min(1.0, conditioning_to_strength))
            # Linear interpolation blends the tensors
            tw = torch.lerp(t0, t1, clamped_to_strength)

            # Copy the dictionary from conditioning_to item to preserve its metadata (like area conditioning)
            output_dict = d1.copy()

            # --- Average pooled outputs if both exist ---
            if pooled_output_from is not None and pooled_output_to is not None:
                if pooled_output_from.shape == pooled_output_to.shape:
                    output_dict["pooled_output"] = torch.lerp(pooled_output_from, pooled_output_to, clamped_to_strength)
                else:
                    logging.warning(f"WeightedConditioningAverage: Pooled output shapes mismatch ({pooled_output_from.shape} vs {pooled_output_to.shape}). Using conditioning_to's pooled output if available.")
                    # Keep pooled_output_to if it exists, otherwise remove the key
                    if pooled_output_to is None:
                         if "pooled_output" in output_dict: del output_dict["pooled_output"]
                    # else: pooled_output_to remains in output_dict via the initial copy
            elif pooled_output_from is not None and pooled_output_to is None:
                 # If only 'from' has pooled output, blend it towards zero based on its weight
                 output_dict["pooled_output"] = pooled_output_from * (1.0 - clamped_to_strength) # Scaled pool from 'from'
            # else: pooled_output_to remains if it existed (from copy), or remains absent

            # --- Apply Overall Strength Multiplier ---
            # --- MODIFIED ---
            # Directly set the final strength based *only* on the overall_strength parameter.
            # This ignores the original strengths of the input conditionings.
            final_strength = overall_strength
            output_dict["strength"] = final_strength
            # --- END MODIFIED ---

            # Create the new conditioning item [tensor, dict]
            new_cond_item = [tw, output_dict]
            out.append(new_cond_item)

        return (out,)

# --- Add Node to Mappings ---
NODE_CLASS_MAPPINGS["WeightedConditioningAverage"] = WeightedConditioningAverage
NODE_DISPLAY_NAME_MAPPINGS["WeightedConditioningAverage"] = "Conditioning Average (Weighted Output)"

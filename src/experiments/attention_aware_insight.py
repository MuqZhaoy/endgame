import torch
import numpy as np
import json
import os
from tqdm import tqdm
from pathlib import Path
from .base import Experiment
from config import device_configs # Assuming device_configs provides device access
from evaluator import EvaluationResult
from quantizer import Quantizer, build_quantizers
from matplotlib import pyplot as plt
from dataclasses import asdict
from functools import cached_property
from transformers.cache_utils import DynamicCache # Import added based on code usage

class AttentionAwareInsight(Experiment):
    """
    Experiment to gain insights into Attention-Aware (Attn) quantization.
    Compares Attn-Aware uniform quantization against standard fixed-bit uniform quantization.
    Visualizes bit allocation and performance trade-offs.
    """

    @cached_property
    def quantizer_list(self) -> list[tuple[Quantizer, Quantizer]]:
        # --- Configuration for Comparison ---
        # We want to compare Non-Attn with fixed bits vs Attn with varying target errors.
        # Fixed parameters for focused comparison:
        focus_level = "head"
        focus_symmetric = False
        focus_outliers = 0.01
        focus_method = "uniform"
        
        # Non-Attention-Aware Configurations (Fixed Bits)
        non_attn_base_config = {
            "use_attentions": [False],
            "method": [focus_method],
            "level": [focus_level],
            "symmetric": [focus_symmetric],
            "outliers_ratio": [focus_outliers],
            "n_bits_uniform": [2, 3, 4, 5, 6, 7, 8], # Varying fixed bits
            # --- Fixed N/A params ---
            "group_size": [None],
            "last_n_attentions": [None], "target_quantization_error": [None],
            "n_bits_min": [None], "n_bits_max": [None], "q_norm": [None],
        }

        # Attention-Aware Configurations (Varying Target Error)
        attn_base_config = {
            "use_attentions": [True],
            "method": [focus_method],
            "level": [focus_level],
            "symmetric": [focus_symmetric],
            "outliers_ratio": [focus_outliers],
            "last_n_attentions": [5], # Fixed example value
            "target_quantization_error": [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0], # Vary target error
            "n_bits_min": [1],        # Fixed example range
            "n_bits_max": [8],        # Fixed example range
            "q_norm": [300],          # Fixed example value (key only)
            # --- Fixed N/A params ---
            "group_size": [None],
            "n_bits_uniform": [None],
        }

        # --- Build Key and Value Quantizers ---
        all_configs = [non_attn_base_config, attn_base_config]

        print("Building Key Quantizers for Attention-Aware Insight...")
        key_quantizers = build_quantizers([{"key_or_value_cache": ["key"], **cfg} for cfg in all_configs])
        print(f"Built {len(key_quantizers)} Key Quantizers.")

        print("Building Value Quantizers for Attention-Aware Insight...")
        value_quantizers = build_quantizers([{"key_or_value_cache": ["value"], **cfg} for cfg in all_configs])
        print(f"Built {len(value_quantizers)} Value Quantizers.")

        # --- Pair Key and Value Quantizers (Symmetrical Pairing) ---
        quantizer_pairs = []
        if len(key_quantizers) != len(value_quantizers):
            print(f"Warning: Mismatch in generated key ({len(key_quantizers)}) and value ({len(value_quantizers)}) quantizers. Pairing might be incorrect.")
            min_len = min(len(key_quantizers), len(value_quantizers))
            key_quantizers = key_quantizers[:min_len]
            value_quantizers = value_quantizers[:min_len]

        for k, v in zip(key_quantizers, value_quantizers):
             # Ensure pairing makes sense (e.g., both Non-Attn or both Attn)
             if k.params['use_attentions'] == v.params['use_attentions']:
                 quantizer_pairs.append((k, v))
             else:
                  print(f"Skipping mismatched pair: Key Attn={k.params['use_attentions']}, Value Attn={v.params['use_attentions']}")


        total_pairs = len(quantizer_pairs)
        print(f"Generated {total_pairs} quantizer pairs for Attention-Aware Insight.")
        if total_pairs == 0:
             print("Warning: No quantizer pairs were generated.")
        
        return quantizer_pairs

    def process_result(self, results: list[EvaluationResult]):
        if not results:
            print("No results to process.")
            return

        # Ensure necessary directories exist
        Path("figs/attn_aware_insight").mkdir(parents=True, exist_ok=True)
        Path("data").mkdir(parents=True, exist_ok=True)

        # --- Data Separation and Processing ---
        plot_data = []
        non_attn_results = {"avg_bits": [], "accuracy": [], "quant_error": [], "attn_error": [], "avg_size": []}
        attn_results = {"avg_bits": [], "accuracy": [], "quant_error": [], "attn_error": [], "avg_size": [], "target_error": []}
        
        attn_quantizer_for_viz = None # To store one quantizer for bit viz

        for (key_q, val_q), result in zip(self.quantizer_list, results):
            # Assuming symmetrical K/V for simplicity in analysis
            q_params = key_q.params # Use key quantizer params as representative
            is_attn = q_params["use_attentions"]
            
            # Store data for JSON
            entry = {
                "params": q_params,
                "results": asdict(result)
            }
            plot_data.append(entry)

            # Separate data for plotting
            avg_bits = result.average_n_bits # Use the reported average bits
            accuracy = result.accuracy
            quant_error = result.quantization_error
            attn_error = result.attention_error
            avg_size = result.average_size
            
            if is_attn:
                attn_results["avg_bits"].append(avg_bits)
                attn_results["accuracy"].append(accuracy)
                attn_results["quant_error"].append(quant_error)
                attn_results["attn_error"].append(attn_error)
                attn_results["avg_size"].append(avg_size)
                attn_results["target_error"].append(q_params.get("target_quantization_error", None))
                if attn_quantizer_for_viz is None: # Get one for visualization
                     attn_quantizer_for_viz = key_q # Use key quantizer
            else:
                # For non-attn, the 'x' axis is technically n_bits_uniform, but avg_bits is better for comparison
                non_attn_results["avg_bits"].append(avg_bits)
                non_attn_results["accuracy"].append(accuracy)
                non_attn_results["quant_error"].append(quant_error)
                non_attn_results["attn_error"].append(attn_error)
                non_attn_results["avg_size"].append(avg_size)
        
        # Sort non-attn results by average bits for potentially smoother line plots
        non_attn_indices = np.argsort(non_attn_results["avg_bits"])
        for key in non_attn_results:
             non_attn_results[key] = np.array(non_attn_results[key])[non_attn_indices]

        # --- Bit Tensor Visualization ---
        if attn_quantizer_for_viz:
            print("Visualizing bit allocation for one Attention-Aware configuration...")
            try:
                model = self.get_model(0) # Get a model instance
                device = next(model.parameters()).device # Get device from model
                question = self.questions[0] # Use the first question
                input_ids = question.input_ids[:1].to(device) # Use first batch/sequence
                question_len = question.question_length

                with torch.no_grad():
                    # Run first forward pass to get cache and attentions
                    outputs = model.forward(input_ids, use_cache=True, output_attentions=True, return_dict=True)
                    past_key_values = outputs.past_key_values
                    attentions = outputs.attentions
                    
                    # Extract key cache (matching quantizer)
                    key_cache_list = []
                    is_dynamic_cache = isinstance(past_key_values, DynamicCache)
                    is_legacy_cache = isinstance(past_key_values, (list, tuple))
                    if is_dynamic_cache:
                         key_cache_list = [k_layer[:, :, :question_len, :].to(device) for k_layer in past_key_values.key_cache]
                    elif is_legacy_cache:
                         key_cache_list = [key[:, :, :question_len, :].to(device) for key, _ in past_key_values]
                    else: raise TypeError("Unsupported cache format")
                    key_cache = torch.stack(key_cache_list) # L, B, H, S, E
                    
                    # Prepare attentions needed by quantizer
                    question_attentions = [attn[:,:,:question_len,:question_len].to(device) for attn in attentions]

                    # Calculate outlier mask
                    outlier_mask = attn_quantizer_for_viz._calc_outlier_mask(key_cache)
                    
                    # Calculate the bit tensor
                    attn_quantizer_for_viz.set_dtype_and_device(key_cache.dtype, device) # Ensure quantizer has dtype/device
                    n_bits_tensor = attn_quantizer_for_viz._calc_quantization_bits(question_attentions, key_cache, outlier_mask)
                    # Expected shape depends on level, e.g., (B, S, L, H) for head level
                    
                    # Average over Batch and Sequence length for Layer-Head heatmap
                    if n_bits_tensor.dim() >= 4 and attn_quantizer_for_viz.level == 'head':
                        # Permute to (L, H, B, S) or similar if needed for mean
                        # Example assuming (B, S, L, H) -> mean over (0, 1)
                        bits_per_layer_head = n_bits_tensor.float().mean(dim=(0, 1)).cpu().numpy()
                        
                        # Save the averaged tensor data
                        np.save("data/attn_aware_bits_layer_head.npy", bits_per_layer_head)

                        # Plot heatmap
                        plt.figure(figsize=(10, 8))
                        plt.imshow(bits_per_layer_head, cmap='viridis', aspect='auto')
                        plt.colorbar(label='Average Bits')
                        plt.xlabel("Head Index")
                        plt.ylabel("Layer Index")
                        plt.title(f"Avg Bit Allocation (Attn-Aware, Level={attn_quantizer_for_viz.level}, TargetErr={attn_quantizer_for_viz.params.get('target_quantization_error')})")
                        plt.savefig("figs/attn_aware_insight/bit_distribution.png", dpi=150)
                        plt.close()
                        print("Saved bit allocation heatmap to figs/attn_aware_insight/bit_distribution.png")
                    else:
                         print(f"Could not visualize bit tensor: Level '{attn_quantizer_for_viz.level}' or tensor shape {n_bits_tensor.shape} not suitable for Layer-Head heatmap.")

            except Exception as e:
                print(f"Error during bit tensor visualization: {e}")
        else:
             print("Skipping bit tensor visualization as no Attention-Aware configuration was found/run.")

        # --- Performance Comparison Plotting ---
        print("Generating performance comparison plots...")
        metrics_to_plot = [
             ("Accuracy", "accuracy"),
             ("Quantization Error", "quant_error"),
             ("Attention Error", "attn_error")
        ]
        x_axis_options = [
             ("Average Bits per Value", "avg_bits"),
             ("Average KV Cache Size per Token (bits)", "avg_size")
        ]

        for x_label, x_key in x_axis_options:
             fig, axs = plt.subplots(1, len(metrics_to_plot), figsize=(7 * len(metrics_to_plot), 6), tight_layout=True)
             fig.suptitle(f'Attention-Aware vs Non-Aware ({attn_quantizer_for_viz.level}-Level)') # Use level from viz quantizer

             for i, (y_label, y_key) in enumerate(metrics_to_plot):
                  ax = axs[i]
                  # Plot Non-Attn results (line plot as it follows fixed bits)
                  if len(non_attn_results[x_key]) > 1:
                       ax.plot(non_attn_results[x_key], non_attn_results[y_key], marker='o', linestyle='-', label='Non-Attn (Fixed Bits)')
                  elif len(non_attn_results[x_key]) == 1: # Single point case
                       ax.scatter(non_attn_results[x_key], non_attn_results[y_key], marker='o', label='Non-Attn (Fixed Bits)')
                  
                  # Plot Attn results (scatter plot)
                  ax.scatter(attn_results[x_key], attn_results[y_key], marker='x', label='Attn-Aware')

                  # Add labels for target error on attn points (optional, can be messy)
                  # for j, txt in enumerate(attn_results['target_error']):
                  #      ax.annotate(f'{txt:.1f}', (attn_results[x_key][j], attn_results[y_key][j]))

                  ax.set_xlabel(x_label)
                  ax.set_ylabel(y_label)
                  ax.set_title(f'{y_label} vs {x_label}')
                  ax.legend()
                  ax.grid(True, linestyle='--', alpha=0.6)

             plot_filename = f"figs/attn_aware_insight/performance_vs_{x_key}.png"
             plt.savefig(plot_filename, dpi=150)
             plt.close(fig)
             print(f"Saved performance plot to {plot_filename}")

        # --- JSON Data Storage ---
        json_filename = "data/attn_aware_results.json"
        try:
            with open(json_filename, "w") as f:
                json.dump(plot_data, f, indent=4)
            print(f"Saved detailed results to {json_filename}")
        except Exception as e:
            print(f"Error saving results to JSON: {e}") 
import torch
import numpy as np
import json
import os
from tqdm import tqdm
from pathlib import Path
from .base import Experiment
from config import device_configs
from evaluator import EvaluationResult
from quantizer import Quantizer, build_quantizers
from matplotlib import pyplot as plt
from dataclasses import asdict
from functools import cached_property

class GroupingInsight(Experiment):
    """
    Experiment to investigate the impact of Grouped Quantization (group_size).
    Compares performance across different group sizes under fixed settings.
    """

    @cached_property
    def quantizer_list(self) -> list[tuple[Quantizer, Quantizer]]:
        # --- Configuration for Comparison ---
        # Fixed parameters for focused comparison:
        focus_level = "head"
        focus_symmetric = False
        focus_outliers = 0.01
        focus_use_attentions = False
        focus_method = "uniform"
        focus_bits = 4 # Fixed bit depth for comparison
        
        # !!! IMPORTANT: Adjust group_size values based on your model's embed_size_per_head !!!
        # Ensure values divide embed_size_per_head. None means no grouping.
        group_sizes_to_test = [32, 64, 128, None] # ADJUST THESE VALUES!

        # Base config varying only group_size
        base_config = {
            "use_attentions": [focus_use_attentions],
            "method": [focus_method],
            "level": [focus_level],
            "symmetric": [focus_symmetric],
            "outliers_ratio": [focus_outliers],
            "n_bits_uniform": [focus_bits],
            "group_size": group_sizes_to_test, # Vary group size
            # --- Fixed N/A params ---
            "last_n_attentions": [None], "target_quantization_error": [None],
            "n_bits_min": [None], "n_bits_max": [None], "q_norm": [None],
        }

        # --- Build Key and Value Quantizers --- 
        all_configs = [base_config]

        print("Building Key Quantizers for Grouping Insight...")
        key_quantizers = build_quantizers([{"key_or_value_cache": ["key"], **cfg} for cfg in all_configs])
        print(f"Built {len(key_quantizers)} Key Quantizers.")

        print("Building Value Quantizers for Grouping Insight...")
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
            quantizer_pairs.append((k, v))

        total_pairs = len(quantizer_pairs)
        print(f"Generated {total_pairs} quantizer pairs for Grouping Insight.")
        if total_pairs == 0:
            print("Warning: No quantizer pairs were generated.")
        
        # TODO: Add a check here or in build_quantizers to ensure group sizes are valid for the model?
        #       Requires model config access during experiment setup.

        return quantizer_pairs

    def process_result(self, results: list[EvaluationResult]):
        if not results:
            print("No results to process for GroupingInsight.")
            return

        # Ensure necessary directories exist
        Path("figs/grouping_insight").mkdir(parents=True, exist_ok=True)
        Path("data").mkdir(parents=True, exist_ok=True)

        # --- Data Separation and Processing by Group Size ---
        plot_data = []
        # Use a dictionary where keys are group sizes (or 'None' string)
        grouping_results = {}

        for (key_q, _), result in zip(self.quantizer_list, results):
            q_params = key_q.params # Use key quantizer params as representative
            group_size = q_params.get("group_size", "disabled") # Get group_size, handle None/disabled
            group_size_key = str(group_size) # Use string representation as dict key
            
            if group_size_key not in grouping_results:
                 grouping_results[group_size_key] = {
                      "avg_bits": [], "accuracy": [], "quant_error": [], "attn_error": [], "avg_size": []
                 }
            
            # Store data for JSON
            entry = {
                "params": q_params,
                "results": asdict(result)
            }
            plot_data.append(entry)

            # Separate data for plotting
            grouping_results[group_size_key]["avg_bits"].append(result.average_n_bits)
            grouping_results[group_size_key]["accuracy"].append(result.accuracy)
            grouping_results[group_size_key]["quant_error"].append(result.quantization_error)
            grouping_results[group_size_key]["attn_error"].append(result.attention_error)
            grouping_results[group_size_key]["avg_size"].append(result.average_size)
        
        # --- Performance Comparison Plotting ---
        print("Generating performance comparison plots for different group sizes...")
        metrics_to_plot = [
             ("Accuracy", "accuracy"),
             ("Quantization Error", "quant_error"),
             ("Attention Error", "attn_error")
        ]
        # Plotting against avg_size is often most informative for grouping trade-offs
        x_axis_options = [
             # ("Group Size (None treated as 0)", "group_size_numeric"), # Hard to plot None
             ("Average Bits per Value (Reported)", "avg_bits"),
             ("Average KV Cache Size per Token (bits)", "avg_size")
        ]
        
        # Get sorted list of group size keys (handle 'None' appropriately for legend)
        group_keys_sorted = sorted(grouping_results.keys(), key=lambda x: float(x) if x != 'None' and x != 'disabled' else float('inf'))

        for x_label, x_key in x_axis_options:
            fig, axs = plt.subplots(1, len(metrics_to_plot), figsize=(7 * len(metrics_to_plot), 6), tight_layout=True)
            fixed_params = self.quantizer_list[0][0].params # Get fixed params from first quantizer
            fig.suptitle(f'Grouping Comparison (Level={fixed_params["level"]})') 

            for i, (y_label, y_key) in enumerate(metrics_to_plot):
                ax = axs[i]
                for group_key in group_keys_sorted:
                    if group_key in grouping_results and len(grouping_results[group_key][x_key]) > 0:
                        # Extract data for this group size
                        x_values = grouping_results[group_key][x_key]
                        y_values = grouping_results[group_key][y_key]
                        
                        label = f"Group {group_key}" if group_key not in ['None', 'disabled'] else "No Grouping"
                        
                        # Since we only have one point per group size in this setup,
                        # use scatter plot instead of line plot.
                        ax.scatter(x_values, y_values, marker='o', s=80, label=label)
                        # Add text label next to the point
                        # for k in range(len(x_values)):
                        #      ax.text(x_values[k], y_values[k], f'  {label}', verticalalignment='center')

                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                ax.set_title(f'{y_label} vs {x_label}')
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.6)

            plot_filename = f"figs/grouping_insight/performance_vs_{x_key}.png"
            plt.savefig(plot_filename, dpi=150)
            plt.close(fig)
            print(f"Saved performance plot to {plot_filename}")

        # --- JSON Data Storage ---
        json_filename = "data/grouping_results.json"
        try:
            with open(json_filename, "w") as f:
                json.dump(plot_data, f, indent=4)
            print(f"Saved detailed results to {json_filename}")
        except Exception as e:
            print(f"Error saving results to JSON: {e}") 
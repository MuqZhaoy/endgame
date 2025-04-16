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

class AdaptiveMethodInsight(Experiment):
    """
    Experiment to compare Adaptive quantization against Uniform and Normal methods.
    Focuses on performance differences at various bit depths under fixed settings.
    """

    @cached_property
    def quantizer_list(self) -> list[tuple[Quantizer, Quantizer]]:
        # --- Configuration for Comparison ---
        # Fixed parameters for focused comparison:
        focus_level = "head"
        focus_symmetric = False
        focus_outliers = 0.01
        focus_use_attentions = False
        focus_group_size = None
        bit_range = [2, 3, 4, 5, 6, 7, 8] # Bit depths to compare

        # Base config shared by all methods in this experiment
        base_config_shared = {
            "use_attentions": [focus_use_attentions],
            "level": [focus_level],
            "symmetric": [focus_symmetric],
            "outliers_ratio": [focus_outliers],
            "n_bits_uniform": bit_range, # Varying bits
            # --- Fixed N/A params ---
            "group_size": [focus_group_size],
            "last_n_attentions": [None], "target_quantization_error": [None],
            "n_bits_min": [None], "n_bits_max": [None], "q_norm": [None],
        }

        # Create specific configs for each method
        uniform_config = {**base_config_shared, "method": ["uniform"]}
        normal_config = {**base_config_shared, "method": ["normal"]}
        adaptive_config = {**base_config_shared, "method": ["adaptive"]}

        # --- Build Key and Value Quantizers --- 
        all_configs = [uniform_config, normal_config, adaptive_config]

        print("Building Key Quantizers for Adaptive Method Insight...")
        key_quantizers = build_quantizers([{"key_or_value_cache": ["key"], **cfg} for cfg in all_configs])
        print(f"Built {len(key_quantizers)} Key Quantizers.")

        print("Building Value Quantizers for Adaptive Method Insight...")
        value_quantizers = build_quantizers([{"key_or_value_cache": ["value"], **cfg} for cfg in all_configs])
        print(f"Built {len(value_quantizers)} Value Quantizers.")

        # --- Pair Key and Value Quantizers (Symmetrical Pairing) ---
        quantizer_pairs = []
        if len(key_quantizers) != len(value_quantizers):
            print(f"Warning: Mismatch in generated key ({len(key_quantizers)}) and value ({len(value_quantizers)}) quantizers. Pairing might be incorrect.")
            min_len = min(len(key_quantizers), len(value_quantizers))
            key_quantizers = key_quantizers[:min_len]
            value_quantizers = value_quantizers[:min_len]

        # Pair based on matching config (implicitly done by zipping after building from same config list)
        for k, v in zip(key_quantizers, value_quantizers):
            quantizer_pairs.append((k, v))

        total_pairs = len(quantizer_pairs)
        print(f"Generated {total_pairs} quantizer pairs for Adaptive Method Insight.")
        if total_pairs == 0:
            print("Warning: No quantizer pairs were generated.")
        
        return quantizer_pairs

    def process_result(self, results: list[EvaluationResult]):
        if not results:
            print("No results to process for AdaptiveMethodInsight.")
            return

        # Ensure necessary directories exist
        Path("figs/adaptive_method_insight").mkdir(parents=True, exist_ok=True)
        Path("data").mkdir(parents=True, exist_ok=True)

        # --- Data Separation and Processing by Method ---
        plot_data = []
        method_results = {
            "uniform": {"n_bits": [], "avg_bits": [], "accuracy": [], "quant_error": [], "attn_error": [], "avg_size": []},
            "normal": {"n_bits": [], "avg_bits": [], "accuracy": [], "quant_error": [], "attn_error": [], "avg_size": []},
            "adaptive": {"n_bits": [], "avg_bits": [], "accuracy": [], "quant_error": [], "attn_error": [], "avg_size": []}
        }

        for (key_q, _), result in zip(self.quantizer_list, results):
            q_params = key_q.params # Use key quantizer params as representative
            method = q_params["method_name"]
            n_bits = q_params.get("n_bits_uniform") # Configured bit depth
            
            if method not in method_results:
                print(f"Warning: Unexpected method '{method}' found in results.")
                continue
            
            # Store data for JSON
            entry = {
                "params": q_params,
                "results": asdict(result)
            }
            plot_data.append(entry)

            # Separate data for plotting
            method_results[method]["n_bits"].append(n_bits)
            method_results[method]["avg_bits"].append(result.average_n_bits)
            method_results[method]["accuracy"].append(result.accuracy)
            method_results[method]["quant_error"].append(result.quantization_error)
            method_results[method]["attn_error"].append(result.attention_error)
            method_results[method]["avg_size"].append(result.average_size)
        
        # Sort results within each method by n_bits for smoother plots
        for method in method_results:
            sort_indices = np.argsort(method_results[method]["n_bits"])
            for key in method_results[method]:
                method_results[method][key] = np.array(method_results[method][key])[sort_indices]

        # --- Performance Comparison Plotting ---
        print("Generating performance comparison plots for different methods...")
        metrics_to_plot = [
             ("Accuracy", "accuracy"),
             ("Quantization Error", "quant_error"),
             ("Attention Error", "attn_error")
        ]
        x_axis_options = [
             ("Configured Bits (n_bits_uniform)", "n_bits"),
             ("Average Bits per Value (Reported)", "avg_bits"),
             ("Average KV Cache Size per Token (bits)", "avg_size")
        ]
        methods_to_include = ["uniform", "normal", "adaptive"]
        colors = {"uniform": "blue", "normal": "green", "adaptive": "red"}
        markers = {"uniform": "o", "normal": "s", "adaptive": "^"}

        for x_label, x_key in x_axis_options:
            fig, axs = plt.subplots(1, len(metrics_to_plot), figsize=(7 * len(metrics_to_plot), 6), tight_layout=True)
            fig.suptitle(f'Quantization Method Comparison (Level={self.quantizer_list[0][0].level})') # Use level from first quantizer

            for i, (y_label, y_key) in enumerate(metrics_to_plot):
                ax = axs[i]
                for method in methods_to_include:
                    if method_results[method][x_key].size > 0:
                        ax.plot(
                            method_results[method][x_key],
                            method_results[method][y_key],
                            marker=markers[method],
                            linestyle='-',
                            color=colors[method],
                            label=method.capitalize()
                        )

                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                ax.set_title(f'{y_label} vs {x_label}')
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.6)

            plot_filename = f"figs/adaptive_method_insight/performance_vs_{x_key}.png"
            plt.savefig(plot_filename, dpi=150)
            plt.close(fig)
            print(f"Saved performance plot to {plot_filename}")

        # --- JSON Data Storage ---
        json_filename = "data/adaptive_method_results.json"
        try:
            with open(json_filename, "w") as f:
                json.dump(plot_data, f, indent=4)
            print(f"Saved detailed results to {json_filename}")
        except Exception as e:
            print(f"Error saving results to JSON: {e}") 
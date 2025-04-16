from tqdm import tqdm
from .base import Experiment
from dataclasses import asdict
from matplotlib import pyplot as plt
from functools import cached_property
from evaluator import EvaluationResult
from quantizer import Quantizer, build_quantizers

params = [
    "level",
    "symmetric",
    "method_name",
    "outliers_ratio",
    "use_attentions",
    "n_bits_min",
    "n_bits_max",
    "last_n_attentions",
    "target_quantization_error",
    "q_norm",
    "n_bits_uniform",
    "group_size",
]
relations = [
    ("accuracy", "average_size"),
    ("accuracy", "quantization_error"),
    ("accuracy", "attention_error"),
    ("answer_log_probability", "average_size"),
    ("answer_log_probability", "quantization_error"),
    ("answer_log_probability", "attention_error"),
    ("average_size", "quantization_error"),
    ("average_size", "attention_error"),
]
translation = {
    "level": "Level",
    "symmetric": "Symmetric",
    "method_name": "Method name",
    "outliers_ratio": "Outliers ratio",
    "use_attentions": "Attention-aware",
    "n_bits_min": "Min # of bits",
    "n_bits_max": "Max # of bits",
    "last_n_attentions": "Last n attentions",
    "target_quantization_error": "Target error",
    "q_norm": "2-norm of query tensor",
    "n_bits_uniform": "Uniform # of bits",
    "group_size": "Group Size",
    "accuracy": "Accuracy",
    "answer_log_probability": "Answer log probability",
    "average_size": "KVcache size",
    "quantization_error": "Quantization error",
    "attention_error": "Attention error",
}


class GridSearch(Experiment):
    @cached_property
    def quantizer_list(self) -> list[tuple[Quantizer, Quantizer]]:

        # --- Define Configuration Blocks (Targeting ~100 combos each) ---

        # Block 1: Baseline Focus (Non-Attn, Uniform, No Grouping)
        # Explores basic level, symmetry, outlier, bit depth effects.
        baseline_focus = {
            "use_attentions": [False],          # Baseline
            "method": ["uniform"],              # Baseline
            "group_size": [None],               # Baseline
            "level": ["token", "layer", "head"],# (3 options)
            "symmetric": [False, True],         # (2 options)
            "outliers_ratio": [0, 0.01, 0.05],  # (3 options)
            "n_bits_uniform": [2, 4, 6, 8],     # (4 options) -> 3*2*3*4 = 72 combos
            # --- Fixed N/A params ---
            "last_n_attentions": [None], "target_quantization_error": [None],
            "n_bits_min": [None], "n_bits_max": [None], "q_norm": [None],
        }

        # Block 2: Attention-Aware Focus (Attn enabled, Uniform, No Grouping)
        # Explores Attn parameters and interactions. Reduced level/outliers to control size.
        attn_focus = {
            "use_attentions": [True],           # Feature enabled
            "method": ["uniform"],              # Baseline method
            "group_size": [None],               # Baseline grouping
            "level": ["layer", "head"],         # (2 options) - Focus where effect might differ more
            "symmetric": [False, True],         # (2 options)
            "outliers_ratio": [0.01],           # (1 option) - Reduced
            "last_n_attentions": [1, 5],        # (2 options) - Attn param
            "target_quantization_error": [1.0, 20.0], # (2 options) - Attn param
            "n_bits_min": [1, 4],               # (2 options) - Attn param
            "n_bits_max": [4, 8],               # (2 options) - Attn param
            "q_norm": [300],                    # (1 option) - Attn param (key only)
            # --- Fixed N/A params ---
            "n_bits_uniform": [None],
            # Combinations: 2 * 2 * 1 * 2 * 2 * 2 * 2 = 64 combos
        }

        # Block 3: Adaptive Focus (Non-Attn, Adaptive enabled, No Grouping)
        # Explores Adaptive method vs baseline parameters.
        adaptive_focus = {
            "use_attentions": [False],          # Baseline Attn
            "method": ["adaptive"],             # Feature enabled
            "group_size": [None],               # Baseline grouping
            "level": ["token", "layer", "head"],# (3 options)
            "symmetric": [False, True],         # (2 options)
            "outliers_ratio": [0, 0.01, 0.05],  # (3 options)
            "n_bits_uniform": [4, 6, 8],        # (3 options) - Base bits for adaptive
            # --- Fixed N/A params ---
            "last_n_attentions": [None], "target_quantization_error": [None],
            "n_bits_min": [None], "n_bits_max": [None], "q_norm": [None],
            # Combinations: 3 * 2 * 3 * 3 = 54 combos
        }

        # Block 4: Grouping Focus (Non-Attn, Uniform, Grouping enabled)
        # Explores Grouping vs baseline parameters.
        # !!! IMPORTANT: Adjust group_size based on your model's embed_size_per_head !!!
        grouping_focus = {
            "use_attentions": [False],          # Baseline Attn
            "method": ["uniform"],              # Baseline method
            "group_size": [32, 64, 128],        # (3 options) - Feature enabled (!!! ADJUST !!!)
            "level": ["layer", "head"],         # (2 options) - Grouping relevant levels
            "symmetric": [False, True],         # (2 options)
            "outliers_ratio": [0, 0.01, 0.05],  # (3 options)
            "n_bits_uniform": [4, 6],           # (2 options)
            # --- Fixed N/A params ---
            "last_n_attentions": [None], "target_quantization_error": [None],
            "n_bits_min": [None], "n_bits_max": [None], "q_norm": [None],
            # Combinations: 3 * 2 * 2 * 3 * 2 = 72 combos
        }

        # --- Pairwise Combination Blocks ---

        # Block 5: Attention-Aware + Grouping (Attn enabled, Uniform, Grouping enabled)
        # Explores interaction of Attn and Grouping. Reduced some params to control size.
        # !!! IMPORTANT: Adjust group_size based on your model's embed_size_per_head !!!
        attn_grouping = {
            "use_attentions": [True],           # Feature 1 enabled
            "method": ["uniform"],              # Baseline method
            "group_size": [32, 64, 128],        # (3 options) - Feature 2 enabled (!!! ADJUST !!!)
            "level": ["layer", "head"],         # (2 options)
            "symmetric": [False, True],         # (2 options)
            "outliers_ratio": [0.01],           # (1 option) - Reduced
            "last_n_attentions": [5],           # (1 option) - Reduced
            "target_quantization_error": [1.0, 20.0], # (2 options)
            "n_bits_min": [1, 4],               # (2 options)
            "n_bits_max": [4, 8],               # (2 options)
            "q_norm": [300],                    # (1 option)
            # --- Fixed N/A params ---
            "n_bits_uniform": [None],
            # Combinations: 3 * 2 * 2 * 1 * 1 * 2 * 2 * 2 * 1 = 96 combos
        }

        # Block 6: Adaptive + Grouping (Non-Attn, Adaptive enabled, Grouping enabled)
        # Explores interaction of Adaptive and Grouping.
        # !!! IMPORTANT: Adjust group_size based on your model's embed_size_per_head !!!
        adaptive_grouping = {
            "use_attentions": [False],          # Baseline Attn
            "method": ["adaptive"],             # Feature 1 enabled
            "group_size": [32, 64, 128],        # (3 options) - Feature 2 enabled (!!! ADJUST !!!)
            "level": ["layer", "head"],         # (2 options)
            "symmetric": [False, True],         # (2 options)
            "outliers_ratio": [0, 0.01, 0.05],  # (3 options)
            "n_bits_uniform": [4, 6, 8],        # (3 options) - Base bits
            # --- Fixed N/A params ---
            "last_n_attentions": [None], "target_quantization_error": [None],
            "n_bits_min": [None], "n_bits_max": [None], "q_norm": [None],
            # Combinations: 3 * 2 * 2 * 3 * 3 = 108 combos (Slightly over, acceptable)
        }

        # Note: Attn + Adaptive combination block is omitted due to likely implementation constraints.
        # Note: Three-way combination (Attn + Adaptive + Grouping) is also omitted.

        # --- Combine all configuration blocks ---
        all_configs = [
            baseline_focus,    # Block 1 (~72)
            attn_focus,        # Block 2 (~64)
            adaptive_focus,    # Block 3 (~54)
            grouping_focus,    # Block 4 (~72)
            attn_grouping,     # Block 5 (~96)
            adaptive_grouping, # Block 6 (~108)
        ]

        # --- Build Key and Value Quantizers ---
        print("Building key quantizer configurations...")
        key_quantizer_configs_full = [{"key_or_value_cache": ["key"], **cfg} for cfg in all_configs]
        key_quantizers = build_quantizers(key_quantizer_configs_full)
        print(f"Built {len(key_quantizers)} key quantizers.")

        print("Building value quantizer configurations...")
        value_quantizer_configs_full = [{"key_or_value_cache": ["value"], **cfg} for cfg in all_configs]
        value_quantizers = build_quantizers(value_quantizer_configs_full)
        print(f"Built {len(value_quantizers)} value quantizers.")

        # --- Pair Key and Value Quantizers ---
        quantizer_pairs = []
        if len(key_quantizers) != len(value_quantizers):
            print(f"Warning: Mismatch in generated key ({len(key_quantizers)}) and value ({len(value_quantizers)}) quantizers. Pairing might be incorrect.")
            min_len = min(len(key_quantizers), len(value_quantizers))
            print(f"Pairing up to minimum length: {min_len}")
            key_quantizers = key_quantizers[:min_len]
            value_quantizers = value_quantizers[:min_len]

        for k, v in zip(key_quantizers, value_quantizers):
            quantizer_pairs.append((k, v))

        total_pairs = len(quantizer_pairs)
        estimated_total = 72 + 64 + 54 + 72 + 96 + 108 # Recalculate estimate here
        print(f"Generated {total_pairs} quantizer pairs for grid search. (Estimated: {estimated_total})" )
        if total_pairs > 1000:
             print("Warning: The number of configurations is large. Grid search may take a very long time.")
        elif total_pairs == 0:
             print("Warning: No quantizer pairs were generated. Check configuration blocks.")

        # TODO: Before running, double-check group_size compatibility with your model's embed_size_per_head.
        #       Modify the group_size lists in Blocks 4, 5, 6 if necessary.

        return quantizer_pairs

    def process_result(self, results: list[EvaluationResult]):
        plt.figure(figsize=(5*len(relations), 5*2*len(params)))
        for param_idx, param_name in enumerate(tqdm(params)):
            for relation_idx, (metric_name_x, metric_name_y) in enumerate(relations):
                key_x, key_y, value_x, value_y = {}, {}, {}, {}
                for (key_quantizers, value_quantizers), result in zip(self.quantizer_list, results):
                    result = asdict(result)
                    if param_name in key_quantizers.params:
                        key_param_data = key_quantizers.params[param_name]
                        if key_param_data not in key_x:
                            key_x[key_param_data] = []
                        key_x[key_param_data].append(result[metric_name_x])
                        if key_param_data not in key_y:
                            key_y[key_param_data] = []
                        key_y[key_param_data].append(result[metric_name_y])
                    if param_name in value_quantizers.params:
                        value_param_data = value_quantizers.params[param_name]
                        if value_param_data not in value_x:
                            value_x[value_param_data] = []
                        value_x[value_param_data].append(result[metric_name_x])
                        if value_param_data not in value_y:
                            value_y[value_param_data] = []
                        value_y[value_param_data].append(result[metric_name_y])
                ax = plt.subplot(2*len(params), len(relations), (2*param_idx) * len(relations) + (relation_idx+1))
                for label in key_x:
                    ax.scatter(key_x[label], key_y[label], label=label)
                if len(key_x) > 0:
                    ax.legend()
                ax.set_title(f"{translation[param_name]} (Key)")
                ax.set_xlabel(translation[metric_name_x])
                ax.set_ylabel(translation[metric_name_y])
                ax.set_box_aspect(1)
                ax = plt.subplot(2*len(params), len(relations), (2*param_idx+1) * len(relations) + (relation_idx+1))
                for label in value_x:
                    ax.scatter(value_x[label], value_y[label], label=label)
                if len(value_x) > 0:
                    ax.legend()
                ax.set_title(f"{translation[param_name]} (Value)")
                ax.set_xlabel(translation[metric_name_x])
                ax.set_ylabel(translation[metric_name_y])
                ax.set_box_aspect(1)
        print(f"Rendering {2*len(params)*len(relations)} figures, it may take about 30 seconds...")
        plt.tight_layout()
        plt.savefig("figs/grid_search_results.png", dpi=100)

import os
import gc
import json
import math
import torch
from tqdm import tqdm
from models import CausalLM
from typing import Optional, Any
from torch.nn import functional as F
from functools import cached_property
from dataclasses import dataclass, asdict
from qa_dataset import QADataset, Question
from quantizer import Quantizer, AttentionType
from transformers.cache_utils import Cache, DynamicCache


@dataclass
class EvaluationResult:
    accuracy: float = 0.0
    accuracy_confidence: float = 0.0
    answer_log_probability: float = 0.0
    quantization_error: float = 0.0
    key_quantization_error: float = 0.0
    value_quantization_error: float = 0.0
    attention_error: float = 0.0
    logit_error: float = 0.0
    average_n_bits: float = 0.0
    key_average_n_bits: float = 0.0
    value_average_n_bits: float = 0.0
    average_size: float = 0.0
    key_average_size: float = 0.0
    value_average_size: float = 0.0


class Evaluator:
    def __init__(self, device: torch.device,
                 version: str,
                 model_name: str,
                 datasets: QADataset,
                 key_quantizer: Quantizer,
                 value_quantizer: Quantizer):
        self.device = device
        self.version = version
        self.model_name = model_name
        self.datasets = datasets
        self.key_quantizer = key_quantizer
        self.value_quantizer = value_quantizer

    @cached_property
    def params(self) -> dict[str, Any]:
        res: dict[str, Any] = {}
        res["version"] = self.version
        res["model_name"] = self.model_name
        res["dataset_name"] = self.datasets.dataset_name
        res["question_count"] = self.datasets.question_count
        res["key_quantizer"] = self.key_quantizer.params
        res["value_quantizer"] = self.value_quantizer.params
        return res

    def _calc_tensor_error(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
        return ((tensor1.to(self.device) - tensor2.to(self.device)) ** 2).mean().item()

    def _calc_attention_error(self, attention1: AttentionType, attention2: AttentionType) -> float:
        return sum(self._calc_tensor_error(attn1, attn2) for attn1, attn2 in zip(attention1, attention2)) / len(attention1)

    def _evaluate_single(self, model: CausalLM, question: Question) -> EvaluationResult:
        question_len = question.question_length
        # Forward before quantization
        input_ids = question.input_ids.to(self.device)
        result = model.forward(input_ids, use_cache=True, output_attentions=True, return_dict=True)
        # Quantize key/value cache
        question_attentions = [attn[:,:,:question_len,:question_len].to(self.device) for attn in result.attentions]
        
        # MODIFICATION START: Handle modern Cache object format
        past_key_values = result.past_key_values # This might be a Cache object or legacy tuple
        key_cache_list = []
        value_cache_list = []
        
        if hasattr(past_key_values, 'key_cache') and hasattr(past_key_values, 'value_cache'): # Heuristic check for Cache object
             # Assuming structure like DynamicCache where key_cache/value_cache are lists of tensors per layer
             for k_layer, v_layer in zip(past_key_values.key_cache, past_key_values.value_cache):
                  key_cache_list.append(k_layer[:,:,:question_len,:].to(self.device))
                  value_cache_list.append(v_layer[:,:,:question_len,:].to(self.device))
        elif isinstance(past_key_values, (list, tuple)): # Handle legacy tuple/list format
             for key, value in past_key_values:
                  key_cache_list.append(key[:,:,:question_len,:].to(self.device))
                  value_cache_list.append(value[:,:,:question_len,:].to(self.device))
        else:
             raise TypeError(f"Unsupported past_key_values format: {type(past_key_values)}")
             
        key_cache = torch.stack(key_cache_list) # Shape (L, B, H, S_kv, E)
        value_cache = torch.stack(value_cache_list) # Shape (L, B, H, S_kv, E)
        # Note: Quantizer expects (B, S, L, H, E), need to handle potential shape mismatch if quantizer expects permuted input
        # Let's assume quantizer.quantize now correctly handles (L, B, H, S, E) input based on previous fixes.
        # If not, permute here before passing to quantizer.
        # key_cache = key_cache.permute(1, 3, 0, 2, 4) # B, S, L, H, E
        # value_cache = value_cache.permute(1, 3, 0, 2, 4) # B, S, L, H, E

        quantized_key_cache_stack, key_average_n_bits = self.key_quantizer.quantize(key_cache, question_attentions)
        quantized_value_cache_stack, value_average_n_bits = self.value_quantizer.quantize(value_cache, question_attentions)

        # If quantizer returned permuted shape, permute back before creating cache
        # quantized_key_cache_stack = quantized_key_cache_stack.permute(2, 0, 3, 1, 4) # L, B, H, S, E
        # quantized_value_cache_stack = quantized_value_cache_stack.permute(2, 0, 3, 1, 4) # L, B, H, S, E
        
        # MODIFICATION START: Reconstruct past_key_values in the ORIGINAL format
        quantized_kvcache: Optional[Cache | tuple[tuple[torch.Tensor]]] = None

        is_dynamic_cache = isinstance(past_key_values, DynamicCache)
        is_legacy_cache = isinstance(past_key_values, (list, tuple)) # Keep this check for older transformers or edge cases
        num_layers_in_cache = 0
        if is_dynamic_cache:
             num_layers_in_cache = len(past_key_values.key_cache)
        elif is_legacy_cache:
             num_layers_in_cache = len(past_key_values)
        else: # Might be another Cache type or unsupported
             # Attempt to get length generically, might fail
             try:
                  num_layers_in_cache = len(past_key_values.key_cache) if hasattr(past_key_values, 'key_cache') else len(past_key_values)
             except Exception:
                  raise TypeError(f"Cannot determine number of layers for past_key_values type: {type(past_key_values)}")

        if num_layers_in_cache != quantized_key_cache_stack.shape[0]:
             # Sanity check
             raise ValueError(f"Number of layers mismatch: original cache ({num_layers_in_cache}) vs quantized stack ({quantized_key_cache_stack.shape[0]}) - Cache Type: {type(past_key_values)}")

        # Prepare lists for the new cache
        new_key_cache_list = []
        new_value_cache_list = []
        for idx in range(num_layers_in_cache): # Iterate through layers
            # Determine the original device for this specific layer
            layer_device: Optional[torch.device] = None
            if is_dynamic_cache:
                layer_device = past_key_values.key_cache[idx].device
            elif is_legacy_cache:
                layer_device = past_key_values[idx][0].device
            else: # Fallback for other Cache types or error
                 try:
                      layer_device = past_key_values.key_cache[idx].device
                 except Exception:
                      raise TypeError(f"Could not determine layer device for past_key_values type: {type(past_key_values)}")
            
            # Append the quantized tensors for this layer to the list, on the correct device
            new_key_cache_list.append(quantized_key_cache_stack[idx].to(layer_device))
            new_value_cache_list.append(quantized_value_cache_stack[idx].to(layer_device))

        # Now construct the final cache in the original format
        if is_dynamic_cache:
             quantized_kvcache = DynamicCache(key_cache=new_key_cache_list, value_cache=new_value_cache_list)
        elif is_legacy_cache:
             # Reconstruct legacy tuple format
             quantized_kvcache_legacy = list(zip(new_key_cache_list, new_value_cache_list))
             quantized_kvcache = tuple(quantized_kvcache_legacy) # Convert list of tuples to tuple of tuples
        else:
             # If it's another Cache type, we might need specific handling.
             # For now, raise an error if it wasn't DynamicCache or legacy.
             # TODO: Add support for other Cache types if necessary (e.g., StaticCache)
             raise TypeError(f"Unsupported cache type for reconstruction: {type(past_key_values)}. Only DynamicCache and legacy tuples currently supported.")
        # MODIFICATION END

        # Forward after quantization using the reconstructed kv cache
        quantized_result = model.forward(input_ids[:,question_len:], past_key_values=quantized_kvcache, use_cache=True, output_attentions=True, return_dict=True)
        # Calculate log probabilities
        first_word_log_softmax = F.log_softmax(result.logits[:,question_len-1], dim=-1)
        quantized_log_softmax = F.log_softmax(quantized_result.logits, dim=-1)
        max_log_probability, max_choice_idx, answer_log_probability = None, None, None
        for choice_idx, choice_len in enumerate(question.choice_length):
            quantized_log_probability = first_word_log_softmax[choice_idx, input_ids[choice_idx, question_len]].item()
            quantized_log_probability += quantized_log_softmax[choice_idx, torch.arange(choice_len-1), input_ids[choice_idx,question_len+1:question_len+choice_len]].sum().item()
            quantized_log_probability /= choice_len
            if choice_idx == question.answer_idx:
                answer_log_probability = quantized_log_probability
            if max_log_probability is None or quantized_log_probability > max_log_probability:
                max_log_probability = quantized_log_probability
                max_choice_idx = choice_idx
        # Calculate quantization metrics
        key_quantization_error = self._calc_tensor_error(key_cache, quantized_key_cache_stack)
        value_quantization_error = self._calc_tensor_error(value_cache, quantized_value_cache_stack)
        attention_error = self._calc_attention_error(
            [attn[:,:,question_len:,:question_len].to(self.device) for attn in result.attentions],
            [attn[:,:,:,:question_len].to(self.device) for attn in quantized_result.attentions],
        )
        logit_error = self._calc_tensor_error(result.logits[:,question_len:,:], quantized_result.logits)

        # Get model dimensions for size calculation
        num_layers = model.config.num_hidden_layers
        num_heads = model.config.num_attention_heads
        # Calculate embed_size_per_head, ensure integer division
        embed_size_per_head = model.config.hidden_size // num_heads
        if model.config.hidden_size % num_heads != 0:
             # This shouldn't happen with standard transformer models
             print(f"Warning: hidden_size ({model.config.hidden_size}) is not divisible by num_heads ({num_heads}). Size calculation might be inaccurate.")

        # Use the new signature for calc_quantized_cache_size_per_token
        key_average_size = self.key_quantizer.calc_quantized_cache_size_per_token(
            num_layers=num_layers,
            num_heads=num_heads,
            embed_size_per_head=embed_size_per_head
        )
        value_average_size = self.value_quantizer.calc_quantized_cache_size_per_token(
            num_layers=num_layers,
            num_heads=num_heads,
            embed_size_per_head=embed_size_per_head
        )
        return EvaluationResult(
            accuracy=1.0 if max_choice_idx == question.answer_idx else 0.0,
            answer_log_probability=answer_log_probability,
            quantization_error=(key_quantization_error + value_quantization_error) / 2,
            key_quantization_error=key_quantization_error,
            value_quantization_error=value_quantization_error,
            attention_error=attention_error,
            logit_error=logit_error,
            average_size=(key_average_size + value_average_size) / 2,
            key_average_size=key_average_size,
            value_average_size=value_average_size,
            average_n_bits=(key_average_n_bits + value_average_n_bits) / 2,
            key_average_n_bits=key_average_n_bits,
            value_average_n_bits=value_average_n_bits,
        )

    def evaluate(self, model: CausalLM, use_tqdm: bool) -> EvaluationResult:
        assert model.name_or_path == self.model_name
        result = EvaluationResult()
        total_tokens = 0
        with torch.no_grad():
            for idx, question in enumerate(tqdm(self.datasets.questions) if use_tqdm else self.datasets.questions):
                single_result = self._evaluate_single(model, question)
                n_tokens = question.question_length
                total_tokens += n_tokens
                result.accuracy += single_result.accuracy
                result.answer_log_probability += single_result.answer_log_probability
                result.quantization_error += single_result.quantization_error
                result.key_quantization_error += single_result.key_quantization_error
                result.value_quantization_error += single_result.value_quantization_error
                result.attention_error += single_result.attention_error
                result.logit_error += single_result.logit_error
                result.average_size += single_result.average_size * n_tokens
                result.key_average_size += single_result.key_average_size * n_tokens
                result.value_average_size += single_result.value_average_size * n_tokens
                result.average_n_bits += single_result.average_n_bits * n_tokens
                result.key_average_n_bits += single_result.key_average_n_bits * n_tokens
                result.value_average_n_bits += single_result.value_average_n_bits * n_tokens
                if (idx + 1) % 100 == 0:
                    gc.collect()
        result.accuracy /= self.datasets.question_count
        # Calculate 95% confidence interval
        result.accuracy_confidence = 1.96 * math.sqrt(result.accuracy * (1.0 - result.accuracy) / self.datasets.question_count)
        result.answer_log_probability /= self.datasets.question_count
        result.quantization_error /= self.datasets.question_count
        result.key_quantization_error /= self.datasets.question_count
        result.value_quantization_error /= self.datasets.question_count
        result.attention_error /= self.datasets.question_count
        result.logit_error /= self.datasets.question_count
        result.average_size /= total_tokens
        result.key_average_size /= total_tokens
        result.value_average_size /= total_tokens
        result.average_n_bits /= total_tokens
        result.key_average_n_bits /= total_tokens
        result.value_average_n_bits /= total_tokens
        return result
    
    def get_cached_result(self, cache_file: Optional[str]) -> Optional[EvaluationResult]:
        if cache_file is None or not os.path.exists(cache_file):
            return None
        with open(cache_file, "r") as f:
            cached_results = json.load(f)
            for entry in cached_results:
                if entry["params"] == self.params:
                    return EvaluationResult(**entry["results"])
        return None

    def cache_result(self, cache_file: Optional[str], result: EvaluationResult):
        if cache_file is None:
            return
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                cached_results = json.load(f)
        else:
            cached_results = []
        cached_results.append({
            "params": self.params,
            "results": asdict(result),
        })
        with open(cache_file, "w") as f:
            json.dump(cached_results, f, indent=4, separators=(", ", ": "))

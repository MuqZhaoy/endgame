import math
import torch
import numpy as np
from models import CausalLM
from scipy.stats import norm
from itertools import product
from functools import cached_property
from typing import Literal, Optional, Any


AttentionType = list[torch.Tensor]
QuantizationLevels = Literal["no-quantization", "token", "layer", "head"]
QuantizationMethods = Literal["uniform", "normal", "adaptive"]


class Quantizer:
    def __init__(self,
                 # Key cache or value cache
                 key_or_value_cache: Optional[Literal["key", "value"]] = None,
                 # no-quantization, token-level, layer-level, or head-level
                 level: Optional[QuantizationLevels] = None,
                 # True: assume cache is already zero-centered, only scale it
                 # False: zero-center cache and then scale it
                 symmetric: Optional[bool] = None,
                 # uniform: assume normalized cache values obbey uniform distribution between max value and min value
                 # normal: assume normalized cache values obbey standard normal distribution
                 # adaptive: derive quantization levels from empirical distribution
                 method: Optional[QuantizationMethods] = None,
                 # Percentage of outliers (including both lowest and highest)
                 outliers_ratio: Optional[float] = None,
                 # Whether to enable attention-aware quantization
                 use_attentions: Optional[bool] = None,
                 # (only applicable for uniform quantization)
                 # The uniform quantization bits
                 n_bits_uniform: Optional[int] = None,
                 # (only applicable for attention-aware quantization)
                 # Use the last n rows of attentions to calculate quantization bits
                 last_n_attentions: Optional[int] = None,
                 # (only applicable for attention-aware quantization)
                 # Target quantization error
                 target_quantization_error: Optional[float] = None,
                 # (only applicable for attention-aware quantization)
                 # Minimum allowed quantization bits
                 n_bits_min: Optional[int] = None,
                 # (only applicable for attention-aware quantization)
                 # Maximum allowed quantization bits
                 n_bits_max: Optional[int] = None,
                 # (only applicable for attention-aware quantization of key cache)
                 # 2-norm of query tensor used in the formula
                 q_norm: Optional[float] = None,
                 # --- NEW PARAMETERS ---
                 # MODIFICATION START: Add group_size parameter
                 # (Optional) Size of feature groups within the embedding dimension for grouped quantization.
                 # If None or <= 0, grouping is disabled.
                 group_size: Optional[int] = None
                 # MODIFICATION END
                 ):
        # Set key_or_value_cache
        assert key_or_value_cache is not None
        self.key_or_value_cache = key_or_value_cache
        # Early exit for no quantization
        assert level is not None
        self.level = level
        if level == "no-quantization":
            return
        # Set level
        if level == "token":
            self.quantize_dims = (-3, -2, -1)
        elif level == "layer":
            self.quantize_dims = (-2, -1)
        elif level == "head":
            self.quantize_dims = (-1,)
        # Set symmetric
        assert symmetric is not None
        self.symmetric = symmetric
        # Set outliers_ratio
        assert outliers_ratio is not None
        self.outliers_ratio = outliers_ratio
        # Set use_attentions:
        assert use_attentions is not None
        self.use_attentions = use_attentions

        # MODIFICATION START: Store group_size and validate
        self.group_size = group_size if group_size is not None and group_size > 0 else -1
        self.use_grouping = self.group_size > 0
        if self.use_grouping:
            # Grouping currently makes most sense conceptually with head-level or maybe layer-level.
            # For token-level, the entire embedding dimension is already treated as one block.
            # Let's allow it for layer/head for now.
            assert self.level in ["layer", "head"], "Grouping is typically used with layer or head level quantization."
        # MODIFICATION END

        if use_attentions:
            # Set last_n_attentions
            assert last_n_attentions is not None
            assert last_n_attentions > 0
            self.last_n_attentions = last_n_attentions
            # Set target_quantization_error
            assert target_quantization_error is not None
            assert target_quantization_error > 0.0
            self.target_quantization_error = target_quantization_error
            # Set n_bits_min
            assert n_bits_min is not None
            assert 0 <= n_bits_min <= 16
            self.n_bits_min = n_bits_min
            # Set n_bits_max
            assert n_bits_max is not None
            assert n_bits_min <= n_bits_max <= 16
            self.n_bits_max = n_bits_max
            if self.key_or_value_cache == "key":
                # Set q_norm
                assert q_norm is not None
                assert q_norm > 0
                self.q_norm = q_norm
        else:
            # Set n_bits_uniform
            assert n_bits_uniform is not None
            assert 0 <= n_bits_uniform <= 16
            self.n_bits_uniform = n_bits_uniform
        # Set method
        assert method is not None
        self.method_name = method
        if method == "uniform":
            self.quantization_method = self._uniform_quantize
        elif method == "normal":
            self.quantization_method = self._normal_quantize
        # MODIFICATION START: Added adaptive quantization method
        elif method == "adaptive":
            self.quantization_method = self._adaptive_quantize
            # Adaptive method typically uses std-dev based normalization
            self.normalization_method_for_adaptive = "std" # Default normalization for adaptive
        # MODIFICATION END

    def set_dtype_and_device(self, dtype: torch.dtype, device: torch.device):
        self.dtype = dtype
        self.device = device
        # MODIFICATION START: Adapt precomputation logic for adaptive method (if using normal quantiles)
        if self.level != "no-quantization":
            # Precompute normal quantiles if normal method is used OR if adaptive method uses std normalization
            needs_normal_quantiles = (self.method_name == "normal") or \
                                     (self.method_name == "adaptive" and getattr(self, 'normalization_method_for_adaptive', None) == "std")

            if needs_normal_quantiles:
                 if self.use_attentions:
                     n_bits_range = range(self.n_bits_min, self.n_bits_max+1)
                 else:
                     n_bits_range = range(self.n_bits_uniform, self.n_bits_uniform+1)

                 # Filter out n_bits = 0 as 2**0 = 1 quantile is problematic
                 valid_n_bits_range = [n for n in n_bits_range if n > 0]

                 if valid_n_bits_range: # Ensure the range is not empty and contains bits > 0
                     self.normal_quantiles_upper_bound = {
                         n: torch.tensor(norm.ppf(np.arange(0, 1, 1/(2**n)) + 1/(2**n)), dtype=dtype, device=device)
                         for n in valid_n_bits_range
                     }
                     self.normal_quantiles_center = {
                         n: torch.tensor(norm.ppf(np.arange(0, 1, 1/(2**n)) + 0.5/(2**n)), dtype=dtype, device=device)
                         for n in valid_n_bits_range
                     }
                 else:
                     # Handle case where no valid n_bits > 0 are specified
                     self.normal_quantiles_upper_bound = {}
                     self.normal_quantiles_center = {}
            # else: # No need to precompute normal quantiles if method is uniform or adaptive (not std-based)
            #     pass
        # MODIFICATION END

    @cached_property
    def params(self) -> dict[str, Any]:
        res: dict[str, Any] = {}
        res["key_or_value_cache"] = self.key_or_value_cache
        res["level"] = self.level
        if self.level == "no-quantization":
            return res
        res["symmetric"] = self.symmetric
        res["method_name"] = self.method_name
        res["outliers_ratio"] = self.outliers_ratio
        res["use_attentions"] = self.use_attentions
        # MODIFICATION START: Add group_size to params
        res["group_size"] = self.group_size if self.use_grouping else "disabled"
        # MODIFICATION END
        if self.use_attentions:
            res["n_bits_min"] = self.n_bits_min
            res["n_bits_max"] = self.n_bits_max
            res["last_n_attentions"] = self.last_n_attentions
            res["target_quantization_error"] = self.target_quantization_error
            if self.key_or_value_cache == "key":
                res["q_norm"] = self.q_norm
        else:
            res["n_bits_uniform"] = self.n_bits_uniform
        return res

    def _calc_quantization_bits(self, attentions: AttentionType, cache: torch.Tensor, outlier_mask: torch.Tensor) -> torch.Tensor:
        # cache/outlier_mask.shape: (n_batch, seq_len, n_layer, n_head, embed_size_per_head)
        n_batch, seq_len, n_layer, n_head, _ = cache.shape
        if not self.use_attentions:
            if self.level == "token":
                shape = (n_batch, seq_len)
            elif self.level == "layer":
                shape = (n_batch, seq_len, n_layer)
            elif self.level == "head":
                shape = (n_batch, seq_len, n_layer, n_head)
            return torch.ones(shape, dtype=torch.int64, device=self.device) * self.n_bits_uniform
        if self.key_or_value_cache == "key":
            max_error = math.sqrt(12.0 / self.q_norm * math.log(seq_len**3/(seq_len-1) * self.target_quantization_error**2 + 1))
            # max_error.shape: (n_batch, seq_len) or (n_batch, seq_len, n_layer) or (n_batch, seq_len, n_layer, n_head)
        elif self.key_or_value_cache == "value":
            attentions = torch.stack(attentions)
            # attentions.shape: (n_layer, n_batch, n_head, seq_len, seq_len)
            attentions = attentions[:, :, :, -self.last_n_attentions:, :]
            # attentions.shape: (n_layer, n_batch, n_head, last_n_attentions, seq_len)
            attentions = attentions.permute(1, 4, 0, 2, 3)
            # attentions.shape: (n_batch, seq_len, n_layer, n_head, last_n_attentions)
            attentions = attentions.amax(dim=self.quantize_dims)
            # attentions.shape: (n_batch, seq_len) or (n_batch, seq_len, n_layer) or (n_batch, seq_len, n_layer, n_head)
            max_error = math.sqrt(12.0 / seq_len) * self.target_quantization_error / attentions
            # max_error.shape: (n_batch, seq_len) or (n_batch, seq_len, n_layer) or (n_batch, seq_len, n_layer, n_head)
        cache = torch.masked.masked_tensor(cache, torch.logical_not(outlier_mask))
        # NOTE: PyTorch's bug: https://github.com/pytorch/pytorch/issues/115624
        quantize_dims = [x + len(cache.shape) for x in self.quantize_dims]
        if self.method_name == "uniform":
            if self.symmetric:
                max_value = cache.abs().amax(dim=quantize_dims)
                scale_value = 2 * max_value
            else:
                max_value = cache.amax(dim=quantize_dims)
                min_value = cache.amin(dim=quantize_dims)
                scale_value = max_value - min_value
            assert scale_value.get_mask().all()
            scale_value = scale_value.get_data()
            # scale_value.shape: (n_batch, seq_len) or (n_batch, seq_len, n_layer) or (n_batch, seq_len, n_layer, n_head)
            n_bits = torch.log2(scale_value / (2 * max_error) + 1)
            # n_bits.shape: (n_batch, seq_len) or (n_batch, seq_len, n_layer) or (n_batch, seq_len, n_layer, n_head)
        elif self.method_name == "normal":
            raise NotImplementedError()
        n_bits = torch.clamp(torch.ceil(n_bits).to(torch.int64), self.n_bits_min, self.n_bits_max)
        # n_bits.shape: (n_batch, seq_len) or (n_batch, seq_len, n_layer) or (n_batch, seq_len, n_layer, n_head)
        # The last (last_n_attentions-1) tokens do not have enough history attentions so we do not quantize them
        if self.last_n_attentions > 1:
            n_bits[:, -self.last_n_attentions+1:] = self.n_bits_max
        return n_bits

    def _calc_outlier_mask(self, cache: torch.Tensor) -> torch.Tensor:
        if self.outliers_ratio == 0.0:
            return torch.zeros_like(cache, dtype=torch.bool, device=self.device)

        # cache.shape: (n_batch, seq_len, n_layer, n_head, embed_size_per_head)
        original_shape = cache.shape

        # MODIFICATION START: Calculate outliers per group if grouping enabled
        if self.use_grouping:
            # Reshape for grouping: (..., embed_size_per_head) -> (..., num_groups, group_size)
            cache_grouped = self._reshape_for_grouping(cache)
            # Calculate thresholds along the group_size dimension (-1)
            group_dim_size = cache_grouped.shape[-1]
            k_lower = int(math.ceil(self.outliers_ratio / 2 * group_dim_size))
            k_upper = int(math.floor((1.0 - self.outliers_ratio / 2) * group_dim_size))

            # Initialize masks
            lower_mask = torch.zeros_like(cache_grouped, dtype=torch.bool)
            upper_mask = torch.zeros_like(cache_grouped, dtype=torch.bool)

            # Calculate lower mask only if k_lower > 0
            if k_lower > 0:
                if k_lower > group_dim_size: k_lower = group_dim_size # Clamp k if needed
                lower_threshold = torch.kthvalue(cache_grouped, k=k_lower, dim=-1, keepdim=True).values # keepdim for broadcasting
                lower_mask = cache_grouped <= lower_threshold
            
            # Calculate upper mask only if k_upper < group_dim_size
            # If k_upper >= group_dim_size, it means we keep all values up to the max, so no upper outliers
            if k_upper < group_dim_size:
                if k_upper <= 0: k_upper = 1 # Clamp k if needed (should be at least 1st element)
                # kthvalue finds the k-th smallest; we want values *greater* than the k_upper-th value
                upper_threshold = torch.kthvalue(cache_grouped, k=k_upper, dim=-1, keepdim=True).values # keepdim for broadcasting
                upper_mask = cache_grouped > upper_threshold

            mask_grouped = lower_mask | upper_mask
            # mask_grouped shape: (..., num_groups, group_size)

            # Reshape mask back to original cache shape
            mask = self._reshape_from_grouping(mask_grouped)

        else:
            # Original logic: calculate outliers over the entire quantization block
            # Flatten according to self.quantize_dims relative to the end
            # Make sure quantize_dims are negative indices relative to the end
            if not all(d < 0 for d in self.quantize_dims):
                 raise ValueError("quantize_dims must contain negative indices for flattening logic.")
            start_flatten_dim = min(self.quantize_dims)
            cache_flat = cache.flatten(start_dim=cache.dim() + start_flatten_dim) # Flatten dimensions defined by quantize_dims
            # cache_flat.shape example: (B, S, L, H*E) for level='head'
            flat_dim_size = cache_flat.shape[-1]
            k_lower = int(math.ceil(self.outliers_ratio / 2 * flat_dim_size))
            k_upper = int(math.floor((1.0 - self.outliers_ratio / 2) * flat_dim_size))

            # Initialize flat masks
            lower_mask_flat = torch.zeros_like(cache_flat, dtype=torch.bool)
            upper_mask_flat = torch.zeros_like(cache_flat, dtype=torch.bool)

            if k_lower > 0:
                if k_lower > flat_dim_size: k_lower = flat_dim_size
                lower_threshold = torch.kthvalue(cache_flat, k=k_lower, dim=-1, keepdim=True).values
                lower_mask_flat = cache_flat <= lower_threshold

            if k_upper < flat_dim_size:
                if k_upper <= 0: k_upper = 1
                upper_threshold = torch.kthvalue(cache_flat, k=k_upper, dim=-1, keepdim=True).values
                upper_mask_flat = cache_flat > upper_threshold

            mask_flat = lower_mask_flat | upper_mask_flat
            # Reshape mask back to original cache shape
            mask = mask_flat.view(*original_shape)
            # mask.shape = (n_batch, seq_len, n_layer, n_head, embed_size_per_head)
        # MODIFICATION END

        # Ensure output mask has the same shape as input cache
        assert mask.shape == original_shape, f"Outlier mask shape mismatch: {mask.shape} vs {original_shape}"
        return mask

    # Returns (normalized cache, mean value, scale value)
    # Mean/scale have shapes compatible for denormalization (e.g., keep group dim if grouped)
    def _normalize(self, cache: torch.Tensor, method: Literal["minmax", "std"], n_bits: int, outlier_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # cache/outlier_mask assumed to be slices corresponding to one quantization unit,
        # potentially with multiple groups if self.use_grouping is True.
        # Shape examples (before potential grouping):
        # level=head: (embed_size_per_head,)
        # level=layer: (n_head, embed_size_per_head)
        # level=token: (n_layer, n_head, embed_size_per_head)
        
        # MODIFICATION START: Handle grouping within normalization
        original_shape = cache.shape
        if self.use_grouping:
            # Grouping assumes the input cache/mask has the original shape for the unit
            # e.g., for level=head, input shape is (embed_size_per_head,)
            # We reshape it here.
            cache_grouped = self._reshape_for_grouping(cache)
            outlier_mask_grouped = self._reshape_for_grouping(outlier_mask)
            # Normalization dimension is now the group size dimension (-1)
            norm_dims = (-1,)
            # Target shape for stats needs to broadcast across group_size dim
            target_stat_shape = cache_grouped.shape[:-1] + (1,) # Shape: (..., num_groups, 1)
            masked_input = torch.where(outlier_mask_grouped, torch.nan, cache_grouped.float())
        else:
            # Normalize over all dimensions defined by self.quantize_dims relative to the original FULL cache tensor.
            # Since 'cache' here is just a slice, we should normalize over ALL its dimensions.
            norm_dims = tuple(range(cache.dim()))
            target_stat_shape = (1,) * cache.dim() # Shape: (1, 1, ...) broadcastable
            masked_input = torch.where(outlier_mask, torch.nan, cache.float())

        # --- Calculate Statistics --- 
        # Use float() for masked mean/std/etc to avoid potential type issues with masked tensors
        #masked_cache = torch.masked.masked_tensor(cache, torch.logical_not(outlier_mask)) # Using nan instead

        num_valid = torch.sum(~torch.isnan(masked_input), dim=norm_dims, keepdim=True)
        # Ensure num_valid is at least 1 to avoid division by zero in mean/std calculation
        num_valid_clamped = torch.clamp(num_valid, min=1)
        
        if self.symmetric:
            mean_value = torch.zeros(target_stat_shape, dtype=self.dtype, device=self.device)

            if method == "minmax":
                 abs_masked = torch.abs(masked_input)
                 # nanmax equivalent: replace nan with -inf before max
                 abs_max = torch.where(torch.isnan(abs_masked), -torch.inf, abs_masked).max(dim=norm_dims, keepdim=True).values
                 # Handle case where all values in the block/group were NaN (abs_max is -inf)
                 abs_max = torch.where(torch.isinf(abs_max), torch.tensor(0.0, device=self.device, dtype=self.dtype), abs_max)
                 # Avoid division by zero for scale if n_bits=0 (shouldn't happen with checks, but safe)
                 denominator = torch.clamp(torch.tensor(2**n_bits, dtype=self.dtype, device=self.device), min=1)
                 scale_value = 2 * abs_max / denominator
            elif method == "std":
                 # Manual nanstd: var = nansum((x - nanmean)^2) / (N_valid - 1)
                 # For symmetric, assume mean is 0 for std calculation relative to 0
                 # variance = torch.nansum(masked_input**2, dim=norm_dims, keepdim=True) / torch.clamp(num_valid - 1, min=1)
                 # Let's use the standard deviation of the absolute values instead for symmetric scaling
                 abs_masked = torch.abs(masked_input)
                 mean_abs = torch.nansum(abs_masked, dim=norm_dims, keepdim=True) / num_valid_clamped
                 variance = torch.nansum((abs_masked - mean_abs)**2, dim=norm_dims, keepdim=True) / torch.clamp(num_valid - 1, min=1)
                 scale_value = torch.sqrt(variance)
                 # Alternative std approach: calculate std dev as if centered, use that as scale?
                 # mean_for_std = torch.nansum(masked_input, dim=norm_dims, keepdim=True) / num_valid_clamped
                 # variance = torch.nansum((masked_input - mean_for_std)**2, dim=norm_dims, keepdim=True) / torch.clamp(num_valid - 1, min=1)
                 # scale_value = torch.sqrt(variance)

        else: # Non-symmetric
            mean_value = torch.nansum(masked_input, dim=norm_dims, keepdim=True) / num_valid_clamped

            if method == "minmax":
                 # nanmax/nanmin equivalents
                 max_value = torch.where(torch.isnan(masked_input), -torch.inf, masked_input).max(dim=norm_dims, keepdim=True).values
                 min_value = torch.where(torch.isnan(masked_input), torch.inf, masked_input).min(dim=norm_dims, keepdim=True).values
                 max_value = torch.where(torch.isinf(max_value), torch.tensor(0.0, device=self.device, dtype=self.dtype), max_value)
                 min_value = torch.where(torch.isinf(min_value), torch.tensor(0.0, device=self.device, dtype=self.dtype), min_value)
                 denominator = torch.clamp(torch.tensor(2**n_bits, dtype=self.dtype, device=self.device), min=1)
                 scale_value = (max_value - min_value) / denominator
            elif method == "std":
                 # Variance calculation uses the mean_value already calculated
                 variance = torch.nansum((masked_input - mean_value)**2, dim=norm_dims, keepdim=True) / torch.clamp(num_valid - 1, min=1)
                 scale_value = torch.sqrt(variance)

        # Replace NaN scale/mean with defaults (e.g., 1.0 and 0.0) if all values in the group/block were outliers
        # This happens if num_valid was 0 initially.
        scale_value = torch.nan_to_num(scale_value.to(self.dtype), nan=1.0) # Use 1.0 to avoid division by zero later
        mean_value = torch.nan_to_num(mean_value.to(self.dtype), nan=0.0)

        # Ensure scale is not too small to prevent division by near-zero or large normalized values
        scale_value = torch.clamp(scale_value, min=1e-8)

        # --- Perform Normalization --- 
        # Apply normalization using original cache data type
        if self.use_grouping:
             # Use the grouped cache and broadcast mean/scale (shape: ..., num_groups, 1)
             normalized_cache_grouped = (cache_grouped.to(self.dtype) - mean_value) / scale_value
             # Reshape back to original slice shape
             normalized_cache = self._reshape_from_grouping(normalized_cache_grouped)
        else:
             # Broadcast mean/scale (shape: 1, 1, ...)
             normalized_cache = (cache.to(self.dtype) - mean_value) / scale_value

        # normalized_cache should have original_shape
        # mean_value, scale_value have target_stat_shape
        
        # Final check on shapes (optional)
        # assert normalized_cache.shape == original_shape, f"Normalized shape mismatch: {normalized_cache.shape} vs {original_shape}"
        # assert mean_value.shape == target_stat_shape, f"Mean shape mismatch: {mean_value.shape} vs {target_stat_shape}"
        # assert scale_value.shape == target_stat_shape, f"Scale shape mismatch: {scale_value.shape} vs {target_stat_shape}"

        return normalized_cache, mean_value, scale_value
        # MODIFICATION END

    def _denormalize(self, normalized_cache: torch.Tensor, mean_value: torch.Tensor, scale_value: torch.Tensor) -> torch.Tensor:
        # normalized_cache has the original slice shape (e.g., embed_size_per_head)
        # mean_value/scale_value have the shape calculated in _normalize:
        #   - Grouped: (..., num_groups, 1)
        #   - Not Grouped: (1, 1, ...)
        
        # MODIFICATION START: Handle potentially grouped mean/scale
        if self.use_grouping:
             # Ensure mean/scale have the expected grouped shape ending in 1
             # Their shape should match target_stat_shape from _normalize
             if not (mean_value.shape == scale_value.shape and mean_value.shape[-1] == 1):
                  raise ValueError(f"Grouped mean/scale shape mismatch or not ending in 1. Mean: {mean_value.shape}, Scale: {scale_value.shape}")

             # Reshape normalized_cache temporarily to match group structure for broadcasting
             # Input normalized_cache has shape e.g., (embed_size_per_head,)
             # Needs to be reshaped to (num_groups, group_size)
             normalized_cache_grouped = self._reshape_for_grouping(normalized_cache)
             
             # Check if grouped shapes are compatible for broadcasting
             # normalized_cache_grouped shape: (..., num_groups, group_size)
             # mean/scale shape:             (..., num_groups, 1)
             if normalized_cache_grouped.shape[:-1] != mean_value.shape[:-1]:
                  raise ValueError(
                      f"Grouped shape mismatch for denormalization. "
                      f"Normalized grouped base shape: {normalized_cache_grouped.shape[:-1]}, "
                      f"Mean/Scale base shape: {mean_value.shape[:-1]}"
                  )

             denormalized_grouped = normalized_cache_grouped * scale_value + mean_value
             # Reshape back to original slice shape
             return self._reshape_from_grouping(denormalized_grouped)
        else:
             # Standard denormalization (mean/scale are broadcastable)
             # Ensure mean/scale are broadcastable (e.g., shape (1,1,...))
             if not (mean_value.shape == scale_value.shape and all(d == 1 for d in mean_value.shape)):
                  # Allow for scalar case as well
                  if not (mean_value.numel() == 1 and scale_value.numel() == 1):
                    print(f"Warning: Non-grouped mean/scale shape might not broadcast correctly. Mean: {mean_value.shape}, Scale: {scale_value.shape}")
                    # Attempt broadcasting anyway

             return normalized_cache * scale_value + mean_value
        # MODIFICATION END

    def _uniform_quantize(self, cache: torch.Tensor, n_bits: int, outlier_mask: torch.Tensor) -> torch.Tensor:
        # cache/outlier_mask.shape: (n_count, n_layer, n_head, embed_size_per_head) or (n_count, n_head, embed_size_per_head), (n_count, embed_size_per_head)
        normalized_cache, mean_value, scale_value = self._normalize(cache, "minmax", n_bits, outlier_mask)
        quantized_cache = torch.clamp(torch.round(normalized_cache).to(torch.int32), -(2 ** (n_bits-1)), 2 ** (n_bits-1) - 1)
        dequantized_cache = quantized_cache.to(self.dtype)
        denormalized_cache = self._denormalize(dequantized_cache, mean_value, scale_value)
        # denormalized_cache.shape: (n_count, n_layer, n_head, embed_size_per_head) or (n_count, n_head, embed_size_per_head), (n_count, embed_size_per_head)
        return torch.where(outlier_mask, cache, denormalized_cache)

    def _normal_quantize(self, cache: torch.Tensor, n_bits: int, outlier_mask: torch.Tensor) -> torch.Tensor:
        # cache/outlier_mask.shape: (n_count, n_layer, n_head, embed_size_per_head) or (n_count, n_head, embed_size_per_head), (n_count, embed_size_per_head)
        normalized_cache, mean_value, scale_value = self._normalize(cache, "std", n_bits, outlier_mask)
        quantized_cache = torch.searchsorted(self.normal_quantiles_upper_bound[n_bits], normalized_cache.contiguous())
        dequantized_cache = self.normal_quantiles_center[n_bits][quantized_cache]
        denormalized_cache = self._denormalize(dequantized_cache, mean_value, scale_value)
        # denormalized_cache.shape: (n_count, n_layer, n_head, embed_size_per_head) or (n_count, n_head, embed_size_per_head), (n_count, embed_size_per_head)
        return torch.where(outlier_mask, cache, denormalized_cache)

    # MODIFICATION START: Implement adaptive quantization method
    def _adaptive_quantize(self, cache: torch.Tensor, n_bits: int, outlier_mask: torch.Tensor) -> torch.Tensor:
        # cache/outlier_mask are slices for a specific quantization unit

        # Step 1: Handle n_bits = 0
        if n_bits == 0:
            # Normalize to get mean value if needed (non-symmetric)
            _, mean_value, scale_value = self._normalize(cache, self.normalization_method_for_adaptive, 1, outlier_mask)
            if self.symmetric:
                 denormalized_cache = torch.zeros_like(cache)
            else:
                 zeros_normalized = torch.zeros_like(cache)
                 denormalized_cache = self._denormalize(zeros_normalized, mean_value, scale_value)
            # Restore outliers
            return torch.where(outlier_mask, cache, denormalized_cache)

        # Step 2: Normalize data (excluding outliers)
        # Use the normalization method specified for adaptive (e.g., 'std')
        # Pass n_bits=1? The exact scaling might not matter as much since we derive quantiles from the normalized data itself.
        # Let's pass n_bits, _normalize handles it.
        normalized_cache, mean_value, scale_value = self._normalize(
            cache, self.normalization_method_for_adaptive, n_bits, outlier_mask
        )
        
        # Create a NaN mask from the original outlier mask for quantile calculation
        nan_mask = outlier_mask # Where True, the value was an outlier

        # Step 3: Calculate Empirical Quantile Boundaries
        num_quantiles = 2 ** n_bits
        # Quantile points (e.g., for n_bits=2, num_quantiles=4, points=[0, 0.25, 0.5, 0.75, 1.0])
        quantile_points = torch.linspace(0, 1, num_quantiles + 1, device=self.device, dtype=self.dtype)

        boundaries = None # Initialize boundaries
        boundaries_list = [] # Use list for iterative building

        if self.use_grouping:
            # Calculate quantiles per group
            normalized_cache_grouped = self._reshape_for_grouping(normalized_cache)
            nan_mask_grouped = self._reshape_for_grouping(nan_mask)

            # Target shape for boundaries: (..., num_groups, num_quantiles + 1)
            target_boundaries_shape = normalized_cache_grouped.shape[:-1] + (num_quantiles + 1,)
            # boundaries = torch.zeros(target_boundaries_shape, dtype=self.dtype, device=self.device) # Use list instead

            # Iterate over groups (potential performance bottleneck)
            # Flatten leading dimensions for easier iteration
            num_total_groups = math.prod(normalized_cache_grouped.shape[:-1])
            flat_grouped_cache = normalized_cache_grouped.view(num_total_groups, self.group_size)
            flat_grouped_mask = nan_mask_grouped.view(num_total_groups, self.group_size)

            for i in range(num_total_groups):
                group_data = flat_grouped_cache[i]
                group_mask = flat_grouped_mask[i]
                # Select non-outlier (non-NaN) data for this group
                valid_data = group_data[~group_mask]

                if valid_data.numel() == 0:
                    # Handle empty group (all outliers or input was empty)
                    # Assign default boundaries (e.g., zeros or replicate neighbors later?)
                    # Let's use zeros for now.
                    group_boundaries = torch.zeros(num_quantiles + 1, device=self.device, dtype=self.dtype)
                else:
                    # Compute quantiles for this group's valid data
                    group_boundaries = torch.quantile(
                        valid_data.float(), # quantile expects float
                        quantile_points,
                        interpolation='linear'
                    )
                boundaries_list.append(group_boundaries)
            
            if boundaries_list: # Check if list is not empty
                boundaries = torch.stack(boundaries_list) # Shape: (num_total_groups, num_quantiles + 1)
                # Reshape back to target shape
                boundaries = boundaries.view(*target_boundaries_shape)
            else: # Handle case where input cache was empty
                boundaries = torch.zeros(target_boundaries_shape, dtype=self.dtype, device=self.device)

        else: # No grouping, calculate quantiles over the whole slice
            # Apply NaN mask
            masked_normalized_cache = torch.where(nan_mask, torch.nan, normalized_cache.float())
            # Select valid data
            valid_data = masked_normalized_cache[~torch.isnan(masked_normalized_cache)]

            if valid_data.numel() == 0:
                # Handle empty slice
                boundaries = torch.zeros(num_quantiles + 1, device=self.device, dtype=self.dtype)
            else:
                boundaries = torch.quantile(
                    valid_data,
                    quantile_points,
                    interpolation='linear'
                )
            # Reshape boundaries for broadcasting consistency later (e.g., (1, 1, ..., num_quantiles + 1))
            boundaries = boundaries.view((1,) * normalized_cache.dim() + (num_quantiles + 1,))

        # Step 4: Calculate Centroids (Midpoints of boundaries)
        centroids = (boundaries[..., :-1] + boundaries[..., 1:]) / 2.0
        # Handle cases where boundaries are equal (results in NaN centroid) - replace with boundary value
        centroids = torch.where(torch.isnan(centroids), boundaries[..., :-1], centroids)
        # Centroids shape: (..., num_groups, num_quantiles) or (1, ..., 1, num_quantiles)

        # Step 5: Quantize (Find bin index for each value)
        # Use searchsorted on the upper boundaries (excluding the lowest boundary, effectively -inf)
        upper_boundaries = boundaries[..., 1:].contiguous() # Shape (..., num_q+1) -> (..., num_q)

        # Apply searchsorted. Needs careful broadcasting if grouping.
        if self.use_grouping:
            normalized_cache_grouped = self._reshape_for_grouping(normalized_cache).contiguous()
            # Output indices should match normalized_cache_grouped shape
            # quantized_indices_grouped = torch.zeros_like(normalized_cache_grouped, dtype=torch.int64) # Use list
            indices_list = []
            
            # Iterate again (vectorization tricky here)
            num_total_groups = math.prod(normalized_cache_grouped.shape[:-1])
            flat_grouped_cache = normalized_cache_grouped.view(num_total_groups, self.group_size)
            # Ensure upper_boundaries is also flattened consistently
            flat_upper_boundaries = upper_boundaries.view(num_total_groups, num_quantiles)

            for i in range(num_total_groups):
                group_boundaries = flat_upper_boundaries[i] # Shape (num_quantiles,)
                group_values = flat_grouped_cache[i]       # Shape (group_size,)
                # Find indices for this group
                group_indices = torch.searchsorted(group_boundaries, group_values, right=True)
                indices_list.append(group_indices)

            if indices_list:
                 quantized_indices_grouped = torch.stack(indices_list).view(*normalized_cache_grouped.shape)
            else: # Handle empty cache case
                 quantized_indices_grouped = torch.zeros_like(normalized_cache_grouped, dtype=torch.int64)
            # Clamp indices
            quantized_indices_grouped = torch.clamp(quantized_indices_grouped, 0, num_quantiles - 1)

        else: # No grouping
            # Use broadcastable searchsorted (PyTorch >= 1.10)
            # Ensure upper_boundaries has a compatible shape for broadcasting with normalized_cache
            # upper_boundaries: (1, ..., 1, num_quantiles)
            # normalized_cache: (...) original slice shape
            search_boundaries = upper_boundaries # Shape (1, ..., 1, num_quantiles)
            search_values = normalized_cache.unsqueeze(-1) # Add dim for comparison: (..., 1)
            
            # Perform broadcasted comparison (searchsorted handles this)
            # Need boundaries on the dim being searched. Let's ensure boundary dim is last.
            quantized_indices = torch.searchsorted(search_boundaries, search_values, right=True)
            quantized_indices = quantized_indices.squeeze(-1) # Remove added dimension
            # Clamp indices
            quantized_indices = torch.clamp(quantized_indices, 0, num_quantiles - 1)

        # Step 6: Dequantize (Gather Centroids based on indices)
        # Need indices shape to align with centroids shape for gather
        # Centroids shape: (..., num_groups, num_quantiles) or (1, ..., 1, num_quantiles)
        # Indices shape: (..., num_groups, group_size) or (...)
        
        if self.use_grouping:
             # Use iteration again for gather
             dequantized_grouped_list = []
             flat_indices = quantized_indices_grouped.view(num_total_groups, self.group_size)
             # Ensure centroids is also flattened consistently
             flat_centroids = centroids.view(num_total_groups, num_quantiles)

             for i in range(num_total_groups):
                  group_centroids = flat_centroids[i] # Shape (num_quantiles,)
                  group_indices = flat_indices[i]    # Shape (group_size,)
                  # Gather along the quantile dimension (0 for 1D centroids)
                  dequantized_group = torch.gather(group_centroids, 0, group_indices)
                  dequantized_grouped_list.append(dequantized_group)

             if dequantized_grouped_list:
                  dequantized_grouped = torch.stack(dequantized_grouped_list) # Shape (num_total_groups, group_size)
                  # Reshape back to (..., num_groups, group_size)
                  dequantized_grouped = dequantized_grouped.view(*quantized_indices_grouped.shape) 
                  # Reshape back to original slice shape
                  dequantized_cache = self._reshape_from_grouping(dequantized_grouped)
             else: # Handle empty cache
                  dequantized_cache = torch.zeros_like(normalized_cache)

        else: # No grouping
             # Centroids shape (1,...,1, N), indices shape (...)
             # We need to gather along the last dimension of centroids using indices
             target_gather_shape = quantized_indices.shape + (num_quantiles,)
             broadcasted_centroids = centroids.expand(target_gather_shape)
             quantized_indices_unsqueezed = quantized_indices.unsqueeze(-1) # Add dim for index
             
             dequantized_cache = torch.gather(broadcasted_centroids, -1, quantized_indices_unsqueezed)
             dequantized_cache = dequantized_cache.squeeze(-1) # Remove added dimension

        # Ensure dequantized cache has the same dtype as normalization outputs
        dequantized_cache = dequantized_cache.to(self.dtype)

        # Step 7: Denormalize
        denormalized_cache = self._denormalize(dequantized_cache, mean_value, scale_value)

        # Step 8: Restore outliers
        final_cache = torch.where(outlier_mask, cache, denormalized_cache)

        # Ensure final output shape matches input slice shape
        assert final_cache.shape == cache.shape, f"Final cache shape mismatch in adaptive: {final_cache.shape} vs {cache.shape}"

        return final_cache
    # MODIFICATION END
        
    # Returns (quantized kvcache, average n_bits)
        if self.level != "no-quantization":
             default_param_bits = float(torch.finfo(self.dtype).bits) # Bits for storing scale/mean (e.g., 16 for float16)
             # Number of parameters (scale, mean/zero-point) per quantization block
             n_params_per_block = 1.0 if self.symmetric else 2.0 # Scale + Mean/Zero-point
             
             num_quant_blocks = 0.0 # Total number of blocks requiring separate parameters per token

             # Determine the number of base quantization blocks based on the level
             if self.level == "token":
                 num_base_blocks = 1.0
             elif self.level == "layer":
                 num_base_blocks = float(num_layers)
             elif self.level == "head":
                 num_base_blocks = float(num_layers * num_heads)
             else: # Should not happen due to Literal type check
                 num_base_blocks = 0.0 

             # If grouping is enabled, multiply base blocks by the number of feature groups within each block
             if self.use_grouping:
                  # Ensure group_size is valid and compatible
                  if self.group_size <= 0:
                      raise ValueError("group_size must be positive when use_grouping is True.")
                  if embed_size_per_head % self.group_size != 0:
                       raise ValueError(
                           f"Embed size per head ({embed_size_per_head}) not divisible by group_size ({self.group_size}) "
                           f"in size calculation (Level: {self.level})"
                       )
                  num_feature_groups_per_embed = embed_size_per_head // self.group_size
                  
                  # Adjust the number of blocks based on the level and feature groups
                  if self.level == "token":
                      # For token level, the "base block" covers all layers/heads.
                      # Grouping applies within each head's embedding dim across all heads/layers.
                      num_quant_blocks = num_base_blocks * num_layers * num_heads * num_feature_groups_per_embed
                  elif self.level == "layer":
                      # For layer level, each base block is a layer.
                      # Grouping applies within each head's embedding dim across all heads in that layer.
                      num_quant_blocks = num_base_blocks * num_heads * num_feature_groups_per_embed
                  elif self.level == "head":
                      # For head level, each base block is a head.
                      # Grouping applies within that head's embedding dimension.
                      num_quant_blocks = num_base_blocks * num_feature_groups_per_embed
             else:
                  # No grouping, the number of blocks is just the number of base blocks
                  num_quant_blocks = num_base_blocks

             overhead_bits = num_quant_blocks * n_params_per_block * default_param_bits

             # TODO: Add overhead for adaptive quantization parameters if centroids/boundaries are stored explicitly.
             # This depends heavily on the specific implementation of _adaptive_quantize.
             # Assuming for now adaptive method overhead is similar to storing scale/mean per group.

        # Total size is data bits + overhead bits
        total_bits = data_bits + overhead_bits
        return total_bits

# MODIFICATION START: Add the missing build_quantizers function
def build_quantizers(config_grid_list: list[dict[str, list]]) -> list[Quantizer]:
    quantizer_list: list[Quantizer] = []
    for config_grid in config_grid_list:
        # Get keys and value lists from the config grid
        keys = list(config_grid.keys())
        value_lists = list(config_grid.values())

        # Create all combinations of parameter values
        for args in product(*value_lists):
            kwargs = {k: v for k, v in zip(keys, args)}
            # Check for potential conflicts or dependencies if needed
            # e.g., n_bits_uniform only relevant if use_attentions=False
            # group_size requires level != token etc. (already checked in init)
            try:
                 # Instantiate the quantizer
                 quantizer = Quantizer(**kwargs)
                 # Check for method='adaptive' and ensure _adaptive_quantize is implemented
                 # if quantizer.method_name == 'adaptive' and not callable(getattr(quantizer, '_adaptive_quantize', None)):
                 #      print(f"Warning: Skipping config with method='adaptive' as implementation is missing: {kwargs}")
                 #      continue # Skip if adaptive method selected but not implemented

                 # Only add if instantiation succeeds and necessary checks pass
                 quantizer_list.append(quantizer)

            except AssertionError as e:
                 # Optionally print skipped invalid combinations due to assert checks
                 # print(f"Skipping invalid config due to AssertionError: {kwargs} - {e}")
                 pass
            except ValueError as e:
                 # Catch ValueErrors from parameter validation (e.g., group_size divisibility)
                 # print(f"Skipping invalid config due to ValueError: {kwargs} - {e}")
                 pass
            except Exception as e:
                 # Catch other unexpected errors during instantiation
                 print(f"Error creating quantizer with config: {kwargs} - {type(e).__name__}: {e}")
                 # Decide whether to skip or raise, currently skipping

    return quantizer_list
# MODIFICATION END


Okay, let's start modifying the code in `QAQ-KVCacheQuantization/src/quantizer.py` to implement both **feature grouping** (using `group_size`) and the **adaptive distribution quantization** method.

I will add comments within `''' '''` blocks to highlight the changes and explain the logic.

```python
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
''' MODIFICATION START: Added "adaptive" to quantization methods '''
QuantizationMethods = Literal["uniform", "normal", "adaptive"]
''' MODIFICATION END '''


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
                 ''' MODIFICATION START: Added group_size parameter '''
                 # (Optional) Size of feature groups within the embedding dimension for grouped quantization.
                 # If None or <= 0, grouping is disabled.
                 group_size: Optional[int] = None
                 ''' MODIFICATION END '''
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

        ''' MODIFICATION START: Store group_size and validate '''
        self.group_size = group_size if group_size is not None and group_size > 0 else -1
        self.use_grouping = self.group_size > 0
        if self.use_grouping:
            # Grouping currently makes most sense conceptually with head-level or maybe layer-level.
            # For token-level, the entire embedding dimension is already treated as one block.
            # Let's allow it for layer/head for now.
            assert self.level in ["layer", "head"], "Grouping is typically used with layer or head level quantization."
        ''' MODIFICATION END '''

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
        ''' MODIFICATION START: Added adaptive quantization method '''
        elif method == "adaptive":
            self.quantization_method = self._adaptive_quantize
            # Adaptive method typically uses std-dev based normalization
            self.normalization_method_for_adaptive = "std"
        ''' MODIFICATION END '''

    def set_dtype_and_device(self, dtype: torch.dtype, device: torch.device):
        self.dtype = dtype
        self.device = device
        if self.level != "no-quantization":
             # Precompute normal quantiles if normal method is used OR if adaptive method uses std normalization (often does)
            if self.method_name == "normal" or (self.method_name == "adaptive" and self.normalization_method_for_adaptive == "std"):
                 if self.use_attentions:
                     n_bits_range = range(self.n_bits_min, self.n_bits_max+1)
                 else:
                     n_bits_range = range(self.n_bits_uniform, self.n_bits_uniform+1)

                 ''' MODIFICATION START: Check if range is valid before proceeding '''
                 if n_bits_range: # Ensure the range is not empty
                     self.normal_quantiles_upper_bound = {
                         n: torch.tensor(norm.ppf(np.arange(0, 1, 1/(2**n)) + 1/(2**n)), dtype=dtype, device=device)
                         for n in n_bits_range if n > 0 # Avoid 2**0 = 1 quantile
                     }
                     self.normal_quantiles_center = {
                         n: torch.tensor(norm.ppf(np.arange(0, 1, 1/(2**n)) + 0.5/(2**n)), dtype=dtype, device=device)
                         for n in n_bits_range if n > 0
                     }
                 else:
                     # Handle case where no valid n_bits are specified (e.g., n_bits_uniform=0)
                     self.normal_quantiles_upper_bound = {}
                     self.normal_quantiles_center = {}
                 ''' MODIFICATION END '''


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
        ''' MODIFICATION START: Added group_size to params '''
        res["group_size"] = self.group_size if self.use_grouping else "disabled"
        ''' MODIFICATION END '''
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

    ''' MODIFICATION START: Helper function to reshape for grouping '''
    def _reshape_for_grouping(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.use_grouping:
            return tensor
        # tensor shape: (..., embed_size_per_head)
        shape = tensor.shape
        embed_size_per_head = shape[-1]
        assert embed_size_per_head % self.group_size == 0, \
            f"Embedding size ({embed_size_per_head}) must be divisible by group_size ({self.group_size})"
        num_groups = embed_size_per_head // self.group_size
        # Reshape to (..., num_groups, group_size)
        return tensor.view(*shape[:-1], num_groups, self.group_size)

    def _reshape_from_grouping(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.use_grouping or len(tensor.shape) < 2: # Check if tensor has group dims
            return tensor
        # tensor shape: (..., num_groups, group_size)
        shape = tensor.shape
        # Reshape back to (..., embed_size_per_head)
        return tensor.reshape(*shape[:-2], -1) # Merges the last two dimensions
    ''' MODIFICATION END '''


    def _calc_quantization_bits(self, attentions: AttentionType, cache: torch.Tensor, outlier_mask: torch.Tensor) -> torch.Tensor:
        # cache/outlier_mask.shape: (n_batch, seq_len, n_layer, n_head, embed_size_per_head)
        n_batch, seq_len, n_layer, n_head, embed_size_per_head = cache.shape

        ''' MODIFICATION START: Validate group_size if grouping is enabled '''
        if self.use_grouping:
             assert embed_size_per_head % self.group_size == 0, \
                 f"Embedding size ({embed_size_per_head}) must be divisible by group_size ({self.group_size})"
        ''' MODIFICATION END '''

        if not self.use_attentions:
            if self.level == "token":
                shape = (n_batch, seq_len)
            elif self.level == "layer":
                shape = (n_batch, seq_len, n_layer)
            elif self.level == "head":
                shape = (n_batch, seq_len, n_layer, n_head)
            return torch.ones(shape, dtype=torch.int64, device=self.device) * self.n_bits_uniform

        # --- Attention-aware bit calculation ---
        # NOTE: Current implementation calculates bits per token/layer/head.
        # It does NOT yet calculate bits per *group*. Implementing per-group
        # adaptive bits would require significant changes here and in how
        # `n_bits` is used later. For now, we calculate bits at the specified
        # `level` and apply the *same* n_bits to all groups within that level's unit.

        if self.key_or_value_cache == "key":
            # This formula estimates error based on norm, assuming uniform quantization error.
            # Adapting it perfectly for grouped quantization or other methods might need theoretical work.
            # We use the existing formula as an approximation for now.
            max_error = math.sqrt(12.0 / self.q_norm * math.log(seq_len**3/(seq_len-1) * self.target_quantization_error**2 + 1))
            # max_error shape: scalar
        elif self.key_or_value_cache == "value":
            attentions_tensor = torch.stack(attentions)
            # attentions.shape: (n_layer, n_batch, n_head, seq_len, seq_len)
            attentions_tensor = attentions_tensor[:, :, :, -self.last_n_attentions:, :]
            # attentions.shape: (n_layer, n_batch, n_head, last_n_attentions, seq_len)
            attentions_tensor = attentions_tensor.permute(1, 4, 0, 2, 3)
            # attentions.shape: (n_batch, seq_len, n_layer, n_head, last_n_attentions)

            # Reduce attentions based on the quantization level to get max attention weight per unit
            reduce_dims = [d + len(attentions_tensor.shape) for d in self.quantize_dims] # Adjust dims based on current shape
            # We take the max attention value across the dimensions that will be quantized together
            # Example: level=head, quantize_dims=(-1,) -> reduce over last_n_attentions dim
            # Example: level=layer, quantize_dims=(-2, -1) -> reduce over n_head, last_n_attentions
            max_attention_weights = attentions_tensor.amax(dim=reduce_dims)
            # max_attention_weights.shape: (n_batch, seq_len) or (n_batch, seq_len, n_layer) or (n_batch, seq_len, n_layer, n_head)

            # Formula estimates max allowed error based on attention weights.
            # Again, using the existing formula as approximation.
            max_error = math.sqrt(12.0 / seq_len) * self.target_quantization_error / max_attention_weights
            # max_error.shape: (n_batch, seq_len) or (n_batch, seq_len, n_layer) or (n_batch, seq_len, n_layer, n_head)

        # Calculate n_bits based on the max_error tolerance
        masked_cache = torch.masked.masked_tensor(cache, torch.logical_not(outlier_mask))
        # Adjust quantize_dims relative to cache's current shape
        cache_quantize_dims = [d + len(masked_cache.shape) for d in self.quantize_dims]

        # Estimate required bits assuming uniform quantization for simplicity in bit calculation
        # This calculation might not be perfectly accurate for 'normal' or 'adaptive' methods,
        # but serves as a heuristic for dynamic bit allocation.
        if self.symmetric:
            abs_max_val = masked_cache.abs().amax(dim=cache_quantize_dims)
            scale_val = 2 * abs_max_val
        else:
            max_val = masked_cache.amax(dim=cache_quantize_dims)
            min_val = masked_cache.amin(dim=cache_quantize_dims)
            scale_val = max_val - min_val

        # Ensure scale_val is valid (no NaNs/Infs from empty masks, etc.)
        # If a whole quantization block was masked as outlier, scale_val might be NaN.
        # Handle potential division by zero or log(<=0) if max_error is non-positive or scale_val is zero
        valid_scale = scale_val.get_data() > 1e-9 # Use a small epsilon
        valid_error = (max_error > 1e-9) if isinstance(max_error, torch.Tensor) else (max_error > 1e-9)
        valid_mask = valid_scale & valid_error & scale_val.get_mask()

        # Calculate n_bits only where valid
        n_bits = torch.zeros_like(scale_val.get_data(), dtype=self.dtype)
        scale_value_data = scale_val.get_data()[valid_mask]
        max_error_data = max_error[valid_mask] if isinstance(max_error, torch.Tensor) else max_error

        # log2(scale / (2 * error) + 1) -> bits needed for uniform quantization steps
        # Adding 1 inside log2 ensures positive result even if scale <= 2*error
        n_bits[valid_mask] = torch.log2(scale_value_data / (2 * max_error_data) + 1.0)

        # Clamp bits and handle invalid cases (assign max bits)
        n_bits = torch.clamp(torch.ceil(n_bits).to(torch.int64), self.n_bits_min, self.n_bits_max)
        n_bits[~valid_mask] = self.n_bits_max # Assign max bits if scale or error was invalid

        # n_bits.shape: (n_batch, seq_len) or (n_batch, seq_len, n_layer) or (n_batch, seq_len, n_layer, n_head)

        # The last tokens might lack sufficient history for stable attention scores
        if self.last_n_attentions > 1 and seq_len >= self.last_n_attentions:
             if n_bits.dim() == 2: # (n_batch, seq_len)
                 n_bits[:, -self.last_n_attentions+1:] = self.n_bits_max
             elif n_bits.dim() == 3: # (n_batch, seq_len, n_layer)
                 n_bits[:, -self.last_n_attentions+1:, :] = self.n_bits_max
             elif n_bits.dim() == 4: # (n_batch, seq_len, n_layer, n_head)
                 n_bits[:, -self.last_n_attentions+1:, :, :] = self.n_bits_max

        return n_bits


    def _calc_outlier_mask(self, cache: torch.Tensor) -> torch.Tensor:
        if self.outliers_ratio == 0.0:
            return torch.zeros_like(cache, dtype=torch.bool, device=self.device)

        # cache.shape: (n_batch, seq_len, n_layer, n_head, embed_size_per_head)
        original_shape = cache.shape

        ''' MODIFICATION START: Calculate outliers per group if grouping enabled '''
        if self.use_grouping:
            # Reshape for grouping: (..., embed_size_per_head) -> (..., num_groups, group_size)
            cache_grouped = self._reshape_for_grouping(cache)
            # Calculate thresholds along the group_size dimension (-1)
            group_dim_size = cache_grouped.shape[-1]
            k_lower = int(math.ceil(self.outliers_ratio / 2 * group_dim_size))
            k_upper = int(math.floor((1.0 - self.outliers_ratio / 2) * group_dim_size))

            # Handle edge case where k is 0 or > size
            if k_lower <= 0 or k_upper > group_dim_size:
                 return torch.zeros_like(cache, dtype=torch.bool, device=self.device)
            if k_lower > group_dim_size: k_lower = group_dim_size # Clamp k if needed
            if k_upper <= 0: k_upper = 1 # Clamp k if needed


            lower_threshold = torch.kthvalue(cache_grouped, k=k_lower, dim=-1).values # Shape: (..., num_groups)
            upper_threshold = torch.kthvalue(cache_grouped, k=k_upper, dim=-1).values # Shape: (..., num_groups)

            # Unsqueeze to allow broadcasting against the grouped cache
            lower_threshold = lower_threshold.unsqueeze(-1) # Shape: (..., num_groups, 1)
            upper_threshold = upper_threshold.unsqueeze(-1) # Shape: (..., num_groups, 1)

            # Calculate mask based on grouped thresholds
            mask_grouped = (cache_grouped <= lower_threshold) | (cache_grouped > upper_threshold)
            # mask_grouped shape: (..., num_groups, group_size)

            # Reshape mask back to original cache shape
            mask = self._reshape_from_grouping(mask_grouped)

        else:
            # Original logic: calculate outliers over the entire quantization block
            cache_flat = cache.flatten(start_dim=-len(self.quantize_dims))
            # cache_flat.shape: (n_batch, seq_len, n_layer*n_head*embed_size_per_head) etc.
            flat_dim_size = cache_flat.shape[-1]
            k_lower = int(math.ceil(self.outliers_ratio / 2 * flat_dim_size))
            k_upper = int(math.floor((1.0 - self.outliers_ratio / 2) * flat_dim_size))

            if k_lower <= 0 or k_upper > flat_dim_size:
                 return torch.zeros_like(cache, dtype=torch.bool, device=self.device)
            if k_lower > flat_dim_size: k_lower = flat_dim_size
            if k_upper <= 0: k_upper = 1

            lower_threshold = torch.kthvalue(cache_flat, k=k_lower, dim=-1).values
            upper_threshold = torch.kthvalue(cache_flat, k=k_upper, dim=-1).values
            # lower/upper_threshold.shape: (n_batch, seq_len) or (..., n_layer) or (..., n_head)

            # Expand thresholds to match cache shape for comparison
            view_shape = list(lower_threshold.shape) + [1] * len(self.quantize_dims)
            lower_threshold = lower_threshold.view(*view_shape)
            upper_threshold = upper_threshold.view(*view_shape)
            # lower/upper_threshold shape matches cache for broadcasting

            mask = (cache <= lower_threshold) | (cache > upper_threshold)
            # mask.shape = (n_batch, seq_len, n_layer, n_head, embed_size_per_head)
        ''' MODIFICATION END '''

        return mask

    # Returns (normalized cache, mean value, scale value)
    # Mean/scale have shapes compatible for denormalization (e.g., keep group dim if grouped)
    def _normalize(self, cache: torch.Tensor, method: Literal["minmax", "std"], n_bits: int, outlier_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ''' MODIFICATION START: Handle grouping within normalization '''
        original_shape = cache.shape
        if self.use_grouping:
            cache = self._reshape_for_grouping(cache)
            outlier_mask = self._reshape_for_grouping(outlier_mask)
            # Normalization dimension is now the group size dimension (-1)
            norm_dims = (-1,)
        else:
            # Original dimensions relative to the full tensor passed in
            # Need to figure out how dims relate to the potentially sliced input tensor
            # Assuming input 'cache' here is a slice corresponding to one quantization unit
            # If level='head', cache shape might be (embed_size_per_head), dims=(-1,)
            # If level='layer', cache shape (n_head, embed_size_per_head), dims=(-2,-1)
            # If level='token', cache shape (n_layer, n_head, embed_size_per_head), dims=(-3,-2,-1)
            # Let's assume the input `cache` here is already appropriately sliced/shaped
            # such that the normalization should happen over ALL its dimensions.
            norm_dims = tuple(range(cache.dim())) # Normalize over all dims of the input slice

        # Apply outlier mask
        # Use float() for masked mean/std/etc to avoid potential type issues with masked tensors
        masked_cache = torch.where(outlier_mask, torch.nan, cache.float())

        if self.symmetric:
            mean_value = torch.zeros((1,)*cache.dim(), dtype=self.dtype, device=self.device)
            # Make mean broadcastable if grouping
            if self.use_grouping:
                 # Shape (1, ..., 1, 1) -> needs to match (..., num_groups, 1) for scale
                 mean_value = torch.zeros(*cache.shape[:-1], 1, dtype=self.dtype, device=self.device)

            if method == "minmax":
                # abs_max = masked_cache.abs().nanmax(dim=norm_dims, keepdim=True)
                 # nanmax equivalent:
                abs_max = torch.where(torch.isnan(masked_cache), -torch.inf, masked_cache.abs()).max(dim=norm_dims, keepdim=True).values
                abs_max = torch.where(torch.isinf(abs_max), torch.tensor(0.0, device=self.device), abs_max) # Handle case where all were NaN
                scale_value = 2 * abs_max / (2 ** n_bits)
            elif method == "std":
                # scale_value = masked_cache.nanstd(dim=norm_dims, keepdim=True) # nanstd not directly available
                # Manual nanstd:
                num_valid = torch.sum(~torch.isnan(masked_cache), dim=norm_dims, keepdim=True)
                mean = torch.nansum(masked_cache, dim=norm_dims, keepdim=True) / torch.clamp(num_valid, min=1)
                variance = torch.nansum((masked_cache - mean)**2, dim=norm_dims, keepdim=True) / torch.clamp(num_valid - 1, min=1)
                scale_value = torch.sqrt(variance)

        else: # Non-symmetric
            num_valid = torch.sum(~torch.isnan(masked_cache), dim=norm_dims, keepdim=True)
            mean_value = torch.nansum(masked_cache, dim=norm_dims, keepdim=True) / torch.clamp(num_valid, min=1)

            if method == "minmax":
                # max_value = masked_cache.nanmax(dim=norm_dims, keepdim=True)
                # min_value = masked_cache.nanmin(dim=norm_dims, keepdim=True)
                max_value = torch.where(torch.isnan(masked_cache), -torch.inf, masked_cache).max(dim=norm_dims, keepdim=True).values
                min_value = torch.where(torch.isnan(masked_cache), torch.inf, masked_cache).min(dim=norm_dims, keepdim=True).values
                max_value = torch.where(torch.isinf(max_value), torch.tensor(0.0, device=self.device), max_value)
                min_value = torch.where(torch.isinf(min_value), torch.tensor(0.0, device=self.device), min_value)

                scale_value = (max_value - min_value) / torch.clamp(torch.tensor(2 ** n_bits), min=1) # Avoid division by zero if n_bits=0
            elif method == "std":
                 # Manual nanstd:
                 variance = torch.nansum((masked_cache - mean_value)**2, dim=norm_dims, keepdim=True) / torch.clamp(num_valid - 1, min=1)
                 scale_value = torch.sqrt(variance)


        # Replace NaN scale/mean with defaults (e.g., 1.0 and 0.0) if all values were outliers
        scale_value = torch.nan_to_num(scale_value.to(self.dtype), nan=1.0) # Use 1.0 to avoid division by zero
        mean_value = torch.nan_to_num(mean_value.to(self.dtype), nan=0.0)

        # Ensure scale is not too small
        scale_value = torch.clamp(scale_value, min=1e-8)

        # Perform normalization using original cache data (not the float version used for stats)
        normalized_cache = (cache - mean_value) / scale_value

        # Reshape normalized_cache back if grouping was used, keep mean/scale grouped
        if self.use_grouping:
            normalized_cache = self._reshape_from_grouping(normalized_cache)

        # Final check on shapes
        # normalized_cache should have original_shape
        # mean_value, scale_value should have shape compatible for broadcasting in _denormalize
        # e.g., if grouped: (..., num_groups, 1), if not grouped: (..., 1, 1, 1) matching reduced dims

        return normalized_cache, mean_value, scale_value
        ''' MODIFICATION END '''


    def _denormalize(self, normalized_cache: torch.Tensor, mean_value: torch.Tensor, scale_value: torch.Tensor) -> torch.Tensor:
        ''' MODIFICATION START: Handle potentially grouped mean/scale '''
        # Assume normalized_cache has the original shape (ungrouped)
        # mean_value/scale_value might have shape (..., num_groups, 1) or broadcastable shape
        if self.use_grouping and mean_value.shape != scale_value.shape:
             # This shouldn't happen if _normalize is correct, but adding safety check
             raise ValueError("Grouped mean and scale shapes mismatch in denormalize")

        if self.use_grouping and mean_value.dim() > normalized_cache.dim():
             # Need to reshape normalized_cache temporarily for broadcasted denormalization
             grouped_norm_cache = self._reshape_for_grouping(normalized_cache)
             denormalized_grouped = grouped_norm_cache * scale_value + mean_value
             return self._reshape_from_grouping(denormalized_grouped)
        else:
             # Standard denormalization (works if mean/scale are broadcastable without grouping dim)
             return normalized_cache * scale_value + mean_value
        ''' MODIFICATION END '''


    def _uniform_quantize(self, cache: torch.Tensor, n_bits: int, outlier_mask: torch.Tensor) -> torch.Tensor:
        ''' MODIFICATION START: Uniform quant using potentially grouped normalize/denormalize '''
        if n_bits == 0: # Special case: zero bits means set to zero (or mean if not symmetric)
             _, mean_value, _ = self._normalize(cache, "minmax", 1, outlier_mask) # Need mean if not symmetric
             denormalized_cache = torch.zeros_like(cache) if self.symmetric else mean_value.expand_as(cache)
        elif n_bits > 0:
             normalized_cache, mean_value, scale_value = self._normalize(cache, "minmax", n_bits, outlier_mask)

             # Determine quantization range based on symmetry
             if self.symmetric:
                 q_min = -(2 ** (n_bits - 1))
                 q_max = 2 ** (n_bits - 1) - 1
             else:
                 # Asymmetric: Map range [min_val, max_val] to [0, 2**n_bits - 1]
                 # Normalization maps to approx [0, 2**n_bits] if mean is min_value? No, depends on scale def.
                 # Let's assume _normalize with minmax maps non-outliers to roughly centered range scaled by 2**n_bits.
                 # For now, stick to symmetric range for simplicity, can refine asymmetric later.
                 # TODO: Revisit asymmetric uniform range if needed. Using symmetric range for now.
                 q_min = -(2 ** (n_bits - 1))
                 q_max = 2 ** (n_bits - 1) - 1
                 # Alternative: q_min = 0, q_max = 2**n_bits - 1. Requires careful normalization alignment.


             quantized_cache = torch.clamp(torch.round(normalized_cache).to(torch.int32), q_min, q_max)
             dequantized_cache = quantized_cache.to(self.dtype) # Convert back to float for denormalization
             denormalized_cache = self._denormalize(dequantized_cache, mean_value, scale_value)
        else:
             raise ValueError(f"Unsupported n_bits: {n_bits}")

        # Restore outliers
        return torch.where(outlier_mask, cache, denormalized_cache)
        ''' MODIFICATION END '''


    def _normal_quantize(self, cache: torch.Tensor, n_bits: int, outlier_mask: torch.Tensor) -> torch.Tensor:
        ''' MODIFICATION START: Normal quant using potentially grouped normalize/denormalize '''
        if n_bits == 0:
             _, mean_value, _ = self._normalize(cache, "std", 1, outlier_mask)
             denormalized_cache = torch.zeros_like(cache) if self.symmetric else mean_value.expand_as(cache)
        elif n_bits > 0:
             # Ensure precomputed quantiles exist for this n_bits
             if n_bits not in self.normal_quantiles_upper_bound or n_bits not in self.normal_quantiles_center:
                  raise ValueError(f"Precomputed normal quantiles not available for n_bits={n_bits}. Check set_dtype_and_device.")

             normalized_cache, mean_value, scale_value = self._normalize(cache, "std", n_bits, outlier_mask)

             # Quantize based on precomputed standard normal quantiles
             # Reshape normalized cache for group processing if needed
             if self.use_grouping:
                 normalized_cache_grouped = self._reshape_for_grouping(normalized_cache)
             else:
                 normalized_cache_grouped = normalized_cache # No grouping needed

             quantized_indices = torch.searchsorted(self.normal_quantiles_upper_bound[n_bits], normalized_cache_grouped.contiguous())
             # Dequantize using the center points of the normal bins
             dequantized_grouped = self.normal_quantiles_center[n_bits][quantized_indices]

             # Reshape back if grouping was used
             if self.use_grouping:
                 dequantized_cache = self._reshape_from_grouping(dequantized_grouped)
             else:
                 dequantized_cache = dequantized_grouped

             denormalized_cache = self._denormalize(dequantized_cache, mean_value, scale_value)
        else:
             raise ValueError(f"Unsupported n_bits: {n_bits}")


        # Restore outliers
        return torch.where(outlier_mask, cache, denormalized_cache)
        ''' MODIFICATION END '''


    ''' MODIFICATION START: Added adaptive quantization method '''
    def _adaptive_quantize(self, cache: torch.Tensor, n_bits: int, outlier_mask: torch.Tensor) -> torch.Tensor:
        if n_bits == 0:
            _, mean_value, _ = self._normalize(cache, self.normalization_method_for_adaptive, 1, outlier_mask)
            denormalized_cache = torch.zeros_like(cache) if self.symmetric else mean_value.expand_as(cache)
        elif n_bits > 0:
            # Step 1: Normalize (using configured method, likely 'std')
            # Note: n_bits in _normalize might affect scaling, but adaptive re-quantiles anyway. Pass n_bits=1? Or n_bits? Let's pass n_bits.
            normalized_cache, mean_value, scale_value = self._normalize(cache, self.normalization_method_for_adaptive, n_bits, outlier_mask)

            # Step 2: Grouping and Empirical Quantiles
            if self.use_grouping:
                normalized_cache_grouped = self._reshape_for_grouping(normalized_cache)
                target_tensor = normalized_cache_grouped
                quantile_dim = -1 # Calculate quantiles over group_size dimension
            else:
                target_tensor = normalized_cache # Calculate quantiles over the whole block
                quantile_dim = tuple(range(target_tensor.dim())) # Quantiles over all dims


            num_quantiles = 2 ** n_bits
            # Create quantile points (0.0, 1/N, 2/N, ..., 1.0)
            quantile_values = torch.linspace(0, 1, num_quantiles + 1, device=self.device, dtype=self.dtype)

            # Calculate empirical quantile boundaries from the data
            # Need to handle potential NaNs from outliers masked during normalization
            non_nan_mask = ~torch.isnan(target_tensor)
            if not torch.any(non_nan_mask): # All values are NaN (likely all outliers)
                 boundaries = torch.zeros(*target_tensor.shape[:-1] if self.use_grouping else (1,)*(target_tensor.dim()), num_quantiles + 1, device=self.device, dtype=self.dtype)
            else:
                # Flatten data for quantile calculation if needed (torch.quantile needs 1D or specified dim)
                if self.use_grouping: # Calculate per group
                     boundaries_list = []
                     # Iterate over groups (can be slow, consider vectorization if possible)
                     # Shape (..., num_groups, group_size) -> iterate over (...) and num_groups
                     flat_groups = target_tensor.reshape(-1, self.group_size) # Flatten leading dims
                     valid_mask_flat = non_nan_mask.reshape(-1, self.group_size)

                     for i in range(flat_groups.shape[0]):
                          group_data = flat_groups[i][valid_mask_flat[i]] # Get valid data for this group
                          if group_data.numel() == 0: # Handle empty group case
                               group_boundaries = torch.zeros(num_quantiles + 1, device=self.device, dtype=self.dtype)
                          else:
                               group_boundaries = torch.quantile(group_data, quantile_values, interpolation='linear')
                          boundaries_list.append(group_boundaries)
                     boundaries = torch.stack(boundaries_list) # Shape (num_total_groups, num_quantiles + 1)
                     # Reshape boundaries to match target_tensor shape (..., num_groups, num_quantiles + 1)
                     boundaries = boundaries.view(*target_tensor.shape[:-1], num_quantiles + 1)

                else: # Calculate over the whole block
                     block_data = target_tensor[non_nan_mask]
                     boundaries = torch.quantile(block_data, quantile_values, interpolation='linear')
                     # Reshape boundaries for broadcasting if needed (e.g., scalar block)
                     boundaries = boundaries.view((1,)*target_tensor.dim() + (num_quantiles+1,))


            # Step 3: Determine Centroids (Midpoint for simplicity)
            centroids = (boundaries[..., :-1] + boundaries[..., 1:]) / 2.0
            # Handle cases where boundary[i] == boundary[i+1] -> centroid = boundary[i]
            # Replace NaN centroids (if boundaries were equal) with the boundary value
            centroids = torch.where(torch.isnan(centroids), boundaries[..., :-1], centroids)

            # Ensure centroids have the correct shape for gather: (..., num_groups, num_quantiles) or (..., num_quantiles)

            # Step 4: Quantize - Find bin index for each value
            # Use searchsorted on the upper boundaries (excluding the first one, effectively -inf)
            # Add epsilon to boundaries to handle edge cases where value == boundary
            boundary_search = boundaries[..., 1:] + 1e-9
            quantized_indices = torch.searchsorted(boundary_search.contiguous(), target_tensor.contiguous(), right=True)
            # Clamp indices just in case (shouldn't be needed if boundaries cover range)
            quantized_indices = torch.clamp(quantized_indices, 0, num_quantiles - 1)


            # Step 5: Dequantize (Gather Centroids)
            # Need to expand indices to match centroid dimensions for gather
            if self.use_grouping:
                 # centroids shape: (..., num_groups, num_quantiles)
                 # indices shape: (..., num_groups, group_size)
                 # gather needs index shape compatible with centroid dims excluding gather dim
                 # This requires iterating or complex indexing. Let's try iterating over groups again.

                 dequantized_grouped_list = []
                 flat_indices = quantized_indices.reshape(-1, self.group_size)
                 flat_centroids = centroids.reshape(-1, num_quantiles) # Shape (num_total_groups, num_quantiles)

                 for i in range(flat_indices.shape[0]):
                      group_centroids = flat_centroids[i].unsqueeze(0) # Shape (1, num_quantiles)
                      group_indices = flat_indices[i].unsqueeze(0)    # Shape (1, group_size)
                      # Use gather along the quantile dimension (-1)
                      dequantized_group = torch.gather(group_centroids, -1, group_indices)
                      dequantized_grouped_list.append(dequantized_group.squeeze(0))

                 dequantized_grouped = torch.stack(dequantized_grouped_list) # Shape (num_total_groups, group_size)
                 dequantized_grouped = dequantized_grouped.view(*target_tensor.shape) # Reshape back

            else:
                 # centroids shape: (..., num_quantiles)
                 # indices shape: (...) same as target_tensor
                 # Unsqueeze indices to match centroid dims for gather if needed
                 dequantized_cache = torch.gather(centroids, -1, quantized_indices)


            # Reshape back if grouping was used
            if self.use_grouping:
                dequantized_cache = self._reshape_from_grouping(dequantized_grouped)
            # else: dequantized_cache is already set

            # Step 6: Denormalize
            denormalized_cache = self._denormalize(dequantized_cache, mean_value, scale_value)
        else:
            raise ValueError(f"Unsupported n_bits: {n_bits}")

        # Step 7: Restore outliers
        return torch.where(outlier_mask, cache, denormalized_cache)
    ''' MODIFICATION END '''


    # Returns (quantized kvcache, average n_bits)
    def quantize(self, cache: torch.Tensor, attentions: AttentionType) -> tuple[torch.Tensor, float]:
        if self.level == "no-quantization":
            return cache, torch.finfo(self.dtype).bits
        # cache.shape: (n_layer, n_batch, n_head, seq_len, embed_size_per_head)
        # Transpose to put sequence and batch first for easier processing
        cache = cache.permute(1, 3, 0, 2, 4)
        # cache.shape: (n_batch, seq_len, n_layer, n_head, embed_size_per_head)

        outlier_mask = self._calc_outlier_mask(cache)
        # outlier_mask.shape: (n_batch, seq_len, n_layer, n_head, embed_size_per_head)

        n_bits_tensor = self._calc_quantization_bits(attentions, cache, outlier_mask)
        # n_bits_tensor.shape: (n_batch, seq_len) or (..., n_layer) or (..., n_head)

        # Calculate average bits before quantization loop
        # This average reflects the bits used *before* considering outliers are stored at full precision
        avg_bits_non_outlier = n_bits_tensor.mean(dtype=self.dtype).item()
        # Calculate final average bits considering outlier ratio
        average_n_bits = avg_bits_non_outlier * (1.0 - self.outliers_ratio) + torch.finfo(self.dtype).bits * self.outliers_ratio

        n_bits_min_val, n_bits_max_val = n_bits_tensor.min().item(), n_bits_tensor.max().item()

        # Clone cache to perform quantization in-place on the copy
        quantized_cache = cache.clone()

        # Iterate through the unique bit values calculated
        for n in range(n_bits_min_val, n_bits_max_val + 1):
            # Find where this number of bits should be applied
            indices = torch.where(n_bits_tensor == n)

            if len(indices[0]) == 0: # Skip if no elements need this bit count
                continue

            # Extract the corresponding cache slices and outlier masks
            # The shape of cache_slice and mask_slice depends on the level
            # Example: level='head', indices shape ~ (num_matches,), n_bits shape (B, S, L, H)
            # cache_slice shape needs to be (num_matches, embed_size_per_head)
            # outlier_mask_slice shape needs to be (num_matches, embed_size_per_head)

            ''' MODIFICATION START: Apply quantization method to selected slices '''
            # Get the slices from the cache and mask corresponding to the current bit number 'n'
            # The indices refer to the dimensions of n_bits_tensor (e.g., B, S, L, H for head level)
            # We need to select the corresponding slices from the full cache tensor
            cache_to_quantize = cache[indices] # This gathers based on the indices
            mask_for_slice = outlier_mask[indices] # Gather corresponding mask parts

            # Check if gathered slice is empty before proceeding
            if cache_to_quantize.numel() == 0:
                continue

            # Apply the chosen quantization method to this slice
            quantized_slice = self.quantization_method(
                cache=cache_to_quantize,
                n_bits=n,
                outlier_mask=mask_for_slice
            )

            # Put the quantized slice back into the result tensor
            quantized_cache[indices] = quantized_slice
            ''' MODIFICATION END '''


        # Transpose back to original format
        quantized_cache = quantized_cache.permute(2, 0, 3, 1, 4)
        # cache.shape: (n_layer, n_batch, n_head, seq_len, embed_size_per_head)
        return quantized_cache, average_n_bits


    def calc_quantized_cache_size_per_token(self, average_n_bits: float, model: CausalLM) -> float:
        ''' MODIFICATION START: Account for grouping in overhead calculation '''
        # Base size from quantized data bits
        num_layers = model.config.num_hidden_layers
        num_heads = model.config.num_attention_heads
        hidden_size = model.config.hidden_size # Note: hidden_size = num_heads * embed_size_per_head

        # Calculate bits for the data itself (average_n_bits already includes outlier full precision bits)
        data_bits = average_n_bits * num_layers * hidden_size

        # Calculate overhead bits for storing scale and mean/zero-point
        overhead_bits = 0
        default_n_bits = torch.finfo(self.dtype).bits # Bits for storing scale/mean (e.g., 16 for float16)
        n_params_per_group = 0 if self.symmetric else 1 # Only scale if symmetric
        n_params_per_group += 1 # Always store scale

        if self.level != "no-quantization":
            num_quant_groups = 0
            if self.level == "token":
                num_quant_groups = 1
            elif self.level == "layer":
                num_quant_groups = num_layers
            elif self.level == "head":
                num_quant_groups = num_layers * num_heads

            if self.use_grouping:
                 embed_size_per_head = hidden_size // num_heads
                 if embed_size_per_head % self.group_size != 0:
                      # This should have been caught earlier, but double-check
                      raise ValueError("Embedding size not divisible by group_size in size calculation")
                 num_feature_groups_per_head = embed_size_per_head // self.group_size
                 # Multiply the number of base groups by the number of feature groups within them
                 if self.level == "head":
                      num_quant_groups *= num_feature_groups_per_head
                 elif self.level == "layer":
                      # If layer level, each layer has multiple heads, each with feature groups
                      num_quant_groups *= num_heads * num_feature_groups_per_head
                 elif self.level == "token":
                      # All layers, all heads, all feature groups
                      num_quant_groups *= num_layers * num_heads * num_feature_groups_per_head
                 else: # Should not happen
                     pass


            overhead_bits = num_quant_groups * n_params_per_group * default_n_bits

            # TODO: Add overhead for adaptive quantization parameters if centroids/boundaries are stored explicitly.
            # This depends heavily on the specific implementation of _adaptive_quantize.
            # Assuming for now adaptive method overhead is similar to storing scale/mean per group.


        return data_bits + overhead_bits
        ''' MODIFICATION END '''


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
            # group_size only relevant if level != token etc.
            try:
                 quantizer_list.append(Quantizer(**kwargs))
            except AssertionError as e:
                 # Optionally print skipped invalid combinations
                 # print(f"Skipping invalid config: {kwargs} - {e}")
                 pass
            except Exception as e:
                 print(f"Error creating quantizer with config: {kwargs} - {e}")
                 # Decide whether to skip or raise


    return quantizer_list

```

**Summary of Changes:**

1.  **`__init__`:**
    *   Added `group_size` parameter and `self.use_grouping` flag.
    *   Added `"adaptive"` to `QuantizationMethods`.
    *   Initialized `self.normalization_method_for_adaptive`.
2.  **`set_dtype_and_device`:** Slightly adjusted logic for precomputing normal quantiles based on method.
3.  **`params`:** Added `group_size` to the output dictionary.
4.  **Grouping Helpers:** Added `_reshape_for_grouping` and `_reshape_from_grouping`.
5.  **`_calc_quantization_bits`:** Added validation for `group_size`. Kept bit calculation at the token/layer/head level for now, not per group. Refined calculation logic slightly.
6.  **`_calc_outlier_mask`:** Modified to calculate outliers per group if `self.use_grouping` is True.
7.  **`_normalize`:**
    *   Major changes to handle grouping: reshapes input, calculates stats over `group_size` dimension, ensures output `mean_value` and `scale_value` have the group dimension.
    *   Uses `torch.where` and `nan` handling for masked tensor operations.
8.  **`_denormalize`:** Adapted to handle potentially grouped `mean_value` and `scale_value` by temporarily reshaping the input `normalized_cache` if needed.
9.  **`_uniform_quantize`, `_normal_quantize`:** Updated to use the potentially group-aware `_normalize` and `_denormalize`. Added handling for `n_bits=0`. `_normal_quantize` applies quantiles to group-normalized data.
10. **`_adaptive_quantize`:**
    *   New method implemented.
    *   Uses `_normalize`.
    *   Calculates empirical quantiles per group (if `use_grouping`) or per block. Uses iteration for grouping currently (potential performance bottleneck).
    *   Calculates centroids (midpoint method).
    *   Uses `torch.searchsorted` to find indices.
    *   Uses `torch.gather` (via iteration for groups) to dequantize using centroids.
    *   Uses `_denormalize`.
    *   Restores outliers.
11. **`quantize`:**
    *   Modified the loop applying quantization: gathers slices based on `n_bits_tensor`, applies the selected `self.quantization_method` to the slice, and updates the result tensor.
12. **`calc_quantized_cache_size_per_token`:** Updated to calculate overhead based on the number of groups (considering both `level` and `group_size`).
13. **`build_quantizers`:** Added basic error handling during instantiation.

**Next Steps & Considerations:**

*   **Testing:** Thoroughly test these changes, especially the grouping logic in `_normalize`, `_calc_outlier_mask`, `_denormalize`, and the new `_adaptive_quantize` method. Pay attention to edge cases (e.g., `n_bits=0`, all values being outliers).
*   **Performance:** The iterative approach in `_adaptive_quantize` for grouping might be slow. Explore vectorized alternatives using advanced indexing or libraries like `vmap` if performance becomes an issue.
*   **Feature Importance:** This implementation lays the groundwork (grouping). Integrating feature importance would require:
    *   A way to get importance scores (new parameter/input).
    *   Modifying `_calc_quantization_bits` to distribute bits based on importance *across groups* within a head/layer.
    *   Potentially adjusting `_normalize` or the quantization methods further based on importance.
*   **Configuration (`config.py`):** Remember to update your configuration files to include `group_size` and allow `"adaptive"` for the `method` parameter.

This is a significant update. Please review carefully and test incrementally.

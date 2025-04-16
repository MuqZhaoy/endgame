# KVCache 量化之特征分组 (Feature Grouping)

## 1. 动机

标准的 KVCache 量化方法（如 Head-level 或 Layer-level）通常为整个量化单元（例如一个 Attention Head 的所有特征维度 `embed_size_per_head`）计算一组量化参数（如 `scale` 和 `mean`/`zero-point`）。这种方式假设在一个 Head 或 Layer 内部，所有特征维度的数值分布特性是相似的。

然而，在实践中，`embed_size_per_head` 这个维度内部可能存在显著的数值差异。某些特征可能具有较大的动态范围，而另一些则可能范围较小。使用单一的 `scale` 和 `mean` 可能无法精确地适配所有特征，导致某些特征的量化误差较大。

**特征分组 (Feature Grouping)** 的提出正是为了解决这个问题。其核心思想是将 `embed_size_per_head` 维度进一步划分为更小的组，并为每个组独立地计算和应用量化参数。

**目标:** 通过在特征维度上实现更细粒度的量化，捕捉局部数值特性，从而提高量化精度，减少信息损失。

## 2. 核心机制

特征分组引入了一个新的参数 `group_size`。当启用分组时（即 `group_size` 被设置为一个大于 0 的正整数），对于每个基础量化单元（例如一个 Head），其 `embed_size_per_head` 维度会被划分为 `num_groups = embed_size_per_head / group_size` 个连续的组。

**关键变化:**

*   **独立参数计算:** 不再为整个 Head 计算单一的 `scale` 和 `mean`，而是为 **每个特征组** 单独计算它们自己的 `scale_group` 和 `mean_group`。
*   **局部适应:** 每个组的量化参数能更好地反映该组内特征的实际数值范围和分布。

**示例:** 假设 `embed_size_per_head = 128`，`group_size = 32`。那么每个 Head 会被分成 `128 / 32 = 4` 个特征组。在进行归一化时，会计算出 4 组独立的 `scale` 和 `mean` 值。

## 3. 实现细节 (代码关联)

在 `src/quantizer.py` 的 `Quantizer` 类中，特征分组主要通过以下方式实现：

*   **启用参数:**
    *   `__init__` 方法接收 `group_size: Optional[int] = None` 参数。
    *   `self.use_grouping` 标志根据 `group_size` 的值设置。
    *   当前实现要求 `embed_size_per_head` 必须能被 `group_size` 整除。
    *   目前主要设计用于 `level="head"` 或 `level="layer"` 的量化级别，因为 `level="token"` 已经将整个 Tensor 视为一个块处理。
*   **形状变换辅助函数:**
    *   `_reshape_for_grouping(tensor)`: 将输入张量的最后一个维度（`embed_size_per_head`）变换为 `(num_groups, group_size)`。
    *   `_reshape_from_grouping(tensor)`: 将带有组维度的张量恢复到原始的 `embed_size_per_head` 维度。
*   **核心函数修改:**
    *   **`_calc_outlier_mask(cache)`:** 如果 `self.use_grouping` 为 `True`，则会先调用 `_reshape_for_grouping`，然后在 `group_size` 维度上计算 `kthvalue` 来确定 **每个组内部** 的离群点阈值，最后调用 `_reshape_from_grouping` 恢复掩码形状。
    *   **`_normalize(cache, ..., outlier_mask)`:**
        *   这是受影响最大的函数之一。
        *   如果分组，输入 `cache` 和 `outlier_mask` 会被 reshape。
        *   统计量（`mean`, `scale` 等）沿着 `group_size` 维度 (`dim=-1`) 计算，并保持 `num_groups` 维度 (`keepdim=True`)。
        *   返回的 `mean_value` 和 `scale_value` 将具有额外的 `num_groups` 维度（例如 `(..., num_groups, 1)`），以便后续按组应用。
        *   返回的 `normalized_cache` 形状与输入 `cache` 保持一致。
    *   **`_denormalize(normalized_cache, mean_value, scale_value)`:**
        *   如果分组，它会临时 reshape `normalized_cache`，然后利用 `mean_value` 和 `scale_value` 中额外的 `num_groups` 维度进行广播，实现按组的反归一化，最后恢复 `normalized_cache` 的原始形状。
    *   **`_uniform_quantize`, `_normal_quantize`, `_adaptive_quantize`:** 这些量化方法本身的核心逻辑（如四舍五入、查表）通常是逐元素的。它们依赖于 `_normalize` 提供了按组归一化的数据，并使用 `_denormalize` 进行按组的恢复。因此，它们本身不需要大量修改就能利用分组带来的好处。
    *   **`calc_quantized_cache_size_per_token(...)`:** 开销计算被修改，以反映需要存储更多组的 `scale` 和 `mean` 参数所带来的额外比特数。总的参数数量等于 `num_base_blocks * num_groups_per_block * n_params_per_group`。

## 4. 优势与代价

**优势:**

*   **提升量化精度:** 通过更细粒度的参数调整，可以更好地拟合特征维度内的局部数据分布，有望减少量化误差。
*   **捕捉局部特征:** 对于那些内部特征差异较大的 Head/Layer，分组量化能更有效地处理这种差异。
*   **未来扩展基础:** 为更高级的、基于特征重要性的比特分配或参数调整策略奠定了基础（例如，可以给更重要的组分配更多比特或更精细的量化方案）。

**代价:**

*   **参数开销增加:** 主要代价是需要存储更多的 `scale` 和 `mean`/`zero-point` 参数。参数量变为原来的 `num_groups` 倍。这会增加 KVCache 的总大小，尽管增加的是开销部分，而不是数据本身。
*   **计算开销:** 计算统计量（`mean`, `std`, `min`, `max`）以及应用参数的次数增加了 `num_groups` 倍。虽然实现中尽量使用向量化操作，但仍可能带来一定的计算延迟。
*   **实现复杂度:** 代码逻辑相对更复杂。
*   **限制:** `embed_size_per_head` 必须能被 `group_size` 整除。

## 5. 如何使用

在构建 `Quantizer` 对象或配置实验时，设置 `group_size` 参数为一个正整数即可启用特征分组。`group_size` 的选择是一个超参数，需要根据模型和性能目标进行调整。

*   `group_size` 越小，分组越细，参数开销越大，理论上精度潜力越高，但计算开销也可能增加。
*   `group_size` 等于 `embed_size_per_head` 时，等效于不分组。
*   `group_size` 未设置或小于等于 0 时，禁用分组。

**示例配置 (假设在 `config.py` 或类似文件中):**

```python
quantizer_config = {
    "key_or_value_cache": "key",
    "level": "head",           # 分组通常用于 head 或 layer 级别
    "symmetric": True,
    "method": "uniform",
    "outliers_ratio": 0.01,
    "use_attentions": False,
    "n_bits_uniform": 4,
    "group_size": 32           # 启用分组，每 32 个特征为一组
}

# 或者在 build_quantizers 的 grid 中
config_grid = {
    # ... other parameters ...
    "level": ["head"],
    "group_size": [16, 32, 64, None] # 测试不同的 group_size，None 表示不分组
}
```

## 6. 与其他参数的关系

*   **`level`:** 如前所述，分组主要影响 `head` 和 `layer` 级别，因为它作用于 `embed_size_per_head` 维度。
*   **`method`:** 不同的量化方法 (`uniform`, `normal`, `adaptive`) 都会受益于按组计算的 `scale` 和 `mean`，因为它们都依赖于归一化步骤。
*   **`symmetric`:** 决定了 **每个组** 是否需要存储 `mean` 值（非对称）或仅存储 `scale` 值（对称）。
*   **`outliers_ratio`:** 决定了 **每个组内部** 有多少比例的值被视为离群点。

通过调整 `group_size`，可以在量化精度和参数开销之间进行权衡。 
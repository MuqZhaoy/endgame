# KVCache 量化之自适应分布量化 (Adaptive Distribution Quantization)

## 1. 动机

传统的量化方法，如均匀量化 (`uniform`) 或正态量化 (`normal`)，都基于一个先验假设：归一化后的数据遵循某种特定的概率分布（分别是均匀分布和标准正态分布）。然后根据这个假设的分布来确定量化区间的边界和中心点。

然而，KVCache 中特征的实际数值分布可能非常复杂，并不严格符合均匀或正态分布。当实际分布与假设分布偏差较大时，基于错误假设设计的量化方案可能会引入显著的量化误差，影响模型性能。

**自适应分布量化 (Adaptive Distribution Quantization)** 的核心思想是放弃固定的分布假设，转而直接**从实际观测到的数据中学习**量化方案。它旨在让量化边界和中心点能更好地拟合数据本身的特性。

**目标:** 通过使量化方案适应数据的局部经验分布，进一步减少量化误差，提高压缩效率。

## 2. 核心机制

自适应分布量化的关键在于使用**经验累积分布函数 (Empirical CDF)** 或其逆函数（即**经验分位数 Empirical Quantiles**）来确定量化边界。其主要步骤如下：

1.  **数据准备:** 获取需要量化的数据块（例如一个 Head 或一个特征组），并排除预先识别出的离群点 (Outliers)。
2.  **归一化:** 对非离群数据进行归一化处理（通常使用标准差归一化，即 `method="std"`，由 `normalization_method_for_adaptive` 控制），得到 `normalized_cache`。
3.  **计算经验分位数边界 (`boundaries`):**
    *   根据目标量化比特数 `n_bits`，确定需要 `num_quantiles = 2 ** n_bits` 个量化区间。
    *   创建等间距的分位数点 `q = torch.linspace(0, 1, num_quantiles + 1)`（例如，`n_bits=2` 时为 `[0, 0.25, 0.5, 0.75, 1.0]`）。
    *   使用 `torch.quantile(normalized_cache (non-outliers), q)` 直接从归一化后的**有效数据**中计算出对应这些分位数点的值。这些值就构成了 `num_quantiles` 个量化区间的边界。
4.  **计算量化中心点 (`centroids`):**
    *   每个量化区间的代表值（中心点）通常取该区间相邻两个边界的算术平均值: `centroid[i] = (boundary[i] + boundary[i+1]) / 2`。
5.  **量化 (查找索引):**
    *   对于 `normalized_cache` 中的每一个值，使用 `torch.searchsorted(boundaries, value)` 找到它所属的量化区间索引 `quantized_indices` (范围从 0 到 `num_quantiles - 1`)。
6.  **反量化 (查找中心点):**
    *   使用 `quantized_indices` 从 `centroids` 数组中查找对应的中心点值，得到 `dequantized_cache`。
7.  **反归一化:**
    *   使用原始计算得到的 `mean_value` 和 `scale_value` 对 `dequantized_cache` 进行反归一化。
8.  **合并离群点:**
    *   将反归一化后的结果与原始离群点合并，得到最终的量化后 KVCache。

## 3. 实现细节 (代码关联)

该方法在 `src/quantizer.py` 的 `Quantizer` 类中由 `_adaptive_quantize` 方法实现：

*   **方法选择:** 在 `__init__` 中，如果 `method="adaptive"` 被指定，`self.quantization_method` 会指向 `_adaptive_quantize`。
*   **归一化依赖:** 方法首先调用 `self._normalize`，并使用 `self.normalization_method_for_adaptive` (默认为 `"std"`) 作为归一化方式。
*   **分位数计算 (`torch.quantile`):** 这是核心步骤。代码计算 `quantile_points`，并对 **排除了离群点** 的 `normalized_cache` 调用 `torch.quantile` 来获取 `boundaries`。
*   **中心点计算:** 通过相邻边界的平均值计算 `centroids`。
*   **索引查找 (`torch.searchsorted`):** 使用计算出的 `boundaries`（通常是上边界）对 `normalized_cache` 进行 `torch.searchsorted` 以获得每个值的量化索引。
*   **中心点查找 (`torch.gather`):** 使用量化索引在 `centroids` 张量中查找对应的中心点值。对于非分组情况，使用了 PyTorch 的广播和 `gather`；对于分组情况，由于每个组有独立的 `centroids` 和索引，当前实现使用 **迭代** 来完成查找。
*   **与分组的交互:**
    *   如果 `self.use_grouping` 为 `True`，`_adaptive_quantize` 会在内部调用 `_reshape_for_grouping`。
    *   **关键:** 分位数边界 (`boundaries`) 和中心点 (`centroids`) 的计算是 **在每个特征组内部独立完成的**。这通常是通过迭代每个组的数据来实现的。
    *   类似地，`searchsorted` 和 `gather` 操作也需要在每个组上使用其对应的边界和中心点进行，当前实现也采用 **迭代** 处理。
*   **迭代性能:** 需要注意，当前实现中，当启用分组时，计算分位数、查找索引和查找中心点的步骤涉及 Python 循环遍历所有组，这可能成为 **性能瓶颈**，尤其是在 `group_size` 较小（即组数很多）的情况下。未来的优化可以探索更高级的向量化方法。

## 4. 优势与代价

**优势:**

*   **数据拟合性强:** 量化方案直接由数据驱动，能更好地适应各种奇特的、非理想的数值分布，理论上可以达到比固定分布假设方法更高的精度。
*   **无需先验假设:** 不需要猜测数据符合哪种分布。

**代价:**

*   **计算成本高:** `torch.quantile` 和 `torch.searchsorted` 本身比简单的算术运算更耗时。当结合分组并且使用迭代实现时，计算开销会显著增加。
*   **潜在的参数开销:** 虽然当前实现是动态计算边界和中心点，但如果为了推理效率而需要预计算并存储它们，将会带来额外的内存开销（每个量化块/组都需要存储 `num_quantiles` 个边界或中心点）。
*   **对数据量敏感:** 经验分位数的准确性依赖于有效数据的数量。如果某个块/组内的非离群点很少，计算出的边界可能不稳定或不准确。
*   **实现复杂度:** 逻辑比均匀或正态量化更复杂，尤其是在处理分组和广播时。

## 5. 如何使用

在构建 `Quantizer` 对象或配置实验时，将 `method` 参数设置为 `"adaptive"` 即可启用此方法。

```python
quantizer_config = {
    # ... other parameters ...
    "method": "adaptive",        # 启用自适应分布量化
    "group_size": 32           # 可选：结合特征分组
}

# 或在 grid search 中
config_grid = {
    # ... other parameters ...
    "method": ["uniform", "normal", "adaptive"],
    "group_size": [32, None]
}
```

## 6. 与其他参数的关系

*   **`group_size`:** 如上所述，自适应量化可以与特征分组结合。启用分组时，自适应过程（计算边界、中心点等）将在每个特征组内部独立进行。
*   **`normalization_method_for_adaptive`:** `Quantizer` 初始化时为此方法设置了一个默认的归一化方式（当前为 `"std"`）。理论上可以修改为支持其他归一化方式，但这会影响计算出的经验分位数。
*   **`n_bits`:** 直接决定了量化区间的数量 (`2**n_bits`)，从而影响分位数计算的粒度和最终精度。
*   **`outliers_ratio`:** 决定了哪些数据点不参与经验分位数的计算。

自适应分布量化提供了一种更灵活但计算成本更高的方法，其效果可能因数据分布和具体实现（尤其是否向量化）而异。
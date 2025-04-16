
好的，我们来梳理一下当前 `Quantizer` 的核心量化思路：

1.  **目标:** 减少 KVCache 的内存占用，同时尽量减小对模型性能（如准确率、困惑度）的影响。

2.  **核心步骤:** 对 KVCache Tensor 中的浮点数值进行处理，将其转换为较低比特数的表示，并能在需要时近似地恢复回浮点数。这个过程主要包含以下几个关键点：

    *   **确定量化单元 (Granularity - `level`):**
        *   决定哪些数值共享同一套量化参数（主要是 `scale` 和 `mean`/`zero-point`）。
        *   `token`: 每个 token 的所有层、所有 head 的 K/V Cache 使用一套参数。
        *   `layer`: 每个 token、每一层的 K/V Cache 使用一套参数。
        *   `head`: 每个 token、每一层、每一个 head 的 K/V Cache 使用一套参数（最细粒度）。
        *   这个选择体现在 `quantize_dims` 上，后续计算统计量（如 min, max, mean, std）会沿着这些维度进行。

    *   **处理离群点 (Outlier Handling - `outliers_ratio`):**
        *   在计算量化参数（scale/mean）之前，识别并“屏蔽”掉一部分极端值（最大和最小的 `outliers_ratio / 2` 百分比）。
        *   这些离群点不参与主体的量化过程，而是直接以原始精度存储。
        *   **核心目的:** 防止极少数异常值过度拉大数值范围，导致大部分正常值的量化精度下降。这是常见的提升量化效果的技巧。这一步在 `_calc_outlier_mask` 实现。

    *   **数值归一化/标准化 (Normalization - `_normalize`):**
        *   将**非离群点**的数值从原始范围映射到一个更适合量化的标准范围。
        *   **对称 (`symmetric=True`):** 假设数据关于 0 对称，只计算一个 `scale` 因子。通常基于绝对值的最大值 (`method="minmax"`) 或标准差 (`method="std"`)。`mean_value` 固定为 0。
        *   **非对称 (`symmetric=False`):** 同时计算 `mean_value` 和 `scale_value`。`scale` 通常基于最大值和最小值的差 (`method="minmax"`) 或标准差 (`method="std"`)。
        *   **核心目的:** 将不同范围的数据映射到相似的尺度，方便后续用固定位数表示。`scale` 和 `mean` 是关键的量化参数，需要存储下来用于反量化。

    *   **映射到量化值 (Quantization Mapping):**
        *   **均匀量化 (`_uniform_quantize`):**
            *   通常配合 Min-Max Normalization 使用。
            *   将归一化后的值四舍五入到最接近的整数。整数的范围由 `n_bits` 决定（例如，对于 n bits 对称量化，范围是 `[-2^(n-1), 2^(n-1)-1]`）。
            *   **核心思想:** 数据在归一化后的范围内是均匀分布的。
        *   **正态量化 (`_normal_quantize`):**
            *   通常配合 Standardization (基于均值和标准差) 使用。
            *   假设归一化后的数据服从标准正态分布。
            *   使用预先计算好的标准正态分布分位数 (`normal_quantiles_upper_bound`) 来确定每个值应该映射到哪个“桶”(bin)。每个桶用其中心值 (`normal_quantiles_center`) 来表示。
            *   **核心思想:** 如果数据分布更接近正态分布，这种方法可能比均匀量化更有效。

    *   **反量化/恢复 (Dequantization - `_denormalize`):**
        *   使用存储的 `scale` 和 `mean`，将整数（或正态量化的中心值）映射回近似的原始浮点数值。
        *   `denormalized_value = quantized_value * scale + mean`

    *   **整合结果:**
        *   将反量化后的值与之前被屏蔽的离群点（以原始精度存储）合并，得到最终的、模拟量化过程后的 KVCache。

    *   **动态比特数分配 (Attention-Aware - `use_attentions=True`):**
        *   这是一个可选的高级功能，核心点在于**不使用固定的 `n_bits_uniform`**，而是根据量化可能引入的误差来动态决定每个量化单元使用多少比特 (`_calc_quantization_bits`)。
        *   它试图将 Key 或 Value 的量化误差控制在一个目标阈值 (`target_quantization_error`) 内。
        *   对于 Key Cache，误差估计基于其对 Attention Score 计算的影响（涉及到 `q_norm`）。
        *   对于 Value Cache，误差估计基于 Value 的误差如何通过 Attention 权重 (`attentions`) 影响最终输出。Attention 权重大的 Value 可能需要更高的精度（更多的 bits）。
        *   计算出的比特数会被限制在 `n_bits_min` 和 `n_bits_max` 之间。
        *   **核心目的:** 在保证一定性能（由 `target_quantization_error` 控制）的前提下，自适应地使用比特数，可能比固定比特数更节省内存。

**总结来说，最核心的点是：**

1.  **选定粒度 (Level):** 决定参数共享范围。
2.  **分离离群点:** 保护主流量化精度。
3.  **归一化/标准化:** 使用 `scale` 和 `mean` 将数据映射到标准范围。
4.  **映射到低比特表示:** 通过均匀分层或基于分布（如正态）的分位数进行映射。
5.  **(可选) 动态比特分配:** 基于 Attention 感知或误差估计来决定比特数。

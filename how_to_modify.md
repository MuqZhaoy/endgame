
好的，我来分析一下整个代码仓库，判断实现新方案可能涉及的文件改动：

1.  **`quantizer.py`**: 这是**主要修改**的地方。
    *   需要添加新的初始化参数（如 `group_size`，特征重要性相关参数，自适应分布量化开关等）。
    *   需要修改 `_normalize` 以支持特征组级别的归一化。
    *   可能需要修改 `_calc_outlier_mask` 以支持组级别离群点。
    *   需要修改 `_calc_quantization_bits` 以集成特征重要性感知和组级别比特分配。
    *   需要实现新的 `_adaptive_quantize` 方法。
    *   需要修改 `_uniform_quantize` 和 `_normal_quantize` 以处理组级别量化。
    *   需要修改 `quantize` 方法的主流程来协调特征分组、重要性评估和选择合适的量化映射。
    *   需要更新 `params` 属性以包含新参数。
    *   需要更新 `calc_quantized_cache_size_per_token` 来计算可能更复杂的参数存储开销。
    *   `build_quantizers` 函数需要能接收和处理新的配置选项。

2.  **`config.py`** (或类似的配置文件): **需要修改**。
    *   必须在这里添加新的配置选项，例如：
        *   `group_size`: 定义特征分组大小。
        *   `feature_importance_method`: 选择静态或动态特征重要性评估方法（以及相关参数）。
        *   `quantization_mapping_method`: 允许选择 "uniform", "normal", 或 "adaptive"。
        *   可能还有自适应分布量化相关的参数（如分位数数量等）。
    *   需要更新配置网格 (`config_grid_list` 或类似结构) 以包含这些新选项的组合，以便 `build_quantizers` 能创建包含新配置的 `Quantizer` 实例。

3.  **`evaluator.py`** (或负责评估的脚本): **可能需要少量修改**。
    *   核心的模型推理和调用 `quantizer.quantize` 的流程**不一定**需要改变，前提是 `quantize` 方法的输入输出接口保持兼容（输入 KVCache Tensor，输出同形状的 Tensor）。
    *   **但是**，如果想在评估结果中记录或分析与新方案相关的指标（例如，不同特征组的平均比特数、使用的量化映射方法、特征重要性分布等），就需要修改日志记录或结果保存部分。

4.  **`main.py` / `run.py`** (或主执行脚本): **可能需要非常微小的修改**。
    *   主要是确保从 `config.py` 加载的配置能正确传递给 `build_quantizers`。如果配置加载和传递机制是通用的，可能无需修改。

5.  **`models.py`**: **大概率不需要修改**。
    *   只要 `Quantizer` 类仍然接收标准格式的 KVCache 并返回相同外部形状的 KVCache，模型部分的实现（如何生成和使用 KVCache）应该不受影响。

6.  **`analysis.md`**: **需要更新** (我们已经做过了)。

**结论:**

实现这个新方案**不仅仅是修改 `quantizer.py`**。虽然 `quantizer.py` 包含核心算法的绝大部分改动，但**至少还需要修改 `config.py`** 来定义和管理新的配置选项。此外，根据你是否需要在评估中追踪新引入的细节，**可能还需要少量修改 `evaluator.py`**。主脚本和模型定义部分基本不需要改动。

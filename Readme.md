# 🚀 LLaDA MoE Optimization

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

High-performance optimization suite for LLaDA (Large Language Diffusion Assistant) Mixture-of-Experts (MoE) layers. This implementation transforms the inference regime from a CPU-bound state to a hardware-saturated, compute-bound state
---

## Overview

The standard LLaDA MoE implementation often struggle with low GPU utilization due to Python-side dispatching loops and fragmented memory access. When scaling LLaDA to multiple experts, these bottlenecks become critical at num_group = 1 geneartion

`FastLLaDAMoE` addresses these issues by:
- Memory Coalescing: Grouping expert weights into unified buffers for faster VRAM access
- Token Rearrangement: Efficiently grouping tokens by assigned experts to maximize compute density

## 📈 Performance Benchmarks

[cite_start]Experiments conducted on an **NVIDIA A100 (40/80GB)** with `transformers == 4.57.6`[cite: 12, 80]:

* [cite_start]**CUDA Speedup**: Achieved a mean execution time ratio of **$1.8883 \pm 0.0030$**[cite: 87, 172]
* [cite_start]**Memory Throughput**: Nearly doubled effective memory bandwidth utilization with a ratio of **$1.9251 \pm 0.0066$**[cite: 83, 172]
* [cite_start]**Hardware Efficiency**: Increased SM utilization by a factor of **$1.2541 \pm 0.0009$**[cite: 83, 174]
* [cite_start]**Mathematical Parity**: Validated on **GSM8K** with identical **50% accuracy**, confirming full parity with the native MoE block[cite: 102, 178]

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/llada-moe-optimization.git
   cd llada-moe-optimization
   ```

2. Install dependencies:
   ```bash
   pip install torch transformers packaging
   ```
   *Note: This optimization is validated against `transformers==4.57.6`.*

## Quick Start

```python
import torch
from transformers import AutoModelForCausalLM
from llada_moe_optimization import optimize_llada_moe

#your LLaDA MoE model
model = AutoModelForCausalLM.from_pretrained("inclusionAI/LLaDA-MoE-7B-A1B-Instruct", device_map="auto")

#patch the model with optimized MoE blocks
optimized_model = optimize_llada_moe(model)

##use code from https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Base from section №2 No Speedup: transformers
```

## Performance Tuning

This optimization is particularly effective for:
- **Different GPUs**: Was released only on torch
- **High Expert Count**: Parallelizes the "Expert Loop" logic more effectively
- **Triton Integration**: (Coming Soon) Custom kernels for specialized MoE activation

## Compatibility Warning

This implementation relies on monkey-patching specific internal classes of the `LLaDAMoESparseMoeBlock`
| Component | Tested Version | Status |
|-----------|----------------|--------|
| Transformers | 4.57.6 | Stable |
| PyTorch | 2.0+ | Stable |

## Contributing

Contributions are welcome! If you have interesting ideas for further optimizations (e.g., Triton kernels for the router), please open an issue or submit a PR

gmail: alexeymanakonov@gmail.com
tg: @alexeo_0man

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

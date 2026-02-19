# LLaDA MoE Optimization

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

High-performance optimization suite for LLaDA (Large Language Diffusion Assistant) Mixture-of-Experts (MoE) layers. This implementation transforms the inference regime from a CPU-bound state to a hardware-saturated, compute-bound state
---

## Performance Benchmarks

### Environment & Setup
All experiments were conducted on an **NVIDIA A100 (40GB/80GB)**
Measurements were averaged over eight independent runs to ensure statistical consistency
**Transformers Version**: 4.57.6
**Task**: LLaDA-MoE inference (max length 128)

### Acceleration Metrics
By transitioning from fragmented memory access to the **Sort-Compute-Scatter** pipeline, we achieved:
* **CUDA Speedup**: Mean execution time ratio of **$1.8883 \pm 0.0030$**
* **Memory Throughput**: Effective bandwidth utilization increased by a ratio of **$1.9251 \pm 0.0066$**
* **Hardware Efficiency**: SM utilization improved by a factor of **$1.2541 \pm 0.0009$** (at 1 batch size)

### Mathematical Parity
The optimization maintains full mathematical equivalence with the native implementation:
* **Benchmark**: GSM8K test set (50 samples)
* **Accuracy**: **50%** for both baseline and optimized versions
* **Verification**: Confirmed that Sort-Compute-Scatter does not compromise model performance
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

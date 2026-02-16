# 🚀 LLaDA MoE Optimization

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance optimization suite for **LLaDA (Large Language Diffusion Assistant)** Mixture-of-Experts (MoE) layers. This implementation focuses on reducing kernel launch overhead and improving memory throughput during parallel expert execution.

---

## Overview

The standard LLaDA MoE implementation can suffer from performance bottlenecks due to Python-level loops and fragmented memory access when processing multiple experts. 

`FastLLaDAMoE` addresses these issues by:
- Unified Weight Buffers: Coalescing expert weights into continuous memory.
- Batched Dispatch: Grouping tokens by expert assignment to maximize GPU occupancy.

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
model = AutoModelForCausalLM.from_pretrained("path/to/llada-model", device_map="auto")

#patch the model with optimized MoE blocks
optimized_model = optimize_llada_moe(model)

##use code from https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Base from section №2 No Speedup: transformers
```

## Performance Tuning

This optimization is particularly effective for:
- **Different GPUs**: Was released only on torch.
- **High Expert Count**: Parallelizes the "Expert Loop" logic more effectively.


## Compatibility Warning

This implementation relies on monkey-patching specific internal classes of the `LLaDAMoESparseMoeBlock`. 
| Component | Tested Version | Status |
|-----------|----------------|--------|
| Transformers | 4.57.6 | ✅ Stable |
| PyTorch | 2.0+ | ✅ Stable |
| CUDA | 12.1+ | ✅ Recommended |

## Contributing

Contributions are welcome! If you have ideas for further optimizations (e.g., Triton kernels for the router), please open an issue or submit a PR.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

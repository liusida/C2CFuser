# Cache-to-Cache (C2C)

Simplified reproduction of [Cache-to-Cache: Direct Semantic Communication Between Large Language Models](https://arxiv.org/abs/2510.03215).

## Warning

This is an experimental reimplementation of the paper’s method. Training is not yet stable and may diverge. If you spot issues or have fixes, please open an issue.

## Quick Start

```bash
# Train
python train.py

# Test
python test.py

# Benchmark
python benchmark.py
```

## Files

-   `train.py` - Training on OpenHermes-2.5
-   `test.py` - Testing on OpenHermes-2.5
-   `benchmark.py` - Evaluating on MMLU Redux
-   `models.py` - C2C model definitions
-   `utils.py` - Utility functions

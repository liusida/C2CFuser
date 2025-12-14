# Cache-to-Cache (C2C)

Simplified reproduction of [Cache-to-Cache: Direct Semantic Communication Between Large Language Models](https://arxiv.org/abs/2510.03215).

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

## My Quick Results

MMLU-Redux has 2880 valid questions (and 120 invalid)

The raw receiver Qwen3-0.6B got 39.27% (1131/2880), I am going to check my trained model.

my model has all gates on, and the result is:34.90% (1005/2880)

I'll train again to record the parameters.

"""Reproducibility study package for the Vietnamese Healthcare NLP survey (Section 6).

Instruction-tunes a single Llama model (Llama-3.1-8B-Instruct by default) on
instruction-formatted benchmark datasets, one dataset per run, with LoRA (bf16 for the
reported runs; 4-bit QLoRA loading is optional). Designed for GPU environments
supported by the installed Unsloth and CUDA versions. See README.md.
"""

__version__ = "1.0.0"

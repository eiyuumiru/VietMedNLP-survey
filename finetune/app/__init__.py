"""Reproducibility study package for the Vietnamese Healthcare NLP survey (Section 6).

Instruction-tunes a single Llama model (Llama-3.1-8B-Instruct by default) on the
HEALTHDOMAIN instruct datasets, one dataset per run, with QLoRA. Designed for GPU
environments supported by the installed Unsloth and CUDA versions. See README.md.
"""

__version__ = "1.0.0"

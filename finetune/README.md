# Section 6.1 — dataset-specific fine-tuning

This folder contains the training, inference, evaluation, and aggregation code
for the fine-tuning experiments. The runner uses
`unsloth/Meta-Llama-3.1-8B-Instruct` and trains a separate LoRA adapter for each
dataset using the natural-language instruction and target columns.

## Environment and data

Use a GPU environment supported by Unsloth and the installed CUDA/PyTorch stack;
the original runs used Google Colab with an A100. Start from a clean GPU runtime
and install this folder's dependencies:

```bash
python -m pip install -r requirements.txt
```

Unsloth selects a CUDA-compatible training stack, so its version is not pinned
in this portable install file. Every completed run records the installed
Unsloth, TRL, Transformers, PyTorch, PEFT, and Datasets versions in
`metrics.json`.

The data is not included. Obtain authorized copies of the HEALTHDOMAIN
`Instruct_Datasets` files and place them in a local directory. It must
contain `train/`, `dev/`, and `test/` directories with the CSV files named by
`python -m app.run --list-datasets`. The runner validates the expected columns
and reads the `instruction_prompt_nl` / `output_prompt_nl` fields.
[`DATA_FORMAT.md`](DATA_FORMAT.md) gives the exact folder layout and CSV schema.

Set `DATA_ROOT` to the local `Instruct_Datasets` directory:

```bash
DATA_ROOT="/path/to/Instruct_Datasets"
python -m app.validate_data --data-root "$DATA_ROOT"
```

Validation can report known unusable data such as `ViMQ_NER`; the ten datasets
reported in Section 6.1 do not include `ViMQ_NER` or `ViSP_Sentence_Paraphrases`.
The CSV files remain outside this repository. The original experiments used a
Colab A100 runtime; other compatible GPU environments can run the same recipe.

## Reproduce a reported run

Run commands from `finetune/`. The selected Section 6.1 runs used bf16 LoRA,
seed 42, learning rate `2e-4`, training batch 16, gradient accumulation 1,
LoRA rank 16 / alpha 32 / dropout 0.05, and validation on up to 1,000 `dev`
examples. The `test` split is evaluated in full. `--no-4bit` is required to
match the reported non-quantized bf16 LoRA setup; `--no-auto-batch` preserves
the recorded batch settings.

```bash
python -m app.run \
  --dataset ViMedNLI_ViMedNLI \
  --data-root "$DATA_ROOT" \
  --output-root results \
  --no-4bit --no-auto-batch \
  --epochs 3 --seed 42 --learning-rate 2e-4 \
  --train-batch-size 16 --grad-accum 1 --eval-batch-size 32 \
  --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
  --max-val-samples 1000
```

Use `--epochs 2` for `ViMedAQA_Abstract_QA`; the other reported runs use three
epochs. `python -m app.run --list-datasets` lists every dataset in the code's
registry, including datasets outside the Section 6.1 result table.

The split roles are `train` for fitting, `dev` for best-checkpoint selection by
validation loss, and `test` for final evaluation. The runner writes each run to
`results/<dataset>/` with `metrics.json`, `predictions.jsonl`,
`run_config.json`, and the adapter. These generated files are ignored by Git.
`run_config.json` records requested values; `metrics.json` records the effective
precision, accepted TRL settings, and package versions from the actual runtime.

## Post-process and summarize

Section 6.1 reports BERTScore for three generation tasks. Compute it from the
saved predictions after those runs finish, then regenerate the combined tables:

```bash
python -m app.score_bertscore --results-root results --lang vi
python -m app.aggregate --results-root results \
  --model-label "Llama-3.1-8B (ours)"
```

The BERTScore script reads saved prediction/reference pairs; it does not load
the adapters or generate new predictions. `summary.csv` and `summary.tex` are
written under the results folder.

## Main modules

- `app/run.py`: single-dataset train-and-evaluate CLI.
- `app/train.py`, `app/modeling.py`, `app/infer.py`: Unsloth/TRL fine-tuning and
  generation.
- `app/metrics.py`, `app/evaluate.py`: task metrics and evaluation.
- `app/validate_data.py`: dataset schema and content checks.
- `app/score_bertscore.py`, `app/aggregate.py`: generation score post-processing
  and table generation.

No training or model downloads happen when importing the package. The selected
run configuration is saved alongside each result for later inspection.

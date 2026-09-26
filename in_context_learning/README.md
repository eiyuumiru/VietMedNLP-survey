# Section 6.2 — in-context learning

This folder contains the ICL-v3 experiment runner, task prompts and evaluator,
results audit, and offline regression tests. The batch covers ten datasets,
four model profiles, and two prompting conditions: `direct` and `cot` (a brief
explanation plus a separate final answer). Each dataset uses the same sampled
training demonstrations across model profiles and prompting conditions.

[`PROMPTS.md`](PROMPTS.md) documents the task instructions, shot counts,
sampling and scoring protocol.

## Install

Use a Python environment separate from `finetuning/`:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux:        source .venv/bin/activate
python -m pip install -r requirements.txt
```

The first `tiktoken` use may download its tokenizer data. Offline tests need
neither credentials nor benchmark data.

## Prepare benchmark data

Obtain authorized copies of the benchmark CSV files and place them outside
this repository using this layout:

```text
icl-data/
├── train/
│   ├── ViMedNLI_ViMedNLI.csv
│   └── ...
└── test/
    ├── ViMedNLI_ViMedNLI.csv
    └── ...
```

Each CSV must have non-empty `input` and `output` columns. The runner streams
the files from disk and records their SHA-256 fingerprints in each experiment
manifest; it does not copy dataset contents into this repository. Pass the
folder with `--data-root` or set `ICL_DATA_ROOT` in `.env`.

## Configure a model API

Copy `.env.example` to `.env`, then set values for an API you are authorized to
use. Keep `.env` local; Git ignores it. The API adapter supports:

- `openai-compatible` (default): set `ICL_API_KEY`; optionally set
  `ICL_API_BASE_URL`. Leave it empty for the default OpenAI endpoint.
- `azure-openai`: also set `ICL_API_BASE_URL` and `ICL_API_VERSION`.

For a single run, set `ICL_MODEL` or pass `--model` with the model identifier
(for Azure OpenAI, its deployment name). For the paper batch, set the four
profile variables below. Each value is the identifier expected by the chosen
provider; for Azure OpenAI, use the deployment name.

| Paper model profile | Environment variable |
|---|---|
| GPT-4o | `ICL_MODEL_GPT_4O` |
| GPT-4.1 | `ICL_MODEL_GPT_4_1` |
| GPT-5.2 | `ICL_MODEL_GPT_5_2` |
| GPT-5.4 | `ICL_MODEL_GPT_5_4` |

## Run the Section 6.2 batch

From `in_context_learning/`, run:

```bash
python src/run_all.py --data-root /path/to/icl-data
```

This schedules four model profiles × ten datasets × two methods = 80 runs,
using seed 42, task-specific shot counts, and each full test split. It allows
four request workers per model profile and caps total concurrency at 16. The
two prompting conditions run sequentially. API charges may apply. Rerunning
the command resumes compatible checkpoints and skips completed runs.

By default, outputs go to `results/all_models_icl-v3/` in this folder. Choose a
different directory or a smoke-run subset with:

```bash
python src/run_all.py \
  --data-root /path/to/icl-data \
  --results-root /path/to/experiment-output \
  --limit 20
```

The batch summary is `<results-root>/summary.json`. Predictions, manifests,
metrics, caches, and logs are experiment outputs and excluded from Git.

To run one task with a paper model profile:

```bash
python src/icl_inference.py \
  --data-root /path/to/icl-data \
  --dataset ViMedNLI_ViMedNLI \
  --model-family gpt-4o \
  --method direct --shots 6 \
  --output results/single/vimednli-gpt-4o
```

Or pass a provider-specific model identifier directly:

```bash
python src/icl_inference.py \
  --data-root /path/to/icl-data \
  --dataset ViMedNLI_ViMedNLI \
  --model my-model-id \
  --method direct --shots 6 \
  --output results/single/vimednli-custom
```

Omit `--limit` for the full test split. `--prepare-only` selects and saves
demonstrations without making an API request; it still needs the local data and
a model identifier (or configured model profile).

## Audit and tests

After a complete batch, create the Section 6.2 score table and ICL-v3 audit:

```bash
python src/audit_icl_v3.py \
  --results-root /path/to/experiment-output \
  --analysis-root /path/to/analysis-output
```

The audit requires 80 complete runs under the selected results folder. The
defaults match the batch runner's default output and this folder's `analysis/`
directory. It checks manifests, summary metrics, and prediction counts, then
writes the LaTeX table and audit bundle to the analysis folder; it does not
write to the manuscript directory. Review the generated table before copying
it into the paper.

Run offline tests from `in_context_learning/`:

```bash
python -B -m unittest discover -s tests -p "test_*.py"
```

Tests use mocked clients and temporary CSV fixtures. They do not call a model
API, download datasets, or create repository predictions.

## Code availability

The source code and experiment instructions for Section 6.2 will be available
at `<repository URL>` after the repository is published.

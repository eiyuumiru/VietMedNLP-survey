# Section 6.2 — in-context learning

This folder contains the ICL-v3 runner, task prompts and local evaluator,
results audit, and offline regression tests. It supports the ten Section 6.2
datasets with four model families and two prompting conditions: `direct` and
`cot` (a brief explanation plus a separate final answer). The same selected
training demonstrations are used for every model and both conditions.

See [`PROMPTS.md`](PROMPTS.md) for the prompt templates, task shot counts,
scoring rules, and protocol details.

## Install

Use a Python environment separate from `finetune/`:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux:        source .venv/bin/activate
python -m pip install -r requirements.txt
```

The first `tiktoken` use may download its vocabulary cache. The ICL tests do not
need API credentials or data access.

## Configure credentials, deployments, and data

Copy `.env.example` to `.env`, then fill it with values for an API-compatible
chat endpoint and a Drive folder you are authorized to access. Keep `.env` local;
Git ignores it.

Required endpoint settings are `LLM_FARM_ENDPOINT`, `LLM_FARM_API_KEY`, and
`LLM_FARM_CHAT_API_VERSION`. For a single experiment, set
`LLM_FARM_CHAT_DEPLOYMENT` or pass `--deployment`. To run the complete batch,
set each deployment mapping below to the gateway deployment that serves the
corresponding model family:

| Model family | Environment variable |
|---|---|
| GPT-4o | `ICL_DEPLOYMENT_GPT_4O` |
| GPT-4.1 | `ICL_DEPLOYMENT_GPT_4_1` |
| GPT-5.2 | `ICL_DEPLOYMENT_GPT_5_2` |
| GPT-5.4 | `ICL_DEPLOYMENT_GPT_5_4` |

`ICL_DATASET_FOLDER_ID` accepts a Drive folder ID or folder URL. The folder
must contain the authorized `train/` and `test/` CSV files for the ten enabled
tasks. The ICL code enumerates the files through Drive; it does not copy the
dataset into this repository. The `.env.example` contains only sample values.

## Run the Section 6.2 batch

From `in_context_learning/`, run:

```bash
python src/run_all.py
```

This schedules 4 model families × 10 datasets × 2 methods = 80 experiments.
It uses seed 42, task-specific shot counts, and the full test splits. Within
each model it allows four request workers and caps total concurrency at 16;
`direct` and `cot` run sequentially. API charges may apply. Rerunning the
command resumes verified incomplete runs and skips completed runs.

Results are written under
`results/all_models_icl-v3/<dataset>/<model-family>/`; CoT outputs use a
`cot/` subfolder. The batch summary is
`results/all_models_icl-v3/summary.json`. Predictions, manifests, metrics, and
logs are local experiment outputs and are excluded from Git.

To run one task/model family instead, for example:

```bash
python src/icl_inference.py \
  --dataset ViMedNLI_ViMedNLI \
  --model-family gpt-4o \
  --method direct --shots 6 \
  --output results/single/vimednli-gpt-4o
```

Use `--limit 20` for a small API smoke run and a fresh output folder. Omit
`--limit` for the full test split. `--prepare-only` saves the selected shots
without making a chat completion request, but still requires data access and a
configured client in the current CLI.

## Audit and tests

After a complete batch, create the Section 6.2 score table and ICL-v3 audit:

```bash
python src/audit_icl_v3.py
```

The audit requires the 80 complete runs under `results/all_models_icl-v3/`. It
checks manifests, summary metrics, and prediction counts. It writes the LaTeX
table and audit bundle to `analysis/` inside this folder; it does not write to
the manuscript directory. Inspect the generated table before deciding whether
to copy it into the paper.

Run the offline tests from `in_context_learning/`:

```bash
python -B -m unittest discover -s tests -p "test_*.py"
```

The tests use mocked clients and temporary fixtures; they do not call the API,
download datasets, or create repository predictions.

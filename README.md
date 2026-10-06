# Companion repository: Vietnamese medical NLP survey

This repository accompanies the survey *Towards Reliable Medical NLP in
Low-Resource Languages: A Systematic Survey of Vietnamese Healthcare Text
Processing*. It contains the review protocol, the corpus and codebook, the code
for the re-evaluation in Sections 6 and 7, and the aggregate results printed in
those sections.

```text
.
├── 01_review_protocol/        research questions and search strategy
├── 02_corpus_and_codebook/    included studies, codebook, taxonomy mapping, benchmark audit
├── 03_reevaluation/
│   ├── src/
│   │   ├── finetuning/          Llama 3.1 8B LoRA, one adapter per dataset
│   │   └── in_context_learning/ answer-only and explanation-request prompting
│   └── scripts/
│       ├── rebuild.py           recomputes every file in 04_results/ from the saved runs
│       └── test_rebuild.py
├── 04_results/                aggregate outputs behind the re-evaluation tables
├── CITATION.cff
└── LICENSE
```

## What is released, and what is not

**Released (`04_results/`).** Aggregates only; no dataset text, predictions, or
model outputs.

| File | Content |
|---|---|
| `bootstrap_intervals.csv` | 90 primary-score 95% intervals (10 fine-tuning runs, 80 ICL configurations; 400 resamples, seed 20260923) |
| `paired_prompt_differences.csv` | 40 paired explanation-minus-answer-only contrasts with 95% intervals |
| `fine_tuning_manifest.csv` | per-task run settings and SHA-256 of the saved config, metrics, and predictions files |
| `icl_run_manifest.csv` | per-configuration model and version, prompt condition (`answer_only` or `explanation_request`), decoding, SDK, prompt and demonstration fingerprints, experiment ID, SHA-256 of saved files, invalid and truncated counts |
| `error_counts_by_condition.csv`, `error_counts_by_configuration.csv` | error numerators and denominators, pooled and per configuration |
| `output_validity_by_configuration.csv`, `dataset_output_coverage.csv` | invalid and truncated outputs, explanation-format compliance, 21,418 test rows × 8 configurations = 171,344 predictions |
| `case_trace.csv` | row and test-hash references for the four examples in Section 7 |

The fine-tuning runs come from two training batches. Seven tasks are from the
first (`results_1.zip`); the three generation tasks (UIT-ViCoV19QA, ViMedAQA,
FAQ summarization) are from the second (`results_2.zip`). UIT-ViCoV19QA reached
20.60 ROUGE-L in the first batch; the reported 20.73 comes from the second.

ICL models are named `gpt-4o`, `gpt-4.1`, `gpt-5.2`, and `gpt-5.4`; the
`model_version` column of `icl_run_manifest.csv` gives the dated version. The
two prompt conditions are labelled `answer_only` and `explanation_request`.

**Recomputable only with the withheld runs.** Test-set scores, bootstrap
intervals, paired contrasts, and error counts are recomputed from the saved
fine-tuning archives and ICL predictions. Dataset text, predictions and adapters
are withheld because redistribution rights and record-level personal-information
review for these materials have not been completed. With them in place,
`rebuild.py --check` confirms that every released CSV is reproduced exactly and,
with `--table-dir`, that the manuscript tables match.

**Not reproducible from the saved runs.** BERTScore values are checked against
archived metric files only; the encoder revision was not pinned. The base-model
weight revision, the exact provider-side model snapshots, a dependency lock for
the fine-tuning runs, and full hashes of the input split files were not
recorded, and are not reconstructed from later installations.

The intervals describe variation over test rows conditional on the saved runs
(one fine-tuning seed, one demonstration selection). The 40 paired contrasts are
exploratory and not adjusted for multiple comparisons.

## Recompute

Install `numpy` and `rouge-score`, and the dependencies of the two pipelines
(`03_reevaluation/src/*/requirements.txt`). Then, with the saved runs in local
directories:

```bash
python 03_reevaluation/scripts/rebuild.py --check \
  --finetune-results <dir with results_1.zip and results_2.zip> \
  --icl-root <dir containing the per-dataset ICL run folders>
```

Drop `--check` to regenerate `04_results/`. If the model folders inside each
dataset folder are not named after the model aliases, pass `--run-dir-map` with
a JSON object mapping each alias to its folder name. The offline tests run with
`python -B -m unittest discover -s 03_reevaluation/scripts` and
`python -B -m unittest discover -s 03_reevaluation/src/in_context_learning/tests`.

## Running the experiments

Each pipeline has its own environment and data requirements; see
[`finetuning/`](03_reevaluation/src/finetuning/README.md) and
[`in_context_learning/`](03_reevaluation/src/in_context_learning/README.md).
Both expect you to obtain authorized benchmark data and place it in the local
layout those READMEs describe. Install each method's dependencies in its own
environment and run commands from the method folder. Generated results and
local configuration are excluded by `.gitignore`.

## Citation

If you use this code, use the software citation metadata in
[`CITATION.cff`](CITATION.cff).

## License

This project is licensed under the [MIT License](LICENSE).

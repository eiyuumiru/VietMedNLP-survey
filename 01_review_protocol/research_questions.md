# Research Questions

## Scope

The survey examines Vietnamese healthcare text-processing research published from 1 January 2021 through 31 May 2026. Its final corpus contains 33 canonical primary publications and 39 task-level annotations. A publication is counted once in publication summaries but may receive multiple task labels. Two studies with substantive Vietnamese healthcare text components and broader or audio-centered primary scopes are flagged as boundary cases.

The five questions guide evidence extraction and descriptive synthesis. RQ4 and RQ5 also use the controlled re-evaluation, which is distinct from the literature review. RQ5 focuses on in-context learning (ICL) outputs. None of the questions presupposes a causal effect.

## Research questions and evidence

| ID | Research question | Evidence considered | Paper section |
| --- | --- | --- | --- |
| **RQ1 — Landscape and taxonomy** | What task families, methodological configurations, and data-construction patterns characterize Vietnamese medical NLP, and how are they distributed across the reviewed corpus? | Included publications; task annotations and families; model backbone, system composition, adaptation strategy, evidence grounding, and data-construction pathway; and the relevant publication- and task-level denominators. | Section 3: Taxonomy of Vietnamese Medical NLP |
| **RQ2 — Data construction and quality** | How are Vietnamese medical datasets and knowledge resources sourced, constructed, and validated, and what access or documentation information is reported? | Reported provenance, construction and supervision, annotation agreement, expert involvement, evidence validation, access, documentation, and benchmark protocols. | Section 4: Datasets and Benchmarks |
| **RQ3 — Methods and conditions of effectiveness** | Which modeling paradigms are used, and under what task, data, and evidence conditions do they appear effective? | Modeling configurations and adaptation alongside the evaluated task, data provenance, evidence access, baselines, metrics, and protocol conditions. | Section 5: Methodological Configurations and Performance under Benchmark Conditions |
| **RQ4 — Comparability and reproducibility** | To what extent can reported results be compared across studies, and what does a controlled re-evaluation reveal when representative datasets are tested under a common framework? | Dataset and split definitions, inputs and outputs, preprocessing, evidence access, model and evaluator settings, metrics, and controlled re-evaluation results. | Section 4: Datasets and Benchmarks; Section 6: Reproducibility study |
| **RQ5 — Failure patterns and methodological implications** | Which output errors occur under the controlled ICL evaluation, and which configuration-level implications follow under the stated evaluation conditions? | ICL predictions and references, evaluation decisions, error counts and examples, prompt conditions, model configurations, and their relation to the methodological taxonomy. | Section 7: Error Analysis and Methodological Implications |

## Interpretation

- **Counting units:** each canonical publication contributes once to publication summaries. Distinct evaluated outputs can contribute multiple task annotations; the final counts are 33 publications and 39 task annotations.
- **Sensitivity analysis:** publication- and task-level summaries are examined for the full corpus, the core-only corpus excluding the two boundary studies, the peer-reviewed-only corpus retaining archival peer-reviewed publications, and the strict-core corpus applying both restrictions. These restrictions are not applied to the separate 25-resource benchmark audit or the controlled re-evaluation.
- **Comparability:** reported results are interpreted with their task, data, evidence, output, metric, and protocol conditions. Scores from incompatible settings are not pooled into a cross-study ranking.
- **Clinical scope:** agreement with benchmark references is distinct from clinical correctness or readiness for deployment. The controlled re-evaluation does not isolate the causal effect of fine-tuning versus prompting.

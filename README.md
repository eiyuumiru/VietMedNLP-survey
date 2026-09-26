# Reproducibility source for Sections 6.1 and 6.2

This repository contains the experiment code accompanying the fine-tuning and
in-context learning evaluations in Sections 6.1 and 6.2 of the survey
*Towards Reliable Medical NLP in Low-Resource Languages: A Systematic Survey of
Vietnamese Healthcare Text Processing*.

The two experiment pipelines have separate dependencies and data access:

- [`finetuning/`](finetuning/README.md) trains one Llama 3.1 8B LoRA adapter per
  dataset and evaluates on its test split.
- [`in_context_learning/`](in_context_learning/README.md) runs direct few-shot
  and few-shot explanation-and-answer prompting with four model families.

Both pipelines expect you to obtain authorized benchmark data and place it in
the local layout described by each README. The repository contains no
datasets, predictions, trained adapters, API credentials, or private Drive
folder identifiers. Configure data access locally as described in each
folder's README.

## Quick orientation

```text
source/
├── CITATION.cff
├── finetuning/
│   ├── README.md
│   ├── requirements.txt
│   └── app/
└── in_context_learning/
    ├── README.md
    ├── PROMPTS.md
    ├── requirements.txt
    ├── .env.example
    ├── src/
    └── tests/
```

Install each method's dependencies in its own environment. Run commands from
the relevant method folder so output paths stay with the corresponding
pipeline. Generated results and local configuration are excluded by
`.gitignore`.

## Code availability statement for the paper

After this repository has an approved public URL, the following wording can be
adapted for the manuscript:

> The source code for the Section 6.1 fine-tuning and Section 6.2 in-context
> learning experiments is available at **[repository URL]**. Access to the
> benchmark data is subject to the dataset providers' terms.

The bracketed URL is documentation guidance only; no placeholder was added to
the manuscript. This local repository has no `LICENSE` file pending confirmation
of the authors' and organizations' rights to distribute the code. Add a license
and publish the repository only after that review.

## Citation

`CITATION.cff` records the article authors for repository citation metadata.
Add the approved repository URL, version, and DOI there when those become
available. No GitHub remote or release is configured in this local copy.

# Reproducibility source for Sections 6.1 and 6.2

This repository contains the experiment code accompanying the fine-tuning and
in-context learning evaluations in Sections 6.1 and 6.2 of the survey
*Towards Reliable Medical NLP in Low-Resource Languages: A Systematic Survey of
Vietnamese Healthcare Text Processing*.

The two experiment pipelines have separate dependencies and data access:

- [`finetuning/`](finetuning/README.md) trains one Llama 3.1 8B LoRA adapter per
  dataset and evaluates on its test split.
- [`in_context_learning/`](in_context_learning/README.md) runs answer-only
  and explanation-and-answer prompting with four model profiles.

Both pipelines expect you to obtain authorized benchmark data and place it in
the local layout described by each README. The repository contains no
datasets, predictions, trained adapters, or API credentials. Configure data
access locally as described in each folder's README. Historical run manifests,
bootstrap interval tables, and review records are not included.

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

## Citation

If you use this code, use the software citation metadata in
[`CITATION.cff`](CITATION.cff).

## License

This project is licensed under the [MIT License](LICENSE).

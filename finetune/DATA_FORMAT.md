# Fine-tuning data format

The Section 6.1 runner consumes the authorized HEALTHDOMAIN
`Instruct_Datasets` release. It does not download datasets or construct this
release from raw benchmark files.

Set `--data-root` to a directory with this layout:

```text
Instruct_Datasets/
├── train/<dataset>.csv
├── dev/<dataset>.csv
└── test/<dataset>.csv
```

`<dataset>` is one of the names printed by:

```bash
python -m app.run --list-datasets
```

Each CSV must contain exactly these ten columns:

```text
input, output,
instruction_prompt_json, output_prompt_json,
instruction_prompt_nl, output_prompt_nl,
instruction_prompt_ps, output_prompt_ps,
task, dataset
```

The training and evaluation runner uses only `instruction_prompt_nl` as the
user message and `output_prompt_nl` as the target. The other columns are kept
because they belong to the validated HEALTHDOMAIN release. Values must be
non-empty strings after trimming; `None` remains a valid literal target for
NER examples.

Run this check before training:

```bash
python -m app.validate_data --data-root /path/to/Instruct_Datasets
```

Keep all CSV files outside the source repository. The data provider's access
terms govern redistribution and any conversion from original benchmark files.

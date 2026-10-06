# Fine-tuning data format

The fine-tuning branch (Section 6) consumes authorized
instruction-formatted copies of the ten benchmark datasets (one instruction
prompt and one target response per row). It does not download datasets or
construct these copies from raw benchmark files.

Set `--data-root` to a directory with this layout:

```text
instruction-data/
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
because they are part of the instruction-formatted copies. Values must be
non-empty strings after trimming; `None` remains a valid literal target for
NER examples.

Run this check before training:

```bash
python -m app.validate_data --data-root /path/to/instruction-data
```

Keep all CSV files outside the source repository. The data provider's access
terms govern redistribution and any conversion from original benchmark files.

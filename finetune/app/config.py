"""Central configuration: dataset registry, task -> metric mapping, model defaults.

A single Llama model is instruction-tuned once per dataset. Each dataset is described
by a `DatasetSpec` that tells the pipeline (a) which CSV file to read, (b) what kind of
task it is, (c) which metric is primary, and (d) sensible sequence / generation lengths.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------------------
# Model defaults
# --------------------------------------------------------------------------------------

# Unsloth-optimized Llama-3.1-8B-Instruct (usually ungated). load_in_4bit=True auto-uses
# the matching 4-bit build; load_in_4bit=False loads it in 16-bit for bf16 LoRA (A100).
# The original gated weights "meta-llama/Llama-3.1-8B-Instruct" also work (needs HF_TOKEN).
DEFAULT_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"

# LoRA is applied to all linear projections of the attention + MLP blocks.
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# We only ever use the natural-language prompt columns (confirmed for this study).
PROMPT_STYLE = "nl"
INSTRUCTION_COLUMN = "instruction_prompt_nl"
TARGET_COLUMN = "output_prompt_nl"

# The strict 10-column schema every task CSV must follow (see dataset documentation).
EXPECTED_COLUMNS = [
    "input",
    "output",
    "instruction_prompt_json",
    "output_prompt_json",
    "instruction_prompt_nl",
    "output_prompt_nl",
    "instruction_prompt_ps",
    "output_prompt_ps",
    "task",
    "dataset",
]

# Strings that mean "no answer / empty target" (case-insensitive) across tasks.
EMPTY_TARGET_SENTINELS = {"", "none", "nan", "null", "[]", "{}", "n/a", "na"}

# Optional short system message prepended via the chat template. Set to "" to disable.
SYSTEM_PROMPT = (
    "Bạn là trợ lý xử lý ngôn ngữ y khoa tiếng Việt. "
    "Hãy đọc kỹ yêu cầu và chỉ trả về kết quả đúng định dạng được yêu cầu, không giải thích thêm."
)

# Fixed date passed to the Llama-3.1 chat template. Without it, the template injects the
# wall-clock "Today Date" via strftime_now, making rendered prompts (and thus results)
# non-reproducible across days. Pinning it keeps every run byte-identical.
CHAT_DATE_STRING = "26 Jul 2024"


# --------------------------------------------------------------------------------------
# Task types and their primary metric
# --------------------------------------------------------------------------------------
# task_type -> primary metric key produced by metrics.py
PRIMARY_METRIC = {
    "ner": "entity_micro_f1",
    "classification": "macro_f1",
    "nli": "accuracy",
    "mcqa": "accuracy",
    "extractive_qa": "squad_f1",
    "generation": "rougeL",
}


@dataclass
class DatasetSpec:
    """Description of one task dataset (one CSV per split)."""

    stem: str                # file name without extension, e.g. "ViMedNLI_ViMedNLI"
    display: str             # human-readable name for tables
    task_type: str           # key into PRIMARY_METRIC
    max_seq_len: int = 1024  # max tokens for the (prompt + target) training example
    max_new_tokens: int = 128  # generation budget at eval time
    # Reference best result reported in the survey (for context only; methods differ).
    paper_sota: dict | None = None

    @property
    def primary_metric(self) -> str:
        return PRIMARY_METRIC[self.task_type]


# --------------------------------------------------------------------------------------
# Dataset registry — the 12 task CSVs in New_Data/Instruct_Datasets/{train,dev,test}/
# `paper_sota` values are stored as fractions in [0, 1]; "method" names the SOTA system
# from the survey tables (often a different paradigm than our fine-tuned Llama).
# --------------------------------------------------------------------------------------
DATASETS: dict[str, DatasetSpec] = {
    spec.stem: spec
    for spec in [
        # ---- Named Entity Recognition ----
        DatasetSpec(
            "PhoNER_COVID19_NER", "PhoNER-COVID19 (NER)", "ner",
            max_seq_len=1024, max_new_tokens=256,
            paper_sota={"metric": "Micro-F1", "value": 0.9476, "method": "ViPubmedDeBERTa-base"},
        ),
        DatasetSpec(
            "ViMedNER_NER", "ViMedNER (NER)", "ner",
            max_seq_len=1024, max_new_tokens=256,
            paper_sota={"metric": "Micro-F1", "value": 0.725, "method": "XLM-R-large"},
        ),
        DatasetSpec(
            "ViMQ_NER", "ViMQ (NER)", "ner",
            max_seq_len=1024, max_new_tokens=256,
            paper_sota={"metric": "Micro-F1", "value": 0.8065, "method": "ViPubmedDeBERTa-base"},
        ),
        # ---- Text classification ----
        DatasetSpec(
            "ViMQ_intent_classification", "ViMQ (Intent)", "classification",
            max_seq_len=512, max_new_tokens=32,
            paper_sota={"metric": "Macro-F1", "value": 0.9165, "method": "PhoBERT (self-sup.)"},
        ),
        DatasetSpec(
            "ViMedNLI_ViMedNLI", "ViMedNLI (NLI)", "nli",
            max_seq_len=1024, max_new_tokens=16,
            paper_sota={"metric": "Accuracy", "value": 0.8165, "method": "ViPubmedT5"},
        ),
        DatasetSpec(
            "vihealthbert_acrDrAid", "acrDrAid (Acronym)", "classification",
            max_seq_len=1024, max_new_tokens=64,
            paper_sota={"metric": "Macro-F1", "value": 0.8696, "method": "ViPubmedDeBERTa"},
        ),
        DatasetSpec(
            "VMHQA_Multiple_Choice_QA", "VMHQA (MC-QA)", "mcqa",
            max_seq_len=2048, max_new_tokens=32,
            paper_sota={"metric": "Accuracy", "value": 0.9150, "method": "Qwen-7B (LoRA)"},
        ),
        # ---- Extractive QA (SQuAD-style) ----
        DatasetSpec(
            "ViNewsQA_Extractive_QA", "ViNewsQA (Extractive QA)", "extractive_qa",
            max_seq_len=2048, max_new_tokens=128,
            paper_sota={"metric": "F1", "value": 0.9184, "method": "XLM-R + BiLSTM"},
        ),
        # ---- Generation (community/abstractive QA, summarization, paraphrase) ----
        DatasetSpec(
            "UIT-ViCoV19QA_QA", "UIT-ViCoV19QA (QA)", "generation",
            max_seq_len=2048, max_new_tokens=256,
            paper_sota={"metric": "ROUGE-L", "value": 0.3395, "method": "RNN-2 (Luong attn.)"},
        ),
        DatasetSpec(
            "ViMedAQA_Abstract_QA", "ViMedAQA (Abstractive QA)", "generation",
            max_seq_len=2048, max_new_tokens=256,
            paper_sota={"metric": "ROUGE-L", "value": 0.5989, "method": "VinaLlama-7B"},
        ),
        DatasetSpec(
            "vihealthbert_Summarization", "ViHealthBERT-FAQ (Summarization)", "generation",
            max_seq_len=2048, max_new_tokens=256,
            paper_sota={"metric": "ROUGE-L", "value": 0.4385, "method": "ViHealthBERT (word, MLM)"},
        ),
        DatasetSpec(
            "ViSP_Sentence_Paraphrases", "ViSP (Paraphrase)", "generation",
            max_seq_len=1024, max_new_tokens=128,
            paper_sota={"metric": "ROUGE-2", "value": 0.7578, "method": "BARTpho-word-large"},
        ),
    ]
}


def get_spec(dataset: str) -> DatasetSpec:
    """Resolve a dataset by its file stem (case-insensitive, tolerant of .csv)."""
    key = dataset.strip()
    if key.lower().endswith(".csv"):
        key = key[:-4]
    if key in DATASETS:
        return DATASETS[key]
    # tolerant lookup
    lowered = {k.lower(): k for k in DATASETS}
    if key.lower() in lowered:
        return DATASETS[lowered[key.lower()]]
    raise KeyError(
        f"Unknown dataset '{dataset}'. Available: {', '.join(DATASETS)}"
    )


def list_datasets() -> list[str]:
    return list(DATASETS)


# --------------------------------------------------------------------------------------
# Training / evaluation hyper-parameter container
# --------------------------------------------------------------------------------------
@dataclass
class RunConfig:
    """All knobs for one (dataset) run. Populated from CLI in run.py."""

    dataset: str
    data_root: str                      # folder holding train/ dev/ test/ subdirs
    output_root: str = "results"
    model_name: str = DEFAULT_MODEL

    # data
    train_split: str = "train"
    eval_split: str = "test"
    val_split: str = "dev"
    max_seq_len: int | None = None      # None -> use DatasetSpec default
    max_new_tokens: int | None = None   # None -> use DatasetSpec default
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    # Use the dev split during training for eval_loss + best-checkpoint selection.
    eval_during_train: bool = True
    max_val_samples: int | None = 1000  # cap dev (e.g. ViSP dev has ~391k rows)

    # quantization / LoRA. These defaults match the CLI defaults.
    use_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # optimization
    epochs: float = 3.0
    max_steps: int = -1                 # >0 overrides epochs
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    per_device_eval_batch_size: int = 8
    no_auto_batch: bool = False   # True: honor batch sizes as-is (don't shrink for seq 2048)
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 20
    save_adapter: bool = True

    # misc
    seed: int = 42
    bf16: bool = True
    use_bertscore: bool = False
    merge_and_save: bool = False        # merge LoRA into base and save full model

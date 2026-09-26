"""Offline protocol checks for the in-context learning implementation."""

import json
import io
import os
import sys
import tempfile
import threading
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from urllib3.response import HTTPResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import icl_inference as runner
import run_all
from icl_inference import select_examples
from icl_tasks import TASKS, cot_compliance, evaluate, messages, parse_answer


class ProtocolTests(unittest.TestCase):
    def test_vmhqa_uses_its_extractable_gold_answer(self):
        kind, _, labels, _ = TASKS["VMHQA_Multiple_Choice_QA"]
        self.assertEqual(kind, "extractive")
        self.assertFalse(labels)

    def test_input_content_filter_is_a_scored_invalid_response(self):
        error = RuntimeError("prompt rejected: content_filter")
        self.assertTrue(runner.is_input_content_filter(error))

    def test_checkpoint_tail_recovery_preserves_original_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            good = b'{"row": 0}\n'
            torn = b'{"row": 1, "text": "\xe1\xbb'
            path.write_bytes(good + torn)
            runner.recover_checkpoint_tail(path)
            self.assertEqual(path.read_bytes(), good)
            backups = list(Path(directory).glob("*.bak"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), torn)

            complete = b'{"row": 0}'
            path.write_bytes(complete)
            runner.recover_checkpoint_tail(path)
            self.assertEqual(path.read_bytes(), complete + b"\n")

            for corrupt in (b"broken\n" + good, good + b"broken\n"):
                path.write_bytes(corrupt)
                with self.assertRaisesRegex(ValueError, "manual inspection"):
                    runner.recover_checkpoint_tail(path)
                self.assertEqual(path.read_bytes(), corrupt)

    def test_fingerprints_cover_example_order_and_full_prompt(self):
        task = TASKS["ViMedNLI_ViMedNLI"]
        examples = [
            {"input": "a", "output": "neutral"},
            {"input": "b", "output": "entailment"},
        ]
        direct = runner.experiment_fingerprints(examples, task, "direct")
        self.assertNotEqual(
            direct,
            runner.experiment_fingerprints(examples[::-1], task, "direct"),
        )
        self.assertNotEqual(
            direct, runner.experiment_fingerprints(examples, task, "cot")
        )

    def test_decimal_diagnostic_does_not_silently_replace_benchmark_metric(
        self,
    ):
        score = evaluate(
            [{"gold": "1.5 mg", "prediction": "15 mg", "valid": True}],
            TASKS["ViNewsQA_Extractive_QA"],
        )
        self.assertEqual(score["metrics"]["exact_match"], 100)
        self.assertEqual(score["diagnostics"]["punctuation_preserving_em"], 0)
        self.assertEqual(
            score["diagnostics"]["punctuation_preserving_token_f1"], 50
        )
        self.assertIn("unverified", score["benchmark_comparability"])

    def test_cot_compliance_does_not_change_answer_accuracy(self):
        rows = [
            {
                "gold": "neutral",
                "prediction": "neutral",
                "valid": True,
                "raw_prediction": '{"answer":"neutral"}',
            },
            {
                "gold": "neutral",
                "prediction": "neutral",
                "valid": True,
                "raw_prediction": '{"answer":"neutral","explanation":"Thiếu thông tin."}',
            },
        ]
        score = evaluate(rows, TASKS["ViMedNLI_ViMedNLI"], "cot")
        self.assertEqual(score["metrics"]["accuracy"], 100)
        self.assertEqual(score["cot_format"]["rate_percent"], 50)
        self.assertFalse(
            cot_compliance('{"answer":"neutral","explanation":[]}')
        )
        self.assertFalse(
            cot_compliance('{"answer":"neutral","explanation":" "}')
        )

    def test_decoding_caps_all_four_models_without_changing_cot_budget(self):
        for model in run_all.MODELS:
            options = runner.decoding_options(model)
            self.assertEqual(options["max_completion_tokens"], 8192)
            if model.startswith("gpt-5."):
                self.assertEqual(options["reasoning_effort"], "none")
                self.assertNotIn("temperature", options)
            else:
                self.assertEqual(options["temperature"], 0)

    def test_empty_evaluation_is_rejected(self):
        for task in TASKS.values():
            with self.assertRaisesRegex(ValueError, "empty test"):
                evaluate([], task)

    def test_run_rejects_empty_short_and_over_budget_inputs_and_counts_truncation(
        self,
    ):
        dataset = "ViMedNLI_ViMedNLI"
        examples = [
            {"input": f"Train {label}", "output": label}
            for label in TASKS[dataset][2]
        ]
        for case in (
            "empty",
            "short",
            "over_budget",
            "truncated",
            "input_filtered",
        ):
            with self.subTest(
                case=case
            ), tempfile.TemporaryDirectory() as directory:
                rows = (
                    []
                    if case == "empty"
                    else [{"input": "Test input", "output": "neutral"}]
                )

                @contextmanager
                def remote(file_id):
                    yield iter(rows)

                client = MagicMock()
                client.chat.completions.create.return_value = SimpleNamespace(
                    usage=None,
                    choices=[
                        SimpleNamespace(
                            finish_reason=(
                                "length" if case == "truncated" else "stop"
                            ),
                            message=SimpleNamespace(
                                content='{"answer":"neutral"}'
                            ),
                        )
                    ],
                )
                if case == "input_filtered":
                    client.chat.completions.create.side_effect = RuntimeError(
                        "prompt rejected: content_filter"
                    )
                args = SimpleNamespace(
                    dataset=dataset,
                    deployment="test-model",
                    shots=3,
                    output=Path(directory),
                    seed=42,
                    method="direct",
                    limit=2 if case == "short" else None,
                    max_prompt_chars=60000,
                    resume=True,
                    prepare_only=False,
                )
                files = {
                    f"{split}/{dataset}.csv": split
                    for split in ("train", "test")
                }
                with patch.object(
                    runner, "make_client", return_value=(client, "test-model")
                ), patch.object(runner, "remote_csv", remote), patch.object(
                    runner,
                    "MAX_PROMPT_TOKENS",
                    1 if case == "over_budget" else 32000,
                ), redirect_stdout(
                    io.StringIO()
                ):
                    if case not in {"truncated", "input_filtered"}:
                        with self.assertRaises(ValueError):
                            runner.run_experiment(args, files, (examples, {}))
                        self.assertFalse(
                            (Path(directory) / "metrics.json").exists()
                        )
                        if case in {"empty", "over_budget"}:
                            client.chat.completions.create.assert_not_called()
                    else:
                        runner.run_experiment(args, files, (examples, {}))
                        score = json.loads(
                            (Path(directory) / "metrics.json").read_text(encoding="utf-8")
                        )
                        self.assertEqual(score["n_scored"], 1)
                        self.assertEqual(score["n_invalid"], 1)
                        self.assertEqual(
                            score["n_truncated"], case == "truncated"
                        )
                        self.assertEqual(score["metrics"]["accuracy"], 0)

    def test_four_workers_overlap_but_results_keep_test_order(self):
        barrier = threading.Barrier(4)
        lock = threading.Lock()
        active = peak = 0

        def work(index):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            barrier.wait(timeout=5)
            with lock:
                active -= 1
            return index

        self.assertEqual(
            list(runner.ordered_results(work, range(8), 4)), list(range(8))
        )
        self.assertEqual(peak, 4)

    def test_batch_covers_all_models_and_datasets(self):
        def fake_run(args, files, prepared=None):
            args.output.mkdir(parents=True, exist_ok=True)
            runner.write_json(
                args.output / "experiment.json",
                {"examples": [], "sampling": {}},
            )
            runner.write_json(
                args.output / "metrics.json",
                {"metrics": {"accuracy": 100}, "n_scored": 2, "n_invalid": 0},
            )

        with tempfile.TemporaryDirectory() as directory:
            environment = {
                "ICL_DATASET_FOLDER_ID": "https://example.invalid/test-datasets",
                "ICL_DEPLOYMENT_GPT_4O": "test-gpt-4o",
                "ICL_DEPLOYMENT_GPT_4_1": "test-gpt-4.1",
                "ICL_DEPLOYMENT_GPT_5_2": "test-gpt-5.2",
                "ICL_DEPLOYMENT_GPT_5_4": "test-gpt-5.4",
            }
            with patch.object(
                run_all, "RESULTS", Path(directory)
            ), patch.object(
                run_all, "ROOT", Path(directory)
            ), patch.object(
                run_all.gdown, "download_folder", return_value=[]
            ), patch.object(
                run_all, "run_experiment", side_effect=fake_run
            ) as run, patch.dict(os.environ, environment), redirect_stdout(
                io.StringIO()
            ):
                run_all.main()
            inference_calls = [
                call
                for call in run.call_args_list
                if not call.args[0].prepare_only
            ]
            pairs = {
                (call.args[0].dataset, call.args[0].model_family)
                for call in inference_calls
            }
            self.assertEqual(
                pairs,
                {
                    (dataset, model)
                    for dataset in TASKS
                    for model in run_all.MODELS
                },
            )
            self.assertEqual(len(inference_calls), 80)
            combinations = {
                (
                    call.args[0].dataset,
                    call.args[0].model_family,
                    call.args[0].method,
                )
                for call in inference_calls
            }
            self.assertEqual(
                combinations,
                {
                    (dataset, model, method)
                    for dataset in TASKS
                    for model in run_all.MODELS
                    for method in run_all.METHODS
                },
            )
            self.assertEqual(
                len({call.args[0].output for call in inference_calls}), 80
            )
            for dataset in TASKS:
                selected = [
                    call.args[2]
                    for call in inference_calls
                    if call.args[0].dataset == dataset
                ]
                self.assertTrue(all(item is selected[0] for item in selected))
            self.assertTrue(
                all(
                    call.args[0].workers == run_all.WORKERS_PER_MODEL
                    for call in inference_calls
                )
            )
            summary = json.loads(
                (Path(directory) / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(summary), 80)

    def test_cot_uses_same_examples_and_scores_only_answer(self):
        for task in TASKS.values():
            kind, _, labels, _ = task
            gold = (
                f"{labels[0]}: bệnh"
                if kind == "ner"
                else labels[0] if labels else "bệnh tim"
            )
            examples = [{"input": "Train input", "output": gold}]
            direct = messages("Test input", examples, task, "direct")
            cot = messages("Test input", examples, task, "cot")
            self.assertEqual(direct[1:], cot[1:])
            self.assertIn('"explanation"', cot[0]["content"])
            raw = json.dumps(
                {"explanation": "Unrelated explanation", "answer": gold}
            )
            self.assertEqual(parse_answer(raw, task), (gold, True))
            self.assertEqual(
                parse_answer(json.dumps({"explanation": gold}), task),
                ("", False),
            )

    def test_remote_csv_reads_multiline_cells_through_eof(self):
        payload = 'input,output\n"first\nsecond",label\nlast,label\n'.encode()
        raw = HTTPResponse(body=io.BytesIO(payload), preload_content=False)
        response = MagicMock(raw=raw, headers={"Content-Type": "text/csv"})
        with patch.object(runner.requests, "Session") as session:
            session.return_value.__enter__.return_value.get.return_value = (
                response
            )
            with runner.remote_csv("test") as rows:
                result = list(rows)
        self.assertEqual(
            result,
            [
                {"input": "first\nsecond", "output": "label"},
                {"input": "last", "output": "label"},
            ],
        )

    def test_remote_csv_uses_usercontent_after_drive_403(self):
        blocked = MagicMock(status_code=403)
        blocked.raise_for_status.side_effect = runner.requests.HTTPError("403")
        payload = b"input,output\nquestion,label\n"
        raw = HTTPResponse(body=io.BytesIO(payload), preload_content=False)
        downloaded = MagicMock(raw=raw, headers={"Content-Type": "text/csv"})
        with patch.object(runner.requests, "Session") as session:
            session.return_value.__enter__.return_value.get.side_effect = [
                blocked,
                downloaded,
            ]
            with runner.remote_csv("test") as rows:
                self.assertEqual(list(rows), [{"input": "question", "output": "label"}])
        self.assertEqual(
            session.return_value.__enter__.return_value.get.call_count, 2
        )

    def test_balance_reaches_late_minority_and_is_reproducible(self):
        rows = [
            {"input": f"Câu hỏi {i}", "output": "entailment"}
            for i in range(400)
        ]
        rows += [
            {"input": f"Khác {label} {i}", "output": label}
            for label in ("contradiction", "neutral")
            for i in range(10)
        ]
        task = TASKS["ViMedNLI_ViMedNLI"]
        first = select_examples(iter(rows), task, 6, 42)
        self.assertEqual(first, select_examples(iter(rows), task, 6, 42))
        self.assertEqual(
            first[1]["selected_groups"],
            {"entailment": 2, "contradiction": 2, "neutral": 2},
        )

    def test_ner_updates_all_types_and_deduplicates_examples(self):
        task = ("ner", 2, ("A", "B", "C"), "NER")
        rows = [
            {"input": "alpha beta", "output": "A: alpha, B: beta"},
            {"input": "gamma", "output": "C: gamma"},
        ]
        examples, stats = select_examples(iter(rows), task, 2, 42)
        self.assertEqual(len({row["input"] for row in examples}), 2)
        self.assertEqual(stats["selected_groups"], {"a": 1, "b": 1, "c": 1})

    def test_prompt_has_train_gold_but_no_test_gold_parameter(self):
        task = TASKS["ViMedNLI_ViMedNLI"]
        example = {"input": "TRAIN INPUT", "output": "entailment"}
        prompt = messages("TEST INPUT", [example], task, "direct")
        self.assertEqual(prompt[-1], {"role": "user", "content": "TEST INPUT"})
        self.assertEqual(
            json.loads(prompt[2]["content"]), {"answer": "entailment"}
        )

    def test_invalid_outputs_stay_in_denominator(self):
        task = TASKS["ViMedNLI_ViMedNLI"]
        self.assertEqual(
            parse_answer('{"answer": "unknown"}', task), ("", False)
        )
        records = [
            {"gold": "entailment", "prediction": "entailment", "valid": True},
            {"gold": "neutral", "prediction": "", "valid": False},
        ]
        score = evaluate(records, task)
        self.assertEqual(score["metrics"]["accuracy"], 50)
        self.assertEqual(score["n_invalid"], 1)

    def test_exact_gold_scores_for_every_task(self):
        for dataset, task in TASKS.items():
            kind, _, labels, _ = task
            gold = (
                f"{labels[0]}: bệnh"
                if kind == "ner"
                else labels[0] if labels else "bệnh tim"
            )
            raw = json.dumps({"answer": gold}, ensure_ascii=False)
            pred, valid = parse_answer(raw, task)
            score = evaluate(
                [{"gold": gold, "prediction": pred, "valid": valid}], task
            )
            self.assertTrue(valid, dataset)
            primary = (
                "micro_f1"
                if kind in {"ner", "intent"}
                else (
                    "macro_f1"
                    if kind == "acronym"
                    else (
                        "accuracy"
                        if labels
                        else (
                            "exact_match"
                            if kind == "extractive"
                            else "rouge_l_f1"
                        )
                    )
                )
            )
            self.assertEqual(score["metrics"][primary], 100, dataset)

    def test_qa_whitespace_normalization(self):
        score = evaluate(
            [{"gold": "Bệnh,  tim", "prediction": "bệnh tim", "valid": True}],
            TASKS["ViNewsQA_Extractive_QA"],
        )
        self.assertEqual(score["metrics"]["exact_match"], 100)

    def test_api_failure_then_resume_preserves_ground_truth_alignment(self):
        train = [
            {"input": f"Train {label}", "output": label}
            for label in ("entailment", "contradiction", "neutral")
        ]
        test = [
            {"input": "Test one", "output": "entailment"},
            {"input": "Test two", "output": "neutral"},
        ]

        @contextmanager
        def remote(file_id):
            yield iter(train if file_id == "train" else test)

        def response(answer):
            return SimpleNamespace(
                usage=None,
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(
                            content=json.dumps({"answer": answer})
                        ),
                    )
                ],
            )

        dataset = "ViMedNLI_ViMedNLI"
        files = [
            SimpleNamespace(path=f"{split}/{dataset}.csv", id=split)
            for split in ("train", "test")
        ]
        client = MagicMock()
        with tempfile.TemporaryDirectory() as directory:
            args = [
                "icl",
                "--dataset",
                dataset,
                "--shots",
                "3",
                "--output",
                directory,
            ]
            with patch.dict(os.environ, {"ICL_DATASET_FOLDER_ID": "https://example.invalid/test-datasets"}), patch.object(runner, "ROOT", Path(directory)), patch.object(
                runner, "remote_csv", remote
            ), patch.object(
                runner, "make_client", return_value=(client, "test-model")
            ), patch.object(
                runner.gdown, "download_folder", return_value=files
            ), redirect_stdout(
                io.StringIO()
            ):
                client.chat.completions.create.side_effect = [
                    response("entailment"),
                    RuntimeError("API down"),
                ]
                with patch("sys.argv", args), self.assertRaisesRegex(
                    RuntimeError, "API down"
                ):
                    runner.main()
                self.assertFalse((Path(directory) / "metrics.json").exists())
                self.assertEqual(
                    len(
                        list(
                            runner.read_jsonl(
                                Path(directory) / "predictions.jsonl"
                            )
                        )
                    ),
                    1,
                )
                manifest_path = Path(directory) / "experiment.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                modified = json.loads(manifest_path.read_text(encoding="utf-8"))
                modified["examples"][0]["input"] = "tampered input"
                runner.write_json(manifest_path, modified)
                with patch(
                    "sys.argv", args + ["--resume"]
                ), self.assertRaisesRegex(
                    ValueError, "Prompt/examples changed"
                ):
                    runner.main()
                runner.write_json(manifest_path, manifest)

                def changed_prompt(*values):
                    prompt = messages(*values)
                    prompt[0]["content"] += " Changed CoT instruction."
                    return prompt

                with patch("sys.argv", args + ["--resume"]), patch.object(
                    runner, "messages", changed_prompt
                ), self.assertRaisesRegex(
                    ValueError, "Prompt/examples changed"
                ):
                    runner.main()
                with patch("sys.argv", args + ["--resume"]), patch.object(
                    runner, "EVALUATOR_VERSION", "changed"
                ), self.assertRaisesRegex(
                    ValueError, "configuration/code differs"
                ):
                    runner.main()

                client.chat.completions.create.side_effect = [
                    response("neutral")
                ]
                with patch("sys.argv", args + ["--resume"]):
                    runner.main()
                score = json.loads(
                    (Path(directory) / "metrics.json").read_text(encoding="utf-8")
                )
                self.assertEqual(score["n_scored"], 2)
                self.assertEqual(score["metrics"]["accuracy"], 100)
                records = list(
                    runner.read_jsonl(Path(directory) / "predictions.jsonl")
                )
                self.assertEqual(
                    [r["gold"] for r in records], ["entailment", "neutral"]
                )
                self.assertTrue(
                    all(
                        r["experiment_id"] == score["experiment_id"]
                        for r in records
                    )
                )
                self.assertEqual(score["n_test_rows"], 2)
                self.assertEqual(
                    client.chat.completions.create.call_args.kwargs[
                        "max_completion_tokens"
                    ],
                    8192,
                )
                calls_before = client.chat.completions.create.call_count
                with patch("sys.argv", args + ["--resume"]):
                    runner.main()
                self.assertEqual(
                    client.chat.completions.create.call_count, calls_before
                )
                records[0]["prediction"] = "contradiction"
                prediction_path = Path(directory) / "predictions.jsonl"
                prediction_path.write_text(
                    "".join(json.dumps(row) + "\n" for row in records)
                )
                with patch(
                    "sys.argv", args + ["--resume"]
                ), self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    runner.main()


if __name__ == "__main__":
    unittest.main()

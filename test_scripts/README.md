# Expanded-release evaluation

Run from the repository root with Python, `tqdm` and `requests` installed.
No data-generation pipeline or CLOCQ service is needed to evaluate this release.

```sh
python test_scripts/test.py --auto_eval_with_openai --openai_base_url http://localhost:8000/v1 --openai_model YOUR_SERVED_MODEL --output_dir runs/model/context
python test_scripts/nocontext_test.py --auto_eval_with_openai --openai_base_url http://localhost:8000/v1 --openai_model YOUR_SERVED_MODEL --output_dir runs/model/nocontext
```

Both commands default to the complete `benchmark/test.json` (1,214 turns), with
few-shot examples exclusively from `benchmark/train.json`. Use `--input
benchmark/dev.json` for development, or `--train_input` for another training
file. IDs are local to each split. Merged `full.json` is rejected.
Use a different output directory for each model, condition and repetition.
Relative `--preds_out` and `--log_path` paths resolve inside `--output_dir`.

Evaluate saved structured predictions or raw JSONL without model calls:

```sh
python test_scripts/test.py --eval_test_preds path/to/predictions.json --output_dir runs/rescore
```

Predictions must identify turns in the selected evaluation split. Missing turns
count as errors; duplicate or unknown prediction IDs are rejected. With
`--max_test_turns N`, only the first N turns are evaluated, so supply predictions
for that same subset. This option is for smoke tests, not final results.

## Protocol retained from the existing scripts

- Context evaluation uses `question` and previous **questions**, not previous
  answers. The standalone condition uses `rephrased_question` without history.
- Default `--shots_conversations 3` means three complete training conversations,
  not three individual question/answer pairs. Context examples may be trimmed
  by the token budget; standalone examples are not trimmed. Set an adequate
  `--shots_token_budget` and `--model_context_tokens` for controlled comparisons.
  Actual example turn counts and arguments are recorded in `run_manifest.json`.
- Entity EM and the original permissive Fuzzy score are both retained. They are
  not a newly implemented SQuAD scoring protocol.
- TCR uses inclusive days: intersection length / predicted interval length.
  TCR accuracy uses threshold 0.5. Relation accuracy excludes first turns.
  Context reports include S1-S4 groups, using multi-label assignment by default.
- Model errors produce empty predictions and remain in the metric denominator.

Each run writes predictions (for inference), `evaluation_report.json`,
`run_manifest.json`, and `test_correct.json`; optional raw dumps are kept there.
The CLI does not train models. Fine-tuning and strategy-removal training code
is not included in this repository. Expanded training requires new checkpoints;
existing prompted baselines do not use all training turns automatically.

Run offline compatibility and metric regression checks:

```sh
python -m unittest discover -s test_scripts -p "test_expanded_release.py" -v
```

## Retained annotation anomalies

The supplied training split contains three reversed main evidence intervals
(`47-3`, `47-5`, `184-1`). Loading emits a warning; annotations remain byte-for-byte
identical to the authoritative release. Resolve these with source evidence before
using them as temporal supervision. The test split has no such reversed intervals.

## Bailian Qwen3.7-Plus full test

Set `DASHSCOPE_API_KEY` in the process environment, then run:

```sh
python test_scripts/run_bailian.py --output_dir runs/qwen3.7-plus/context
```

The runner evaluates all 1,214 test turns using the context protocol above,
three training conversations (18 example turns under the default budget),
temperature 0, non-thinking mode, JSON output, and a 512-token output limit.
Six HTTP requests run concurrently; question-only histories are precomputed and
identical to sequential evaluation. This is one full main-condition run, not
repeated trials or the standalone ablation.

Each response is flushed to `responses.jsonl`. Rerunning the same command
resumes only unfinished requests and rejects changed configuration or data/code
hashes. Transport failures are retried up to six times and prevent the run from
being declared complete. Model formatting failures remain scored errors.
`completion_audit.json` records coverage, truncations, invalid JSON, returned
model IDs and token usage. API keys are never serialized.

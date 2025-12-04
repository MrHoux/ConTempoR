# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def ensure_big_list(data: Any) -> List[Any]:
    if not isinstance(data, list):
        raise ValueError("Pipeline expects the input JSON top-level to be a list.")
    return data


def _format_date_string(value: Any) -> Any:
    """Convert YYYYMMDD ints/strings to YYYY-MM-DD strings."""
    if isinstance(value, int):
        text = f"{value:08d}"
    elif isinstance(value, str):
        text = value.strip()
    else:
        return value
    if len(text) == 8 and text.isdigit():
        return f"{text[0:4]}-{text[4:6]}-{text[6:8]}"
    return value


def normalize_evidence_dates(data: List[Any]) -> None:
    """Normalize evidence.main/constraint date ranges in place."""
    def _convert_block(block: Any) -> None:
        if not isinstance(block, list):
            return
        for item in block:
            if isinstance(item, list) and len(item) >= 3:
                date_range = item[2]
                if isinstance(date_range, list) and len(date_range) >= 2:
                    date_range[0] = _format_date_string(date_range[0])
                    date_range[1] = _format_date_string(date_range[1])

    for entry in data:
        if isinstance(entry, dict):
            turns = entry.get("turns")
            if isinstance(turns, list):
                normalize_evidence_dates(turns)
                continue
            evidence = entry.get("evidence")
            if isinstance(evidence, dict):
                _convert_block(evidence.get("main"))
                _convert_block(evidence.get("constraint"))
        elif isinstance(entry, list):
            normalize_evidence_dates(entry)


# ---------------------------------------------------------------------------
# Stage 2 – filter on signal_check + minimum dict count
# ---------------------------------------------------------------------------
def _stage2_is_dict(obj: Any) -> bool:
    return isinstance(obj, dict)


def _stage2_is_list(obj: Any) -> bool:
    return isinstance(obj, list)


def _stage2_filter_small_list(small: List[Any], stats: Dict[str, int]) -> List[Any]:
    new_small: List[Any] = []
    for item in small:
        if _stage2_is_dict(item):
            if item.get("signal_check") is False:
                stats["dict_removed_false_signal"] += 1
                continue
            new_small.append(item)
        else:
            new_small.append(item)
    return new_small


def _stage2_dict_count(lst: List[Any]) -> int:
    return sum(1 for x in lst if _stage2_is_dict(x))


def stage2_filter_signal(big_list: List[Any], min_dicts: int) -> Tuple[List[Any], Dict[str, int]]:
    stats = {
        "dict_removed_false_signal": 0,
        "lists_removed_too_few_dicts": 0,
        "lists_kept": 0,
        "lists_input_total": 0,
        "lists_output_total": 0,
    }
    output: List[Any] = []
    for small in big_list:
        if not _stage2_is_list(small):
            output.append(small)
            continue
        stats["lists_input_total"] += 1
        filtered = _stage2_filter_small_list(small, stats)
        dict_count = _stage2_dict_count(filtered)
        if dict_count < min_dicts:
            stats["lists_removed_too_few_dicts"] += 1
            continue
        stats["lists_kept"] += 1
        output.append(filtered)
    stats["lists_output_total"] = stats["lists_kept"]
    return output, stats


# ---------------------------------------------------------------------------
# Stage 3 – drop items whose question_entity contains too many dicts
# ---------------------------------------------------------------------------
def _count_dicts_in_list(lst: Any) -> int:
    if not isinstance(lst, list):
        return 0
    return sum(1 for x in lst if isinstance(x, dict))


def stage3_trim_question_entities(
    data: List[Any], max_qe_dicts: int
) -> Tuple[List[Any], Dict[str, int]]:
    trimmed: List[Any] = []
    removed_items = 0
    for chain in data:
        if not isinstance(chain, list):
            trimmed.append([])
            continue
        new_chain: List[Any] = []
        for item in chain:
            if not isinstance(item, dict):
                new_chain.append(item)
                continue
            qe = item.get("question_entity", [])
            if _count_dicts_in_list(qe) > max_qe_dicts:
                removed_items += 1
                continue
            new_chain.append(item)
        trimmed.append(new_chain)
    lengths = [len(chain) if isinstance(chain, list) else 0 for chain in trimmed]
    stats = {
        "items_removed": removed_items,
        "chains_total": len(trimmed),
        "min_chain_length": min(lengths) if lengths else 0,
        "max_chain_length": max(lengths) if lengths else 0,
    }
    return trimmed, stats


# ---------------------------------------------------------------------------
# Stage 4 – drop turns whose first answer label is a Q-id, and prune short convs
# ---------------------------------------------------------------------------
QID_PREFIX = "Q"


def _is_qid_label_turn(turn: Dict[str, Any]) -> bool:
    ans = turn.get("answer")
    if not isinstance(ans, list) or not ans:
        return False
    first = ans[0]
    if not isinstance(first, dict):
        return False
    label = first.get("label")
    if not isinstance(label, str):
        return False
    return label.startswith(QID_PREFIX) and label[1:].isdigit()


def _normalize_to_conversations(data: List[Any]) -> List[Dict[str, Any]]:
    convs: List[Dict[str, Any]] = []
    for idx, item in enumerate(data, start=1):
        if isinstance(item, dict) and isinstance(item.get("turns"), list):
            convs.append(
                {
                    "conversation_id": item.get("conversation_id", idx),
                    "turns": [t for t in item["turns"] if isinstance(t, dict)],
                    "original_shape_objects": True,
                }
            )
        elif isinstance(item, list):
            convs.append(
                {
                    "conversation_id": idx,
                    "turns": [t for t in item if isinstance(t, dict)],
                    "original_shape_objects": False,
                }
            )
    return convs


def _restore_original_shape(cleaned: List[Dict[str, Any]]) -> List[Any]:
    out: List[Any] = []
    for conv in cleaned:
        if conv.get("original_shape_objects"):
            out.append(
                {
                    "conversation_id": conv.get("conversation_id"),
                    "turns": conv.get("turns", []),
                }
            )
        else:
            out.append(conv.get("turns", []))
    return out


def stage4_drop_qid_turns(
    data: List[Any], min_turns: int
) -> Tuple[List[Any], Dict[str, int]]:
    convs = _normalize_to_conversations(data)
    cleaned: List[Dict[str, Any]] = []
    removed_turns = 0
    removed_convs = 0
    kept_turns = 0

    for conv in convs:
        turns = conv["turns"]
        filtered = [t for t in turns if not _is_qid_label_turn(t)]
        removed_turns += len(turns) - len(filtered)
        if len(filtered) < min_turns:
            removed_convs += 1
            continue
        kept_turns += len(filtered)
        cleaned.append(
            {
                "conversation_id": conv.get("conversation_id"),
                "turns": filtered,
                "original_shape_objects": conv.get("original_shape_objects", False),
            }
        )

    stats = {
        "conversations_in": len(convs),
        "conversations_kept": len(cleaned),
        "conversations_removed": removed_convs,
        "turns_removed": removed_turns,
        "turns_kept": kept_turns,
    }
    restored = _restore_original_shape(cleaned)
    return restored, stats


# ---------------------------------------------------------------------------
# Stage 5 – drop chains shorter than threshold
# ---------------------------------------------------------------------------
def stage5_prune_short_chains(
    data: List[Any], min_len: int
) -> Tuple[List[Any], Dict[str, int]]:
    filtered = [
        chain for chain in data if isinstance(chain, list) and len(chain) >= min_len
    ]
    stats = {
        "chains_in": len(data),
        "chains_removed": len(data) - len(filtered),
        "chains_kept": len(filtered),
        "min_len_required": min_len,
    }
    return filtered, stats


# ---------------------------------------------------------------------------
# Stage 6 – question connection (reuse original module via importlib)
# ---------------------------------------------------------------------------
_STAGE6_MODULE = None


def _load_stage6_module():
    global _STAGE6_MODULE
    if _STAGE6_MODULE is None:
        module_path = Path(__file__).resolve().parent / "question_connect.py"
        if not module_path.is_file():
            raise FileNotFoundError(f"Cannot locate {module_path} for stage 6.")
        spec = importlib.util.spec_from_file_location(
            "ConTempoR_question_rephrase_stage6", module_path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        _STAGE6_MODULE = module
    return _STAGE6_MODULE


def stage6_connect_questions(
    data: List[Any],
    seed: int,
    api_key: Optional[str],
    gender_model: str,
    polish_model: str,
    temperature: float,
    gender_cache: Optional[str],
    disable_gpt_polish: bool,
    max_tries: int,
    max_polish_tries: int,
    s1_prob: float,
    progress: bool,
) -> List[Any]:
    module = _load_stage6_module()
    stage_args = SimpleNamespace(
        seed=seed,
        api_key=api_key,
        gender_model=gender_model,
        polish_model=polish_model,
        temperature=temperature,
        gender_cache=gender_cache,
        disable_gpt_polish=disable_gpt_polish,
        max_tries=max_tries,
        max_polish_tries=max_polish_tries,
        s1_prob=s1_prob,
        progress=progress,
    )
    return module.process_big_list(data, stage_args)


# ---------------------------------------------------------------------------
# Stage 7 – reform to conversation objects
# ---------------------------------------------------------------------------
def stage7_restructure_conversations(data: List[Any]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for conv_idx, chain in enumerate(data, start=1):
        if not isinstance(chain, list):
            continue
        conversation = {"conversation_id": conv_idx, "turns": []}
        for turn_idx, item in enumerate(chain, start=1):
            if not isinstance(item, dict):
                continue
            question = item.get("question", "")
            rephrased = item.get("rephrased_question", "")
            new_item = dict(item)
            ordered = {
                "turn_id": f"{conv_idx}-{turn_idx}",
                "question": question,
                "rephrased_question": rephrased,
            }
            for k, v in new_item.items():
                if k not in ordered:
                    ordered[k] = v
            conversation["turns"].append(ordered)
        output.append(conversation)
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run scripts 2-7 in a single shot pipeline."
    )
    parser.add_argument("--input", required=True, help="Input JSON file.")
    parser.add_argument("--output", required=True, help="Final output JSON file.")
    parser.add_argument(
        "--intermediate-dir",
        help="Optional directory to dump intermediate stage outputs (JSON).",
    )
    parser.add_argument(
        "--stop-after-stage",
        type=int,
        choices=[2, 3, 4, 5, 6, 7],
        help="Stop after the specified stage number.",
    )

    # Stage-specific knobs
    parser.add_argument(
        "--stage2-min-dicts",
        type=int,
        default=5,
        help="Stage2: minimum dict count per chain.",
    )
    parser.add_argument(
        "--stage3-max-question-entities",
        type=int,
        default=4,
        help="Stage3: max allowed dict count inside question_entity.",
    )
    parser.add_argument(
        "--stage4-min-turns",
        type=int,
        default=5,
        help="Stage4: minimum turns per conversation after filtering.",
    )
    parser.add_argument(
        "--stage5-min-length",
        type=int,
        default=5,
        help="Stage5: drop chains shorter than this length.",
    )

    # Stage6 knobs mirror the original script
    parser.add_argument("--stage6-seed", type=int, default=42)
    parser.add_argument(
        "--stage6-api-key",
        default=None,
        help="OpenAI API key (defaults to OPENAI_API_KEY env when omitted).",
    )
    parser.add_argument(
        "--stage6-gender-model",
        default="gpt-4o-mini",
        help="Model used for gender probing.",
    )
    parser.add_argument(
        "--stage6-polish-model",
        default="gpt-4o-mini",
        help="Model used for optional polishing.",
    )
    parser.add_argument("--stage6-temperature", type=float, default=0.0)
    parser.add_argument(
        "--stage6-gender-cache",
        default=".cache/gender.json",
        help="Cache file for gender lookups.",
    )
    parser.add_argument(
        "--stage6-disable-gpt-polish",
        action="store_true",
        help="Disable GPT polishing in stage6.",
    )
    parser.add_argument("--stage6-max-tries", type=int, default=9)
    parser.add_argument("--stage6-max-polish-tries", type=int, default=1)
    parser.add_argument("--stage6-s1-prob", type=float, default=0.33)
    parser.add_argument("--stage6-progress", action="store_true")

    return parser


def maybe_dump_intermediate(
    directory: Optional[Path], stage_id: int, data: Any
) -> None:
    if not directory:
        return
    directory.mkdir(parents=True, exist_ok=True)
    filename = {
        2: "stage2_signal_filter.json",
        3: "stage3_question_entity_filter.json",
        4: "stage4_qid_prune.json",
        5: "stage5_length_filter.json",
        6: "stage6_question_connect.json",
    }.get(stage_id, f"stage{stage_id}.json")
    dump_json(directory / filename, data)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    intermediate_dir = Path(args.intermediate_dir) if args.intermediate_dir else None
    stop_stage = args.stop_after_stage

    data = ensure_big_list(load_json(input_path))
    normalize_evidence_dates(data)
    current_stage = 1

    # Stage 2
    current_stage = 2
    data, stats2 = stage2_filter_signal(data, args.stage2_min_dicts)
    print(f"[Stage 2] stats: {stats2}")
    maybe_dump_intermediate(intermediate_dir, 2, data)
    if stop_stage == 2:
        dump_json(output_path, data)
        print(f"Pipeline stopped after stage 2. Output: {output_path}")
        return

    # Stage 3
    current_stage = 3
    data, stats3 = stage3_trim_question_entities(
        data, args.stage3_max_question_entities
    )
    print(f"[Stage 3] stats: {stats3}")
    maybe_dump_intermediate(intermediate_dir, 3, data)
    if stop_stage == 3:
        dump_json(output_path, data)
        print(f"Pipeline stopped after stage 3. Output: {output_path}")
        return

    # Stage 4
    current_stage = 4
    data, stats4 = stage4_drop_qid_turns(data, args.stage4_min_turns)
    print(f"[Stage 4] stats: {stats4}")
    maybe_dump_intermediate(intermediate_dir, 4, data)
    if stop_stage == 4:
        dump_json(output_path, data)
        print(f"Pipeline stopped after stage 4. Output: {output_path}")
        return

    # Stage 5
    current_stage = 5
    data, stats5 = stage5_prune_short_chains(data, args.stage5_min_length)
    print(f"[Stage 5] stats: {stats5}")
    maybe_dump_intermediate(intermediate_dir, 5, data)
    if stop_stage == 5:
        dump_json(output_path, data)
        print(f"Pipeline stopped after stage 5. Output: {output_path}")
        return

    # Stage 6
    current_stage = 6
    stage6_api_key = args.stage6_api_key or os.environ.get("OPENAI_API_KEY")
    data = stage6_connect_questions(
        data=data,
        seed=args.stage6_seed,
        api_key=stage6_api_key,
        gender_model=args.stage6_gender_model,
        polish_model=args.stage6_polish_model,
        temperature=args.stage6_temperature,
        gender_cache=args.stage6_gender_cache,
        disable_gpt_polish=args.stage6_disable_gpt_polish,
        max_tries=args.stage6_max_tries,
        max_polish_tries=args.stage6_max_polish_tries,
        s1_prob=args.stage6_s1_prob,
        progress=args.stage6_progress,
    )
    print(f"[Stage 6] processed {len(data)} conversations.")
    maybe_dump_intermediate(intermediate_dir, 6, data)
    if stop_stage == 6:
        dump_json(output_path, data)
        print(f"Pipeline stopped after stage 6. Output: {output_path}")
        return

    # Stage 7
    current_stage = 7
    final_data = stage7_restructure_conversations(data)
    dump_json(output_path, final_data)
    print(f"[Stage 7] Completed. Final output: {output_path}")


if __name__ == "__main__":
    main()

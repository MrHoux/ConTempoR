# -*- coding: utf-8 -*-
import argparse
import json
import random
from collections import defaultdict, deque, OrderedDict
from pathlib import Path
from typing import Any, Deque, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Optional[Path], data: Any) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------
def is_dict(obj: Any) -> bool:
    return isinstance(obj, dict)


def normalize_string(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def iter_question_dicts(raw: Any) -> Iterable[Dict[str, Any]]:
    """
    Yield question-like dicts. Supports:
    - List of dicts.
    - List of lists (e.g., [entity_dict, question_dict, ...]).
    """
    if not isinstance(raw, list):
        raise ValueError("Input JSON must be a list at the top level.")
    for item in raw:
        if isinstance(item, dict):
            if "pseudo_question_construction" in item:
                yield item
        elif isinstance(item, list):
            for sub in item:
                if isinstance(sub, dict) and "pseudo_question_construction" in sub:
                    yield sub


def ensure_question_list(raw: Any) -> List[Dict[str, Any]]:
    return list(iter_question_dicts(raw))


# ---------------------------------------------------------------------------
# clean_data.py logic
# ---------------------------------------------------------------------------
def load_json_list(path: Path) -> List[Dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list at top level.")
    return data


def evidence_main_len(sample: Dict[str, Any]) -> int:
    ev = sample.get("evidence", {})
    main = ev.get("main", [])
    return len(main) if isinstance(main, list) else 0


def build_signature(sample: Dict[str, Any]) -> FrozenSet[str]:
    tokens: List[str] = []
    for key in ("question_entity", "answer"):
        items = sample.get(key, [])
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            token = normalize_string(item.get("id") or item.get("label"))
            if token:
                tokens.append(token)
    return frozenset(tokens)


def clean_samples(data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    total_in = len(data)
    removed_by_main = 0
    removed_duplicates = 0
    seen = set()
    cleaned: List[Dict[str, Any]] = []
    for sample in data:
        if evidence_main_len(sample) >= 2:
            removed_by_main += 1
            continue
        sig = build_signature(sample)
        if sig in seen:
            removed_duplicates += 1
            continue
        seen.add(sig)
        cleaned.append(sample)
    stats = {
        "input_total": total_in,
        "removed_main_ge_2": removed_by_main,
        "removed_signature_dups": removed_duplicates,
        "kept": len(cleaned),
    }
    return cleaned, stats


# ---------------------------------------------------------------------------
# deduplicate.py logic
# ---------------------------------------------------------------------------
def deduplicate_by_topic_answer(data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    seen = set()
    result: List[Dict[str, Any]] = []
    removed = 0
    for item in data:
        topic_id = item.get("topic_entity", {}).get("id")
        answer_id = None
        answers = item.get("answer")
        if isinstance(answers, list) and answers:
            answer_id = answers[0].get("id")
        key = (topic_id, answer_id)
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        result.append(item)
    stats = {"input_total": len(data), "removed_dups": removed, "kept": len(result)}
    return result, stats


# ---------------------------------------------------------------------------
# pseudo_deduplicate.py logic
# ---------------------------------------------------------------------------
def deduplicate_by_field(data: List[Dict[str, Any]], field: str) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    seen = set()
    result: List[Dict[str, Any]] = []
    removed = 0
    for item in data:
        value = item.get(field)
        if value in seen:
            removed += 1
            continue
        seen.add(value)
        result.append(item)
    stats = {"input_total": len(data), "removed_dups": removed, "kept": len(result)}
    return result, stats


# ---------------------------------------------------------------------------
# reorder.py logic (field ordering + optional grouping)
# ---------------------------------------------------------------------------
def reorder_fields(item: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure pseudo_question_construction appears before evidence."""
    if "pseudo_question_construction" not in item:
        return item
    if "evidence" not in item:
        return item
    ordered = OrderedDict()
    ordered["pseudo_question_construction"] = item["pseudo_question_construction"]
    ordered["evidence"] = item["evidence"]
    for key, value in item.items():
        if key in ("pseudo_question_construction", "evidence"):
            continue
        ordered[key] = value
    return dict(ordered)


def group_by_topic_id(data: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    groups = defaultdict(list)
    for item in data:
        topic_id = item.get("topic_entity", {}).get("id", "__NO_TOPIC__")
        groups[topic_id].append(reorder_fields(item))
    grouped = []
    for topic_id, items in groups.items():
        grouped.append([{"topic_entity_id": topic_id}] + items)
    return grouped


# ---------------------------------------------------------------------------
# Agroup_data.py logic
# ---------------------------------------------------------------------------
def get_topic_label(sample: Dict[str, Any]) -> str:
    return normalize_string(sample.get("topic_entity", {}).get("label"))


def get_answer_labels(sample: Dict[str, Any]) -> Set[str]:
    labels: Set[str] = set()
    answers = sample.get("answer", [])
    if isinstance(answers, list):
        for item in answers:
            if isinstance(item, dict):
                lab = normalize_string(item.get("label"))
                if lab:
                    labels.add(lab)
    return labels


def build_groups(data: List[Dict[str, Any]]) -> Tuple[Dict[str, Deque[Dict[str, Any]]], Dict[str, Set[str]], Dict[str, Set[str]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sample in data:
        buckets[get_topic_label(sample)].append(sample)
    groups: Dict[str, Deque[Dict[str, Any]]] = {}
    for topic_label, items in buckets.items():
        random.shuffle(items)
        groups[topic_label] = deque(items)

    incoming: Dict[str, Set[str]] = defaultdict(set)
    outgoing: Dict[str, Set[str]] = defaultdict(set)
    label_to_topic = {label: label for label in groups.keys()}
    for topic_label, dq in groups.items():
        answers = set()
        for sample in dq:
            answers.update(get_answer_labels(sample))
        for lab in answers:
            dst = label_to_topic.get(lab)
            if dst and dst != topic_label:
                outgoing[topic_label].add(dst)
                incoming[dst].add(topic_label)
    return groups, incoming, outgoing


def pick_topic(groups: Dict[str, Deque[Dict[str, Any]]], exclude: Optional[Set[str]] = None) -> Optional[str]:
    exclude = exclude or set()
    candidates = [t for t, dq in groups.items() if dq and t not in exclude]
    if not candidates:
        return None
    return random.choice(candidates)


def has_remaining(groups: Dict[str, Deque[Dict[str, Any]]]) -> bool:
    return any(dq for dq in groups.values())


def build_chains(data: List[Dict[str, Any]]) -> Tuple[List[List[Dict[str, Any]]], int]:
    groups, incoming, outgoing = build_groups(data)
    result: List[List[Dict[str, Any]]] = []
    bridge_count = 0

    while has_remaining(groups):
        chain: List[Dict[str, Any]] = []
        result.append(chain)
        start_topic = pick_topic(groups)
        if not start_topic:
            break
        current_topic = start_topic
        visited_topics = {current_topic}

        while current_topic:
            dq = groups[current_topic]
            if not dq:
                break
            chain.append(dq.popleft())
            # once we have >6 entries, freeze this chain
            if len(chain) > 6:
                break
            # attempt to follow answer→topic edges
            next_topic = None
            candidates = list(outgoing.get(current_topic, []))
            random.shuffle(candidates)
            for cand in candidates:
                if cand in visited_topics:
                    continue
                if groups.get(cand):
                    next_topic = cand
                    bridge_count += 1
                    break
            if next_topic is None:
                next_topic = pick_topic(groups, exclude=visited_topics)
            current_topic = next_topic
            if current_topic:
                visited_topics.add(current_topic)
    return result, bridge_count


def post_check_and_repack(nested: List[List[Dict[str, Any]]]) -> Tuple[List[List[Dict[str, Any]]], Dict[str, int]]:
    final_lists: List[List[Dict[str, Any]]] = []
    small_pool: List[Dict[str, Any]] = []
    stats = {
        "deleted_gt_30": 0,
        "split_11_30": 0,
        "kept_chunks_ge_6": 0,
        "pooled_chunks_lt_6": 0,
        "repacked_lists": 0,
    }

    for lst in nested:
        n = len(lst)
        if n == 0:
            continue
        if n > 30:
            stats["deleted_gt_30"] += 1
            continue
        if n > 10:
            stats["split_11_30"] += 1
            start = 0
            while start < n:
                end = min(start + 10, n)
                chunk = lst[start:end]
                if len(chunk) >= 6:
                    final_lists.append(chunk)
                    stats["kept_chunks_ge_6"] += 1
                else:
                    small_pool.extend(chunk)
                    stats["pooled_chunks_lt_6"] += 1
                start = end
        else:
            final_lists.append(lst)

    if small_pool:
        topic_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for sample in small_pool:
            topic_buckets[get_topic_label(sample)].append(sample)
        topics = list(topic_buckets.keys())
        random.shuffle(topics)
        while topics:
            new_list: List[Dict[str, Any]] = []
            while len(new_list) <= 6 and topics:
                topic = topics.pop()
                block = topic_buckets.pop(topic, [])
                new_list.extend(block)
            if new_list:
                final_lists.append(new_list)
                stats["repacked_lists"] += 1

    return final_lists, stats


def agroup_pipeline(data: List[Dict[str, Any]]) -> Tuple[List[List[Dict[str, Any]]], Dict[str, int]]:
    nested, bridge_count = build_chains(data)
    repacked, post_stats = post_check_and_repack(nested)
    post_stats["initial_chain_count"] = len(nested)
    post_stats["bridge_count"] = bridge_count
    post_stats["final_chain_count"] = len(repacked)
    post_stats["total_items"] = sum(len(lst) for lst in repacked)
    return repacked, post_stats


# ---------------------------------------------------------------------------
# extract.py logic (generalized)
# ---------------------------------------------------------------------------
def extract_pqc_from_nested(data: List[Any], min_len: int, dedup: bool) -> List[str]:
    results: List[str] = []
    seen = set()

    def record(question: Dict[str, Any]):
        evidence = question.get("evidence", {})
        main = evidence.get("main", [])
        constraint = evidence.get("constraint", [])
        main_len = len(main) if isinstance(main, list) else 0
        cons_len = len(constraint) if isinstance(constraint, list) else 0
        if main_len >= min_len or cons_len >= min_len:
            pqc = question.get("pseudo_question_construction")
            if pqc is None:
                return
            if dedup:
                if pqc in seen:
                    return
                seen.add(pqc)
            results.append(pqc)

    if isinstance(data, list) and data and all(isinstance(x, dict) for x in data):
        for question in data:
            if isinstance(question, dict):
                record(question)
    else:
        for group in data:
            if isinstance(group, list):
                for question in group:
                    if isinstance(question, dict):
                        record(question)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the precheck pipeline (clean, dedup, reorder, group, extract).")
    parser.add_argument("--input", required=True, help="Input JSON file.")
    parser.add_argument("--output", default="precheck_grouped.json", help="Final grouped JSON output path.")
    parser.add_argument("--grouped-topic-output", help="Optional output for simple topic-based grouping (reorder.py style).")
    parser.add_argument("--pqc-output", help="Optional output path for extracted pseudo_question_construction list.")
    parser.add_argument("--intermediate-dir", help="Directory to dump intermediate stage outputs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for grouping logic.")
    parser.add_argument("--extract-min-len", type=int, default=2, help="Min evidence length for extraction stage.")
    parser.add_argument("--extract-dedup", action="store_true", help="Deduplicate pseudo_question_construction during extraction.")
    return parser


def stage_path(base: Optional[Path], filename: str) -> Optional[Path]:
    if not base:
        return None
    return base / filename


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    intermediate_dir = Path(args.intermediate_dir) if args.intermediate_dir else None
    final_output = Path(args.output)
    grouped_topic_output = Path(args.grouped_topic_output) if args.grouped_topic_output else None
    pqc_output = Path(args.pqc_output) if args.pqc_output else None

    raw = load_json(input_path)
    questions = ensure_question_list(raw)
    print(f"[Stage 0] Collected {len(questions)} question dicts for processing.")

    # Stage 1: clean
    cleaned, stats_clean = clean_samples(questions)
    print(f"[Stage 1] Clean stats: {stats_clean}")
    dump_json(stage_path(intermediate_dir, "stage1_clean.json"), cleaned)

    # Stage 2: dedup by topic+answer
    dedup_ta, stats_ta = deduplicate_by_topic_answer(cleaned)
    print(f"[Stage 2] Topic+answer dedup stats: {stats_ta}")
    dump_json(stage_path(intermediate_dir, "stage2_topic_answer_dedup.json"), dedup_ta)

    # Stage 3: dedup by pseudo_question_construction
    dedup_pqc, stats_pqc = deduplicate_by_field(dedup_ta, "pseudo_question_construction")
    print(f"[Stage 3] Pseudo-question dedup stats: {stats_pqc}")
    dump_json(stage_path(intermediate_dir, "stage3_pqc_dedup.json"), dedup_pqc)

    # Stage 4: reorder fields + optional grouped view
    reordered = [reorder_fields(item) for item in dedup_pqc]
    dump_json(stage_path(intermediate_dir, "stage4_reordered.json"), reordered)
    if grouped_topic_output:
        grouped_topic = group_by_topic_id(reordered)
        dump_json(grouped_topic_output, grouped_topic)
        print(f"[Stage 4] Wrote topic-grouped view with {len(grouped_topic)} groups -> {grouped_topic_output}")

    # Stage 5: apply agroup logic
    random.seed(args.seed)
    grouped, stats_group = agroup_pipeline(reordered)
    print(f"[Stage 5] Grouping stats: {stats_group}")
    dump_json(final_output, grouped)
    print(f"[Stage 5] Final grouped output written to {final_output}")

    # Stage 6: extract pseudo_question_construction list
    if pqc_output:
        pqc_list = extract_pqc_from_nested(grouped, min_len=args.extract_min_len, dedup=args.extract_dedup)
        dump_json(pqc_output, pqc_list)
        print(f"[Stage 6] Extracted {len(pqc_list)} pseudo_question_construction entries -> {pqc_output}")


if __name__ == "__main__":
    main()

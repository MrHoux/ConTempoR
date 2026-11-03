"""Split-safe helpers for the expanded ConTempoR release."""
import json
import re
import warnings
from pathlib import Path
from datetime import datetime

BENCHMARK = Path(__file__).resolve().parents[1] / "benchmark" / "final"


def parse_dataset_date(value):
    text = str(value).strip()
    if re.fullmatch(r"\d{8}", text):
        return datetime.strptime(text, "%Y%m%d")
    return datetime.strptime(text, "%Y-%m-%d")


def load_split(path):
    path = Path(path).resolve()
    if path.name.lower() == "full.json":
        raise ValueError("Use train.json/dev.json/test.json separately; full.json mixes evaluation splits.")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError("Expected a non-empty list of conversations")
    seen_cids, seen_tids = set(), set()
    reversed_ids = []
    for conv in data:
        cid = conv.get("conversation_id")
        if cid is None or cid in seen_cids:
            raise ValueError(f"Missing or duplicate conversation_id: {cid}")
        seen_cids.add(cid)
        if not isinstance(conv.get("turns"), list) or not conv["turns"]:
            raise ValueError(f"Empty or invalid turns: {cid}")
        for turn in conv["turns"]:
            tid = turn.get("turn_id")
            if tid is None or tid in seen_tids:
                raise ValueError(f"Missing or duplicate turn_id: {tid}")
            seen_tids.add(tid)
            start, end = turn["evidence"]["main"][0][2]
            if parse_dataset_date(start) > parse_dataset_date(end):
                reversed_ids.append(tid)
    if reversed_ids:
        warnings.warn(f"{path.name}: {len(reversed_ids)} reversed gold intervals retained unchanged: {reversed_ids}")
    return data


def complete_predictions(dataset, predictions):
    """Score every gold turn; missing predictions are wrong, duplicates are errors."""
    keys = [(c["conversation_id"], t["turn_id"]) for c in dataset for t in c["turns"]]
    expected = set(keys)
    if len(expected) != len(keys):
        raise ValueError("Duplicate gold IDs; evaluate a single split")
    found = {}
    for pred in predictions:
        key = (pred.get("conversation_id"), pred.get("turn_id"))
        if key not in expected:
            raise ValueError(f"Prediction outside evaluation split: {key}")
        if key in found:
            raise ValueError(f"Duplicate prediction: {key}")
        found[key] = pred
    return [found.get(key, {"conversation_id": key[0], "turn_id": key[1],
                           "answer_entity": "", "time_range": {}, "pred_relation": None})
            for key in keys]


def cap_conversations(data, limit):
    if limit is None:
        return data
    if limit <= 0:
        raise ValueError("max_test_turns must be positive")
    result = []
    for conv in data:
        turns = conv["turns"][:limit]
        result.append({**conv, "turns": turns})
        limit -= len(turns)
        if limit == 0:
            break
    return result

# -*- coding: utf-8 -*-
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from tqdm.auto import tqdm

# -----------------------------
# Text normalization
# -----------------------------

QUOTE_MAP = {
    "\\u2019": "'", "\\u2018": "'", "\\u2032": "'", "\\uFF07": "'",
    "\u2019": "'", "\u2018": "'", "\u2032": "'", "\uFF07": "'",
    "’": "'", "‘": "'",
}
PUNCT_RE = re.compile(r"[^\w\s]")

def decode_escaped_unicode(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    try:
        s2 = bytes(s, "utf-8").decode("unicode_escape", errors="ignore")
    except Exception:
        s2 = s
    for k, v in QUOTE_MAP.items():
        s2 = s2.replace(k, v)
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2

def normalize_for_exact(s: str) -> str:
    return decode_escaped_unicode(s).strip().lower()

def normalize_for_fuzzy(s: str) -> str:
    s2 = decode_escaped_unicode(s).lower()
    s2 = PUNCT_RE.sub(" ", s2)
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2

_ACRONYM_STOPWORDS = {
    "of", "the", "and", "a", "an",
    "de", "da", "di", "du", "del",
    "la", "le", "les", "el", "los", "las",
    "von", "van", "der", "den", "und"
}

def _acronym(s: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", normalize_for_fuzzy(s))
    letters = []
    for t in tokens:
        if t in _ACRONYM_STOPWORDS:
            continue
        if t:
            letters.append(t[0])
    return "".join(letters)

_ALIAS_MAP = {
    "us": ["united states", "u s a", "usa", "u s"],
    "uk": ["united kingdom", "u k"],
    "nyc": ["new york city"],
    "un": ["united nations"],
}

def fuzzy_match(pred: str, gold: str) -> bool:
    a = normalize_for_fuzzy(pred)
    b = normalize_for_fuzzy(gold)
    if not a or not b:
        return False
    if a == b or a in b or b in a:
        return True
    ac, bc = _acronym(a), _acronym(b)
    if ac and bc and (ac == bc):
        return True
    if a in _ALIAS_MAP and any(alias in b for alias in _ALIAS_MAP[a]):
        return True
    if b in _ALIAS_MAP and any(alias in a for alias in _ALIAS_MAP[b]):
        return True
    return False

# -----------------------------
# Dates & TCR
# -----------------------------

def parse_dataset_ymd(n: int) -> datetime:
    s = str(n)
    return datetime(int(s[0:4]), int(s[4:6]), int(s[6:8]))

def parse_pred_date(s: Optional[str], which: str) -> Optional[datetime]:
    if s is None:
        if which == "end":
            return datetime(2050, 12, 31)
        return None
    s = s.strip().lower()
    if s in ("open", "none", ""):
        return datetime(2050, 12, 31) if which == "end" else None
    m = re.fullmatch(r"(\d{4})", s)
    if m:
        y = int(m.group(1))
        return datetime(y, 1, 1) if which == "start" else datetime(y, 12, 31)
    m = re.fullmatch(r"(\d{4})-(\d{2})", s)
    if m:
        y = int(m.group(1)); mm = int(m.group(2))
        if which == "start":
            return datetime(y, mm, 1)
        else:
            if mm == 12:
                return datetime(y, 12, 31)
            nxt = datetime(y, mm+1, 1)
            return nxt - timedelta(days=1)
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        y = int(m.group(1)); mm = int(m.group(2)); dd = int(m.group(3))
        try:
            return datetime(y, mm, dd)
        except Exception:
            return None
    return None

def closed_days(a: datetime, b: datetime) -> int:
    return (b - a).days + 1

def temporal_coverage_ratio(pred: Tuple[datetime, datetime], gold: Tuple[datetime, datetime]) -> float:
    if not pred or not gold:
        return 0.0
    ps, pe = pred
    gs, ge = gold
    if not (ps and pe and gs and ge):
        return 0.0
    if pe < ps:
        return 0.0
    os = max(ps, gs)
    oe = min(pe, ge)
    if oe < os:
        return 0.0
    overlap = closed_days(os, oe)
    pred_len = closed_days(ps, pe)
    if pred_len <= 0:
        return 0.0
    return max(0.0, min(1.0, overlap / pred_len))

# -----------------------------
# Gold helpers
# -----------------------------

def get_gold_maps(dataset: List[Dict[str, Any]]):
    m = {}
    for conv in dataset:
        cid = conv.get("conversation_id")
        for t in conv.get("turns", []):
            tid = t.get("turn_id")
            if cid is not None and tid is not None:
                m[(cid, tid)] = t
    return m

def extract_gold_time_range(turn: Dict[str, Any]) -> Optional[Tuple[datetime, datetime]]:
    ev = turn.get("evidence", {}) or {}
    main = ev.get("main") or []
    if not main or not isinstance(main, list) or not main[0] or len(main[0]) < 3:
        return None
    rng = main[0][2]
    if not rng or len(rng) < 2:
        return None
    try:
        start = parse_dataset_ymd(rng[0])
        end = parse_dataset_ymd(rng[1])
        return (start, end)
    except Exception:
        return None

def gold_answer_label(turn: Dict[str, Any]) -> Optional[str]:
    ans = turn.get("answer") or []
    if ans and isinstance(ans, list):
        return ans[0].get("label")
    return None

# -----------------------------
# Few-shot prompt
# -----------------------------

def example_from_turn_flat(turn: Dict[str, Any]) -> str:
    q = turn.get("rephrased_question") or turn.get("question") or ""
    ans = gold_answer_label(turn) or ""
    tr = extract_gold_time_range(turn)
    if tr:
        s = tr[0].strftime("%Y-%m-%d"); e = tr[1].strftime("%Y-%m-%d")
    else:
        s = ""; e = ""
    gold_json = {
        "answer_entity": ans,
        "time_range": {"start": s, "end": e}
    }
    return f"Q: {q}\nA: {json.dumps(gold_json, ensure_ascii=False)}"

def build_prompt_flat(shots: List[Dict[str, Any]], current_rephrased_q: str) -> List[Dict[str, str]]:
    system = (
        "You answer temporal questions. Return ONLY valid JSON with keys: "
        "answer_entity (string), time_range (object with start,end). "
        "Dates must be YYYY or YYYY-MM or YYYY-MM-DD. "
        "Do not add explanations and try your best to provide an valid end time instead of 2025-12-31. "
        "When pronouns appear, resolve them using the immediate context. "
        "You may need to use the previous question to answer the current question. "
        "The time value in the question is a relationship prompt with last question; "
        "time value does not limit the time. "
        "For example, one month later may also be prompted as one week later. "
        "If the answer has aliases, please output all aliases. "
        "If you have multiple answers, output them all. "
    )
    shot_texts = [example_from_turn_flat(t) for t in shots]
    examples = "\n\n".join(shot_texts)
    user = f"{examples}\n\nQ: {current_rephrased_q}\nA:" if examples else f"Q: {current_rephrased_q}\nA:"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]

# -----------------------------
# HTTP client
# -----------------------------

import requests
def _http_chat_completions(messages, model, temperature, api_key, base_url=None,
                           max_completion_tokens=512, json_strict=False,
                           max_retries=3, retry_backoff=1.2) -> str:
    url_root = (base_url.rstrip("/") if base_url else "https://api.openai.com/v1")
    url = url_root + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key and api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": max_completion_tokens,
    }
    if json_strict:
        payload["response_format"] = {"type": "json_object"}
    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
            r.raise_for_status()
            data = r.json()
            return (data["choices"][0]["message"]["content"] or "").strip()
        except Exception as e:
            last_err = e
            time.sleep(retry_backoff * (attempt + 1))
    raise RuntimeError(f"HTTP call failed after {max_retries} attempts: {last_err}")

JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

def parse_model_json(s: str) -> Dict[str, Any]:
    if not s:
        return {}
    m = JSON_BLOCK_RE.search(s)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

# -----------------------------
# JSONL loader
# -----------------------------

def _build_turnid_to_cid_map(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    idx = {}
    for conv in dataset:
        cid = conv.get("conversation_id")
        for t in (conv.get("turns") or []):
            tid = t.get("turn_id")
            if tid is not None and cid is not None:
                idx[str(tid)] = cid
    return idx

def load_predictions_compat(dataset: List[Dict[str, Any]], path: str) -> List[Dict[str, Any]]:
    preds: List[Dict[str, Any]] = []
    # Try JSON array
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            for it in obj:
                preds.append({
                    "conversation_id": it.get("conversation_id"),
                    "turn_id": it.get("turn_id"),
                    "answer_entity": it.get("answer_entity", ""),
                    "time_range": it.get("time_range") or {},
                })
            return preds
    except Exception:
        pass
    # JSONL fallback (supports {"turn_id": "...", "raw": "..."} lines)
    turnid2cid = _build_turnid_to_cid_map(dataset)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            # structured line already
            if all(k in rec for k in ("conversation_id", "turn_id", "answer_entity", "time_range")):
                preds.append({
                    "conversation_id": rec.get("conversation_id"),
                    "turn_id": rec.get("turn_id"),
                    "answer_entity": rec.get("answer_entity", ""),
                    "time_range": rec.get("time_range") or {},
                })
                continue
            # raw-output line
            tid = rec.get("turn_id")
            raw = rec.get("raw") or ""
            if tid is None:
                continue
            obj = parse_model_json(raw) if raw else {}
            answer_entity = obj.get("answer_entity", "") if isinstance(obj, dict) else ""
            tr = obj.get("time_range") if isinstance(obj, dict) else {}
            if not isinstance(tr, dict):
                tr = {}
            preds.append({
                "conversation_id": turnid2cid.get(str(tid)),
                "turn_id": tid,
                "answer_entity": answer_entity,
                "time_range": tr,
            })
    return preds

# -----------------------------
# Evaluation (with tqdm)
# -----------------------------

def evaluate_flat(dataset, preds, tcr_threshold, log_path, mark_test_correct):
    gold_map = get_gold_maps(dataset)
    total = em_hits = fuzzy_hits = tcr_hits = 0
    tcr_sum = 0.0
    correct_ids, log_lines = [], []
    def log(msg): log_lines.append(msg)

    for p in tqdm(preds, desc="Evaluating", unit="turn"):
        cid = p.get("conversation_id")
        tid = p.get("turn_id")
        if cid is None or tid is None:
            continue
        gold_turn = gold_map.get((cid, tid))
        if not gold_turn:
            continue
        total += 1
        # entity
        gold_label = gold_answer_label(gold_turn) or ""
        pred_ans = p.get("answer_entity")
        if isinstance(pred_ans, list):
            cand_list = [str(x) for x in pred_ans if x]
        elif pred_ans is None:
            cand_list = []
        else:
            parts = re.split(r"[;/\|,]", str(pred_ans))
            cand_list = [x.strip() for x in parts if x.strip()]
        exact_ok = False
        fuzzy_ok = False
        for cand in cand_list:
            if normalize_for_exact(cand) == normalize_for_exact(gold_label):
                exact_ok = True; fuzzy_ok = True; break
            if fuzzy_match(cand, gold_label):
                fuzzy_ok = True
        if exact_ok: em_hits += 1
        if fuzzy_ok: fuzzy_hits += 1
        # time
        gold_range = extract_gold_time_range(gold_turn)
        tr = p.get("time_range") or {}
        ps = parse_pred_date(tr.get("start"), "start") if isinstance(tr, dict) else None
        pe = parse_pred_date(tr.get("end"), "end") if isinstance(tr, dict) else None
        pred_range = (ps, pe) if ps and pe and pe >= ps else None
        tcr = temporal_coverage_ratio(pred_range, gold_range) if (pred_range and gold_range) else 0.0
        tcr_sum += tcr
        time_ok = (tcr >= tcr_threshold)
        if time_ok: tcr_hits += 1
        if mark_test_correct and fuzzy_ok and time_ok:
            correct_ids.append(str(tid))

    res = {
        "totals": {"samples": total},
        "entity": {
            "EM": round(em_hits / total, 4) if total else 0.0,
            "Fuzzy": round(fuzzy_hits / total, 4) if total else 0.0
        },
        "time": {
            "TCR_threshold": tcr_threshold,
            "Accuracy": round(tcr_hits / total, 4) if total else 0.0,
            "Mean_TCR": round(tcr_sum / total, 4) if total else 0.0
        }
    }
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("=== Evaluation (flat autoshoot) ===\n")
            for line in log_lines:
                f.write(line + "\n")
            f.write("\n")
    if mark_test_correct:
        with open("test_correct.json", "w", encoding="utf-8") as f:
            json.dump({"turn_ids": correct_ids}, f, ensure_ascii=False, indent=2)
    return res

# -----------------------------
# Main (with tqdm)
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Full dataset JSON (single list of conversations)")
    ap.add_argument("--shots_conversations", type=int, default=10, help="How many earliest conversations used as few-shots")
    # Unused in flat mode, kept for CLI compatibility with related scripts
    ap.add_argument("--per_dialog_shot_limit", type=int, default=-1, help="[deprecated] ignored in flat mode")
    ap.add_argument("--max_shot_examples", type=int, default=-1, help="[deprecated] ignored in flat mode")
    ap.add_argument("--shots_token_budget", type=int, default=1800, help="ignored in flat")
    ap.add_argument("--model_context_tokens", type=int, default=3072, help="ignored in flat")
    ap.add_argument("--reserve_for_current_qa", type=int, default=700, help="ignored in flat")

    ap.add_argument("--auto_eval_with_openai", action="store_true", help="Enable OpenAI/vLLM calls to produce predictions")
    ap.add_argument("--openai_model", default="gpt-4o-mini")
    ap.add_argument("--openai_api_key", default=os.environ.get("OPENAI_API_KEY"))
    ap.add_argument("--openai_base_url", default=None, help="Set to http://localhost:8000/v1 for local vLLM")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_completion_tokens", type=int, default=512)
    ap.add_argument("--json_strict", action="store_true", help="Force JSON response_format if backend supports")
    ap.add_argument("--tcr_threshold", type=float, default=0.5)
    ap.add_argument("--preds_out", default="test_preds.json", help="Where to save predictions if auto-eval")
    ap.add_argument("--log_path", default="log")
    ap.add_argument("--eval_test_preds", help="If provided, evaluate this predictions JSON/JSONL instead of auto eval")
    ap.add_argument("--max_test_turns", type=int, help="Debug only: cap number of test turns for quick run")
    ap.add_argument("--strategy_assignment", choices=["multi", "exclusive"], default="multi",
                    help="Kept for CLI parity; ignored in flat.")
    ap.add_argument("--debug_raw_dump", action="store_true", help="Dump raw model outputs to debug_raw_model_outputs.jsonl")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if not isinstance(dataset, list):
        print("Top-level must be a list of conversations.", file=sys.stderr)
        sys.exit(1)

    # Few-shot & test split
    K = max(0, int(args.shots_conversations))
    shots_convs = dataset[:K]
    test_convs = dataset[450:606]

    # Collect shots
    shot_turns: List[Dict[str, Any]] = []
    for conv in shots_convs:
        for t in conv.get("turns", []):
            shot_turns.append(t)

    # Collect test turns
    test_turns: List[Dict[str, Any]] = []
    for conv in test_convs:
        cid = conv.get("conversation_id")
        for t in conv.get("turns", []):
            t2 = dict(t)
            t2["conversation_id"] = cid
            test_turns.append(t2)

    if args.max_test_turns and args.max_test_turns > 0:
        test_turns = test_turns[:args.max_test_turns]

    preds: List[Dict[str, Any]] = []

    if args.eval_test_preds:
        preds = load_predictions_compat(dataset, args.eval_test_preds)

    elif args.auto_eval_with_openai:
        # OpenAI requires an api_key; local vLLM often works without one depending on service config
        if (not args.openai_base_url or "api.openai.com" in (args.openai_base_url or "")) and not args.openai_api_key:
            print("ERROR: --auto_eval_with_openai requires --openai_api_key for OpenAI API", file=sys.stderr)
            sys.exit(2)

        raw_dump_f = None
        if args.debug_raw_dump:
            raw_dump_f = open("debug_raw_model_outputs.jsonl", "w", encoding="utf-8")

        for t in tqdm(test_turns, desc="Predicting", unit="turn"):
            q = t.get("rephrased_question") or t.get("question") or ""
            cid = t.get("conversation_id")
            tid = t.get("turn_id")

            messages = build_prompt_flat(shot_turns, q)
            try:
                raw = _http_chat_completions(
                    messages,
                    model=args.openai_model,
                    temperature=args.temperature,
                    api_key=args.openai_api_key,
                    base_url=args.openai_base_url,
                    max_completion_tokens=args.max_completion_tokens,
                    json_strict=args.json_strict
                )
                if raw_dump_f:
                    raw_dump_f.write(json.dumps({"turn_id": tid, "raw": raw}, ensure_ascii=False) + "\n")
                obj = parse_model_json(raw)
            except Exception as e:
                if raw_dump_f:
                    raw_dump_f.write(json.dumps({"turn_id": tid, "error": str(e)}, ensure_ascii=False) + "\n")
                obj = {}

            if not isinstance(obj, dict):
                obj = {}
            preds.append({
                "conversation_id": cid,
                "turn_id": tid,
                "answer_entity": obj.get("answer_entity", ""),
                "time_range": obj.get("time_range") or {}
            })

        if raw_dump_f:
            raw_dump_f.close()

        with open(args.preds_out, "w", encoding="utf-8") as f:
            json.dump(preds, f, ensure_ascii=False, indent=2)

    else:
        print("No predictions provided. Use --auto_eval_with_openai or --eval_test_preds.", file=sys.stderr)
        sys.exit(0)

    # Evaluate
    report = evaluate_flat(
        dataset, preds,
        tcr_threshold=args.tcr_threshold,
        log_path=args.log_path,
        mark_test_correct=True
    )

    with open("evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump({"report": report}, f, ensure_ascii=False, indent=2)

    print("\n=== Final Report ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

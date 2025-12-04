# -*- coding: utf-8 -*-
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Set
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

# 严格缩写：忽略停用词，只接受 ac == bc 的匹配
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

# 别名映射（保留，可按需删减）
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

    # 1) 基础：完全相等 / 子串（不涉及缩写）
    if a == b or a in b or b in a:
        return True

    # 2) 严格“缩写对缩写”匹配（只接受 ac == bc）
    ac, bc = _acronym(a), _acronym(b)
    if ac and bc and (ac == bc):
        return True

    # 3) 别名映射（单向或双向）
    if a in _ALIAS_MAP and any(alias in b for alias in _ALIAS_MAP[a]):
        return True
    if b in _ALIAS_MAP and any(alias in a for alias in _ALIAS_MAP[b]):
        return True

    return False

# -----------------------------
# Date parsing & TCR
# -----------------------------

def parse_dataset_ymd(n: int) -> datetime:
    s = str(n)
    return datetime(int(s[0:4]), int(s[4:6]), int(s[6:8]))

def parse_pred_date(s: Optional[str], which: str) -> Optional[datetime]:
    if s is None:
        return datetime(2050, 12, 31) if which == "end" else None
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
    ps, pe = pred; gs, ge = gold
    if not (ps and pe and gs and ge) or pe < ps:
        return 0.0
    os = max(ps, gs); oe = min(pe, ge)
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

def build_turnid_to_cid_map(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """用于 JSONL 仅含 turn_id 的情况，反查 conversation_id。"""
    idx = {}
    for conv in dataset:
        cid = conv.get("conversation_id")
        for t in (conv.get("turns") or []):
            tid = t.get("turn_id")
            if tid is not None and cid is not None:
                idx[str(tid)] = cid
    return idx

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

def gold_first_and_relation(turn: Dict[str, Any]):
    checks = turn.get("checks") or {}
    first = bool(checks.get("first_in_conversation", False))
    tl = turn.get("temporal_link") or checks.get("temporal_link") or {}
    macro = tl.get("macro")
    return first, macro

# -----------------------------
# S-tags（仅从 gold 读取；支持多标签）
# -----------------------------

_S_TAG_CAPTURE = re.compile(r"\b(S[1-4])(?:(?=[_+\-])|\b)")

def extract_strategy_tags_from_gold(turn: Dict[str, Any]) -> Set[str]:
    """
    仅从 gold 的 link_strategy 字段抽取 S1/S2/S3/S4。
    支持字符串或列表；支持 'S1_pronoun+S2_basic_temporal' 的复合形式；忽略 'none'。
    """
    tags: Set[str] = set()
    ls = turn.get("link_strategy")

    def add_from_token(tok: str):
        if not isinstance(tok, str):
            return
        t = tok.strip()
        if not t or t.lower() == "none":
            return
        for m in _S_TAG_CAPTURE.findall(t):
            if m in ("S1", "S2", "S3", "S4"):
                tags.add(m)

    if isinstance(ls, str):
        add_from_token(ls)
    elif isinstance(ls, list):
        for v in ls:
            add_from_token(v)

    return tags

def assign_strategy_tags(turn: Dict[str, Any], mode: str) -> Set[str]:
    """
    返回用于统计的 S-tags 集合。
    mode == 'multi'：保留全部（多标签）。
    mode == 'exclusive'：互斥优先级 S4 > S3 > S2 > S1，仅取一个。
    """
    tags = extract_strategy_tags_from_gold(turn)
    if mode == "exclusive":
        for k in ("S4", "S3", "S2", "S1"):
            if k in tags:
                return {k}
        return set()
    return tags

# -----------------------------
# Few-shot 构建（整轮对话 + 自动预算裁剪）
# -----------------------------

def example_from_turn(turn: Dict[str, Any]) -> str:
    q = turn.get("question") or turn.get("rephrased_question") or ""
    ans = gold_answer_label(turn) or ""
    tr = extract_gold_time_range(turn)
    first, macro = gold_first_and_relation(turn)
    rel = "UNKNOWN" if first else (macro or "UNKNOWN")
    if tr:
        s = tr[0].strftime("%Y-%m-%d")
        e = tr[1].strftime("%Y-%m-%d")
    else:
        s = ""; e = ""
    gold_json = {
        "answer_entity": ans,
        "time_range": {"start": s, "end": e},
        "pred_relation": rel
    }
    return f"Q: {q}\nA: {json.dumps(gold_json, ensure_ascii=False)}"

def estimate_tokens_by_chars(text: str) -> int:
    # 粗略估算：≈4 chars / token，最少给 1
    return max(1, (len(text) + 3) // 4)

def pack_full_conversation_shots(convs: List[Dict[str, Any]],
                                 k_conversations: int,
                                 token_budget_for_shots: int) -> List[Dict[str, Any]]:
    """
    取最早的前 K 个完整对话的所有 turns，拼成 few-shot。
    若 examples 文本 token 数 > 预算，则按“整轮对话”为单位，从最后一轮开始回退，直到落入预算。
    """
    selected_convs = convs[:max(0, int(k_conversations))]
    shots: List[Dict[str, Any]] = []
    for conv in selected_convs:
        shots.extend(conv.get("turns", []) or [])

    if token_budget_for_shots is None or token_budget_for_shots <= 0:
        return shots

    def shots_to_examples_text(turns: List[Dict[str, Any]]) -> str:
        return "\n\n".join(example_from_turn(t) for t in turns)

    turns_now = shots[:]
    while turns_now:
        examples_text = shots_to_examples_text(turns_now)
        est_tokens = estimate_tokens_by_chars(examples_text)
        if est_tokens <= token_budget_for_shots:
            return turns_now
        # 回退一个“完整对话”
        last_cid = None
        for i in range(len(selected_convs)-1, -1, -1):
            c = selected_convs[i]
            if c.get("turns"):
                last_cid = c.get("conversation_id")
                break
        if last_cid is None:
            break
        turns_now = [t for t in turns_now if t.get("turn_id", "").split("-")[0] != str(last_cid)]
        selected_convs = [c for c in selected_convs if c.get("conversation_id") != last_cid]

    return []

# -----------------------------
# Prompt 组装（加入历史问题）
# -----------------------------

def build_prompt(shot_turns: List[Dict[str, Any]],
                 current_question: str,
                 prev_questions: Optional[List[str]] = None) -> List[Dict[str, str]]:
    system = (
        "You answer temporal conversational questions. Return ONLY valid JSON with keys: "
        "answer_entity (string, entity only), time_range (object with start,end), pred_relation "
        "(AFTER|BEFORE|OVERLAP|UNKNOWN). Dates must be YYYY or YYYY-MM or YYYY-MM-DD. "
        "If the turn is the first question in a conversation, use pred_relation='UNKNOWN'. "
        "Do not add explanations. Always include both time_range.start and time_range.end "
        "(if end is unknown, try your best to find a end time). "
        "When pronouns appear, resolve them using the immediate context. "
        "You may need to use the previous question to answer the current question."
        "The time value in the question is a relationship prompt with last question"
        "time value does not limit the time."
        "For example, one month later may also be prompted as one week later. "
        "If the answer has aliases, please output all aliases. "
        "If you have multiple answers, output them all. "
    )
    shot_texts = [example_from_turn(t) for t in shot_turns]
    examples = "\n\n".join(shot_texts)

    history_block = ""
    if prev_questions:
        hist_lines = [str(q).strip() for q in prev_questions if q and str(q).strip()]
        if hist_lines:
            history_block = "\n[Previous Questions]\n" + "\n".join(f"- {h}" for h in hist_lines)

    user_blocks: List[str] = []
    if examples:
        user_blocks.append(examples)
    if history_block:
        user_blocks.append(history_block)
    user_blocks.append(f"Q: {current_question}\nA:")
    user = "\n\n".join(user_blocks)

    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]

# -----------------------------
# OpenAI / vLLM client（HTTP 直连）
# -----------------------------

def extract_first_balanced_json_block(s: str) -> Optional[str]:
    if not s:
        return None
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```$", "", s, flags=re.I)

    stack = []
    start_idx = None
    for i, ch in enumerate(s):
        if ch == '{':
            if not stack:
                start_idx = i
            stack.append('{')
        elif ch == '}':
            if stack:
                stack.pop()
                if not stack and start_idx is not None:
                    return s[start_idx:i+1]
    return None

def parse_model_json(s: str) -> Dict[str, Any]:
    block = extract_first_balanced_json_block(s or "")
    if not block:
        return {}
    try:
        obj = json.loads(block)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

def call_openai(messages: List[Dict[str, str]],
                model: str,
                temperature: float,
                api_key: Optional[str],
                base_url: Optional[str] = None,
                max_completion_tokens: int = 512,
                json_strict: bool = False,
                max_retries: int = 3,
                retry_backoff: float = 1.2) -> str:
    """
    纯 HTTP 直连 vLLM /v1/chat/completions，不依赖 openai SDK。
    """
    import requests
    import json as _json

    url_root = (base_url.rstrip("/") if base_url else "http://localhost:8000/v1")
    url = url_root + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key and api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"  # 本地 vLLM 通常不校验

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
            r = requests.post(url, headers=headers, data=_json.dumps(payload),
                              timeout=120)
            r.raise_for_status()
            data = r.json()
            return (data["choices"][0]["message"]["content"] or "").strip()
        except Exception as e:
            last_err = e
            time.sleep(retry_backoff * (attempt + 1))

    raise RuntimeError(f"HTTP call to vLLM failed after {max_retries} attempts: {last_err}")

# -----------------------------
# Eval helpers: 读取预测，兼容 .json 与 .jsonl
# -----------------------------

def load_predictions_compat(dataset: List[Dict[str, Any]], path: str) -> List[Dict[str, Any]]:
    """
    兼容读取：
    - JSON 数组：直接返回（要求每项含 conversation_id/turn_id/answer_entity/time_range/pred_relation(可无)）
    - JSONL：逐行读取
        * 若已有完整字段，直接采纳
        * 若只有 turn_id + raw，则 parse raw 得到 answer_entity/time_range，再通过 turn_id 反查 conversation_id
    """
    preds: List[Dict[str, Any]] = []

    # 尝试当成 JSON 数组
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            # 简单验证
            for it in obj:
                if not isinstance(it, dict):
                    continue
                # 容错：缺失字段时给默认
                preds.append({
                    "conversation_id": it.get("conversation_id"),
                    "turn_id": it.get("turn_id"),
                    "answer_entity": it.get("answer_entity", ""),
                    "time_range": it.get("time_range") or {},
                    "pred_relation": it.get("pred_relation"),
                })
            return preds
        # 如果不是 list，也继续走 JSONL 逻辑
    except Exception:
        pass

    # 当成 JSONL 读取
    turnid2cid = build_turnid_to_cid_map(dataset)
    # 可复用模型输出解析
    for line in open(path, "r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            # 不可解析，跳过该行
            continue

        # 情况 A：JSONL 已是结构化预测
        if all(k in rec for k in ("conversation_id", "turn_id", "answer_entity", "time_range")):
            preds.append({
                "conversation_id": rec.get("conversation_id"),
                "turn_id": rec.get("turn_id"),
                "answer_entity": rec.get("answer_entity", ""),
                "time_range": rec.get("time_range") or {},
                "pred_relation": rec.get("pred_relation"),
            })
            continue

        # 情况 B：debug_raw_model_outputs.jsonl 格式：{"turn_id": "...", "raw": "..."} 或 {"turn_id": "...", "error": "..."}
        tid = rec.get("turn_id")
        raw = rec.get("raw") or ""
        if tid is None:
            # 缺少 turn_id 无法使用
            continue

        obj = parse_model_json(raw) if raw else {}
        answer_entity = ""
        time_range = {}
        pred_relation = None
        if isinstance(obj, dict):
            answer_entity = obj.get("answer_entity", "") or ""
            tr = obj.get("time_range") or {}
            if isinstance(tr, dict):
                time_range = {"start": tr.get("start"), "end": tr.get("end")}
            pr = obj.get("pred_relation")
            if isinstance(pr, str):
                pred_relation = pr

        cid = turnid2cid.get(str(tid))
        preds.append({
            "conversation_id": cid,
            "turn_id": tid,
            "answer_entity": answer_entity,
            "time_range": time_range,
            "pred_relation": pred_relation
        })

    return preds

# -----------------------------
# Evaluation (with progress + S-tags from gold)
# -----------------------------

def evaluate_test(dataset: List[Dict[str, Any]],
                  preds: List[Dict[str, Any]],
                  tcr_threshold: float,
                  log_path: Optional[str],
                  mark_test_correct: bool,
                  strategy_assignment: str):
    gold_map = get_gold_maps(dataset)

    total = 0
    em_hits = 0
    fuzzy_hits = 0
    tcr_hits = 0
    tcr_sum = 0.0

    rel_total = 0
    rel_hits = 0

    strat_keys = ["S1", "S2", "S3", "S4"]
    by_strategy = {k: {"samples": 0, "entity_em": 0, "entity_fuzzy": 0,
                       "time_tcr_hits": 0, "time_tcr_sum": 0.0,
                       "relation_total": 0, "relation_hits": 0}
                   for k in strat_keys}

    missing_s_tags = 0
    correct_ids: List[str] = []
    log_lines: List[str] = []

    def log(msg: str):
        log_lines.append(msg)

    res = {
        "totals": {"samples": 0, "relation_scored_samples": 0},
        "entity": {"EM": 0.0, "Fuzzy": 0.0},
        "time": {"TCR_threshold": tcr_threshold, "Accuracy": 0.0, "Mean_TCR": 0.0},
        "relation": {"Accuracy": 0.0},
        "by_strategy": {},
        "by_strategy_missing": 0,
        "by_strategy_notes": "S-tags 完全来源于 gold.link_strategy；未标注样本不计入任何桶。样本可能多标签计入（multi 模式），合计可能大于总题量。"
    }

    for p in tqdm(preds, desc="Evaluating", unit="turn"):
        cid = p.get("conversation_id")
        tid = p.get("turn_id")
        if cid is None or tid is None:
            log(f"[WARN] Missing conversation_id/turn_id in prediction entry: {p}")
            continue
        gold_turn = gold_map.get((cid, tid))
        if not gold_turn:
            log(f"[WARN] No gold for ({cid},{tid}); skip.")
            continue

        total += 1

        # Entity
        gold_label = gold_answer_label(gold_turn) or ""
        pred_ans = p.get("answer_entity")
        if isinstance(pred_ans, list):
            cand_list = [str(x) for x in pred_ans if x is not None]
        elif pred_ans is None:
            cand_list = []
        else:
            s = str(pred_ans)
            parts = re.split(r"[;/\|,]", s)
            cand_list = [x.strip() for x in parts if x.strip()] or [s.strip()]

        exact_ok = False
        fuzzy_ok = False
        for cand in cand_list:
            if normalize_for_exact(cand) == normalize_for_exact(gold_label):
                exact_ok = True; fuzzy_ok = True; break
            if fuzzy_match(cand, gold_label):
                fuzzy_ok = True
        if exact_ok: em_hits += 1
        if fuzzy_ok: fuzzy_hits += 1

        # Time (TCR)
        gold_range = extract_gold_time_range(gold_turn)
        tr = p.get("time_range") or {}
        pred_range = None
        if isinstance(tr, dict):
            ps = parse_pred_date(tr.get("start"), "start")
            pe = parse_pred_date(tr.get("end"), "end")
            if ps and pe and pe >= ps:
                pred_range = (ps, pe)
        tcr = temporal_coverage_ratio(pred_range, gold_range) if (pred_range and gold_range) else 0.0
        tcr_sum += tcr
        time_ok = (tcr >= tcr_threshold)
        if time_ok: tcr_hits += 1

        # Relation
        first, gold_rel = gold_first_and_relation(gold_turn)
        relation_ok = None
        if not first:
            rel_total += 1
            pred_rel = p.get("pred_relation")
            if isinstance(pred_rel, str) and pred_rel in ("AFTER","BEFORE","OVERLAP","UNKNOWN"):
                if gold_rel is not None and pred_rel == gold_rel:
                    rel_hits += 1; relation_ok = True
                else:
                    relation_ok = False
            else:
                log(f"[WARN] Invalid/missing pred_relation for ({cid},{tid}): {pred_rel}")
                relation_ok = False

        # S-tags
        buckets = assign_strategy_tags(gold_turn, strategy_assignment)
        if not buckets:
            missing_s_tags += 1
        else:
            for b in buckets:
                if b not in by_strategy:
                    continue
                stat = by_strategy[b]
                stat["samples"] += 1
                if exact_ok: stat["entity_em"] += 1
                if fuzzy_ok: stat["entity_fuzzy"] += 1
                if time_ok:  stat["time_tcr_hits"] += 1
                stat["time_tcr_sum"] += tcr
                if not first:
                    stat["relation_total"] += 1
                    if relation_ok:
                        stat["relation_hits"] += 1

        # Overall correctness for test_correct.json
        overall_ok = False
        if fuzzy_ok and time_ok:
            overall_ok = True if first else (gold_rel is not None and p.get("pred_relation") == gold_rel)
        if mark_test_correct and overall_ok:
            correct_ids.append(str(tid))

    # 汇总
    if total > 0:
        res["entity"]["EM"] = round(em_hits / total, 4)
        res["entity"]["Fuzzy"] = round(fuzzy_hits / total, 4)
        res["time"]["Accuracy"] = round(tcr_hits / total, 4)
        res["time"]["Mean_TCR"] = round(tcr_sum / total, 4)
    if rel_total > 0:
        res["relation"]["Accuracy"] = round(rel_hits / rel_total, 4)
    res["totals"]["samples"] = total
    res["totals"]["relation_scored_samples"] = rel_total
    res["by_strategy_missing"] = missing_s_tags

    for b, stat in by_strategy.items():
        s = stat["samples"]; rt = stat["relation_total"]
        res["by_strategy"][b] = {
            "samples": s,
            "entity": {
                "EM": round(stat["entity_em"]/s, 4) if s else 0.0,
                "Fuzzy": round(stat["entity_fuzzy"]/s, 4) if s else 0.0
            },
            "time": {
                "TCR_threshold": tcr_threshold,
                "Accuracy": round(stat["time_tcr_hits"]/s, 4) if s else 0.0,
                "Mean_TCR": round(stat["time_tcr_sum"]/s, 4) if s else 0.0
            },
            "relation": {
                "Accuracy": round(stat["relation_hits"]/rt, 4) if rt else None,
                "scored_samples": rt
            }
        }

    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("=== Evaluation (local qwen + prevQ) ===\n")
            f.write(json.dumps(res, ensure_ascii=False, indent=2))
            f.write("\n")

    if mark_test_correct:
        with open("test_correct.json", "w", encoding="utf-8") as f:
            json.dump({"turn_ids": correct_ids}, f, ensure_ascii=False, indent=2)

    return res

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Full dataset JSON (single list of conversations)")
    ap.add_argument("--shots_conversations", type=int, default=10, help="How many earliest conversations used as few-shots (完整对话)")
    # 旧参数保留但不再使用（为了兼容命令行）；实际已按完整对话取样
    ap.add_argument("--per_dialog_shot_limit", type=int, default=-1, help="[deprecated] ignored in full-conversation mode")
    ap.add_argument("--max_shot_examples", type=int, default=-1, help="[deprecated] ignored in full-conversation mode")

    ap.add_argument("--shots_token_budget", type=int, default=1800,
                    help="few-shot 预算（token），超出则按整轮对话回退；估算规则约 4 chars ≈ 1 token")
    ap.add_argument("--model_context_tokens", type=int, default=3072,
                    help="模型最大上下文 token 上限（与 vLLM --max-model-len 保持一致，仅用于给预算留余量）")
    ap.add_argument("--reserve_for_current_qa", type=int, default=700,
                    help="为 system + 当前问题 + 模型回答预留的 token；避免超限（经验值，可微调）")

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
    ap.add_argument("--eval_test_preds", help="If provided, evaluate this predictions JSON instead of auto eval")
    ap.add_argument("--max_test_turns", type=int, help="Debug only: cap number of test turns for quick run")
    ap.add_argument("--strategy_assignment", choices=["multi", "exclusive"], default="multi",
                    help="multi(默认，同一题可落入多个桶) / exclusive(互斥，优先级 S4>S3>S2>S1)")
    ap.add_argument("--debug_raw_dump", action="store_true", help="Dump raw model outputs to debug_raw_model_outputs.jsonl")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if not isinstance(dataset, list):
        print("Top-level must be a list of conversations.", file=sys.stderr)
        sys.exit(1)

    # Few-shot conversations（完整对话）
    K = max(0, int(args.shots_conversations))
    shots_convs = dataset[:K]           # 最早的 K 轮对话
    test_convs  = dataset[450:528]      # 对话 451–606 作为测试集（保持你的切片）

    # few-shot 预算（为当前问答留余量）
    examples_budget = max(0, args.shots_token_budget)
    if args.model_context_tokens > 0 and args.reserve_for_current_qa > 0:
        hard_cap = max(0, args.model_context_tokens - args.reserve_for_current_qa)
        examples_budget = min(examples_budget, hard_cap)

    # 用完整对话拼 shots，若超预算则整轮回退
    shot_turns: List[Dict[str, Any]] = pack_full_conversation_shots(
        shots_convs,
        k_conversations=K,
        token_budget_for_shots=examples_budget
    )

    # Flatten test
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
        if not args.openai_base_url:
            print("WARN: --openai_base_url 未设置，若使用本地 vLLM 请加 --openai_base_url http://localhost:8000/v1", file=sys.stderr)

        raw_dump_f = None
        if args.debug_raw_dump:
            raw_dump_f = open("debug_raw_model_outputs.jsonl", "w", encoding="utf-8")

        # 每个 conversation 维护历史提问
        questions_history_by_cid: Dict[Any, List[str]] = {}

        for t in tqdm(test_turns, desc="Predicting", unit="turn"):
            q = t.get("question") or t.get("rephrased_question") or ""
            cid = t.get("conversation_id"); tid = t.get("turn_id")

            prev_qs = questions_history_by_cid.get(cid, [])
            messages = build_prompt(shot_turns, q, prev_questions=prev_qs)

            try:
                raw = call_openai(messages,
                                  model=args.openai_model,
                                  temperature=args.temperature,
                                  api_key=args.openai_api_key,
                                  base_url=args.openai_base_url,
                                  max_completion_tokens=args.max_completion_tokens,
                                  json_strict=args.json_strict)
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
                "time_range": obj.get("time_range") or {},
                "pred_relation": obj.get("pred_relation")
            })

            # 更新历史提问
            history_list = questions_history_by_cid.setdefault(cid, [])
            history_list.append(q)

        if raw_dump_f:
            raw_dump_f.close()

        with open(args.preds_out, "w", encoding="utf-8") as f:
            json.dump(preds, f, ensure_ascii=False, indent=2)

    else:
        print("No predictions provided. Use --auto_eval_with_openai or --eval_test_preds.", file=sys.stderr)
        sys.exit(0)

    # Evaluate
    report = evaluate_test(dataset, preds,
                           tcr_threshold=args.tcr_threshold,
                           log_path=args.log_path,
                           mark_test_correct=True,
                           strategy_assignment=args.strategy_assignment)

    with open("evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump({"report": report}, f, ensure_ascii=False, indent=2)

    print("\n=== Final Report ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

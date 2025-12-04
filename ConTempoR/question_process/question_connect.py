# -*- coding: utf-8 -*-
import os
import re
import json
import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
from datetime import datetime, timedelta

try:
    import openai
except Exception:
    openai = None

DATE_SENTINEL_OPEN_END = 20251231
OPEN_END_YMD = (2025, 12, 31)

# -----------------------------
# Strategy Balancer
# -----------------------------
class StrategyBalancer:
    def __init__(self):
        self.names = ["S2_basic_temporal", "S3_delta_temporal", "S4_explicit_delta"]
        self.counts = {n: 0 for n in self.names}
        self.order = list(self.names)
        self.ptr = 0

    def preferred(self, applicable: List[str]) -> Optional[str]:
        if not applicable:
            return None
        min_cnt = min(self.counts.get(n, 0) for n in applicable)
        ties = [n for n in applicable if self.counts.get(n, 0) == min_cnt]
        for k in range(len(self.order)):
            name = self.order[(self.ptr + k) % len(self.order)]
            if name in ties:
                return name
        return applicable[0]

    def record_success(self, name: Optional[str]) -> None:
        if not name:
            return
        if name in self.counts:
            self.counts[name] += 1
            self.ptr = (self.ptr + 1) % len(self.order)

# -----------------------------
# Date / time utils
# -----------------------------
def yyyymmdd_to_date(n: int) -> datetime:
    s = str(n)
    return datetime(int(s[0:4]), int(s[4:6]), int(s[6:8]))

def get_range_from_evidence(arr: Optional[List]) -> Optional[Tuple[datetime, datetime]]:
    if not arr or not isinstance(arr, list) or len(arr) == 0:
        return None
    item = arr[0]
    if not isinstance(item, list) or len(item) < 3:
        return None
    rng = item[2]
    if not rng or not isinstance(rng, list) or len(rng) < 2:
        return None
    try:
        start = yyyymmdd_to_date(int(rng[0]))
        end = yyyymmdd_to_date(int(rng[1]))
        return (start, end)
    except Exception:
        return None

def ranges_overlap(a: Tuple[datetime, datetime], b: Tuple[datetime, datetime]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])

def macro_relation(prev_main: Optional[Tuple[datetime, datetime]],
                   curr_main: Optional[Tuple[datetime, datetime]]) -> str:
    if not prev_main or not curr_main:
        return "UNKNOWN"
    pms, pme = prev_main
    cms, cme = curr_main
    if cms >= pme:
        return "AFTER"
    if cme <= pms:
        return "BEFORE"
    if ranges_overlap(prev_main, curr_main):
        return "OVERLAP"
    return "UNKNOWN"

# -----------------------------
# Entity aliasing
# -----------------------------
QUOTE_MAP = {"\u2019": "'", "\u2018": "'", "\u2032": "'", "\uFF07": "'", "’": "'", "‘": "'"}
def normalize_quotes(s: str) -> str:
    if not isinstance(s, str):
        return s
    for k, v in QUOTE_MAP.items():
        s = s.replace(k, v)
    return s
PAREN_RE = re.compile(r"\s*\([^)]*\)")

def build_aliases(label: str) -> List[str]:
    label = label or ""
    raw = label.strip()
    norm = normalize_quotes(raw)
    raw_nop = PAREN_RE.sub("", raw).strip()
    norm_nop = PAREN_RE.sub("", norm).strip()
    aliases = set()
    for cand in (raw, norm, raw_nop, norm_nop):
        if cand:
            aliases.add(cand)
    return sorted(aliases, key=lambda x: (-len(x), x.lower()))

def word_boundary_regex(alias: str) -> re.Pattern:
    escaped = re.escape(alias)
    return re.compile(rf"(?<!\w)({escaped})(?=\b)(?:\s*)((?:'|’)[sS])?", flags=re.IGNORECASE)

def replace_entity_with_pronoun(text: str, label: str, pronoun_subject: str, pronoun_possessive: str) -> Tuple[str, int, List[str]]:
    aliases = build_aliases(label)
    hit_aliases: List[str] = []
    count = 0
    for alias in aliases:
        pat = word_boundary_regex(alias)
        def _sub(m):
            nonlocal count
            poss = m.group(2)
            count += 1
            return pronoun_possessive if poss else pronoun_subject
        new_text, n = pat.subn(_sub, text)
        if n > 0:
            hit_aliases.append(alias)
        text = new_text
    text = re.sub(r"\s+", " ", text).strip()
    return text, count, hit_aliases

# -----------------------------
# Pronouns
# -----------------------------
def probe_gender_with_gpt(name: str, api_key: Optional[str], model: str, temperature: float) -> Tuple[str, str]:
    if not api_key or openai is None:
        return ("unknown", "none")
    openai.api_key = api_key
    prompt = (
        "You are a classifier. Given a person's full name, guess their likely gender strictly as one of: "
        '"male", "female", or "unknown". If you are not confident, return "unknown".'
        f"\nName: {name}\n"
        'Answer only JSON: {"gender": "male|female|unknown"}'
    )
    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=10
        )
        txt = resp.choices[0].message.content.strip()
        m = re.search(r'"gender"\s*:\s*"(?P<g>male|female|unknown)"', txt, re.IGNORECASE)
        if m:
            return (m.group("g").lower(), "gpt")
    except Exception:
        pass
    return ("unknown", "none")

def pronouns_for_type(entity_type: Optional[str], gender: Optional[str]) -> Tuple[str, str, str]:
    et = (entity_type or "").lower()
    g = (gender or "unknown").lower()
    if et in ("human", "person"):
        if g == "male":
            return ("he", "his", "him")
        if g == "female":
            return ("she", "her", "her")
        return ("this person", "the person's", "this person")
    return ("it", "its", "it")

# -----------------------------
# Temporal phrases
# -----------------------------
BASIC_AFTER = ["After that,", "Later,", "Subsequently,", "Then,"]
BASIC_BEFORE = ["Before that,", "Earlier,", "Prior to that,"]
BASIC_OVERLAP = ["Meanwhile,", "During that time,", "At the same time,"]

ABOUT_SYNS = ["About", "Approximately", "Roughly"]
LATER_SYNS = ["later", "thereafter", "afterward", "afterwards"]
EARLIER_SYNS = ["earlier", "previously"]

# For S4 overlap text only (S3 never uses these)
OVERLAP_SYNS = ["During that time,", "In that period,", "Over that period,"]

def add_basic_marker(question_text: str, macro: str, rng: random.Random) -> Optional[str]:
    if macro == "AFTER":
        return rng.choice(BASIC_AFTER) + " " + question_text
    if macro == "BEFORE":
        return rng.choice(BASIC_BEFORE) + " " + question_text
    if macro == "OVERLAP":
        return rng.choice(BASIC_OVERLAP) + " " + question_text
    return None

# -----------------------------
# S4 helpers
# -----------------------------
def end_is_open_end(r: Optional[Tuple[datetime, datetime]]) -> bool:
    if not r:
        return False
    return (r[1].year, r[1].month, r[1].day) == OPEN_END_YMD

def earlier_range(a: Tuple[datetime, datetime], b: Tuple[datetime, datetime]) -> Tuple[Tuple[datetime, datetime], Tuple[datetime, datetime]]:
    return (a, b) if a[0] <= b[0] else (b, a)

def calendar_diff_to_ymw(start: datetime, end: datetime) -> Tuple[int, int, int]:
    y = end.year - start.year
    m = end.month - start.month
    d = end.day - start.day
    if d < 0:
        prev_month = end.replace(day=1) - timedelta(days=1)
        d += prev_month.day
        m -= 1
    if m < 0:
        m += 12
        y -= 1
    w = d // 7
    return max(0, y), max(0, m), max(0, w)

def plural(n: int, unit: str) -> str:
    return f"{n} {unit}" + ("" if n == 1 else "s")

def format_ymw_commas(y: int, m: int, w: int) -> str:
    parts = []
    if y > 0:
        parts.append(plural(y, "year"))
    if m > 0:
        parts.append(plural(m, "month"))
    if w > 0:
        parts.append(plural(w, "week"))
    return ", ".join(parts) if parts else "0 weeks"

TEMPORAL_PREFIX_RE = re.compile(
    r"^\s*(?:After that,|Later,|Subsequently,|Then,|Before that,|Earlier,|Prior to that,|Meanwhile,|During that time,|At the same time,)\s+",
    flags=re.IGNORECASE
)

IMPLICIT_TEMP_WORDS = [
    "before", "after", "during", "while", "earlier", "later",
    "previously", "thereafter", "afterward", "afterwards", "meanwhile", "prior"
]
SENT_SPLIT_RE = re.compile(r"([.!?])")

def remove_temporal_subordinate_clauses(q: str) -> str:
    q = TEMPORAL_PREFIX_RE.sub("", q).strip()
    tokens = SENT_SPLIT_RE.split(q)
    if not tokens:
        return q
    rebuilt = []
    i = 0
    while i < len(tokens):
        seg = tokens[i]
        punct = tokens[i+1] if i+1 < len(tokens) else ""
        i += 2

        seg_stripped = seg.lstrip()
        lead_spaces = seg[:len(seg) - len(seg_stripped)]
        low = seg_stripped.lower()

        hit_pos = -1
        for w in IMPLICIT_TEMP_WORDS:
            pos = low.find(w)
            if pos > 0:
                hit_pos = pos
                break

        if hit_pos > 0:
            seg_clean = seg_stripped[:hit_pos].rstrip()
            rebuilt.append(lead_spaces + seg_clean)
        else:
            rebuilt.append(lead_spaces + seg_stripped)

        if punct:
            rebuilt.append(punct)

    out = "".join(rebuilt)
    out = re.sub(r"\s+\.", ".", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out

def build_s4_prefix(prev_main: Tuple[datetime, datetime],
                    curr_main: Tuple[datetime, datetime],
                    rng: random.Random) -> Tuple[Optional[str], Optional[str]]:
    pms, pme = prev_main
    cms, cme = curr_main
    if cms >= pme:
        y, m, w = calendar_diff_to_ymw(pme, cms)
        if (y + m + w) <= 0:
            return (None, None)
        about = rng.choice(ABOUT_SYNS)
        later = rng.choice(LATER_SYNS)
        dur = format_ymw_commas(y, m, w)
        return (f"{about} {dur} {later},", "S4_later")
    if cme <= pms:
        y, m, w = calendar_diff_to_ymw(cme, pms)
        if (y + m + w) <= 0:
            return (None, None)
        about = rng.choice(ABOUT_SYNS)
        earlier = rng.choice(EARLIER_SYNS)
        dur = format_ymw_commas(y, m, w)
        return (f"{about} {dur} {earlier},", "S4_earlier")
    overlap = random.choice(OVERLAP_SYNS)
    return (overlap, "S4_overlap")

def s4_earlier_has_open_end(prev_main: Tuple[datetime, datetime],
                            curr_main: Tuple[datetime, datetime]) -> bool:
    earlier, _ = earlier_range(prev_main, curr_main)
    e_end = earlier[1]
    return (e_end.year, e_end.month, e_end.day) == OPEN_END_YMD

# -----------------------------
# Validation helpers
# -----------------------------
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?\b", re.IGNORECASE)

def has_explicit_dates(text: str) -> bool:
    return bool(YEAR_RE.search(text) or DATE_RE.search(text))

def extract_prev_answer_qid(prev_question: Dict[str, Any]) -> Optional[str]:
    ans = prev_question.get("answer")
    if not ans:
        return None
    if isinstance(ans, str):
        m = re.search(r"\bQ\d+\b", ans)
        return m.group(0) if m else None
    if isinstance(ans, dict):
        for v in ans.values():
            if isinstance(v, str):
                m = re.search(r"\bQ\d+\b", v)
                if m: return m.group(0)
    if isinstance(ans, list):
        for x in ans:
            if isinstance(x, str):
                m = re.search(r"\bQ\d+\b", x)
                if m: return m.group(0)
        for x in ans:
            if isinstance(x, dict):
                q = x.get("id")
                if isinstance(q, str) and re.match(r"^Q\d+$", q):
                    return q
    return None

def text_contains_any_marker(text: str, markers: List[str]) -> bool:
    lowered = text.lower()
    return any(m.lower() in lowered for m in markers)

def text_contains_alias(text: str, label: str) -> bool:
    for alias in build_aliases(label):
        pat = re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)", flags=re.IGNORECASE)
        if pat.search(text):
            return True
    return False

def text_contains_any_of(text: str, needles: List[str]) -> bool:
    low = text.lower()
    return any(n.lower() in low for n in needles)

# -----------------------------
# S3 bucketing (new rules)
# -----------------------------
HALF_YEAR_DAYS = 182
YEAR_DAYS = 365

def _fmt_year_value(v: float) -> str:
    # Render 0.5, 1.0, 1.5 ... without trailing .0
    if abs(v - int(v)) < 1e-9:
        return str(int(v))
    return f"{v:.1f}".rstrip("0").rstrip(".")

def s3_select_bucket_label(delta_days: int, direction: str) -> Optional[str]:
    """
    Return a label like:
      - 'about 1 month later', ..., 'about 6 months later'  (<= 6 months)
      - 'about 0.5 years later', 'about 1 year later', ... up to 10y  (>6m to 10y)
      - 'about 1 decade later', 'about 2 decades later', ...          (>10y)
    Choose the largest bucket that does NOT exceed the actual delta.
    direction: 'later' | 'earlier'
    """
    d = max(1, delta_days)

    # Months region: <= 6 months
    months = d // 30  # floor
    if months == 0:
        months = 1
    if months <= 6:
        unit = "month" if months == 1 else "months"
        return f"about {months} {unit} {direction}"

    # Years region: > 6 months and <= 10 years, use half-year steps
    years = d / YEAR_DAYS
    if years <= 10.0:
        # largest k*0.5 <= years
        k = int((years + 1e-9) // 0.5)  # avoid floating glitch
        if k < 1:
            k = 1
        value = 0.5 * k
        val_str = _fmt_year_value(value)
        unit = "year" if abs(value - 1.0) < 1e-9 else "years"
        return f"about {val_str} {unit} {direction}"

    # Decades region: > 10 years
    decades = int(years // 10)
    if decades < 1:
        decades = 1
    unit = "decade" if decades == 1 else "decades"
    return f"about {decades} {unit} {direction}"

# -----------------------------
# GPT polish
# -----------------------------
def gpt_polish_question(question_text: str, api_key: str, model: str, temperature: float = 0.0) -> Optional[str]:
    if not api_key or openai is None:
        return None
    openai.api_key = api_key
    prompt = (
        "You are a careful editor. Your task is to polish questions that were already constructed "
        "according to strict rules. You must follow these constraints:\n\n"
        "1. Do not change the meaning of the question.\n"
        "2. Do not add, remove, or replace any entities or pronouns. "
        "Entities may already have been replaced by pronouns (he, she, it, this person, etc.) — keep them.\n"
        "3. Do not introduce explicit calendar dates (years like 1990/2020; dd/mm; mm-dd). "
        "Durations like '1 year, 2 months' are allowed.\n"
        "4. If the question begins with a temporal connector, keep it but you may adjust word order "
        "or synonyms to make it natural.\n"
        "5. If the pronoun form sounds awkward, you may adjust wording, but do not change who it refers to.\n"
        "6. Always output exactly one fluent English question.\n\n"
        f"Question to polish:\n{question_text}\n\nPolished Question:"
    )
    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role":"user","content":prompt}],
            temperature=temperature,
            max_tokens=128
        )
        txt = resp.choices[0].message.content.strip()
        return txt
    except Exception as e:
        print(f"[Warning] GPT polish failed: {e}")
        return None

# -----------------------------
# Core
# -----------------------------
def process_conversation(conversation: List[Any], args, rng_global: random.Random,
                         gender_cache: Dict[str,str],
                         balancer: 'StrategyBalancer',
                         progress_cb: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
                         conversation_index: int = 0) -> List[Any]:
    out_conversation: List[Any] = []
    rng = random.Random(rng_global.randint(0, 10**9))

    for idx, q in enumerate(conversation):
        if not isinstance(q, dict):
            out_conversation.append(q)
            if progress_cb:
                try: progress_cb(conversation_index, idx, {})
                except Exception: pass
            continue

        rq = q.get("rephrased_question", "")
        if not isinstance(rq, str) or not rq.strip():
            out_conversation.append(q)
            if progress_cb:
                try: progress_cb(conversation_index, idx, q)
                except Exception: pass
            continue

        ev = q.get("evidence", {}) or {}
        main_r = get_range_from_evidence(ev.get("main"))
        cons_r = get_range_from_evidence(ev.get("constraint"))  # unused for S3 now

        topic = q.get("topic_entity", {}) or {}
        t_id = topic.get("id")
        t_label = topic.get("label", "")
        t_type = topic.get("type")

        # First question → no linking
        if idx == 0:
            final = rq
            if not args.disable_gpt_polish:
                polished = gpt_polish_question(final, args.api_key, args.polish_model, temperature=0.0)
                if polished and not has_explicit_dates(polished):
                    final = polished
            q["question"] = final
            q["link_strategy"] = "none"
            q["checks"] = {"first_in_conversation": True}
            out_conversation.append(q)
            if progress_cb:
                try: progress_cb(conversation_index, idx, q)
                except Exception: pass
            continue

        prev = conversation[idx-1] if idx-1 >= 0 else None
        prev_ev = prev.get("evidence", {}) if isinstance(prev, dict) else {}
        prev_main = get_range_from_evidence(prev_ev.get("main"))
        prev_cons = get_range_from_evidence(prev_ev.get("constraint"))
        prev_answer_qid = extract_prev_answer_qid(prev) if isinstance(prev, dict) else None
        prev_topic = prev.get("topic_entity", {}) if isinstance(prev, dict) else {}
        prev_label = prev_topic.get("label", "")

        macro = macro_relation(prev_main, main_r)
        same_topic = (t_label and prev_label and
                      normalize_quotes(t_label).lower().strip() ==
                      normalize_quotes(prev_label).lower().strip())
        ans_to_entity = bool(prev_answer_qid and t_id and prev_answer_qid == t_id)

        prev_link_strategy = out_conversation[-1].get("link_strategy") if (out_conversation and isinstance(out_conversation[-1], dict)) else None
        s1_forbidden = (prev_link_strategy is not None and "S1_pronoun" in str(prev_link_strategy) and (not same_topic))

        attempts_left = max(1, args.max_tries)
        success = False
        failure_reasons: List[str] = []
        gender_source = "none"

        while attempts_left > 0 and not success:
            attempts_left -= 1

            q_candidate = rq
            link_parts: List[str] = []
            checks_all: Dict[str, Any] = {}

            # ----- S1 (optional) -----
            did_s1 = False
            s1_applicable = (not s1_forbidden) and (same_topic or ans_to_entity)
            if s1_applicable and (rng.random() < float(args.s1_prob)):
                subj, poss, obj = ("it", "its", "it")
                if (t_type or "").lower() in ("human", "person"):
                    g = None
                    if t_id and t_id in gender_cache:
                        g = gender_cache[t_id];  gender_source = "cache"
                    else:
                        g, gender_source = probe_gender_with_gpt(name=t_label, api_key=args.api_key, model=args.gender_model, temperature=0.0)
                        if t_id and g:
                            gender_cache[t_id] = g
                    subj, poss, obj = pronouns_for_type(t_type, g)

                s1_text, rep_count, aliases_hit = replace_entity_with_pronoun(q_candidate, t_label, subj, poss)
                if rep_count > 0 and not has_explicit_dates(s1_text) and not text_contains_alias(s1_text, t_label):
                    q_candidate = s1_text
                    did_s1 = True
                    link_parts.append("S1_pronoun")
                    checks_all["S1"] = {"pre_polish_ok": True, "aliases_hit": aliases_hit, "gender_source": gender_source}
                else:
                    checks_all["S1"] = {"skipped_or_failed": True}

            # ----- Temporal: exactly one of S2/S3/S4 -----
            did_temporal = False
            temporal_used = None
            temporal_meta: Dict[str, Any] = {}

            applicable: List[str] = []

            # S4 applicability (unchanged)
            s4_basic_ok = bool(prev_main and main_r)
            if s4_basic_ok and not s4_earlier_has_open_end(prev_main, main_r):
                applicable.append("S4_explicit_delta")
            else:
                checks_all["S4"] = {"not_applicable_or_banned": True}

            # S2 applicability
            if macro != "UNKNOWN":
                applicable.append("S2_basic_temporal")

            # S3 applicability (MAIN-ONLY; no overlap; earlier main cannot be open-end)
            s3_ok = True
            if macro not in ("AFTER", "BEFORE"):
                s3_ok = False
            if not prev_main or not main_r:
                s3_ok = False
            if s3_ok and s4_earlier_has_open_end(prev_main, main_r):
                s3_ok = False
            if s3_ok:
                applicable.append("S3_delta_temporal")

            # Choose order
            if applicable:
                primary = balancer.preferred(applicable)
                try_order = [primary] + [s for s in applicable if s != primary]
            else:
                try_order = []

            for strat in try_order:
                if strat == "S4_explicit_delta":
                    prefix, variant = build_s4_prefix(prev_main, main_r, rng)
                    if not prefix:
                        failure_reasons.append("S4_prefix_build_failed")
                        continue
                    base = remove_temporal_subordinate_clauses(q_candidate)
                    q2 = f"{prefix} {base}".strip()
                    if has_explicit_dates(q2):
                        failure_reasons.append("S4_explicit_calendar_date_detected")
                        continue
                    q_candidate = q2
                    did_temporal = True
                    temporal_used = "S4_explicit_delta"
                    temporal_meta = {"macro": macro, "variant": variant}
                    balancer.record_success(temporal_used)
                    break

                if strat == "S2_basic_temporal":
                    q2 = add_basic_marker(q_candidate, macro, rng)
                    if not q2:
                        failure_reasons.append("S2_no_marker")
                        continue
                    if has_explicit_dates(q2):
                        failure_reasons.append("S2_explicit_date_in_text")
                        continue
                    markers = BASIC_AFTER if macro=="AFTER" else BASIC_BEFORE if macro=="BEFORE" else BASIC_OVERLAP
                    if not text_contains_any_marker(q2, markers):
                        failure_reasons.append("S2_marker_missing_post")
                        continue
                    q_candidate = q2
                    did_temporal = True
                    temporal_used = "S2_basic_temporal"
                    temporal_meta = {"macro": macro}
                    balancer.record_success(temporal_used)
                    break

                if strat == "S3_delta_temporal":
                    if macro not in ("AFTER", "BEFORE") or (not prev_main) or (not main_r):
                        failure_reasons.append("S3_missing_prereq_main_only")
                        continue

                    pms, pme = prev_main
                    cms, cme = main_r

                    if macro == "AFTER":
                        delta_days = (cms - pme).days
                        if delta_days <= 0:
                            failure_reasons.append("S3_non_positive_delta_days_main_only")
                            continue
                        label = s3_select_bucket_label(delta_days, "later")
                    else:  # BEFORE
                        delta_days = (pms - cme).days
                        if delta_days <= 0:
                            failure_reasons.append("S3_non_positive_delta_days_main_only")
                            continue
                        label = s3_select_bucket_label(delta_days, "earlier")

                    if not label:
                        failure_reasons.append("S3_no_bucket_selected")
                        continue

                    q3 = f"{label[0].upper() + label[1:]}, {q_candidate}"
                    if has_explicit_dates(q3):
                        failure_reasons.append("S3_explicit_date_in_text")
                        continue

                    q_candidate = q3
                    did_temporal = True
                    temporal_used = "S3_delta_temporal"
                    temporal_meta = {"macro": macro, "bucket": label}
                    balancer.record_success(temporal_used)
                    break

            if did_temporal:
                link_parts.append(temporal_used)
                if temporal_used == "S2_basic_temporal":
                    checks_all[temporal_used] = {"pre_polish_ok": True, "macro": macro}
                elif temporal_used == "S3_delta_temporal":
                    checks_all[temporal_used] = {"pre_polish_ok": True}
                    checks_all["S3_meta"] = temporal_meta
                else:
                    checks_all[temporal_used] = {"pre_polish_ok": True, **temporal_meta}

            # ----- GPT polish -----
            polished = None
            if not args.disable_gpt_polish:
                polish_ok = False
                for _ in range(max(1, args.max_polish_tries)):
                    polished = gpt_polish_question(q_candidate, args.api_key, args.polish_model, temperature=0.0)
                    if not polished:
                        continue
                    if has_explicit_dates(polished):
                        continue
                    if did_s1 and text_contains_alias(polished, t_label):
                        continue
                    if did_temporal and temporal_used == "S2_basic_temporal":
                        markers = BASIC_AFTER if macro=="AFTER" else BASIC_BEFORE if macro=="BEFORE" else BASIC_OVERLAP
                        if not text_contains_any_marker(polished, markers):
                            continue
                    if did_temporal and temporal_used == "S3_delta_temporal":
                        low = polished.lower()
                        # Must include later/earlier and a time unit; must NOT include within/during
                        banned = ("within", "during")
                        requires = ("later", "earlier")
                        units = ("month", "months", "year", "years", "decade", "decades")
                        if any(b in low for b in banned):
                            continue
                        if not any(r in low for r in requires):
                            continue
                        if not any(u in low for u in units):
                            continue
                    if did_temporal and temporal_used == "S4_explicit_delta":
                        low = polished.lower()
                        if not (text_contains_any_of(low, LATER_SYNS) or
                                text_contains_any_of(low, EARLIER_SYNS) or
                                text_contains_any_of(low, [s.rstrip(",") for s in OVERLAP_SYNS])):
                            continue
                    polish_ok = True
                    break
                if polish_ok:
                    q_candidate = polished
                    checks_all["gpt_polish"] = "accepted"
                else:
                    checks_all["gpt_polish"] = "rejected_or_failed"
                    failure_reasons.append("polish_invalid")
            else:
                checks_all["gpt_polish"] = "disabled"

            # ----- Final validations -----
            if has_explicit_dates(q_candidate):
                failure_reasons.append("post_explicit_calendar_date")
                continue
            if did_temporal and temporal_used == "S2_basic_temporal":
                markers = BASIC_AFTER if macro=="AFTER" else BASIC_BEFORE if macro=="BEFORE" else BASIC_OVERLAP
                if not text_contains_any_marker(q_candidate, markers):
                    failure_reasons.append("post_S2_marker_missing")
                    continue
            if did_s1 and text_contains_alias(q_candidate, t_label):
                failure_reasons.append("post_S1_label_reintroduced")
                continue

            # Success
            q["question"] = q_candidate
            q["link_strategy"] = "+".join(link_parts) if link_parts else "none"
            q["checks"] = checks_all
            q["gender_source"] = gender_source if did_s1 else "none"
            q["temporal_link"] = temporal_meta if did_temporal else {"macro": macro}
            q["debug_aliases"] = build_aliases(t_label)
            success = True

        if not success:
            q["question"] = rq
            q["link_strategy"] = "none"
            q["checks"] = {"all_attempts_failed": True, "reasons": failure_reasons}
            q["gender_source"] = "none"
            q["temporal_link"] = {"macro": macro}
        out_conversation.append(q)

        if progress_cb:
            try: progress_cb(conversation_index, idx, q)
            except Exception: pass

    return out_conversation

# -----------------------------
# Process big list
# -----------------------------
def process_big_list(data: List[Any], args) -> List[Any]:
    rng_global = random.Random(args.seed)
    gender_cache: Dict[str,str] = {}
    if args.gender_cache and Path(args.gender_cache).is_file():
        try:
            gender_cache = json.loads(Path(args.gender_cache).read_text(encoding="utf-8"))
        except Exception:
            gender_cache = {}

    balancer = StrategyBalancer()

    total_questions = 0
    for conv in data:
        total_questions += len(conv) if isinstance(conv, list) else 1
    processed = 0

    def _progress_cb(conv_idx: int, q_idx: int, q: Dict[str, Any]):
        nonlocal processed, total_questions
        if not getattr(args, "progress", False):
            return
        processed += 1
        topic = (q or {}).get("topic_entity") or {}
        label = topic.get("label") or ""
        label_display = f" — {label}" if label else ""
        print(f"[{processed}/{total_questions}] conversation {conv_idx+1}, question {q_idx+1}{label_display}")

    out: List[Any] = []
    for conv_idx, conv in enumerate(data):
        if isinstance(conv, list):
            out.append(
                process_conversation(
                    conv, args, rng_global, gender_cache,
                    balancer=balancer,
                    progress_cb=_progress_cb,
                    conversation_index=conv_idx
                )
            )
        else:
            out.append(conv)
            _progress_cb(conv_idx, 0, conv if isinstance(conv, dict) else {})

    if args.gender_cache:
        Path(args.gender_cache).parent.mkdir(parents=True, exist_ok=True)
        Path(args.gender_cache).write_text(
            json.dumps(gender_cache, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    return out

# -----------------------------
# CLI
# -----------------------------
def main():
    p = argparse.ArgumentParser(description="Build final questions with pronoun & temporal linking (S1 + exactly one of S2/S3/S4; balanced usage; months/half-years/decades buckets for S3).")
    p.add_argument("--input", required=True, help="Input JSON (list of conversations).")
    p.add_argument("--output", required=True, help="Output JSON.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    p.add_argument("--gender-model", default=os.environ.get("OPENAI_GENDER_MODEL", "gpt-4o-mini"))
    p.add_argument("--polish-model", default=os.environ.get("OPENAI_POLISH_MODEL", "gpt-4o-mini"))
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--gender-cache", default=".cache/gender.json")
    p.add_argument("--disable-gpt-polish", action="store_true")
    p.add_argument("--max-tries", type=int, default=9)
    p.add_argument("--max-polish-tries", type=int, default=1)
    p.add_argument("--s1-prob", type=float, default=0.33)
    p.add_argument("--progress", action="store_true")

    args = p.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Top-level JSON must be a list of conversations.")

    out = process_big_list(data, args)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Done. Wrote {args.output}")

if __name__ == "__main__":
    main()

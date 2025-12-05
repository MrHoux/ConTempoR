# -*- coding: utf-8 -*-
import os
import json
import time
import pickle
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from filelock import FileLock
import openai

from ConTempoR.library.utils import get_logger, format_text

REPHRASE_PROMPT = """Please rephrase the following input question into a more natural question.

Input: What album Sting ( musician ) was released, during, Sting award received German Radio Award?
Question: which album was released by Sting when he won the German Radio Award?

Input: What human President of Bolivia was the second and most recent female president, after, president of Bolivia officeholder Evo Morales?
Question: Which female president succeeded Evo Morales in Bolivia?

Input: What lake David Bowie He moved to Switzerland purchasing a chalet in the hills to the north of , during, David Bowie spouse Angela Bowie?
Question: Close to which lake did David Bowie buy a chalet while he was married to Angela Bowie?

Input: What human Robert Motherwell spouse, during, Robert Motherwell He also edited Paalen 's collected essays Form and Sense as the first issue of Problems of Contemporary Art?
Question: Who was Robert Motherwell's wife when he edited Paalen's collected essays Form and Sense?

Input: What historical country Independent State of Croatia the NDH government signed an agreement with which demarcated their borders, during, Independent State of Croatia?
Question: At the time of the Independent State of Croatia, which country signed an agreement with the NDH government to demarcate their borders?

Input: What U-boat flotilla German submarine U-559 part of, before, German submarine U-559 She moved to the 29th U-boat Flotilla?
Question: Which U-boat flotilla did the German submarine U-559 belong to before being transferred to the 29th U-boat Flotilla?

Input: What human UEFA chairperson, during, UEFA chairperson Sandor Barcs?
Question: Who was the UEFA chairperson after Sandor Barcs?

Input: What human Netherlands head of government, during, Netherlands head of state Juliana of the Netherlands?
Question: During Juliana of the Netherlands' time as queen, who was the prime minister in the Netherlands?

Input: """

SUPPORTED_CHAT_MODELS = {"gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "gpt-4.1", "gpt-4.1-mini"}
SUPPORTED_COMPLETION_MODELS = {"text-davinci-003"}

def check_signal_same(rephrased_question: str, signal: Optional[str]) -> Optional[bool]:
    if not rephrased_question or not signal:
        return None
    rq = rephrased_question.lower()
    if signal == "OVERLAP":
        return not ("after" in rq or "before" in rq)
    elif signal in ("BEFORE", "AFTER"):
        for bad in ("during", "while", "in the year", "when", "at the time"):
            if bad in rq:
                return False
        return True
    return None

def temporal_guardrail_prompt(signal: Optional[str]) -> str:
    """Hard constraint prompt for the second rewrite to preserve the temporal relation."""
    if signal == "OVERLAP":
        return (
            "IMPORTANT: Preserve the temporal relation: OVERLAP. "
            "Do NOT use 'after' or 'before'. "
            "You may use 'during' or equivalent phrasing that implies overlap.\n"
        )
    elif signal == "BEFORE":
        return (
            "IMPORTANT: Preserve the temporal relation: BEFORE. "
            "The question MUST imply that the target happened BEFORE the reference. "
            "Do NOT use words implying simultaneity such as 'during', 'while', 'in the year', 'when', or 'at the time'.\n"
        )
    elif signal == "AFTER":
        return (
            "IMPORTANT: Preserve the temporal relation: AFTER. "
            "The question MUST imply that the target happened AFTER the reference. "
            "Do NOT use words implying simultaneity such as 'during', 'while', 'in the year', 'when', or 'at the time'.\n"
        )
    return ""

class GPTCache:
    def __init__(self, cache_path: Path, logger):
        self.cache_path = cache_path
        self.version_path = cache_path.with_suffix(cache_path.suffix + ".version")
        self.logger = logger
        self.cache: Dict[str, str] = {}
        self.cache_version = ""
        self.cache_changed = False
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_cache()

    def _init_cache(self):
        if self.cache_path.is_file():
            self.logger.info(f"[Cache] Loading: {self.cache_path}")
            with FileLock(str(self.cache_path) + ".lock"):
                self.cache_version = self._read_cache_version()
                self.cache = self._read_cache()
        else:
            self.logger.info(f"[Cache] New: {self.cache_path}")
            with FileLock(str(self.cache_path) + ".lock"):
                self._write_cache({})
                self._write_cache_version()

    def _read_cache(self) -> Dict[str, str]:
        with open(self.cache_path, "rb") as fp:
            return pickle.load(fp)

    def _write_cache(self, cache: Dict[str, str]):
        with open(self.cache_path, "wb") as fp:
            pickle.dump(cache, fp)

    def _read_cache_version(self) -> str:
        if not self.version_path.is_file():
            self._write_cache_version()
        with open(self.version_path, "r") as fp:
            return fp.readline().strip()

    def _write_cache_version(self):
        version = str(time.time())
        with open(self.version_path, "w") as fp:
            fp.write(version)
        self.cache_version = version

    def get(self, key: str) -> Optional[str]:
        return self.cache.get(key)

    def put(self, key: str, val: str):
        self.cache[key] = val
        self.cache_changed = True

    def flush_if_needed(self):
        if not self.cache_changed:
            return
        with FileLock(str(self.cache_path) + ".lock"):
            current_ver = self._read_cache_version()
            if current_ver != self.cache_version:
                updated = self._read_cache()
                updated.update(self.cache)
                self._write_cache(updated)
                self._write_cache_version()
                self.logger.info("[Cache] Merged & wrote.")
            else:
                self._write_cache(self.cache)
                self._write_cache_version()
                self.logger.info("[Cache] Wrote.")

def call_openai_chat(model: str, prompt: str, api_key: str, org: Optional[str],
                     temperature: float, max_tokens: int, logger, retries: int = 3, backoff: float = 1.5) -> Optional[str]:
    openai.api_key = api_key
    if org:
        openai.organization = org
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(1, retries + 1):
        try:
            resp = openai.ChatCompletion.create(model=model, messages=messages,
                                                temperature=temperature, max_tokens=max_tokens)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"[OpenAI] ChatCompletion failed ({attempt}/{retries}): {e}")
            if attempt < retries:
                time.sleep(backoff ** attempt)
    return None

def call_openai_completion(model: str, prompt: str, api_key: str, org: Optional[str],
                           temperature: float, max_tokens: int, logger, retries: int = 3, backoff: float = 1.5) -> Optional[str]:
    openai.api_key = api_key
    if org:
        openai.organization = org
    for attempt in range(1, retries + 1):
        try:
            resp = openai.Completion.create(model=model, prompt=prompt, temperature=temperature,
                                            max_tokens=max_tokens, top_p=1, frequency_penalty=0, presence_penalty=0)
            return resp["choices"][0]["text"].strip()
        except Exception as e:
            logger.warning(f"[OpenAI] Completion failed ({attempt}/{retries}): {e}")
            if attempt < retries:
                time.sleep(backoff ** attempt)
    return None

def gpt_rephrase_once(pseudo_question: str, model: str, api_key: str, org: Optional[str],
                      temperature: float, max_tokens: int, cache: Optional[GPTCache],
                      logger, extra_prefix: str = "",
                      bypass_cache_get: bool = False,  # Added: skip cache reads
                      allow_cache_put: bool = True     # Added: allow writing new results back to cache
                      ) -> Optional[str]:
    """Single rewrite (optionally prepend extra_prefix).
       If bypass_cache_get=True, skip cache reads; if allow_cache_put=True, write new results back to cache.
    """
    pq = format_text(pseudo_question).encode("utf-8").decode("utf-8")
    prompt = (extra_prefix or "") + REPHRASE_PROMPT + pq + "\nQuestion:"

    # Skip cache reads when requested
    if cache and not bypass_cache_get:
        cached = cache.get(prompt)
        if cached:
            logger.info("[Rephrase] Cache hit.")
            return cached

    # Call OpenAI
    if model in SUPPORTED_CHAT_MODELS:
        out = call_openai_chat(model, prompt, api_key, org, temperature, max_tokens, logger)
    elif model in SUPPORTED_COMPLETION_MODELS:
        out = call_openai_completion(model, prompt, api_key, org, temperature, max_tokens, logger)
    else:
        raise ValueError(f"Unsupported model: {model}")

    # Allow writing back to cache (overwrite same key)
    if out and cache and allow_cache_put:
        cache.put(prompt, out)
    return out


def gpt_rephrase_with_signal_retry(pseudo_question: str, signal: Optional[str], model: str, api_key: str, org: Optional[str],
                                   temperature: float, max_tokens: int, cache: Optional[GPTCache], logger,
                                   enable_retry: bool, retry_max: int,
                                   bypass_first_get: bool = False  # Added: skip cache read on first rewrite
                                   ) -> (Optional[str], Optional[bool], int):
    # First attempt: decide whether to bypass cache read
    rq = gpt_rephrase_once(pseudo_question, model, api_key, org, temperature, max_tokens,
                           cache, logger, extra_prefix="",
                           bypass_cache_get=bypass_first_get,  # Critical
                           allow_cache_put=True)
    sc = check_signal_same(rq, signal) if rq else None
    retries = 0
    if sc is not False or not enable_retry:
        return rq, sc, retries

    # Second rewrite with temporal constraint; cache usage is fine because the prompt differs
    prefix = temporal_guardrail_prompt(signal)
    for i in range(retry_max):
        retries += 1
        rq2 = gpt_rephrase_once(pseudo_question, model, api_key, org, temperature, max_tokens,
                                cache, logger, extra_prefix=prefix,
                                bypass_cache_get=False, allow_cache_put=True)
        sc2 = check_signal_same(rq2, signal) if rq2 else None
        if sc2 is not False:
            return rq2, sc2, retries
    return rq, sc, retries

def process_nested_list(big_list: List[Any], args, cache: Optional[GPTCache], logger) -> List[Any]:
    """
    big_list: top-level large list; each element should be a small list;
    for each dict inside a small list, process items that contain 'pseudo_question_construction'.
    """
    if not isinstance(big_list, list):
        raise ValueError("Input JSON top-level must be a list (big list).")

    out_big: List[Any] = []

    for small in big_list:
        if not isinstance(small, list):
            out_big.append(small)
            continue

        new_small: List[Any] = []
        for item in small:
            if isinstance(item, dict) and "pseudo_question_construction" in item:
                # Skip GPT call if it already passed the signal check
                if item.get("signal_check") is True:
                    logger.info("[Skip] Already passed signal check; skip GPT call.")
                    new_small.append(item)
                    continue

                # Decide whether to bypass cache reads: only when signal_check is False
                bypass_first_get = (item.get("signal_check") is False)

                pqc = item.get("pseudo_question_construction")
                rq, sc, retry_cnt = gpt_rephrase_with_signal_retry(
                    pqc,
                    item.get("signal"),
                    model=args.model,
                    api_key=args.api_key,
                    org=args.organization,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    cache=cache,
                    logger=logger,
                    enable_retry=args.signal_retry,
                    retry_max=args.signal_retry_max,
                    bypass_first_get=bypass_first_get  # Critical
                )
                if rq:
                    item["rephrased_question"] = rq
                    item["rephrased_question_length"] = len(rq.split())
                    item["signal_check"] = sc
                    item["signal_retry_count"] = retry_cnt
                    logger.info(f"[Rephrase] retries={retry_cnt} ok={sc} :: {rq}")
                new_small.append(item)
            else:
                new_small.append(item)

        out_big.append(new_small)

    return out_big


def main():
    parser = argparse.ArgumentParser(description="Rephrase pseudo_question_construction for nested (big list -> small list -> dict) JSON, with signal-aware retry.")
    parser.add_argument("--input", required=True, help="Input JSON file (big list -> small list -> dict)")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo"),
                        help="OpenAI model, default gpt-3.5-turbo")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"), help="OpenAI API Key or set OPENAI_API_KEY")
    parser.add_argument("--organization", default=os.environ.get("OPENAI_ORG"), help="OpenAI organization or set OPENAI_ORG (optional)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default 0 for reproducibility)")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens (default 256)")
    parser.add_argument("--use-cache", action="store_true", help="Enable local cache (pickle)")
    parser.add_argument("--cache-dir", default=".cache/rephrase", help="Cache directory (default .cache/rephrase)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing rephrased_question")

    # Additional: retry with temporal constraint when signal check fails
    parser.add_argument("--signal-retry", action="store_true", help="When signal_check=False, enable a second rewrite with temporal constraint")
    parser.add_argument("--signal-retry-max", type=int, default=2, help="Max retry count (default 2)")

    args = parser.parse_args()

    if not args.api_key:
        raise RuntimeError("Missing OpenAI API Key: use --api-key or set OPENAI_API_KEY")

    logger = get_logger(__name__, {"log_level": "INFO"})

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON top-level must be a list (big list).")

    cache = None
    if args.use_cache:
        cache_path = Path(args.cache_dir) / "gpt_cache.pickle"
        cache = GPTCache(cache_path, logger)

    out = process_nested_list(data, args, cache, logger)

    if cache:
        cache.flush_if_needed()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    logger.info(f"Done. Wrote to {args.output}")

if __name__ == "__main__":
    main()

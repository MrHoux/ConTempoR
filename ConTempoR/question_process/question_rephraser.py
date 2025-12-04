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
    """给第二次改写附加‘保持时序关系’的硬约束提示。"""
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
                      bypass_cache_get: bool = False,  # 新增：跳过读缓存
                      allow_cache_put: bool = True     # 新增：允许把新结果写回缓存
                      ) -> Optional[str]:
    """单次改写（可选加前缀约束 extra_prefix）。
       bypass_cache_get=True 时，不读取缓存；allow_cache_put=True 时，会把新结果写回缓存覆盖旧值。
    """
    pq = format_text(pseudo_question).encode("utf-8").decode("utf-8")
    prompt = (extra_prefix or "") + REPHRASE_PROMPT + pq + "\nQuestion:"

    # —— 跳过读缓存（只在需要时启用）
    if cache and not bypass_cache_get:
        cached = cache.get(prompt)
        if cached:
            logger.info("[Rephrase] Cache hit.")
            return cached

    # 调 OpenAI
    if model in SUPPORTED_CHAT_MODELS:
        out = call_openai_chat(model, prompt, api_key, org, temperature, max_tokens, logger)
    elif model in SUPPORTED_COMPLETION_MODELS:
        out = call_openai_completion(model, prompt, api_key, org, temperature, max_tokens, logger)
    else:
        raise ValueError(f"Unsupported model: {model}")

    # —— 允许写回缓存（覆盖同 key 的旧值）
    if out and cache and allow_cache_put:
        cache.put(prompt, out)
    return out


def gpt_rephrase_with_signal_retry(pseudo_question: str, signal: Optional[str], model: str, api_key: str, org: Optional[str],
                                   temperature: float, max_tokens: int, cache: Optional[GPTCache], logger,
                                   enable_retry: bool, retry_max: int,
                                   bypass_first_get: bool = False  # 新增：首次改写是否跳过读缓存
                                   ) -> (Optional[str], Optional[bool], int):
    # 第一次：根据 bypass_first_get 决定是否绕过缓存读取
    rq = gpt_rephrase_once(pseudo_question, model, api_key, org, temperature, max_tokens,
                           cache, logger, extra_prefix="",
                           bypass_cache_get=bypass_first_get,  # ★ 关键
                           allow_cache_put=True)
    sc = check_signal_same(rq, signal) if rq else None
    retries = 0
    if sc is not False or not enable_retry:
        return rq, sc, retries

    # 带时序约束的二次改写（这里可正常使用缓存；prompt不同，多半不会撞到旧缓存）
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
    big_list: 顶层大列表；每个元素应是小列表；
    对每个小列表里的字典：如果含 'pseudo_question_construction' 则处理。
    """
    if not isinstance(big_list, list):
        raise ValueError("输入 JSON 顶层必须是列表（大列表）。")

    out_big: List[Any] = []

    for small in big_list:
        if not isinstance(small, list):
            out_big.append(small)
            continue

        new_small: List[Any] = []
        for item in small:
            if isinstance(item, dict) and "pseudo_question_construction" in item:
                # 已通过检查的直接跳过，不调用 GPT
                if item.get("signal_check") is True:
                    logger.info("[Skip] 已通过检查，跳过 GPT 调用。")
                    new_small.append(item)
                    continue

                # —— 判断是否需要绕过缓存读取：仅当 signal_check 为 False 时
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
                    bypass_first_get=bypass_first_get  # ★ 关键
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
    parser.add_argument("--input", required=True, help="输入 JSON 文件（大列表 -> 小列表 -> 字典）")
    parser.add_argument("--output", required=True, help="输出 JSON 文件路径")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo"),
                        help="OpenAI 模型，默认 gpt-3.5-turbo")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"), help="OpenAI API Key 或设 OPENAI_API_KEY")
    parser.add_argument("--organization", default=os.environ.get("OPENAI_ORG"), help="OpenAI 组织或设 OPENAI_ORG（可选）")
    parser.add_argument("--temperature", type=float, default=0.0, help="生成温度（默认 0，更可复现）")
    parser.add_argument("--max-tokens", type=int, default=256, help="最大 tokens（默认 256）")
    parser.add_argument("--use-cache", action="store_true", help="启用本地缓存（pickle）")
    parser.add_argument("--cache-dir", default=".cache/rephrase", help="缓存目录（默认 .cache/rephrase）")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有 rephrased_question")

    # 新增：时序检查失败时二次改写
    parser.add_argument("--signal-retry", action="store_true", help="当 signal_check=False 时启用带时序约束的二次改写")
    parser.add_argument("--signal-retry-max", type=int, default=2, help="最多重试次数（默认 2）")

    args = parser.parse_args()

    if not args.api_key:
        raise RuntimeError("缺少 OpenAI API Key：使用 --api-key 或设置环境变量 OPENAI_API_KEY")

    logger = get_logger(__name__, {"log_level": "INFO"})

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("输入 JSON 顶层必须是列表（大列表）。")

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

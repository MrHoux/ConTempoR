"""Resumable full-split Bailian evaluation using the existing context protocol."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import requests
import test as evaluation
from dataset_utils import BENCHMARK, load_split


def build_tasks(data, shots, *, context_removed=False, cot_guidance=False):
    tasks = []
    for conv in data:
        history = []
        for turn in conv['turns']:
            question = turn.get('question') or turn.get('rephrased_question') or ''
            if context_removed and not turn.get('question'):
                raise ValueError('Context removal requires original question for every turn')
            messages = evaluation.build_prompt(shots, question, [] if context_removed else history)
            if cot_guidance:
                messages[0]['content'] += (' Before producing the final JSON, reason step by step in your thinking: resolve the subject and references; identify the main event or state and the temporal constraint; distinguish event dates from tenure intervals; apply direction and calendar offsets at their stated precision; check the candidate answer and its time range; determine the relation to the preceding question. Treat explicit dates and offsets as constraints. Return only the final JSON in the answer content.')
            tasks.append((conv['conversation_id'], turn['turn_id'], messages))
            history = history + [question]
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="runs/qwen3.7-plus/context")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--base_url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    ap.add_argument("--model", default="qwen3.7-plus")
    ap.add_argument("--benchmark_dir", default=str(BENCHMARK), help="Frozen benchmark release directory containing train.json and test.json")
    ap.add_argument("--thinking", action="store_true", help="Enable Qwen thinking mode for the CoT-guided condition")
    ap.add_argument("--cot_guidance", action="store_true", help="Explicit temporal reasoning instructions before final JSON")
    ap.add_argument("--thinking_budget", type=int, default=None)
    ap.add_argument("--max_completion_tokens", type=int, default=None)
    ap.add_argument('--context_removed', action='store_true', help='Use original question with no test-conversation history; preserve fixed few-shots and system prompt')
    args = ap.parse_args()
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise SystemExit("Set DASHSCOPE_API_KEY")
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    benchmark_dir = Path(args.benchmark_dir).resolve()
    data = load_split(benchmark_dir / "test.json")
    train = load_split(benchmark_dir / "train.json")
    shots = evaluation.pack_full_conversation_shots(train[:3], 3, 1800)
    completion_tokens = args.max_completion_tokens or (2048 if args.thinking else 512)
    manifest = {"model": args.model, "base_url": args.base_url,
                "temperature": 0, "enable_thinking": args.thinking,
                "condition": "context_removed" if args.context_removed else ("cot_guided" if args.cot_guidance else ("thinking_only" if args.thinking else "ordinary")),
                "cot_guidance": args.cot_guidance,
                "thinking_budget": args.thinking_budget,
                "max_completion_tokens": completion_tokens, "response_format": {"type": "json_object"},
                "shots_conversations": 3, "actual_shot_turns": len(shots),
                "shots_token_budget": 1800, "history": "none" if args.context_removed else "previous questions only",
                "split": "test", "turns": sum(len(c["turns"]) for c in data),
                "benchmark_dir": str(benchmark_dir),
                "dataset_sha256": {s: hashlib.sha256((benchmark_dir / (s + ".json")).read_bytes()).hexdigest() for s in ("train", "test")},
                "code_sha256": {s: hashlib.sha256(Path(__file__).with_name(s).read_bytes()).hexdigest() for s in ("test.py", "dataset_utils.py", "run_bailian.py")}}
    mp = out / "run_manifest.json"
    if mp.exists() and json.loads(mp.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Resume configuration differs; use a new output directory")
    mp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tasks = build_tasks(data, shots, context_removed=args.context_removed, cot_guidance=args.cot_guidance)
    if args.context_removed:
        audit = {'question_field': 'question', 'test_history': 'removed',
                 'fixed_training_examples_retained': len(shots), 'system_prompt_unchanged': True,
                 'tasks': len(tasks), 'history_blocks': sum('[Previous Questions]' in m[-1]['content'] for _, _, m in tasks),
                 'messages_sha256': hashlib.sha256(json.dumps(tasks, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                 'relation_note': 'Prior-turn relation remains diagnostic only when its prior context is unavailable.'}
        assert audit['history_blocks'] == 0
        (out / 'prompt_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    raw_path = out / "responses.jsonl"
    done = {}
    if raw_path.exists():
        # Recover a final incomplete line left by an interrupted write.
        lines = raw_path.read_text(encoding="utf-8").splitlines()
        good = []
        for i, line in enumerate(lines):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if i != len(lines)-1:
                    raise
                break
            identity = (record["conversation_id"], record["turn_id"])
            if identity in done:
                raise ValueError("Duplicate checkpoint ID")
            done[identity] = record
            good.append(line)
        raw_path.write_text("".join(line + "\n" for line in good), encoding="utf-8")
    expected = {(cid, tid) for cid, tid, _ in tasks}
    if not set(done).issubset(expected):
        raise ValueError("Checkpoint contains unknown IDs")

    def predict(task):
        cid, tid, messages = task
        payload = {"model": args.model, "messages": messages, "temperature": 0,
                   "enable_thinking": args.thinking, "max_completion_tokens": completion_tokens,
                   "response_format": {"type": "json_object"}}
        if args.thinking_budget is not None:
            payload['thinking_budget'] = args.thinking_budget
        for attempt in range(6):
            try:
                response = requests.post(args.base_url.rstrip("/") + "/chat/completions",
                                         headers={"Authorization": "Bearer " + key}, json=payload, timeout=(20, 180))
                if response.status_code in (401,403):
                    raise PermissionError("Bailian authorization failed")
                response.raise_for_status()
                body = response.json()
                choice = body["choices"][0]
                return {"conversation_id": cid, "turn_id": tid, "response": body,
                        "prediction": {"conversation_id": cid, "turn_id": tid,
                                       **evaluation.parse_model_json(choice["message"].get("content") or "")}}
            except PermissionError:
                raise
            except Exception as exc:
                if attempt == 5:
                    raise RuntimeError(f"Request failed for {cid}/{tid}: {type(exc).__name__}") from None
                time.sleep(min(30, 2 ** attempt))

    pending = [t for t in tasks if (t[0], t[1]) not in done]
    print(f"START completed={len(done)} pending={len(pending)} total={len(tasks)}", flush=True)
    failures = []
    with raw_path.open("a", encoding="utf-8") as stream, concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(predict, task): task for task in pending}
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                rec = future.result()
            except Exception as exc:
                failures.append({"conversation_id": task[0], "turn_id": task[1], "error": str(exc)})
                print(f"FAILED {task[1]} {type(exc).__name__}", flush=True)
                continue
            stream.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stream.flush()
            done[(task[0], task[1])] = rec
            if len(done) % 25 == 0 or len(done) == len(tasks):
                print(f"PROGRESS {len(done)}/{len(tasks)}", flush=True)
    (out / "request_failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    if failures:
        raise SystemExit(f"{len(failures)} requests failed; rerun to resume")
    predictions = [done[(cid,tid)]["prediction"] for cid,tid,_ in tasks]
    (out / "test_preds.json").write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    report = evaluation.evaluate_test(data, predictions, 0.5, None, False, "multi")
    (out / "evaluation_report.json").write_text(json.dumps({"report":report},indent=2),encoding="utf-8")
    usage = {k:sum(r["response"].get("usage",{}).get(k,0) for r in done.values()) for k in ("prompt_tokens","completion_tokens","total_tokens")}
    audit = {"completed":len(done), "expected":len(tasks), "request_failures":len(failures),
             "invalid_json":sum(not evaluation.parse_model_json(r["response"]["choices"][0]["message"].get("content") or "") for r in done.values()),
             "truncated":sum(r["response"]["choices"][0].get("finish_reason")=="length" for r in done.values()),
             "returned_models": sorted({r["response"].get("model", "") for r in done.values()}), "usage":usage}
    (out/"completion_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    print(json.dumps({"audit":audit,"report":report}),flush=True)

if __name__ == "__main__":
    main()

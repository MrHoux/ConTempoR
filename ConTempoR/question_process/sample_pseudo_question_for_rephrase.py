# -*- coding: utf-8 -*-
import os
import json
from collections import Counter

from ConTempoR.library.utils import (
    get_logger,
    # Keep only these three filters
    filter_question_having_year,
    filter_question_contain_qid,
    filter_questions_have_strange_char,
)


def merge_similar_main_answer(instance: dict) -> dict:
    """
    Merge similar pseudo-questions into the main instance (same logic as before):
    - Merge answer / evidence / timespan / source
    - Deduplicate and extend question_entity[0]
    - Convert main_evidence_id to a list and append
    """
    for item in instance.get("similar_pseudo_question", []):
        instance.setdefault("answer", [])
        instance.setdefault("evidence", [])
        instance.setdefault("timespan", [])
        instance.setdefault("source", [])
        instance.setdefault("question_entity", [[], []])

        instance["answer"] += item.get("answer", [])

        # Insert evidence / timespan / source from similar items at position 1 in the main instance (keeps prior behavior)
        instance["evidence"].insert(1, item.get("evidence"))
        instance["timespan"].insert(1, item.get("timespan"))
        instance["source"].insert(1, item.get("source"))

        for ques_entity in item.get("question_entity", []):
            if ques_entity not in instance["question_entity"][0]:
                instance["question_entity"][0].append(ques_entity)

        # Always convert main_evidence_id to a list then append
        if "main_evidence_id" in instance:
            instance["main_evidence_id"] = [instance["main_evidence_id"]]
        else:
            instance["main_evidence_id"] = []
        instance["main_evidence_id"].append(item.get("main_evidence_id"))
    return instance


class PseudoQuestionSampleRephrase:
    """
    - No GPT calls, no rewriting
    - No sampling: keep all questions per entity
    - Perform "merge + minimal filtering (year/QID/strange characters)"
    - Final output omits question/rephrase_* fields; keep only pseudo_question_construction
    """
    def __init__(self, config: dict):
        self.config = config
        self.logger = get_logger(__name__, config)

        # I/O paths (aligned with the original project)
        self.result_path = self.config["result_path"]
        self.year_start = self.config["year_start"]
        self.year_end = self.config["year_end"]
        self.target_question_total_number = self.config["target_question_number"]
        self.sample_size = self.config["sample_size"]

        self.output_dir = os.path.join(
            self.result_path,
            f"{self.year_start}_{self.year_end}_{self.target_question_total_number}_{self.sample_size}"
        )
        self.pseudo_questions_file_path = os.path.join(
            self.output_dir, self.config["pseudo_questions_in_total_file"]
        )

    def run_merge_filter_and_format(self):
        """
        Main entry: load -> merge -> filter (three checks) -> format -> return List[dict]
        """
        with open(self.pseudo_questions_file_path, "r", encoding="utf-8") as fp:
            pseudo_questions = json.load(fp)

        self.logger.info(f"[merge+filter] Entities read: {len(pseudo_questions)}")

        kept = []
        total = 0
        for _topic_key, instances in pseudo_questions.items():
            for inst in instances:
                total += 1

                # 1) Merge similar pseudo-questions if present
                if inst.get("similar_pseudo_question"):
                    inst = merge_similar_main_answer(inst)

                # 2) Minimal filtering (three rules)
                if not self._pass_filters(inst):
                    continue

                kept.append(inst)

        self.logger.info(f"[merge+filter] Total instances: {total}, kept after filtering: {len(kept)}")

        # 3) Format output (no question / rephrase_* fields)
        formatted = self._format_for_benchmark(kept)
        self.logger.info(f"[format] Output count: {len(formatted)}")
        return formatted

    def _pass_filters(self, instance: dict) -> bool:
        """
        Apply three filters only: contains a year / contains a QID / contains strange characters.
        Drop the instance if any trigger.
        """
        if filter_question_having_year(instance):
            self.logger.info("drop: question has year")
            return False
        if filter_question_contain_qid(instance):
            self.logger.info("drop: question contains qid")
            return False
        if filter_questions_have_strange_char(instance):
            self.logger.info("drop: question has strange char")
            return False
        return True

    def _format_for_benchmark(self, questions: list) -> list:
        """
        Similar to benchmark_format_for_select_questions, but:
        - Do not output rephrase_* and do not generate question
        - Keep only pseudo_question_construction
        - Split evidence into main/constraint
        - If topic_entity.type is a list, take the first element
        - Deduplicate question_entity / answer
        """
        results = []
        for inst in questions:
            new_format = {}

            # Rebuild evidence: last one as constraint, others as main; if lengths mismatch, treat all as main
            evidences = inst.get("evidence", [])
            sources = inst.get("source", [])
            timespans = inst.get("timespan", [])
            new_format["evidence"] = {"main": [], "constraint": []}

            if evidences and sources and timespans and \
               len(evidences) == len(sources) == len(timespans) and len(evidences) >= 1:
                constraint_evidence = evidences[-1]
                constraint_source = sources[-1]
                constraint_timespan = timespans[-1]
                new_format["evidence"]["constraint"].append(
                    [constraint_evidence, constraint_source, constraint_timespan]
                )
                main_evidences = evidences[:-1]
                main_sources = sources[:-1]
                main_timespans = timespans[:-1]
                for i, ev in enumerate(main_evidences):
                    new_format["evidence"]["main"].append([ev, main_sources[i], main_timespans[i]])
            else:
                for i, ev in enumerate(evidences):
                    src = sources[i] if i < len(sources) else None
                    ts = timespans[i] if i < len(timespans) else None
                    new_format["evidence"]["main"].append([ev, src, ts])

            # Signal
            new_format["signal"] = inst.get("signal")

            # Topic entity: if type is a list, take the first element
            topic_entity = inst.get("topic_entity", {})
            new_format["topic_entity"] = topic_entity.copy() if isinstance(topic_entity, dict) else topic_entity
            if isinstance(new_format["topic_entity"], dict):
                t = new_format["topic_entity"].get("type")
                if isinstance(t, list) and t:
                    new_format["topic_entity"]["type"] = t[0]

            # Deduplicate question entities (supports two-part structure)
            new_format["question_entity"] = []
            seen_qe = set()
            qe = inst.get("question_entity", [])
            if isinstance(qe, list) and len(qe) == 2 and isinstance(qe[0], list) and isinstance(qe[1], list):
                flat_qe = qe[0] + qe[1]
            elif isinstance(qe, list):
                flat_qe = qe
            else:
                flat_qe = []

            for item in flat_qe:
                qid = item.get("id")
                if qid and qid not in seen_qe:
                    seen_qe.add(qid)
                    new_format["question_entity"].append({"id": qid, "label": item.get("label")})

            # Deduplicate answers
            new_format["answer"] = []
            seen_ans = set()
            for item in inst.get("answer", []):
                aid = item.get("id")
                if aid and aid not in seen_ans:
                    seen_ans.add(aid)
                    new_format["answer"].append({"id": aid, "label": item.get("label")})

            # Keep only the original pseudo question (no question/rephrase_*)
            new_format["pseudo_question_construction"] = inst.get("pseudo_question_construction", "")

            results.append(new_format)

        return results


# Utility function (kept for interface compatibility if other modules import it)
def count_items(lst):
    return Counter(lst)


def weighted_random_choice(item_weight_dic):
    items = list(item_weight_dic.keys())
    weights = list(item_weight_dic.values())
    import random
    rnd = random.uniform(0, sum(weights))
    cum = 0.0
    for i, w in enumerate(weights):
        cum += w
        if rnd < cum:
            return items[i]

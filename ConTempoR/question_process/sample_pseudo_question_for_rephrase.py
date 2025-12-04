# -*- coding: utf-8 -*-
import os
import json
from collections import Counter

from ConTempoR.library.utils import (
    get_logger,
    # 仅保留这三项过滤
    filter_question_having_year,
    filter_question_contain_qid,
    filter_questions_have_strange_char,
)


def merge_similar_main_answer(instance: dict) -> dict:
    """
    合并相似伪问题到主实例（沿用你原先的逻辑）
    - 合并 answer / evidence / timespan / source
    - question_entity[0] 去重扩充
    - main_evidence_id 改为列表并附加
    """
    for item in instance.get("similar_pseudo_question", []):
        instance.setdefault("answer", [])
        instance.setdefault("evidence", [])
        instance.setdefault("timespan", [])
        instance.setdefault("source", [])
        instance.setdefault("question_entity", [[], []])

        instance["answer"] += item.get("answer", [])

        # 将相似条目的 evidence / timespan / source 插到主实例第 1 位（保持与原实现一致）
        instance["evidence"].insert(1, item.get("evidence"))
        instance["timespan"].insert(1, item.get("timespan"))
        instance["source"].insert(1, item.get("source"))

        for ques_entity in item.get("question_entity", []):
            if ques_entity not in instance["question_entity"][0]:
                instance["question_entity"][0].append(ques_entity)

        # main_evidence_id 统一转为列表并追加
        if "main_evidence_id" in instance:
            instance["main_evidence_id"] = [instance["main_evidence_id"]]
        else:
            instance["main_evidence_id"] = []
        instance["main_evidence_id"].append(item.get("main_evidence_id"))
    return instance


class PseudoQuestionSampleRephrase:
    """
    - 不调用 GPT、不做改写；
    - 不做采样：每个实体的所有问题全部保留；
    - 执行“合并 + 最小过滤（年/QID/奇怪字符）”；
    - 最终输出不包含 question / rephrase_* 字段，仅保留 pseudo_question_construction。
    """
    def __init__(self, config: dict):
        self.config = config
        self.logger = get_logger(__name__, config)

        # I/O 路径（保持与原工程一致）
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
        主入口：读取 -> 合并 -> 过滤(仅三条) -> 格式化 -> 返回 List[dict]
        """
        with open(self.pseudo_questions_file_path, "r", encoding="utf-8") as fp:
            pseudo_questions = json.load(fp)

        self.logger.info(f"[merge+filter] 读取实体数: {len(pseudo_questions)}")

        kept = []
        total = 0
        for _topic_key, instances in pseudo_questions.items():
            for inst in instances:
                total += 1

                # 1) 若存在相似伪问题，先合并
                if inst.get("similar_pseudo_question"):
                    inst = merge_similar_main_answer(inst)

                # 2) 最小过滤（仅三条）
                if not self._pass_filters(inst):
                    continue

                kept.append(inst)

        self.logger.info(f"[merge+filter] 总实例: {total}，过滤后保留: {len(kept)}")

        # 3) 格式化输出（不含 question / rephrase_*）
        formatted = self._format_for_benchmark(kept)
        self.logger.info(f"[format] 输出条数: {len(formatted)}")
        return formatted

    def _pass_filters(self, instance: dict) -> bool:
        """
        仅保留三类过滤：包含年份 / 包含 QID / 含奇怪字符。
        命中任一条则丢弃。
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
        与原 benchmark_format_for_select_questions 类似，但：
        - 不输出 rephrase_*，不生成 question；
        - 仅保留 pseudo_question_construction；
        - evidence 拆成 main/constraint；
        - topic_entity.type 若为 list 取首；
        - question_entity / answer 去重。
        """
        results = []
        for inst in questions:
            new_format = {}

            # 证据重组：最后一条为 constraint，其余为 main；若长度不齐则全塞 main
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

            # 信号
            new_format["signal"] = inst.get("signal")

            # 主题实体：type 若为 list 取首元素
            topic_entity = inst.get("topic_entity", {})
            new_format["topic_entity"] = topic_entity.copy() if isinstance(topic_entity, dict) else topic_entity
            if isinstance(new_format["topic_entity"], dict):
                t = new_format["topic_entity"].get("type")
                if isinstance(t, list) and t:
                    new_format["topic_entity"]["type"] = t[0]

            # 问题实体去重（兼容两段式结构）
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

            # 答案去重
            new_format["answer"] = []
            seen_ans = set()
            for item in inst.get("answer", []):
                aid = item.get("id")
                if aid and aid not in seen_ans:
                    seen_ans.add(aid)
                    new_format["answer"].append({"id": aid, "label": item.get("label")})

            # 仅保留原始伪问题（不输出 question/rephrase_*）
            new_format["pseudo_question_construction"] = inst.get("pseudo_question_construction", "")

            results.append(new_format)

        return results


# 工具函数（若其他模块引用，保留一致接口）
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

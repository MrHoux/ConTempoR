"""Offline checks: no network or model required."""
import json
import unittest
from datetime import datetime
from pathlib import Path
import test as context
import nocontext_test as flat
from dataset_utils import BENCHMARK, load_split, complete_predictions, parse_dataset_date


class ExpandedReleaseTests(unittest.TestCase):
    def test_iou_penalizes_short_and_wide_predictions(self):
        gold = (datetime(2020, 1, 1), datetime(2020, 1, 10))
        for module in (context, flat):
            self.assertAlmostEqual(module.temporal_iou((gold[0], gold[0]), gold), 0.1)
            self.assertAlmostEqual(module.temporal_iou((gold[0], datetime(2020, 1, 20)), gold), 0.5)
            self.assertEqual(module.temporal_iou(gold, gold), 1.0)
            self.assertEqual(module.temporal_iou((datetime(2020, 2, 1), datetime(2020, 2, 2)), gold), 0.0)
            self.assertEqual(module.temporal_iou(gold, (gold[1], gold[0])), 0.0)

    @classmethod
    def setUpClass(cls):
        cls.data = load_split(BENCHMARK / "test.json")
        cls.gold_predictions = [
            {"conversation_id": c["conversation_id"], "turn_id": t["turn_id"],
             "answer_entity": t["answer"][0]["label"],
             "time_range": dict(zip(("start", "end"), t["evidence"]["main"][0][2])),
             "pred_relation": context.gold_first_and_relation(t)[1]}
            for c in cls.data for t in c["turns"]]

    def test_release_counts_and_dates(self):
        for split, nc, nt in [("train",825,6119),("dev",275,2040),("test",275,2039)]:
            data = load_split(BENCHMARK / (split + ".json"))
            self.assertEqual(len(data), nc)
            self.assertEqual(sum(len(c["turns"]) for c in data), nt)
            for c in data:
                for t in c["turns"]:
                    for module in (context, flat):
                        self.assertIsNotNone(module.extract_gold_time_range(t))
        self.assertEqual(parse_dataset_date(20200101), parse_dataset_date("2020-01-01"))

    def test_gold_predictions_perfect(self):
        for report in [context.evaluate_test(self.data,self.gold_predictions,0.5,None,False,"multi"),
                       flat.evaluate_flat(self.data,self.gold_predictions,0.5,None,False)]:
            self.assertEqual(report["totals"]["samples"],2039)
            self.assertEqual(report["entity"]["EM"],1.0)
            self.assertEqual(report["time"]["Mean_IoU"],1.0)
            self.assertEqual(report["time"]["Accuracy"],1.0)
        report=context.evaluate_test(self.data,self.gold_predictions,0.5,None,False,"multi")
        self.assertEqual(report["relation"]["Accuracy"],1.0)
        self.assertEqual(report["totals"]["relation_scored_samples"],1762)
        self.assertEqual(report["totals"]["first_turns_excluded"],275)
        self.assertEqual(set(report["relation"]["missing_gold_turn_ids"]),
                         {"1357-7", "1373-2"})

    def test_relation_uses_position_and_reports_absent_gold(self):
        import copy
        conversation = copy.deepcopy(self.data[0])
        conversation["turns"] = conversation["turns"][:3]
        first, second, third = conversation["turns"]
        first["checks"] = {}
        second["checks"] = {"first_in_conversation": True}
        second["temporal_link"] = {"macro": "AFTER"}
        third["checks"] = {}
        third.pop("temporal_link", None)
        preds = [{"conversation_id": conversation["conversation_id"],
                  "turn_id": second["turn_id"], "pred_relation": "AFTER"}]
        report = context.evaluate_test([conversation], preds, .5, None, False, "multi")
        self.assertEqual(report["relation"]["Accuracy"], 1.0)
        self.assertEqual(report["totals"]["relation_scored_samples"], 1)
        self.assertEqual(report["totals"]["missing_gold_relations"], 1)
        missing = context.evaluate_test([conversation], [], .5, None, False, "multi")
        self.assertEqual(missing["relation"]["Accuracy"], 0.0)

    def test_missing_predictions_count_as_errors(self):
        for report in [context.evaluate_test(self.data,[],0.5,None,False,"multi"),
                       flat.evaluate_flat(self.data,[],0.5,None,False)]:
            self.assertEqual(report["totals"]["samples"],2039)
            self.assertEqual(report["entity"]["EM"],0.0)
            self.assertEqual(report["time"]["Mean_IoU"],0.0)

    def test_unsafe_input_rejected(self):
        with self.assertRaises(ValueError):
            load_split(BENCHMARK / "full.json")
        with self.assertRaises(ValueError):
            complete_predictions(self.data,[self.gold_predictions[0]]*2)
        with self.assertRaises(ValueError):
            complete_predictions(self.data,[{"conversation_id":-1,"turn_id":"missing"}])

    def test_final_manifest_hashes_and_schema(self):
        import hashlib
        manifest=json.loads((BENCHMARK/'manifest.json').read_text(encoding='utf-8'))
        ids=set()
        for split,info in manifest['splits'].items():
            path=BENCHMARK/(split+'.json')
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),info['sha256'])
            data=load_split(path)
            self.assertEqual(len(data),info['conversations'])
            self.assertEqual(sum(len(c['turns']) for c in data),info['turns'])
            for c in data:
                for t in c['turns']:
                    self.assertNotIn(t['turn_id'],ids)
                    ids.add(t['turn_id'])
                    self.assertEqual(list(t),manifest['turn_fields'])


if __name__ == "__main__":
    unittest.main()

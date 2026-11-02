"""Offline checks: no network or model required."""
import json
import unittest
from pathlib import Path
import test as context
import nocontext_test as flat
from dataset_utils import BENCHMARK, load_split, complete_predictions, parse_dataset_date


class ExpandedReleaseTests(unittest.TestCase):
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
        for split, nc, nt in [("train",1119,8192),("dev",100,792),("test",156,1214)]:
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
            self.assertEqual(report["totals"]["samples"],1214)
            self.assertEqual(report["entity"]["EM"],1.0)
            self.assertEqual(report["time"]["Mean_TCR"],1.0)
            self.assertEqual(report["time"]["Accuracy"],1.0)
        report=context.evaluate_test(self.data,self.gold_predictions,0.5,None,False,"multi")
        self.assertEqual(report["relation"]["Accuracy"],1.0)
        self.assertEqual(report["totals"]["relation_scored_samples"],1058)

    def test_missing_predictions_count_as_errors(self):
        for report in [context.evaluate_test(self.data,[],0.5,None,False,"multi"),
                       flat.evaluate_flat(self.data,[],0.5,None,False)]:
            self.assertEqual(report["totals"]["samples"],1214)
            self.assertEqual(report["entity"]["EM"],0.0)
            self.assertEqual(report["time"]["Mean_TCR"],0.0)

    def test_unsafe_input_rejected(self):
        with self.assertRaises(ValueError):
            load_split(BENCHMARK / "full.json")
        with self.assertRaises(ValueError):
            complete_predictions(self.data,[self.gold_predictions[0]]*2)
        with self.assertRaises(ValueError):
            complete_predictions(self.data,[{"conversation_id":-1,"turn_id":"missing"}])

    def test_full_index_matches_splits(self):
        full=json.loads((BENCHMARK/"full.json").read_text(encoding="utf-8"))
        index=json.loads((BENCHMARK/"split_index.json").read_text(encoding="utf-8"))
        for split in ("train","dev","test"):
            selected=[full[row["full_index"]] for row in index["conversations"] if row["split"]==split]
            self.assertEqual(selected,load_split(BENCHMARK/(split+".json")))

if __name__ == "__main__":
    unittest.main()

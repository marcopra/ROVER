import unittest

from experiments.sink_state_ablation.core import (
    largest_visitation_regression,
    summarize_adaptation,
    summarize_pretrain,
    trapezoid_auc,
)
from experiments.sink_state_ablation.analyze import support_interaction


class SinkStateAblationTest(unittest.TestCase):
    def test_largest_previously_covered_state_drop(self):
        rows = [
            {"visitation_distribution": {"a": .6, "b": .4}},
            {"visitation_distribution": {"a": .1, "b": .4, "c": .5}},
        ]
        self.assertAlmostEqual(largest_visitation_regression(rows), .5)

    def test_pretrain_summary(self):
        rows = [
            {
                "event": "coverage", "frame": 0, "optimization_round": 0,
                "covered_states": 2, "feasible_states": 4,
                "coverage_fraction": .5,
                "visitation_distribution": {"a": .5, "b": .5},
            },
            {
                "event": "operator_update", "frame": 1,
                "estimated_sink_mass": .2,
                "low_support_query_fraction": .25,
            },
            {
                "event": "coverage", "frame": 2, "optimization_round": 1,
                "covered_states": 3, "feasible_states": 4,
                "coverage_fraction": .75,
                "visitation_distribution": {"a": .25, "b": .25, "c": .5},
            },
        ]
        result = summarize_pretrain(rows)
        self.assertEqual(result["final_policy_covered_states"], 3)
        self.assertAlmostEqual(result["largest_visitation_regression"], .25)
        self.assertAlmostEqual(result["mean_estimated_sink_mass"], .2)

    def test_downstream_metrics(self):
        rows = [
            {"event": "evaluation", "frame": 1000, "success_rate": 0.0},
            {"event": "first_reward", "frame": 1500, "interactions": 1500},
            {"event": "evaluation", "frame": 2000, "success_rate": 1.0},
        ]
        result = summarize_adaptation(rows, horizon=2000)
        self.assertEqual(result["time_to_first_downstream_reward"], 1500)
        self.assertAlmostEqual(result["downstream_learning_auc"], .25)
        self.assertAlmostEqual(trapezoid_auc([1000, 2000], [0, 1], 2000), .25)

    def test_paired_support_interaction(self):
        rows = []
        for seed in (1, 2):
            for sink, batch, coverage in (
                ("no_sink", 1024, .4), ("fixed", 1024, .7),
                ("no_sink", 5000, .6), ("fixed", 5000, .7),
            ):
                rows.append({
                    "sink": sink, "operator_batch_size": batch, "seed": seed,
                    "final_policy_coverage_fraction": coverage,
                })
        result = support_interaction(rows)["fixed"]
        self.assertAlmostEqual(result["mean"], .2)
        self.assertIsNone(result["standard_error"])


if __name__ == "__main__":
    unittest.main()

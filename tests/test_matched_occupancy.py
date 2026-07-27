import unittest

import numpy as np

from experiments.matched_occupancy.core import (
    auc,
    coverage_metrics,
    discounted_occupancy,
    match_candidates,
    threshold_interactions,
)


class MatchedOccupancyTest(unittest.TestCase):
    def test_buffer_and_discounted_policy_are_separate(self):
        buffer = coverage_metrics([0, 1, 2, 3], 4)
        policy = discounted_occupancy([[0, 0, 1]], 4, gamma=0.5)
        self.assertEqual(buffer["support_fraction"], 1.0)
        self.assertEqual(policy["support_fraction"], 0.5)
        self.assertAlmostEqual(sum(policy["distribution"]), 1.0)

    def test_blind_matching(self):
        def candidate(name, algorithm, buffer_support, effective, p):
            return {
                "candidate_id": name,
                "algorithm": algorithm,
                "n_states": 10,
                "buffer": {"support_fraction": buffer_support},
                "final_policy": {"effective_support": effective, "distribution": p},
            }
        candidates = [
            candidate("rover", "rover", .90, 8.0, [.5, .5, 0, 0]),
            candidate("cic", "cic", .92, 3.0, [0, 0, .5, .5]),
        ]
        pairs = match_candidates(candidates, {
            "max_buffer_support_gap": .05,
            "min_final_effective_support_gap": .1,
            "require_cross_algorithm": True,
        })
        self.assertEqual(len(pairs), 1)
        contaminated = [dict(candidates[0], success_auc=.9), candidates[1]]
        with self.assertRaisesRegex(ValueError, "downstream"):
            match_candidates(contaminated, {
                "max_buffer_support_gap": .05,
                "min_final_effective_support_gap": .1,
            })

    def test_adaptation_metrics(self):
        x, y = [0, 10, 20], [0, .5, 1]
        self.assertAlmostEqual(auc(x, y), .5)
        self.assertEqual(threshold_interactions(x, y, .8), 20)
        self.assertIsNone(threshold_interactions(x, y, 1.1))


if __name__ == "__main__":
    unittest.main()


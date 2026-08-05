import unittest
from unittest.mock import patch

import numpy as np

from lcqaoa import WeightedGraph, lightcone_expectation, lightcone_gradient_adjoint
from lcqaoa.lightcone import plan_checkpoint_schedule


class PublicApiContractTest(unittest.TestCase):
    def setUp(self):
        self.graph = WeightedGraph(
            n=2,
            edges=((0, 1, 0.75),),
            fields=((0, -0.2),),
            objective="qubo",
        )

    def test_depth_must_match_angle_layers(self):
        with self.assertRaisesRegex(ValueError, "p must equal"):
            lightcone_expectation(
                self.graph, [0.1], [0.2], p=2, prefer_gpu=False
            )
        with self.assertRaisesRegex(ValueError, "p must equal"):
            lightcone_gradient_adjoint(
                self.graph, [0.1], [0.2], p=2, prefer_gpu=False
            )

    def test_constant_offset_changes_value_not_gradient(self):
        shifted = WeightedGraph(
            n=self.graph.n,
            edges=self.graph.edges,
            fields=self.graph.fields,
            objective=self.graph.objective,
            constant_offset=3.7,
        )
        gamma = [0.23, -0.11]
        beta = [0.17, 0.09]
        base = lightcone_gradient_adjoint(
            self.graph,
            gamma,
            beta,
            prefer_gpu=False,
            complex_dtype=np.complex128,
            float_dtype=np.float64,
        )
        with_offset = lightcone_gradient_adjoint(
            shifted,
            gamma,
            beta,
            prefer_gpu=False,
            complex_dtype=np.complex128,
            float_dtype=np.float64,
        )
        self.assertAlmostEqual(with_offset.value - base.value, 3.7, places=12)
        np.testing.assert_allclose(
            with_offset.gradient, base.gradient, rtol=0, atol=1e-12
        )

    def test_budget_rejection_precedes_any_batch_execution(self):
        graph = WeightedGraph(
            n=4,
            edges=((0, 1, 0.75), (1, 2, -0.4), (2, 3, 0.3)),
            fields=((0, -0.2),),
            objective="qubo",
        )
        p = 1
        small_plan = plan_checkpoint_schedule(
            policy="budgeted",
            p=p,
            k=2,
            group_size=1,
            max_batch_states=1 << 12,
            complex_dtype=np.complex128,
            float_dtype=np.float64,
            memory_budget_bytes=10**9,
        )
        large_plan = plan_checkpoint_schedule(
            policy="budgeted",
            p=p,
            k=4,
            group_size=1,
            max_batch_states=1 << 12,
            complex_dtype=np.complex128,
            float_dtype=np.float64,
            memory_budget_bytes=10**9,
        )
        budget = (small_plan.predicted_active_bytes + large_plan.predicted_active_bytes) // 2

        with patch(
            "lcqaoa.lightcone._evaluate_batch_with_checkpoint_gradient"
        ) as evaluate_batch:
            with self.assertRaisesRegex(MemoryError, "below the minimum predicted"):
                lightcone_gradient_adjoint(
                    graph,
                    [0.1],
                    [0.2],
                    prefer_gpu=False,
                    complex_dtype=np.complex128,
                    float_dtype=np.float64,
                    checkpoint_policy="budgeted",
                    memory_budget_bytes=budget,
                )
            evaluate_batch.assert_not_called()


if __name__ == "__main__":
    unittest.main()

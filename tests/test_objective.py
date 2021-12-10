"""Test wrapped objective."""
# pylint: disable=missing-class-docstring,missing-function-docstring
import operator
import unittest
from typing import Any

import scipy.optimize

from wrapdisc import Objective
from wrapdisc.var import ChoiceVar, GridVar, QrandintVar, QuniformVar, RandintVar, UniformVar


def _mixed_optimization_objective(x: tuple, *args: Any) -> float:  # pylint: disable=invalid-name
    return float(sum(x_i if isinstance(x_i, (int, float)) else len(str(x_i)) for x_i in (*x, *args)))


class TestObjective(unittest.TestCase):
    def setUp(self) -> None:
        self.objective = Objective(
            _mixed_optimization_objective,
            [
                ChoiceVar(["foobar", "baz"]),
                ChoiceVar([operator.index, abs, operator.invert]),
                ChoiceVar(["x"]),
                GridVar([0.01, 0.1, 1, 10, 100]),
                GridVar(["good", "better", "best"]),
                RandintVar(-8, 10),
                QrandintVar(1, 10, 2),
                UniformVar(1.2, 3.4),
                QuniformVar(-11.1, 9.99, 0.22),
                QuniformVar(4.6, 81.7, 0.2),
            ],
        )

    def test_objective(self):
        # Test bounds
        expected_bounds = (
            *((0.0, 1.0), (0.0, 1.0)),  # ChoiceVar 1
            *((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),  # ChoiceVar 2
            (-0.49999999999999994, 4.499999999999999),  # GridVar 1
            (-0.49999999999999994, 2.4999999999999996),  # GridVar 2
            (-8.499999999999998, 10.499999999999998),  # RandintVar
            (1.0000000000000002, 10.999999999999998),  # QrandintVar
            (1.2, 3.4),  # UniformVar
            (-11.109999999999998, 10.009999999999998),  # QuniformVar
            (4.500000000000001, 81.69999999999999),  # QuniformVar
        )
        self.assertEqual(self.objective.bounds, expected_bounds)

        # Test decoding
        encoded = (
            *(0.3, 0.8),  # ChoiceVar 1
            *(0.11, 0.44, 0.33),  # ChoiceVar 2
            0.0,  # GridVar 1
            2.499,  # GridVar 2
            -3.369,  # RandintVar
            2.0,  # QrandintVar
            1.909,  # UniformVar
            -11.09,  # QuniformVar
            76.55,  # QuinformVar  (required for test coverage of `round_up`)
        )
        expected_decoded = (
            "baz",  # ChoiceVar 1
            abs,  # ChoiceVar 2
            "x",  # ChoiceVar 3
            0.01,  # GridVar 1
            "best",  # GridVar 2
            -3,  # RandintVar
            2,  # QrandintVar
            1.909,  # UniformVar
            -11.0,  # QuniformVar
            76.60000000000001,  # QuniformVar  (!= 76.6 due to floating point limitation in `round_up`)
        )
        actual_decoded = self.objective[encoded]
        self.assertEqual(actual_decoded, expected_decoded)

        # Test function
        self.assertEqual(self.objective(encoded), _mixed_optimization_objective(actual_decoded))
        self.assertEqual(self.objective(encoded), 97.519)

        # Test cache
        self.assertEqual(self.objective.cache_info._asdict(), {"currsize": 1, "hits": 1, "maxsize": None, "misses": 1})

    def test_minimize(self):
        # Test result
        result = scipy.optimize.differential_evolution(self.objective, self.objective.bounds, seed=0)
        self.assertIsInstance(result.fun, float)

        # Test solution
        encoded_solution = result.x
        decoded_solution = self.objective[encoded_solution]
        cache_info = self.objective.cache_info
        expected_decoded_solution = ("baz", abs, "x", 0.01, "good", -8, 2, 1.2, -11.0, 4.6000000000000005)
        self.assertEqual(decoded_solution, expected_decoded_solution)
        self.assertEqual(result.fun, self.objective(encoded_solution))
        self.assertEqual(result.fun, _mixed_optimization_objective(decoded_solution))

        # Test cache
        self.assertGreaterEqual(cache_info.hits, 1)
        expected_nfev = cache_info.currsize + cache_info.hits
        self.assertEqual(result.nfev, expected_nfev)

    def test_minimize_with_fixed_args(self):
        fixed_args = ("arg1", 2, 3.0)
        result = scipy.optimize.differential_evolution(self.objective, bounds=self.objective.bounds, args=fixed_args, seed=0)
        self.assertIsInstance(result.fun, float)

        # Test solution
        encoded_solution = result.x
        decoded_solution = self.objective[encoded_solution]
        cache_info = self.objective.cache_info
        expected_decoded_solution = ("baz", abs, "x", 0.01, "good", -8, 2, 1.2, -11.0, 4.6000000000000005)
        self.assertEqual(decoded_solution, expected_decoded_solution)
        self.assertEqual(result.fun, self.objective(encoded_solution, *fixed_args))
        self.assertEqual(result.fun, _mixed_optimization_objective(decoded_solution, *fixed_args))

        # Test cache
        self.assertGreaterEqual(cache_info.hits, 1)
        expected_nfev = cache_info.currsize + cache_info.hits
        self.assertEqual(result.nfev, expected_nfev)

    def test_minimize_with_multiple_workers(self):
        # Test result
        updating = "deferred"  # Prevents a warning if when using workers > 1
        result = scipy.optimize.differential_evolution(self.objective, self.objective.bounds, seed=0, workers=2, updating=updating)
        self.assertIsInstance(result.fun, float)

        # Test solution
        encoded_solution = result.x
        decoded_solution = self.objective[encoded_solution]
        expected_decoded_solution = ("baz", abs, "x", 0.01, "best", -8, 2, 1.2000041000840649, -11.0, 4.6000000000000005)
        self.assertEqual(decoded_solution, expected_decoded_solution)
        self.assertEqual(result.fun, self.objective(encoded_solution))
        self.assertEqual(result.fun, _mixed_optimization_objective(decoded_solution))

        # Note: Unlike in test_minimize, cache assertions are skipped because a separate cache exists in each worker.

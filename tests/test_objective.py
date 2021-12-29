"""Test wrapped objective."""
# pylint: disable=missing-class-docstring,missing-function-docstring
import math
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
                GridVar(["uno"]),
                RandintVar(-8, 10),
                RandintVar(1, 1),
                QrandintVar(1, 10, 2),
                UniformVar(1.2, 3.4),
                QuniformVar(-11.1, 9.99, 0.22),
                QuniformVar(4.6, 81.7, 0.2),  # Required for test coverage of `round_up`.
            ],
        )
        self.decoded_guess = ("foobar", operator.invert, "x", 10, "better", "uno", 0, 1, 8, 2.33, 8.8, 56.6)

    def test_objective(self):
        self.assertEqual(self.objective.vars.decoded_len, len(self.decoded_guess))

        # Test bounds
        expected_bounds = (
            *((0.0, 1.0), (0.0, 1.0)),  # ChoiceVar 1
            *((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),  # ChoiceVar 2
            (-0.49999999999999994, 4.499999999999999),  # GridVar 1
            (-0.49999999999999994, 2.4999999999999996),  # GridVar 2
            (-8.499999999999998, 10.499999999999998),  # RandintVar
            (1.0000000000000002, 10.999999999999998),  # QrandintVar
            (1.2, 3.4),  # UniformVar
            (-11.109999999999998, 10.009999999999998),  # QuniformVar 1
            (4.500000000000001, 81.69999999999999),  # QuniformVar 2
        )
        self.assertEqual(expected_bounds, self.objective.bounds)

        # Test decoding
        encoded = (
            *(0.3, 0.8),  # ChoiceVar 1
            *(0.11, 0.44, 0.33),  # ChoiceVar 2
            0.0,  # GridVar 1
            2.499,  # GridVar 2
            -3.369,  # RandintVar
            2.0,  # QrandintVar
            1.909,  # UniformVar
            -11.09,  # QuniformVar 1
            76.55,  # QuinformVar 2
        )
        expected_decoded = (
            "baz",  # ChoiceVar 1
            abs,  # ChoiceVar 2
            "x",  # ChoiceVar 3
            0.01,  # GridVar 1
            "best",  # GridVar 2
            "uno",  # GridVar 3
            -3,  # RandintVar 1
            1,  # RandintVar 2
            2,  # QrandintVar
            1.909,  # UniformVar
            -11.0,  # QuniformVar 1
            76.6,  # QuniformVar 2
        )
        actual_decoded = self.objective.decode(encoded)
        self.assertEqual(expected_decoded, actual_decoded)

        # Test encoding
        decoded = expected_decoded
        expected_encoded = (
            *(0.0, 1.0),  # ChoiceVar 1
            *(0.0, 1.0, 0.0),  # ChoiceVar 2
            0.0,  # GridVar 1
            2.0,  # GridVar 2
            -3.0,  # RandintVar
            2.0,  # QrandintVar
            1.909,  # UniformVar
            -11.0,  # QuniformVar 1
            76.6,  # QuinformVar 2
        )
        actual_encoded = self.objective.encode(decoded)
        self.assertEqual(expected_encoded, actual_encoded)

        # Test reversing
        self.assertEqual(decoded, self.objective.decode(self.objective.encode(decoded)))
        self.assertEqual(self.decoded_guess, self.objective.decode(self.objective.encode(self.decoded_guess)))

        # Test function
        self.assertEqual(self.objective(encoded), _mixed_optimization_objective(actual_decoded))
        self.assertAlmostEqual(self.objective(encoded), 101.519)

        # Test cache
        self.assertEqual(self.objective.cache_info._asdict(), {"currsize": 1, "hits": 1, "maxsize": None, "misses": 1})

    def test_encoded_nan(self):
        encoded = [math.nan] * self.objective.vars.encoded_len
        self.assertTrue(math.isnan(self.objective(encoded)))

    def test_duplicated_var(self):
        objective = Objective(_mixed_optimization_objective, [GridVar(["yes", "no"])] * 2)
        self.assertEqual(objective.encode(["yes", "no"]), (0.0, 1.0))
        self.assertEqual(objective.decode((0.0, 1.0)), ("yes", "no"))
        self.assertEqual(objective((0.0, 1.0)), 5.0)
        self.assertEqual(objective.bounds, ((-0.49999999999999994, 1.4999999999999998),) * 2)

    def test_optimize_de_with_optionals(self):
        optional_fixed_args = ("arg1", 2, 3.0)
        optional_encoded_x0 = self.objective.encode(self.decoded_guess)
        result = scipy.optimize.differential_evolution(self.objective, bounds=self.objective.bounds, seed=0, args=optional_fixed_args, x0=optional_encoded_x0)
        self.assertIsInstance(result.fun, float)

        # Test solution
        encoded_solution = result.x
        decoded_solution = self.objective.decode(encoded_solution)
        cache_info = self.objective.cache_info
        expected_decoded_solution = ("baz", abs, "x", 0.01, "good", "uno", -8, 1, 2, 1.2, -11.0, 4.6)
        self.assertEqual(decoded_solution, expected_decoded_solution)
        self.assertEqual(result.fun, self.objective(encoded_solution, *optional_fixed_args))
        self.assertEqual(result.fun, _mixed_optimization_objective(decoded_solution, *optional_fixed_args))

        # Test cache
        self.assertGreaterEqual(cache_info.hits, 1)
        expected_nfev = cache_info.currsize + cache_info.hits
        self.assertEqual(result.nfev, expected_nfev)

    def test_optimize_de_with_multiple_workers(self):
        # Test result
        updating = "deferred"  # Prevents a warning with workers > 1
        result = scipy.optimize.differential_evolution(self.objective, self.objective.bounds, seed=0, workers=2, updating=updating)
        self.assertIsInstance(result.fun, float)

        # Test solution
        encoded_solution = result.x
        decoded_solution = self.objective.decode(encoded_solution)
        expected_decoded_solution = ("baz", abs, "x", 0.01, "best", "uno", -8, 1, 2, 1.2000041000840649, -11.0, 4.6)
        self.assertEqual(decoded_solution, expected_decoded_solution)
        self.assertEqual(result.fun, self.objective(encoded_solution))
        self.assertEqual(result.fun, _mixed_optimization_objective(decoded_solution))

        # Note: Unlike in test_optimize_de, cache assertions are skipped because a separate cache exists in each worker.

    def test_optimize_minimize(self):
        result = scipy.optimize.minimize(self.objective, x0=self.objective.encode(self.decoded_guess), bounds=self.objective.bounds, method="Nelder-Mead")
        self.assertIsInstance(result.fun, float)

        # Test solution
        encoded_solution = result.x
        decoded_solution = self.objective.decode(encoded_solution)
        cache_info = self.objective.cache_info
        expected_decoded_solution = ("foobar", operator.invert, "x", 1, "good", "uno", 0, 1, 2, 1.2, -11.0, 5.4)
        self.assertEqual(decoded_solution, expected_decoded_solution)
        self.assertEqual(result.fun, self.objective(encoded_solution))
        self.assertEqual(result.fun, _mixed_optimization_objective(decoded_solution))

        # Test cache
        self.assertGreaterEqual(cache_info.hits, 1)
        expected_nfev = cache_info.currsize + cache_info.hits
        self.assertEqual(result.nfev, expected_nfev)

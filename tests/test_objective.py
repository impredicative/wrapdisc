"""Test wrapped objective."""
# pylint: disable=missing-class-docstring,missing-function-docstring
import operator
import unittest
from typing import Any

from wrapdisc.var import ChoiceVar, GridVar, QrandintVar, QuniformVar, RandintVar, UniformVar
from wrapdisc.wrapdisc import Objective


def _mixed_optimization_objective(*args: Any) -> float:
    return float(sum(len(str(a)) for a in args))


class TestObjective(unittest.TestCase):
    def setUp(self) -> None:
        self.objective = Objective(
            _mixed_optimization_objective,
            [
                ChoiceVar(["foo", "bar"]),
                ChoiceVar([operator.add, operator.sub, operator.mul]),
                ChoiceVar(["x"]),
                GridVar([0.01, 0.1, 1, 10, 100]),
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
            (-0.49999999999999994, 4.499999999999999),  # GridVar
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
            0.0,  # GridVar
            -3.369,  # RandintVar
            2.0,  # QrandintVar
            1.909,  # UniformVar
            -11.09,  # QuniformVar
            76.55,  # QuinformVar  (required for test coverage of `round_up`)
        )
        expected_decoded = (
            "bar",  # ChoiceVar 1
            operator.sub,  # ChoiceVar 2
            "x",  # ChoiceVar 3
            0.01,  # GridVar
            -3,  # RandintVar
            2,  # QrandintVar
            1.909,  # UniformVar
            -11.0,  # QuniformVar
            76.60000000000001,  # QuniformVar  (!= 76.6 due to floating point limitation in `round_up`)
        )
        actual_decoded = self.objective[encoded]
        self.assertEqual(actual_decoded, expected_decoded)

        # Test function
        self.assertEqual(self.objective(*encoded), _mixed_optimization_objective(*actual_decoded))
        self.assertEqual(self.objective(*encoded), 61.0)

        # Test cache
        self.assertEqual(self.objective.cache_info._asdict(), {"currsize": 1, "hits": 1, "maxsize": None, "misses": 1})

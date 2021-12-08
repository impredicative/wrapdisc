"""Test objective."""
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
                RandintVar(1, 10),
                QrandintVar(1, 10, 2),
                UniformVar(1.2, 3.4),
                QuniformVar(0.1, 9.99, 0.22),
            ],
        )

    def test_decoding(self):
        decoded = self.objective[
            (
                *(0.3, 0.8),  # ChoiceVar 1
                *(0.11, 0.44, 0.33),  # ChoiceVar 2
                0.0,  # GridVar
                10.4999,  # RandintVar
                2.0,  # QrandintVar
                1.909,  # UniformVar
                0.111,  # QuniformVar
            )
        ]
        expected = (
            "bar",  # ChoiceVar 1
            operator.sub,  # ChoiceVar 2
            "x",  # ChoiceVar 3
            0.01,  # GridVar
            10,  # RandintVar
            2,  # QrandintVar
            1.909,  # UniformVar
            0.22,  # QuniformVar
        )
        self.assertEqual(decoded, expected)

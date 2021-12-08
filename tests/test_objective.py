"""Test objective."""
# pylint: disable=missing-class-docstring,missing-function-docstring
import operator
import unittest

from wrapdisc.var import ChoiceVar, GridVar, QrandintVar, QuniformVar, RandintVar, UniformVar
from wrapdisc.wrapdisc import Objective


class TestObjective(unittest.TestCase):
    def test_with_all_vars(self):
        objective = Objective(
            lambda *args: sum(len(str(a)) for a in args),
            [
                ChoiceVar(["foo", "bar"]),
                ChoiceVar([operator.add, operator.sub, operator.mul]),
                GridVar([0.01, 0.1, 1, 10, 100]),
                RandintVar(1, 10),
                QrandintVar(1, 10, 2),
                UniformVar(1.2, 3.4),
                QuniformVar(0, 9.99, 0.2),
            ],
        )

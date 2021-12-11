"""float utilities."""
import math
from fractions import Fraction
from typing import overload


def next_float(val: float, /) -> float:
    """Return the float that's next after the given float."""
    return math.nextafter(val, math.inf)


def prev_float(val: float, /) -> float:
    """Return the float that's just before the given float."""
    return math.nextafter(val, -math.inf)


@overload
def round_nearest(num: float, to: int) -> int:
    pass


@overload
def round_nearest(num: float, to: float) -> float:
    pass


def round_nearest(num, to):
    """Round `num` to the nearest multiple of `to`."""
    # Ref: https://stackoverflow.com/a/70210770/
    num, to = Fraction(str(num)), Fraction(str(to))
    return float(round(num / to) * to)


def round_down(num: float, to: float) -> float:
    """Round `num` down to the nearest multiple of `to`."""
    # Ref: https://stackoverflow.com/a/70210770/
    num, to = Fraction(str(num)), Fraction(str(to))
    return float(math.floor(num / to) * to)


def round_up(num: float, to: float) -> float:
    """Round `num` up to the nearest multiple of `to`."""
    # Ref: https://stackoverflow.com/a/70210770/
    num, to = Fraction(str(num)), Fraction(str(to))
    return float(math.ceil(num / to) * to)

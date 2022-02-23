"""float utilities."""
from decimal import Decimal
from math import ceil, floor, inf, nextafter
from typing import Sequence, overload


def div_float(x: float, y: int | float, /) -> float:
    """Return x divided by y.

    Intermediate string representations are used.
    """
    return float(Decimal(str(x)) / Decimal(str(y)))


def sum_floats(nums: Sequence[int | float]) -> float:
    """Return the sum of the given numbers.

    Intermediate string representations are used.
    """
    # Note: math.fsum is not used because it was observed to not work well for the example [9.9, .05].
    return float(sum(Decimal(str(f)) for f in nums))


def next_float(val: float, /) -> float:
    """Return the float that's next after the given float."""
    return nextafter(val, inf)


def prev_float(val: float, /) -> float:
    """Return the float that's just before the given float."""
    return nextafter(val, -inf)


@overload
def round_nearest(num: float, to: int) -> int:
    pass


@overload
def round_nearest(num: float, to: float) -> float:
    pass


def round_nearest(num, to):
    """Round `num` to the nearest multiple of `to`.

    Intermediate string representations are used.
    """
    # Ref: https://stackoverflow.com/a/70210770/
    num, to = Decimal(str(num)), Decimal(str(to))
    return float(round(num / to) * to)


def round_down(num: float, to: float) -> float:
    """Round `num` down to the nearest multiple of `to`.

    Intermediate string representations are used.
    """
    # Ref: https://stackoverflow.com/a/70210770/
    num, to = Decimal(str(num)), Decimal(str(to))
    return float(floor(num / to) * to)


def round_up(num: float, to: float) -> float:
    """Round `num` up to the nearest multiple of `to`.

    Intermediate string representations are used.
    """
    # Ref: https://stackoverflow.com/a/70210770/
    num, to = Decimal(str(num)), Decimal(str(to))
    return float(ceil(num / to) * to)

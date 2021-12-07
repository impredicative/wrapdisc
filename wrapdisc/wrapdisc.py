"""Package implementation."""

import abc
import itertools
import math
from functools import _CacheInfo as CacheInfo
from functools import cache, cached_property
from typing import Any, Callable, Sequence, overload

BoundType = tuple[float, float]
BoundsType = tuple[BoundType, ...]
EncodingType = Sequence[int | float]


def next_float(val: float, /) -> float:
    """Return the float that's next after the given float."""
    return math.nextafter(val, math.inf)


def prev_float(val: float, /) -> float:
    """Return the float that's just before the given float."""
    return math.nextafter(val, -math.inf)


@overload
def round_nearest(num: float, to: float) -> float:
    ...


@overload
def round_nearest(num: float, to: int) -> int:
    ...


def round_nearest(num, to):
    """Round `num` to the nearest multiple of `to`."""
    # Ref: https://stackoverflow.com/a/70210770/
    return round(num / to) * to


def round_down(num: float, to: float) -> float:
    """Round `num` down to the nearest multiple of `to`."""
    # Ref: https://stackoverflow.com/a/70210770/
    nearest = round_nearest(num, to)
    if math.isclose(num, nearest):
        return num
    return nearest if nearest < num else nearest - to


def round_up(num: float, to: float) -> float:
    """Round `num` up to the nearest multiple of `to`."""
    # Ref: https://stackoverflow.com/a/70210770/
    nearest = round_nearest(num, to)
    if math.isclose(num, nearest):
        return num
    return nearest if nearest > num else nearest + to


class BaseVar(abc.ABC):
    """Abstract class for variable classes."""
    @cache
    def __len__(self) -> int:
        """Return the length of an encoded solution."""
        return len(self.bounds)

    def __getitem__(self, encoded: EncodingType) -> Any:
        """Return the decoded solution from its encoded solution."""
        return self.decode(encoded)

    @cached_property
    @abc.abstractmethod
    def bounds(self) -> BoundsType:
        """Return the encoded bounds to provide to an optimizer such as `scipy.optimize`."""
        return ((0.0, 1.0),)

    @abc.abstractmethod
    def decode(self, encoded: EncodingType, /) -> Any:
        """Return the decoded solution from its encoded solution."""
        return encoded[0]


class UniformVar(BaseVar):
    def __init__(self, lower: float, upper: float):
        """Sample a float value uniformly between `lower` and `upper`."""
        # Motivational reference: https://docs.ray.io/en/latest/tune/api_docs/search_space.html#tune-uniform
        self.lower, self.upper = float(lower), float(upper)
        assert self.lower <= self.upper

    @cached_property
    def bounds(self) -> BoundsType:
        return ((self.lower, self.upper),)

    def decode(self, encoded: EncodingType, /) -> float:
        assert len(encoded) == 1
        assert isinstance(encoded[0], (float, int))
        assert self.bounds[0][0] <= encoded[0] <= self.bounds[0][1]  # Invalid encoded value.
        decoded = float(encoded[0])
        assert self.lower <= decoded <= self.upper, decoded  # Invalid decoded value.
        return decoded


class QuniformVar(BaseVar):
    def __init__(self, lower: float, upper: float, q: float):
        """Sample a float value uniformly between `lower` and `upper`, quantized to an integer multiple of `q`."""
        # Motivational reference: https://docs.ray.io/en/latest/tune/api_docs/search_space.html#tune-quniform
        self.lower, self.upper, self.quantum = float(lower), float(upper), float(q)
        assert self.lower <= self.upper
        assert 0 < self.quantum <= (self.upper - self.lower)

    @cached_property
    def bounds(self) -> BoundsType:
        half_step = self.quantum / 2
        quantized_lower = round_up(self.lower, self.quantum)
        quantized_upper = round_down(self.upper, self.quantum)
        assert self.lower <= quantized_lower <= quantized_upper <= self.upper
        assert self.quantum <= (quantized_upper - quantized_lower)
        return ((next_float(quantized_lower - half_step), prev_float(quantized_upper + half_step)),)
        # Note: Using half_step allows uniform probability for boundary values of encoded range.
        # Note: Using next_float and prev_float prevent decoding a boundary value of encoded range to a decoded value outside the valid decoded range.

    def decode(self, encoded: EncodingType, /) -> float:
        assert len(encoded) == 1
        assert isinstance(encoded[0], (float, int))
        assert self.bounds[0][0] <= encoded[0] <= self.bounds[0][1]  # Invalid encoded value.
        decoded = round_nearest(encoded[0], self.quantum)
        assert isinstance(decoded, float)
        assert self.lower <= decoded <= self.upper, decoded  # Invalid decoded value.
        return decoded


class RandintVar(BaseVar):
    def __init__(self, lower: int, upper: int):
        """Sample an integer value uniformly between `lower` and `upper`, both inclusive.

        As a reminder, unlike in `ray.tune.randint`, `upper` is inclusive.
        """
        # Motivational reference: https://docs.ray.io/en/latest/tune/api_docs/search_space.html#tune-randint
        assert all(isinstance(arg, int) for arg in (lower, upper))
        self.lower, self.upper = lower, upper
        assert self.lower <= self.upper

    @cached_property
    def bounds(self) -> BoundsType:
        half_step = 0.5
        return ((next_float(self.lower - half_step), prev_float(self.upper + half_step)),)
        # Note: Using half_step allows uniform probability for boundary values of encoded range.
        # Note: Using next_float and prev_float prevent decoding a boundary value of encoded range to a decoded value outside the valid decoded range.

    def decode(self, encoded: EncodingType, /) -> int:
        assert len(encoded) == 1
        assert isinstance(encoded[0], (float, int))
        assert self.bounds[0][0] <= encoded[0] <= self.bounds[0][1]  # Invalid encoded value.
        decoded = round(encoded[0])
        assert isinstance(decoded, int)
        assert self.lower <= decoded <= self.upper, decoded  # Invalid decoded value.
        return decoded


class QrandintVar(BaseVar):
    def __init__(self, lower: int, upper: int, q: int):
        """Sample an integer value uniformly between `lower` and `upper`, both inclusive, quantized to an integer multiple of `q`."""
        # Motivational reference: https://docs.ray.io/en/latest/tune/api_docs/search_space.html#tune-qrandint
        assert all(isinstance(arg, int) for arg in (lower, upper, q))
        self.lower, self.upper, self.quantum = lower, upper, q
        assert self.lower <= self.upper
        assert 1 <= self.quantum <= (self.upper - self.lower)

    @cached_property
    def bounds(self) -> BoundsType:
        half_step = self.quantum / 2
        quantized_lower = round_up(self.lower, self.quantum)
        quantized_upper = round_down(self.upper, self.quantum)
        assert self.lower <= quantized_lower <= quantized_upper <= self.upper
        assert self.quantum <= (quantized_upper - quantized_lower)
        return ((next_float(quantized_lower - half_step), prev_float(quantized_upper + half_step)),)
        # Note: Using half_step allows uniform probability for boundary values of encoded range.
        # Note: Using next_float and prev_float prevent decoding a boundary value of encoded range to a decoded value outside the valid decoded range.

    def decode(self, encoded: EncodingType, /) -> int:
        assert len(encoded) == 1
        assert isinstance(encoded[0], (float, int))
        assert self.bounds[0][0] <= encoded[0] <= self.bounds[0][1]  # Invalid encoded value.
        decoded = round_nearest(encoded[0], self.quantum)
        assert isinstance(decoded, int)
        assert self.lower <= decoded <= self.upper, decoded  # Invalid decoded value.
        return decoded


class ChoiceVar(BaseVar):
    def __init__(self, categories: list[Any]):
        """Sample a categorical value.

        The one-max variation of one-hot encoding is used, such that the category with the max encoded value is sampled.
        In the unlikely event that multiple categories share an encoded max value, the decoded value is the first of these categories in the order of the input.
        """
        # Motivational reference: https://docs.ray.io/en/latest/tune/api_docs/search_space.html#tune-choice
        assert categories
        self.categories = categories
        num_categories = len(self.categories)
        assert num_categories == len(set(self.categories))
        self.encoding_len = 0 if (num_categories == 1) else num_categories
        # Note: A boolean representation of a single encoded variable is intentionally not used if there are two categories.

    @cached_property
    def bounds(self) -> BoundsType:
        return ((0.0, 1.0),) * self.encoding_len

    def decode(self, encoded: EncodingType, /) -> Any:
        assert len(encoded) == self.encoding_len
        if self.encoding_len > 1:
            assert all(isinstance(f, (float, int)) for f in encoded)
            assert all((0.0 <= f <= 1.0) for f in encoded)
            index = max(range(len(encoded)), key=encoded.__getitem__)
            return self.categories[index]  # First category having max value is selected.
        assert self.encoding_len == 0
        return self.categories[0]


class GridVar(BaseVar):
    def __init__(self, values: list[Any]):
        """Sample a grid uniformly."""
        # Motivational reference: https://docs.ray.io/en/latest/tune/api_docs/search_space.html#grid-search-api
        assert values
        self.values = sorted(values)
        assert len(self.values) == len(set(self.values))
        self.randint_var = RandintVar(0, len(values) - 1)

    @cached_property
    def bounds(self) -> BoundsType:
        return self.randint_var.bounds

    def decode(self, encoded: EncodingType, /) -> Any:
        decoded = self.randint_var.decode(encoded)
        return self.values[decoded]


class Vars:
    def __init__(self, variables: Sequence[BaseVar]):
        """Return a solution decoder for multiple variables."""
        assert all(isinstance(var, BaseVar) for var in variables)
        self._variables = variables
        variable_lengths = [len(v) for v in self._variables]
        self._variables_slices = {
            var: slice(tot_len - cur_len, tot_len) for var, cur_len, tot_len in zip(self._variables, variable_lengths, itertools.accumulate(variable_lengths))
        }.items()
        self.decoded_len = len(self._variables)
        self.encoded_len = sum(variable_lengths)

    def __getitem__(self, encoded: EncodingType) -> tuple:
        """Return the decoded solution from its encoded solution."""
        assert len(encoded) == self.encoded_len
        decoded = tuple(var[encoded[var_slice]] for var, var_slice in self._variables_slices)
        assert len(decoded) == self.decoded_len
        return decoded


class Objective:
    def __init__(self, func: Callable, variables: Sequence[BaseVar]):
        """Return the wrapped optimization objective callable.

        An unbounded in-memory cache is used over the given input function. This is essential for preventing redundant calls to the input function.

        Note that the wrapped objective function is unsuitable for production use. For production use, the given input function is to be called directly.
        """
        self.func = cache(func)
        self.vars = Vars(variables)

    def __getitem__(self, encoded: EncodingType) -> tuple:
        """Return the decoded solution from its encoded solution."""
        return self.vars[encoded]

    def __call__(self, *encoded: Any) -> float:
        """Return the result from calling the objective function with the decoded solution from its encoded solution.

        This method is the transformed optimization objective.
        """
        return self.func(*self[encoded])

    @property
    def cache_info(self) -> CacheInfo:
        """Return info about the cache over the input function."""
        return self.func.cache_info()

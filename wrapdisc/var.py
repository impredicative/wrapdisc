"""Various variable classes."""

import abc
from functools import cache, cached_property
from typing import Any, Sequence, Union, final

from wrapdisc.util.float import div_float, next_float, prev_float, round_down, round_nearest, round_up, sum_floats

BoundType = tuple[float, float]
BoundsType = Sequence[BoundType]
EncodingType = Sequence[Union[int, float]]


class BaseVar(abc.ABC):
    """Abstract class for variable classes."""

    @final
    @cache
    def __len__(self) -> int:
        """Return the length of an encoded solution."""
        return len(self.bounds)

    @cached_property
    @abc.abstractmethod
    def bounds(self) -> BoundsType:
        """Return the encoded bounds to provide to an optimizer such as `scipy.optimize`."""
        return ((0.0, 1.0),)  # pragma: no cover

    @abc.abstractmethod
    def decode(self, encoded: EncodingType, /) -> Any:
        """Return the decoded solution from its encoded solution.

        Note that multiple encoded solutions can correspond to the same decoded solution, but a decoded solution corresponds to a single encoded solution.
        """
        return encoded[0]  # pragma: no cover

    @abc.abstractmethod
    def encode(self, decoded: Any) -> EncodingType:
        """Return the encoded solution from its decoded solution.

        Note that multiple encoded solutions can correspond to the same decoded solution, but a decoded solution corresponds to a single encoded solution.
        """
        return (decoded,)  # pragma: no cover


class ChoiceVar(BaseVar):
    """Category sampler."""

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
            hot_index = max(range(len(encoded)), key=encoded.__getitem__)
            decoded = self.categories[hot_index]  # First category having max value is selected.
        else:
            assert self.encoding_len == 0
            decoded = self.categories[0]
        return decoded

    def encode(self, decoded: Any) -> EncodingType:
        assert decoded in self.categories
        if self.encoding_len > 1:
            hot_index = self.categories.index(decoded)
            encoded = tuple(1.0 if cur_index == hot_index else 0.0 for cur_index in range(self.encoding_len))
        else:
            assert self.encoding_len == 0
            encoded = ()
        assert decoded == self.decode(encoded)
        return encoded


class UniformVar(BaseVar):
    """Uniform float sampler."""

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

    def encode(self, decoded: float) -> EncodingType:
        assert isinstance(decoded, (float, int))
        assert self.lower <= decoded <= self.upper, decoded  # Invalid decoded value.
        encoded = (float(decoded),)
        assert self.bounds[0][0] <= encoded[0] <= self.bounds[0][1]  # Invalid encoded value.
        assert decoded == self.decode(encoded)
        return encoded


class QuniformVar(BaseVar):
    """Uniform quantized float sampler."""

    def __init__(self, lower: float, upper: float, q: float):  # pylint: disable=invalid-name
        """Sample a float value uniformly between `lower` and `upper`, quantized to an integer multiple of `q`."""
        # Motivational reference: https://docs.ray.io/en/latest/tune/api_docs/search_space.html#tune-quniform
        self.lower, self.upper, self.quantum = float(lower), float(upper), float(q)
        assert self.lower <= self.upper
        assert 0 < self.quantum <= (self.upper - self.lower)

    @cached_property
    def bounds(self) -> BoundsType:
        half_step = div_float(self.quantum, 2)
        quantized_lower = round_up(self.lower, self.quantum)
        quantized_upper = round_down(self.upper, self.quantum)
        assert self.lower <= quantized_lower <= quantized_upper <= self.upper
        assert self.quantum <= (quantized_upper - quantized_lower)
        lower_bound = next_float(sum_floats((quantized_lower, -half_step)))
        upper_bound = prev_float(sum_floats((quantized_upper, half_step)))
        # Note: Using half_step allows uniform probability for boundary values of encoded range.
        # Note: Using next_float and prev_float prevent decoding a boundary value of encoded range to a decoded value outside the valid decoded range.
        return ((lower_bound, upper_bound),)

    def decode(self, encoded: EncodingType, /) -> float:
        assert len(encoded) == 1
        assert isinstance(encoded[0], (float, int))
        assert self.bounds[0][0] <= encoded[0] <= self.bounds[0][1]  # Invalid encoded value.
        decoded = round_nearest(encoded[0], self.quantum)
        assert isinstance(decoded, float)
        assert self.lower <= decoded <= self.upper, decoded  # Invalid decoded value.
        return decoded

    def encode(self, decoded: float) -> EncodingType:
        assert isinstance(decoded, (float, int))
        assert self.lower <= decoded <= self.upper, decoded  # Invalid decoded value.
        assert decoded == round_nearest(decoded, self.quantum)  # Invalid encoded value.
        encoded = (float(decoded),)
        assert self.bounds[0][0] <= encoded[0] <= self.bounds[0][1]  # Invalid encoded value.
        assert decoded == self.decode(encoded)
        return encoded


class RandintVar(BaseVar):
    """Uniform integer sampler."""

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
        lower_bound = next_float(self.lower - half_step)
        upper_bound = prev_float(self.upper + half_step)
        # Note: Using half_step allows uniform probability for boundary values of encoded range.
        # Note: Using next_float and prev_float prevent decoding a boundary value of encoded range to a decoded value outside the valid decoded range.
        return ((lower_bound, upper_bound),)

    def decode(self, encoded: EncodingType, /) -> int:
        assert len(encoded) == 1
        assert isinstance(encoded[0], (float, int))
        assert self.bounds[0][0] <= encoded[0] <= self.bounds[0][1]  # Invalid encoded value.
        decoded = round(encoded[0])
        assert isinstance(decoded, int)
        assert self.lower <= decoded <= self.upper, decoded  # Invalid decoded value.
        return decoded

    def encode(self, decoded: int) -> EncodingType:
        assert isinstance(decoded, int)
        assert self.lower <= decoded <= self.upper, decoded  # Invalid decoded value.
        encoded = (float(decoded),)
        assert self.bounds[0][0] <= encoded[0] <= self.bounds[0][1]  # Invalid encoded value.
        assert decoded == self.decode(encoded)
        return encoded


class QrandintVar(BaseVar):
    """Uniform quantized integer sampler."""

    def __init__(self, lower: int, upper: int, q: int):  # pylint: disable=invalid-name
        """Sample an integer value uniformly between `lower` and `upper`, both inclusive, quantized to an integer multiple of `q`."""
        # Motivational reference: https://docs.ray.io/en/latest/tune/api_docs/search_space.html#tune-qrandint
        assert all(isinstance(arg, int) for arg in (lower, upper, q))
        self.lower, self.upper, self.quantum = lower, upper, q
        assert self.lower <= self.upper
        assert 1 <= self.quantum <= (self.upper - self.lower)

    @cached_property
    def bounds(self) -> BoundsType:
        half_step = div_float(self.quantum, 2)
        quantized_lower = round_up(self.lower, self.quantum)
        quantized_upper = round_down(self.upper, self.quantum)
        assert self.lower <= quantized_lower <= quantized_upper <= self.upper
        assert self.quantum <= (quantized_upper - quantized_lower)
        lower_bound = next_float(sum_floats((quantized_lower, -half_step)))
        upper_bound = prev_float(sum_floats((quantized_upper, half_step)))
        # Note: Using half_step allows uniform probability for boundary values of encoded range.
        # Note: Using next_float and prev_float prevent decoding a boundary value of encoded range to a decoded value outside the valid decoded range.
        return ((lower_bound, upper_bound),)

    def decode(self, encoded: EncodingType, /) -> int:
        assert len(encoded) == 1
        assert isinstance(encoded[0], (float, int))
        assert self.bounds[0][0] <= encoded[0] <= self.bounds[0][1]  # Invalid encoded value.
        decoded = round_nearest(encoded[0], self.quantum)
        assert decoded == round(decoded)
        decoded = int(decoded)
        assert self.lower <= decoded <= self.upper, decoded  # Invalid decoded value.
        return decoded

    def encode(self, decoded: int) -> EncodingType:
        assert isinstance(decoded, int)
        assert self.lower <= decoded <= self.upper, decoded  # Invalid decoded value.
        assert decoded == round_nearest(decoded, self.quantum)  # Invalid encoded value.
        encoded = (float(decoded),)
        assert self.bounds[0][0] <= encoded[0] <= self.bounds[0][1]  # Invalid encoded value.
        assert decoded == self.decode(encoded)
        return encoded


class GridVar(BaseVar):
    """Grid sampler."""

    def __init__(self, values: list[Any]):
        """Sample a grid uniformly.

        `values` are expected to be ordered. They are not sorted by this class.
        """
        # Motivational reference: https://docs.ray.io/en/latest/tune/api_docs/search_space.html#grid-search-api
        assert values
        # Note: values must not be explicitly sorted here in order to support pre-ordered strings, e.g. ["good", "better", "best"]
        self.values = values
        assert len(self.values) == len(set(self.values))
        self.randint_var = RandintVar(0, len(values) - 1)

    @cached_property
    def bounds(self) -> BoundsType:
        return self.randint_var.bounds

    def decode(self, encoded: EncodingType, /) -> Any:
        decoded_index = self.randint_var.decode(encoded)
        decoded = self.values[decoded_index]
        return decoded

    def encode(self, decoded: Any) -> EncodingType:
        assert decoded in self.values
        decoded_index = self.values.index(decoded)
        encoded = self.randint_var.encode(decoded_index)
        assert decoded == self.decode(encoded)
        return encoded

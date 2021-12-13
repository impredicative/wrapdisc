"""Wrapped optimization objective callable."""

import itertools
from functools import _CacheInfo as CacheInfo
from functools import cache, cached_property
from typing import Any, Callable, Sequence

from wrapdisc.var import BaseVar, BoundsType, EncodingType


class Vars:
    """Solution decoder for multiple variables."""

    def __init__(self, variables: Sequence[BaseVar]):
        """Return a solution decoder for multiple variables."""
        assert all(isinstance(var, BaseVar) for var in variables)
        self._variables = variables
        variable_lengths = [len(v) for v in self._variables]
        self._variables_slices = {
            var: slice(tot_len - cur_len, tot_len) for var, cur_len, tot_len in zip(self._variables, variable_lengths, itertools.accumulate(variable_lengths))
        }
        self.decoded_len = len(self._variables)
        self.encoded_len = sum(variable_lengths)

    def decode(self, encoded: EncodingType) -> tuple:
        """Return the decoded solution from its encoded solution.

        Note that multiple encoded solutions can correspond to the same decoded solution, but a decoded solution corresponds to a single encoded solution.
        """
        assert len(encoded) == self.encoded_len
        decoded = tuple(var.decode(encoded[var_slice]) for var, var_slice in self._variables_slices.items())
        assert len(decoded) == self.decoded_len
        return decoded

    def encode(self, decoded: Sequence) -> EncodingType:
        """Return the encoded solution from its decoded solution.

        Note that multiple encoded solutions can correspond to the same decoded solution, but a decoded solution corresponds to a single encoded solution.
        """
        assert len(decoded) == self.decoded_len
        encoded = tuple(itertools.chain(*(var.encode(decoded_var) for var, decoded_var in zip(self._variables, decoded))))
        assert len(encoded) == self.encoded_len
        assert tuple(decoded) == self.decode(encoded)
        return encoded

    @cached_property
    def bounds(self) -> BoundsType:
        """Return the encoded bounds to provide to an optimizer such as `scipy.optimize`."""
        return tuple(itertools.chain(*(v.bounds for v in self._variables)))


class Objective:
    """Wrapped optimization objective callable."""

    def __init__(self, func: Callable[[tuple], float], variables: Sequence[BaseVar]):
        """Return the wrapped optimization objective callable.

        An unbounded in-memory cache is used over the given input function. This is essential for preventing redundant calls to the input function.

        Note that the wrapped objective function is unsuitable for production use. For production use, the given input function is to be called directly.
        """
        self.func = cache(func)
        self.vars = Vars(variables)

    def __getstate__(self) -> dict[str, Any]:
        """Return the state to be pickled."""
        state = self.__dict__.copy()
        state["func"] = self.func.__wrapped__  # Note: self.func is a cache that cannot be pickled.
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the state that was pickled."""
        self.__dict__.update(state)  # pragma: no cover
        self.func = cache(self.func)  # pragma: no cover

    def decode(self, encoded: EncodingType) -> tuple:
        """Return the decoded solution from its encoded solution.

        Note that multiple encoded solutions can correspond to the same decoded solution, but a decoded solution corresponds to a single encoded solution.
        """
        return self.vars.decode(encoded)

    def encode(self, decoded: Sequence) -> EncodingType:
        """Return the encoded solution from its decoded solution.

        Note that multiple encoded solutions can correspond to the same decoded solution, but a decoded solution corresponds to a single encoded solution.
        """
        return self.vars.encode(decoded)

    def __call__(self, encoded: EncodingType, *args: Any) -> float:
        """Return the result from calling the objective function.

        This method makes the instance the transformed optimization objective.

        :param encoded: This is the encoded solution which first gets decoded. The original objective function is then called with the decoded solution.
        :param args: Additional positional parameters, if any, that are given to the objective function.
        """
        decoded = self.decode(encoded)
        return self.func(decoded, *args)

    @property
    def bounds(self) -> BoundsType:
        """Return the encoded bounds to provide to an optimizer such as `scipy.optimize`."""
        return self.vars.bounds

    @property
    def cache_info(self) -> CacheInfo:
        """Return info about the cache over the input function.

        Note that if multiple worker processes are used, the cache is separate in each process.
        """
        return self.func.cache_info()

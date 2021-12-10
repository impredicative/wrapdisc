# wrapdisc
**wrapdisc** is a Python 3.10 package to wrap a discrete optimization objective such that it can be optimized by a continuous optimizer such as [`scipy.optimize`](https://docs.scipy.org/doc/scipy/reference/optimize.html).
It maps the discrete variables into a continuous space, and uses an in-memory cache over the discrete space.
Both discrete and continuous variables are supported, and are motivated by [Ray Tune's search spaces](https://docs.ray.io/en/latest/tune/key-concepts.html#search-spaces).

[![cicd badge](https://github.com/impredicative/wrapdisc/workflows/cicd/badge.svg?branch=master)](https://github.com/impredicative/wrapdisc/actions?query=workflow%3Acicd+branch%3Amaster)

## Limitations
The current implementation has these limitations:
* The wrapped objective function cannot be pickled, and so multiple workers cannot be used for optimization.
* An unbounded in-memory cache is used over the original objective function, imposing a memory requirement.

## Links
| Caption   | Link                                               |
|-----------|----------------------------------------------------|
| Repo      | https://github.com/impredicative/wrapdisc/         |
| Changelog | https://github.com/impredicative/wrapdisc/releases |
| Package   | https://pypi.org/project/wrapdisc/                 |

## Installation
Python ≥3.10 is required. To install, run:

    pip install wrapdisc

## Variables
The following classes of variables are available:

| Space      | Usage                              | Description                                                   | Examples                                                 |
|------------|------------------------------------|---------------------------------------------------------------|----------------------------------------------------------|
| Discrete   | _**ChoiceVar**(items)_             | Unordered categorical                                         | • fn(["USA", "Panama", "Cayman"])                        |
| Discrete   | _**GridVar**(values)_              | Ordinal (ordered categorical)                                 | • fn([2, 4, 8, 16])<br/>• fn(["good", "better", "best"]) |
| Discrete   | _**RandintVar**(lower, upper)_     | Integer from `lower` to `upper`, both inclusive               | • fn(0, 6)<br/>• fn(3, 9)<br/>• fn(-10, 10)              |
| Discrete   | _**QrandintVar**(lower, upper, q)_ | Quantized integer from `lower` to `upper` in multiples of `q` | • fn(0, 12, 3)<br/>• fn(1, 10, 2)<br/>• fn(-10, 10, 4)   |
| Continuous | _**UniformVar**(lower, upper)_     | Float from `lower` to `upper`                                 | • fn(0.0, 5.11)<br/>• fn(0.2, 4.6)<br/>• fn(-10.0, 10.0) |
| Continuous | _**QuniformVar**(lower, upper, q)_ | Quantized float from `lower` to `upper` in multiples of `q`   | • fn(0.0, 5.1, 0.3)<br/>• fn(-5.1, -0.2, 0.3)            |

## Usage
Example:
```python
import operator
from typing import Any

import scipy.optimize

from wrapdisc import Objective
from wrapdisc.var import ChoiceVar, GridVar, QrandintVar, QuniformVar, RandintVar, UniformVar

def your_mixed_optimization_objective(x: tuple, *args: Any) -> float:
    return float(sum(x_i if isinstance(x_i, (int, float)) else len(str(x_i)) for x_i in (*x, *args)))

wrapped_objective = Objective(
            your_mixed_optimization_objective,
            [
                ChoiceVar(["foobar", "baz"]),
                ChoiceVar([operator.index, abs, operator.invert]),
                GridVar([0.01, 0.1, 1, 10, 100]),
                GridVar(["disagreed", "neutral", "agreed"]),
                RandintVar(-8, 10),
                QrandintVar(1, 10, 2),
                UniformVar(1.2, 3.4),
                QuniformVar(-11.1, 9.99, 0.22),
            ],
        )

optional_fixed_args = ("arg1", 2, 3.0)
result = scipy.optimize.differential_evolution(wrapped_objective, wrapped_objective.bounds, args=optional_fixed_args, seed=0)
cache_usage = wrapped_objective.cache_info
encoded_solution = result.x
decoded_solution = wrapped_objective[encoded_solution]
assert result.fun == wrapped_objective(encoded_solution, *optional_fixed_args)
assert result.fun == your_mixed_optimization_objective(decoded_solution, *optional_fixed_args)
```

Output:
```python
>>> wrapped_objective.bounds
((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (-0.49999999999999994, 4.499999999999999), (-0.49999999999999994, 2.4999999999999996), (-8.499999999999998, 10.499999999999998), (1.0000000000000002, 10.999999999999998), (1.2, 3.4), (-11.109999999999998, 10.009999999999998))
>>> result
     fun: 25.210000000000004
     jac: array([0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 1.00000009,
       0.        ])
 message: 'Optimization terminated successfully.'
    nfev: 6789
     nit: 40
 success: True
       x: array([  0.29493233,   0.88254257,   0.12721268,   0.48978776,
         0.39078759,  -0.04540115,   1.87464003,  -8.02943494,
         1.02999311,   1.2       , -10.98560722])

>>> decoded_solution
('baz', <built-in function abs>, 0.01, 'agreed', -8, 2, 1.2, -11.0)
>>> your_mixed_optimization_objective(decoded_solution, *optional_fixed_args)
25.210000000000004

>>> cache_usage
CacheInfo(hits=169, misses=6620, maxsize=None, currsize=6620)
```

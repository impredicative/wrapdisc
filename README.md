# wrapdisc
**wrapdisc** is a Python 3.10 package to wrap a discrete optimization objective such that it can be optimized by a continuous optimizer such as [`scipy.optimize`](https://docs.scipy.org/doc/scipy/reference/optimize.html).
It maps the discrete variables into a continuous space, and uses an in-memory cache over the discrete space.
Both discrete and continuous [variables](#variables) are supported, and are motivated by [Ray Tune's search spaces](https://docs.ray.io/en/latest/tune/key-concepts.html#search-spaces).

[![cicd badge](https://github.com/impredicative/wrapdisc/workflows/cicd/badge.svg?branch=master)](https://github.com/impredicative/wrapdisc/actions?query=workflow%3Acicd+branch%3Amaster)

## Limitations
The current implementation has these limitations:
* Additional fixed parameters needed by the objective function are not supported.
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

| Type       | Class           | Description                                                   | Example                                    |
|------------|-----------------|---------------------------------------------------------------|--------------------------------------------|
| Discrete   | **ChoiceVar**   | Unordered categorical                                         | ["USA", "Panama", "Cayman"]                |
| Discrete   | **GridVar**     | Ordinal (ordered categorical)                                 | [2, 4, 8, 16], ["good", "better", "best"]  |
| Discrete   | **RandintVar**  | Integer from `lower` to `upper`, both inclusive               | `fn(3, 6)` -> [3, 4, 5, 6]                 |
| Discrete   | **QrandintVar** | Quantized integer from `lower` to `upper` in multiples of `q` | `fn(2, 12, 3)` -> [3, 6, 9, 12]            |
| Continuous | **UniformVar**  | Float from `lower` to `upper`                                 | `fn(0.2, 5.11)` -> [4.131, 1.52, 0.61319]  |
| Continuous | **QuniformVar** | Quantized float from `lower` to `upper` in multiples of `q`   | `fn(0.2, 5.1, 0.3)` -> [[5.1, 0.8999, 3.9] |

## Usage
Example:
```python
import operator

import scipy.optimize

from wrapdisc import Objective
from wrapdisc.var import ChoiceVar, GridVar, QrandintVar, QuniformVar, RandintVar, UniformVar

def your_mixed_optimization_objective(x: tuple) -> float:
    return float(sum(x_i if isinstance(x_i, (int, float)) else len(str(x_i)) for x_i in x))

wrapped_objective = Objective(
            your_mixed_optimization_objective,
            [
                ChoiceVar(["foobar", "baz"]),
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

result = scipy.optimize.differential_evolution(wrapped_objective, wrapped_objective.bounds, seed=0)
encoded_solution = result.x
decoded_solution = wrapped_objective[encoded_solution]
```

Output:
```python
>>> wrapped_objective.bounds
((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (-0.49999999999999994, 4.499999999999999), (-8.499999999999998, 10.499999999999998), (1.0000000000000002, 10.999999999999998), (1.2, 3.4), (-11.109999999999998, 10.009999999999998), (4.500000000000001, 81.69999999999999))

>>> result
     fun: 15.810000000000002
     jac: array([0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 1.00000009, 0.        ,
       0.        ])
 message: 'Optimization terminated successfully.'
    nfev: 9264
     nit: 55
 success: True
       x: array([  0.32091421,   0.64403109,   0.92827817,   0.18718745,
         0.76108352,   0.32380381,  -7.60064697,   2.12231176,
         1.2       , -11.03958486,   4.69462919])

>>> decoded_solution
("baz", operator.add, "x", 0.01, -8, 2, 1.2, -11.0, 4.6000000000000005)

>>> wrapped_objective.cache_info
CacheInfo(hits=135, misses=9129, maxsize=None, currsize=9129)
```

"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from typing import Callable, Iterable

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    "$f(x, y) = x * y$"
    return x * y


def id(x: float) -> float:
    "$f(x) = x$"
    return x


def add(x: float, y: float) -> float:
    "$f(x, y) = x + y$"
    return x + y


def neg(x: float) -> float:
    "$f(x) = -x$"
    return x * -1


def lt(x: float, y: float) -> float:
    "$f(x) =$ 1.0 if x is less than y else 0.0"
    if x < y:
        return 1.0
    else:
        return 0.0


def eq(x: float, y: float) -> float:
    "$f(x) =$ 1.0 if x is equal to y else 0.0"
    if x == y:
        return 1.0
    else:
        return 0.0


def max(x: float, y: float) -> float:
    "$f(x) =$ x if x is greater than y else y"
    if x > y:
        return x
    else:
        return y


def is_close(x: float, y: float) -> float:
    "$f(x) = |x - y| < 1e-2$"
    if abs(x - y) < 0.01:
        return True
    else:
        return False


def sigmoid(x: float) -> float:
    """
        $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$

        (See https://en.wikipedia.org/wiki/Sigmoid_function )

        Calculate as
    s
        $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$

        for stability.
    """

    if x < 0:
        result = (math.e**x) / (1.0 + math.e**x)
    else:
        result = 1.0 / (1.0 + math.e ** (x * -1))
    return float(result)


def relu(x: float) -> float:
    """
    $f(x) =$ x if x is greater than 0, else 0

    (See https://en.wikipedia.org/wiki/Rectifier_(neural_networks) .)
    """
    if x > 0:
        return x
    else:
        return 0.0


EPS = 1e-6


def log(x: float) -> float:
    "$f(x) = log(x)$"
    return math.log(x + EPS)


def exp(x: float) -> float:
    "$f(x) = e^{x}$"
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    r"If $f = log$ as above, compute $d \times f'(x)$"
    """
    if x == 0.0:
        return d
    """
    f_prime = 1 / x
    return f_prime * d


def inv(x: float) -> float:
    "$f(x) = 1/x$"
    return 1 / x


def inv_back(x: float, d: float) -> float:
    r"If $f(x) = 1/x$ compute $d \times f'(x)$"
    f_prime = -1 / (x**2)
    return f_prime * d


def relu_back(x: float, d: float) -> float:
    r"If $f = relu$ compute $d \times f'(x)$"
    if x < 0:
        f_prime = 0
    else:
        f_prime = 1
    return f_prime * d


# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """
    Higher-order map.

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: Function from one value to one value.

    Returns:
        A function that takes a list, applies `fn` to each element, and returns a
         new list
    """

    def apply_func(my_list: Iterable[float]) -> Iterable[float]:
        new_list = []
        for x in my_list:
            new_list.append(fn(x))
        return new_list

    return apply_func


def negList(ls: Iterable[float]) -> Iterable[float]:
    "Use `map` and `neg` to negate each element in `ls`"
    return map(neg)(ls)


def zipWith(
    fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """
    Higher-order zipwith (or map2).

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: combine two values

    Returns:
        Function that takes two equally sized lists `ls1` and `ls2`, produce a new list by
         applying fn(x, y) on each pair of elements.

    """

    def apply_func(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        new_list = []
        ls1 = list(ls1)
        ls2 = list(ls2)
        assert len(ls1) == len(ls2)
        for i in range(len(ls1)):
            new_list.append(fn(ls1[i], ls2[i]))
        return new_list

    return apply_func


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    "Add the elements of `ls1` and `ls2` using `zipWith` and `add`"
    return zipWith(add)(ls1, ls2)


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""
    Higher-order reduce.

    Args:
        fn: combine two values
        start: start value $x_0$

    Returns:
        Function that takes a list `ls` of elements
         $x_1 \ldots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2,
         fn(x_1, x_0)))`
    """
    """
    def do_reduce(ls: Iterable[float]) -> float:
        if len(ls) == 1:
            return fn(ls[0], start)
        else:
            return fn(ls[-1], do_reduce(ls[:-1]))
    """

    def do_reduce(ls: Iterable[float]) -> float:
        result = start
        for element in ls:
            result = fn(element, result)
        return result

    return do_reduce


def sum(ls: Iterable[float]) -> float:
    "Sum up a list using `reduce` and `add`."
    return reduce(add, 0)(ls)


def prod(ls: Iterable[float]) -> float:
    "Product of a list using `reduce` and `mul`."
    return reduce(mul, 1)(ls)

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    """
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    i = arg
    forward_vals = list(vals[:i])
    forward_vals.extend([vals[i]+epsilon/2])
    forward_vals.extend(list(vals[i+1:]))

    backward_vals == list(vals[:i])
    backward_vals.extend([vals[i]-epsilon/2])
    backward_vals.extend(list(vals[i+1:]))

    return(f(foward_vals) - f(backward_vals))
    


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    visited = set()
    sorted_list = []
    
    def do_sort(var):
        if (var.unique_id in visited) or var.is_constant():
            return
        if not(var.is_leaf()):
            for parent in var.parents:
                do_sort(parent)
        sorted_list.insert(0, var)
        visited.add(var.unique_id)
        
    do_sort(variable)
    
    return sorted_list


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    sorted_vars = topological_sort(variable)
    der_dict = {var.unique_id : 0 for var in sorted_vars}
    der_dict[variable.unique_id] = deriv
    for var in sorted_vars:
        d_out = der_dict[var.unique_id]
        if var.is_leaf():
            var.accumulate_derivative(d_out)
            if var.unique_id % 100 == 0:
                """
                print('-'*20)
                print('leaf!')
                print('uid:', var.unique_id)
                print('derivative:', var.derivative)
                print('d_out:', d_out)
                print('history:', var.history)
                """
        else:
            back = var.chain_rule(d_out)
            if var.unique_id % 100 == 0:
                """
                print('='*20)
                print('non-leaf!')
                print('var:', var)
                print('var uid:', var.unique_id)
                print('history:', var.history)
                print('d_out:', d_out)
                """
            for back_var, back_d in back:
                der_dict[back_var.unique_id] += back_d
                if var.unique_id % 100 == 0:
                    """
                    print('-'*10)
                    print('back_uid:', back_var.unique_id)
                    print('back_var:', back_var)
                    print('back_hist:', back_var.history)
                    print('back_d:', back_d)
                    print('derdict:', der_dict[back_var.unique_id])
                    """
@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values

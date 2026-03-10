"""Free-standing compute functions exposed to Python.

These use runtime type dispatch via class name to convert PythonObject
back to typed arrays and call the appropriate kernel.
"""

from std.python import PythonObject, Python
from std.python.bindings import PythonModuleBuilder
from std.python import PythonObject, ConvertibleToPython, ConvertibleFromPython
from marrow.arrays import Array
from marrow.kernels.aggregate import  sum_, product, min_, max_, any_, all_
from marrow.kernels.arithmetic import add, sub, mul, div
from marrow.kernels.filter import filter, drop_nulls


fn pyfunction[
    A0: ConvertibleFromPython,
    R: ConvertibleToPython,
    //,
    func: fn (A0) raises -> R,
]() -> fn (PythonObject) raises -> PythonObject:
    """Wrap a one-arg method returning ConvertibleToPython."""

    fn wrapper(arg0: PythonObject) raises -> PythonObject:
        return func(A0(py=arg0)).to_python_object()

    return wrapper

fn pyfunction[
    A0: ConvertibleFromPython,
    A1: ConvertibleFromPython,
    R: ConvertibleToPython,
    //,
    func: fn (A0, A1) raises -> R,
]() -> fn (PythonObject, PythonObject) raises -> PythonObject:
    """Wrap a two-arg method returning ConvertibleToPython."""

    fn wrapper(arg0: PythonObject, arg1: PythonObject) raises -> PythonObject:
        return func(A0(py=arg0), A1(py=arg1)).to_python_object()

    return wrapper



def add_to_module(mut mb: PythonModuleBuilder) raises -> None:
    mb.def_function[pyfunction[add]()]("add", docstring="Add all valid elements.")
    mb.def_function[pyfunction[sum_]()]("sum_", docstring="Sum all valid elements.")
    mb.def_function[pyfunction[product]()]("product", docstring="Product of all valid elements.")
    mb.def_function[pyfunction[min_]()]("min_", docstring="Minimum of all valid elements.")
    mb.def_function[pyfunction[max_]()]("max_", docstring="Maximum of all valid elements.")
    mb.def_function[pyfunction[any_]()]("any_", docstring="True if any valid element is true (bool arrays only).")
    mb.def_function[pyfunction[all_]()]("all_", docstring="True if all valid elements are true (bool arrays only).")



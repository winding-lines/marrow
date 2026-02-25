"""Aggregate (reduction) kernels."""

from marrow.arrays import PrimitiveArray, Array
from marrow.dtypes import DataType, all_numeric_dtypes, materialize
from .kernels import reduce


fn _accumulate[T: DType](acc: Scalar[T], val: Scalar[T]) -> Scalar[T]:
    return acc + val


fn sum[T: DataType](array: PrimitiveArray[T]) -> Scalar[T.native]:
    """Sum all valid (non-null) elements in the array.

    Args:
        array: The input array.

    Returns:
        The sum of all valid elements. Returns 0 if empty or all null.
    """
    return reduce[T, T, _accumulate[T.native]](array, Scalar[T.native](0))


fn sum(array: Array) raises -> Scalar[DType.float64]:
    """Runtime-typed sum: dispatches to the correct typed sum.

    Returns the result as float64 to accommodate any numeric type.

    Args:
        array: The input array (runtime-typed).

    Returns:
        The sum as a float64 scalar.
    """

    @parameter
    for dtype in all_numeric_dtypes:
        if array.dtype == materialize[dtype]():
            var result = sum[dtype](array.as_primitive[dtype]())
            return result.cast[DType.float64]()

    raise Error("sum: unsupported dtype {}".format(array.dtype))

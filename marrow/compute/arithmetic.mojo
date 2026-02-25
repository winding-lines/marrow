"""Scalar (element-wise) arithmetic kernels."""

from marrow.arrays import PrimitiveArray, Array
from marrow.dtypes import DataType, all_numeric_dtypes, materialize
from .kernels import binary


fn _add[T: DType](a: Scalar[T], b: Scalar[T]) -> Scalar[T]:
    return a + b


fn add[T: DataType](
    left: PrimitiveArray[T], right: PrimitiveArray[T]
) raises -> PrimitiveArray[T]:
    """Element-wise addition of two primitive arrays of the same type.

    Args:
        left: Left operand array.
        right: Right operand array.

    Returns:
        A new PrimitiveArray where result[i] = left[i] + right[i].
        Null if either input is null at that position.
    """
    return binary[T, T, T, _add[T.native]](left, right)


fn add(left: Array, right: Array) raises -> Array:
    """Runtime-typed add: dispatches to the correct typed add.

    Args:
        left: Left operand (runtime-typed Array).
        right: Right operand (runtime-typed Array).

    Returns:
        A new Array with the element-wise sum.
    """
    if left.dtype != right.dtype:
        raise Error(
            "add: dtype mismatch: {} vs {}".format(left.dtype, right.dtype)
        )

    @parameter
    for dtype in all_numeric_dtypes:
        if left.dtype == materialize[dtype]():
            return Array(
                add[dtype](
                    left.as_primitive[dtype](),
                    right.as_primitive[dtype](),
                )
            )

    raise Error("add: unsupported dtype {}".format(left.dtype))

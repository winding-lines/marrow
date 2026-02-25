"""Core arity helpers for compute kernels.

These generic loop functions handle buffer allocation, null propagation,
and iteration. Specific kernels (add, sum, etc.) are thin wrappers.
"""

from marrow.arrays import PrimitiveArray
from marrow.dtypes import DataType


fn unary[
    InT: DataType,
    OutT: DataType,
    func: fn(Scalar[InT.native]) -> Scalar[OutT.native],
](array: PrimitiveArray[InT]) -> PrimitiveArray[OutT]:
    """Apply a scalar function element-wise to produce a new array.

    Null propagation: if input element is null, output element is null.

    Parameters:
        InT: Input array DataType.
        OutT: Output array DataType.
        func: The element-wise transformation function.

    Args:
        array: The input array.

    Returns:
        A new PrimitiveArray with the function applied to each valid element.
    """
    var length = len(array)
    var result = PrimitiveArray[OutT](length)
    for i in range(length):
        if array.is_valid(i):
            result.unsafe_set(i, func(array.unsafe_get(i)))
    result.length = length
    return result^


fn binary[
    LT: DataType,
    RT: DataType,
    OutT: DataType,
    func: fn(Scalar[LT.native], Scalar[RT.native]) -> Scalar[OutT.native],
](left: PrimitiveArray[LT], right: PrimitiveArray[RT]) raises -> PrimitiveArray[
    OutT
]:
    """Apply a binary scalar function element-wise to two arrays.

    Both arrays must have the same length. Output has the same length.
    Null propagation: output is null if EITHER input is null.

    Parameters:
        LT: Left input DataType.
        RT: Right input DataType.
        OutT: Output DataType.
        func: The element-wise binary function.

    Args:
        left: The left input array.
        right: The right input array.

    Returns:
        A new PrimitiveArray with the function applied element-wise.
    """
    if len(left) != len(right):
        raise Error(
            "binary kernel: arrays must have the same length, got {} and {}"
            .format(len(left), len(right))
        )
    var length = len(left)
    var result = PrimitiveArray[OutT](length)
    for i in range(length):
        if left.is_valid(i) and right.is_valid(i):
            result.unsafe_set(i, func(left.unsafe_get(i), right.unsafe_get(i)))
    result.length = length
    return result^


fn reduce[
    T: DataType,
    AccT: DataType,
    func: fn(Scalar[AccT.native], Scalar[T.native]) -> Scalar[AccT.native],
](array: PrimitiveArray[T], initial: Scalar[AccT.native]) -> Scalar[
    AccT.native
]:
    """Reduce an array to a scalar value, skipping nulls.

    Parameters:
        T: Input array DataType.
        AccT: Accumulator DataType (may differ from input for widening).
        func: The accumulator function (acc, value) -> acc.

    Args:
        array: The input array.
        initial: The initial accumulator value (e.g. 0 for sum).

    Returns:
        The accumulated scalar result.
    """
    var acc = initial
    for i in range(len(array)):
        if array.is_valid(i):
            acc = func(acc, array.unsafe_get(i))
    return acc

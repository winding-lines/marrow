"""Element-wise arithmetic kernels — CPU SIMD and GPU specializations.

Each public function dispatches based on the optional `ctx` argument:
  - CPU (default): operates on PrimitiveArray[T] using SIMD vectorization.
  - GPU (ctx provided): operates on device-resident PrimitiveArray[T] via a GPU kernel.

The SIMD helper functions (fn[W: Int](SIMD[T, W], ...) -> SIMD[T, W]) are shared
between CPU and GPU paths. Since Scalar[T] = SIMD[T, 1], the GPU kernel calls
each helper with W=1 per thread.
"""

import math
from gpu.host import DeviceContext

from marrow.arrays import PrimitiveArray, Array

from marrow.dtypes import DataType
from . import binary_simd, binary_gpu, unary_simd, binary_array_dispatch


# ---------------------------------------------------------------------------
# SIMD helpers — shared by CPU and GPU paths
# ---------------------------------------------------------------------------

# Binary: fn[W: Int](SIMD[T, W], SIMD[T, W]) -> SIMD[T, W]


fn _add[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
    return a + b


fn _sub[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
    return a - b


fn _mul[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
    return a * b


fn _div[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
    return a / b


fn _floordiv[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
    return a // b


fn _mod[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
    return a % b


fn _min[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
    # TODO(kszucs): consider return (a < b).select(a, b)
    return math.min(a, b)


fn _max[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
    # TODO(kszucs): consider return (a > b).select(a, b)
    return math.max(a, b)


# Unary: fn[W: Int](SIMD[T, W]) -> SIMD[T, W]


fn _neg[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
    return -a


fn _abs[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
    return abs(a)


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------


fn add[
    T: DataType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: Optional[DeviceContext] = None,
) raises -> PrimitiveArray[T]:
    """Element-wise addition of two primitive arrays.

    Args:
        left: Left operand array.
        right: Right operand array.
        ctx: GPU device context. If provided, runs on GPU; otherwise uses CPU SIMD.

    Returns:
        A new PrimitiveArray where result[i] = left[i] + right[i].
        Null if either input is null at that position.
    """
    if ctx:
        return binary_gpu[T, _add[T.native], "add"](left, right, ctx.value())
    else:
        return binary_simd[T, _add[T.native], "add"](left, right)


fn add(
    left: Array, right: Array
) raises -> Array:
    """Runtime-typed add."""
    return binary_array_dispatch["add", add](left, right)


# ---------------------------------------------------------------------------
# sub
# ---------------------------------------------------------------------------


fn sub[
    T: DataType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: Optional[DeviceContext] = None,
) raises -> PrimitiveArray[T]:
    """Element-wise subtraction of two primitive arrays.

    Args:
        left: Left operand array.
        right: Right operand array.
        ctx: GPU device context. If provided, runs on GPU; otherwise uses CPU SIMD.

    Returns:
        A new PrimitiveArray where result[i] = left[i] - right[i].
        Null if either input is null at that position.
    """
    if ctx:
        return binary_gpu[T, _sub[T.native], "sub"](left, right, ctx.value())
    else:
        return binary_simd[T, _sub[T.native], "sub"](left, right)


fn sub(
    left: Array, right: Array
) raises -> Array:
    """Runtime-typed sub."""
    return binary_array_dispatch["sub", sub](left, right)


# ---------------------------------------------------------------------------
# mul
# ---------------------------------------------------------------------------


fn mul[
    T: DataType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: Optional[DeviceContext] = None,
) raises -> PrimitiveArray[T]:
    """Element-wise multiplication of two primitive arrays.

    Args:
        left: Left operand array.
        right: Right operand array.
        ctx: GPU device context. If provided, runs on GPU; otherwise uses CPU SIMD.

    Returns:
        A new PrimitiveArray where result[i] = left[i] * right[i].
        Null if either input is null at that position.
    """
    if ctx:
        return binary_gpu[T, _mul[T.native], "mul"](left, right, ctx.value())
    else:
        return binary_simd[T, _mul[T.native], "mul"](left, right)


fn mul(
    left: Array, right: Array
) raises -> Array:
    """Runtime-typed mul."""
    return binary_array_dispatch["mul", mul](left, right)


# ---------------------------------------------------------------------------
# div
# ---------------------------------------------------------------------------


fn div[
    T: DataType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: Optional[DeviceContext] = None,
) raises -> PrimitiveArray[T]:
    """Element-wise true division of two primitive arrays.

    Args:
        left: Left operand array.
        right: Right operand array.
        ctx: GPU device context. If provided, runs on GPU; otherwise uses CPU SIMD.

    Returns:
        A new PrimitiveArray where result[i] = left[i] / right[i].
        Null if either input is null at that position.
    """
    if ctx:
        return binary_gpu[T, _div[T.native], "div"](left, right, ctx.value())
    else:
        return binary_simd[T, _div[T.native], "div"](left, right)


fn div(
    left: Array, right: Array
) raises -> Array:
    """Runtime-typed div."""
    return binary_array_dispatch["div", div](left, right)


# ---------------------------------------------------------------------------
# floordiv
# ---------------------------------------------------------------------------


fn floordiv[
    T: DataType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: Optional[DeviceContext] = None,
) raises -> PrimitiveArray[T]:
    """Element-wise floor division of two primitive arrays.

    Args:
        left: Left operand array.
        right: Right operand array.
        ctx: GPU device context. If provided, runs on GPU; otherwise uses CPU SIMD.

    Returns:
        A new PrimitiveArray where result[i] = left[i] // right[i].
        Null if either input is null at that position.
    """
    if ctx:
        return binary_gpu[T, _floordiv[T.native], "floordiv"](left, right, ctx.value())
    else:
        return binary_simd[T, _floordiv[T.native], "floordiv"](left, right)


fn floordiv(
    left: Array, right: Array
) raises -> Array:
    """Runtime-typed floordiv."""
    return binary_array_dispatch["floordiv", floordiv](left, right)


# ---------------------------------------------------------------------------
# mod
# ---------------------------------------------------------------------------


fn mod[
    T: DataType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: Optional[DeviceContext] = None,
) raises -> PrimitiveArray[T]:
    """Element-wise modulo of two primitive arrays.

    Args:
        left: Left operand array.
        right: Right operand array.
        ctx: GPU device context. If provided, runs on GPU; otherwise uses CPU SIMD.

    Returns:
        A new PrimitiveArray where result[i] = left[i] % right[i].
        Null if either input is null at that position.
    """
    if ctx:
        return binary_gpu[T, _mod[T.native], "mod"](left, right, ctx.value())
    else:
        return binary_simd[T, _mod[T.native], "mod"](left, right)


fn mod(
    left: Array, right: Array
) raises -> Array:
    """Runtime-typed mod."""
    return binary_array_dispatch["mod", mod](left, right)


# ---------------------------------------------------------------------------
# min_
# ---------------------------------------------------------------------------


fn min_[
    T: DataType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: Optional[DeviceContext] = None,
) raises -> PrimitiveArray[T]:
    """Element-wise minimum of two primitive arrays.

    Args:
        left: Left operand array.
        right: Right operand array.
        ctx: GPU device context. If provided, runs on GPU; otherwise uses CPU SIMD.

    Returns:
        A new PrimitiveArray where result[i] = min(left[i], right[i]).
        Null if either input is null at that position.
    """
    if ctx:
        return binary_gpu[T, _min[T.native], "min_"](left, right, ctx.value())
    else:
        return binary_simd[T, _min[T.native], "min_"](left, right)


fn min_(
    left: Array, right: Array
) raises -> Array:
    """Runtime-typed min_."""
    return binary_array_dispatch["min_", min_](left, right)


# ---------------------------------------------------------------------------
# max_
# ---------------------------------------------------------------------------


fn max_[
    T: DataType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: Optional[DeviceContext] = None,
) raises -> PrimitiveArray[T]:
    """Element-wise maximum of two primitive arrays.

    Args:
        left: Left operand array.
        right: Right operand array.
        ctx: GPU device context. If provided, runs on GPU; otherwise uses CPU SIMD.

    Returns:
        A new PrimitiveArray where result[i] = max(left[i], right[i]).
        Null if either input is null at that position.
    """
    if ctx:
        return binary_gpu[T, _max[T.native], "max_"](left, right, ctx.value())
    else:
        return binary_simd[T, _max[T.native], "max_"](left, right)


fn max_(
    left: Array, right: Array
) raises -> Array:
    """Runtime-typed max_."""
    return binary_array_dispatch["max_", max_](left, right)


# ---------------------------------------------------------------------------
# neg
# ---------------------------------------------------------------------------


fn neg[T: DataType](array: PrimitiveArray[T]) -> PrimitiveArray[T]:
    """Element-wise negation.

    Args:
        array: Input array.

    Returns:
        A new PrimitiveArray where result[i] = -array[i].
        Null if the input is null at that position.
    """
    return unary_simd[T, _neg[T.native]](array)


# ---------------------------------------------------------------------------
# abs_
# ---------------------------------------------------------------------------


fn abs_[T: DataType](array: PrimitiveArray[T]) -> PrimitiveArray[T]:
    """Element-wise absolute value.

    Args:
        array: Input array.

    Returns:
        A new PrimitiveArray where result[i] = |array[i]|.
        Null if the input is null at that position.
    """
    return unary_simd[T, _abs[T.native]](array)

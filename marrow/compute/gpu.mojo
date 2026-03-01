"""GPU-accelerated compute kernels.

This module requires GPU compilation tools (Xcode with Metal on macOS,
CUDA toolkit on Linux).  It is NOT imported by default from
`marrow.compute` — import explicitly when GPU acceleration is desired:

    from marrow.compute.gpu import add
"""

import math
from sys import size_of

from gpu import global_idx
from gpu.host import DeviceBuffer, DeviceContext

from marrow.arrays import PrimitiveArray, Array
from marrow.buffers import Buffer, BitmapBuilder
from marrow.dtypes import DataType, all_numeric_dtypes, materialize
from .arithmetic import _add, _add_no_nulls
from .kernels import binary


# ---------------------------------------------------------------------------
# GPU kernel
# ---------------------------------------------------------------------------


fn _add_gpu_kernel[
    dtype: DType
](
    lhs: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    rhs: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    length: Int,
):
    """GPU kernel for element-wise addition."""
    var tid = global_idx.x
    if tid < UInt(length):
        result[tid] = lhs[tid] + rhs[tid]


fn _add_gpu[
    T: DataType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    length: Int,
    ctx: DeviceContext,
) raises -> PrimitiveArray[T]:
    """GPU-accelerated add, reusing device buffers when already resident."""
    comptime native = T.native
    comptime BLOCK_SIZE = 256

    # Reuse device buffers if already on GPU, otherwise upload
    var lhs_dev: DeviceBuffer[native]
    var rhs_dev: DeviceBuffer[native]
    if left.buffer.has_device():
        lhs_dev = left.buffer.device.value().create_sub_buffer[native](
            0, length
        )
    else:
        lhs_dev = ctx.enqueue_create_buffer[native](length)
        var lhs_ptr = (
            left.buffer.ptr.bitcast[Scalar[native]]()
            + left.offset
            + left.buffer.offset
        )
        ctx.enqueue_copy(lhs_dev, lhs_ptr)

    if right.buffer.has_device():
        rhs_dev = right.buffer.device.value().create_sub_buffer[native](
            0, length
        )
    else:
        rhs_dev = ctx.enqueue_create_buffer[native](length)
        var rhs_ptr = (
            right.buffer.ptr.bitcast[Scalar[native]]()
            + right.offset
            + right.buffer.offset
        )
        ctx.enqueue_copy(rhs_dev, rhs_ptr)

    # Allocate output on device
    var out_dev = ctx.enqueue_create_buffer[native](length)

    # Launch kernel
    var num_blocks = math.ceildiv(length, BLOCK_SIZE)
    comptime kernel = _add_gpu_kernel[native]
    ctx.enqueue_function_experimental[kernel](
        lhs_dev,
        rhs_dev,
        out_dev,
        length,
        grid_dim=num_blocks,
        block_dim=BLOCK_SIZE,
    )

    # Build device-only result (no host copy — call .to_host(ctx) to read)
    var bm = BitmapBuilder.alloc(length)
    bm.unsafe_range_set(0, length, True)
    var device_bytes = length * size_of[native]()
    var buf = Buffer(
        UnsafePointer[UInt8, ImmutExternalOrigin](), device_bytes
    )
    buf.device = out_dev.create_sub_buffer[DType.uint8](0, device_bytes)
    return PrimitiveArray[T](
        length=length, offset=0, bitmap=bm^.freeze(), buffer=buf^
    )


# ---------------------------------------------------------------------------
# Public API — mirrors arithmetic.add but with DeviceContext dispatch
# ---------------------------------------------------------------------------


fn add[
    T: DataType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: Optional[DeviceContext] = None,
) raises -> PrimitiveArray[T]:
    """Element-wise addition with optional GPU acceleration.

    When a DeviceContext is provided, the result stays on device (GPU).
    Call `.to_host(ctx)` on the result to download to CPU memory.
    Without a context, dispatches to the SIMD-optimized CPU path.

    Args:
        left: Left operand array.
        right: Right operand array.
        ctx: Optional GPU device context for acceleration.

    Returns:
        A new PrimitiveArray where result[i] = left[i] + right[i].
        With GPU: result is device-resident; chain ops or call to_host().
        Without GPU: result is host-resident and immediately readable.
    """
    if len(left) != len(right):
        raise Error(
            "add: arrays must have the same length, got {} and {}".format(
                len(left), len(right)
            )
        )

    var no_nulls = left.null_count() == 0 and right.null_count() == 0

    if no_nulls:
        if ctx:
            return _add_gpu[T](left, right, len(left), ctx.value())
        return _add_no_nulls[T](left, right, len(left))

    return binary[T, T, T, _add[T.native]](left, right)


fn add(
    left: Array, right: Array, ctx: Optional[DeviceContext] = None
) raises -> Array:
    """Runtime-typed add with optional GPU acceleration.

    Args:
        left: Left operand (runtime-typed Array).
        right: Right operand (runtime-typed Array).
        ctx: Optional GPU device context for acceleration.

    Returns:
        A new Array with the element-wise sum.
    """
    if left.dtype != right.dtype:
        raise Error(
            "add: dtype mismatch: "
            + String(left.dtype)
            + " vs "
            + String(right.dtype)
        )

    comptime for dtype in all_numeric_dtypes:
        if left.dtype == materialize[dtype]():
            return Array(
                add[dtype](
                    left.as_primitive[dtype](),
                    right.as_primitive[dtype](),
                    ctx,
                )
            )

    raise Error("add: unsupported dtype " + String(left.dtype))

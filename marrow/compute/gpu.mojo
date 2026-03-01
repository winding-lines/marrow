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

from marrow.arrays import PrimitiveArray, FixedSizeListArray, Array
from marrow.buffers import Buffer, BitmapBuilder, MemorySpace
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
    left: PrimitiveArray[T, MemorySpace.DEVICE],
    right: PrimitiveArray[T, MemorySpace.DEVICE],
    length: Int,
    ctx: DeviceContext,
) raises -> PrimitiveArray[T, MemorySpace.DEVICE]:
    """GPU-accelerated add on device-resident arrays."""
    comptime native = T.native
    comptime BLOCK_SIZE = 256

    var lhs_dev = left.buffer.device_buffer().create_sub_buffer[native](
        0, length
    )
    var rhs_dev = right.buffer.device_buffer().create_sub_buffer[native](
        0, length
    )

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

    # Build device-only result
    var bm = BitmapBuilder.alloc(length)
    bm.unsafe_range_set(0, length, True)
    var device_bytes = length * size_of[native]()
    var buf = Buffer[MemorySpace.DEVICE].device_only(
        out_dev.create_sub_buffer[DType.uint8](0, device_bytes), device_bytes
    )
    return PrimitiveArray[T, MemorySpace.DEVICE](
        length=length,
        offset=0,
        bitmap=bm^.freeze().to_device(ctx),
        buffer=buf^,
    )


# ---------------------------------------------------------------------------
# Public API — mirrors arithmetic.add but with DeviceContext dispatch
# ---------------------------------------------------------------------------


fn add[
    T: DataType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
) raises -> PrimitiveArray[T]:
    """Element-wise addition on CPU arrays.

    Args:
        left: Left operand array.
        right: Right operand array.

    Returns:
        A new PrimitiveArray where result[i] = left[i] + right[i].
    """
    if len(left) != len(right):
        raise Error(
            "add: arrays must have the same length, got {} and {}".format(
                len(left), len(right)
            )
        )

    var no_nulls = left.null_count() == 0 and right.null_count() == 0
    if no_nulls:
        return _add_no_nulls[T](left, right, len(left))
    return binary[T, T, T, _add[T.native]](left, right)


fn add[
    T: DataType
](
    left: PrimitiveArray[T, MemorySpace.DEVICE],
    right: PrimitiveArray[T, MemorySpace.DEVICE],
    ctx: DeviceContext,
) raises -> PrimitiveArray[T, MemorySpace.DEVICE]:
    """Element-wise addition on device-resident arrays.

    Args:
        left: Left operand (device-resident).
        right: Right operand (device-resident).
        ctx: GPU device context.

    Returns:
        A new device-resident PrimitiveArray where result[i] = left[i] + right[i].
        Call `.to_host(ctx)` to download to CPU memory.
    """
    if len(left) != len(right):
        raise Error(
            "add: arrays must have the same length, got {} and {}".format(
                len(left), len(right)
            )
        )
    return _add_gpu[T](left, right, len(left), ctx)


fn add(
    left: Array[MemorySpace.CPU], right: Array[MemorySpace.CPU]
) raises -> Array[MemorySpace.CPU]:
    """Runtime-typed add on CPU arrays.

    Args:
        left: Left operand (runtime-typed Array).
        right: Right operand (runtime-typed Array).

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
                    PrimitiveArray[dtype](data=left),
                    PrimitiveArray[dtype](data=right),
                )
            )

    raise Error("add: unsupported dtype " + String(left.dtype))


# ---------------------------------------------------------------------------
# GPU cosine similarity kernel
# ---------------------------------------------------------------------------


fn _cosine_similarity_gpu_kernel[
    dtype: DType
](
    vectors: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    query: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n_vectors: Int,
    dim: Int,
):
    """GPU kernel: each thread computes one vector's cosine similarity."""
    var tid = global_idx.x
    if tid < UInt(n_vectors):
        var offset = Int(tid) * dim
        var dot = Scalar[dtype](0)
        var norm_v = Scalar[dtype](0)
        var norm_q = Scalar[dtype](0)
        for j in range(dim):
            var v = vectors[offset + j]
            var q = query[j]
            dot += v * q
            norm_v += v * v
            norm_q += q * q
        var denom = math.sqrt(norm_v) * math.sqrt(norm_q)
        if denom > 0:
            result[tid] = dot / denom
        else:
            result[tid] = Scalar[dtype](0)


fn _cosine_similarity_gpu[
    T: DataType
](
    vectors: FixedSizeListArray[MemorySpace.DEVICE],
    query: PrimitiveArray[T, MemorySpace.DEVICE],
    n_vectors: Int,
    dim: Int,
    ctx: DeviceContext,
) raises -> PrimitiveArray[T, MemorySpace.DEVICE]:
    """GPU-accelerated batch cosine similarity on device-resident data."""
    comptime native = T.native
    comptime BLOCK_SIZE = 256

    var n_values = n_vectors * dim

    # Data is already on device — get handles directly
    ref child = vectors.values
    var vec_dev = child.buffers[0].device_buffer().create_sub_buffer[native](
        0, n_values
    )
    var query_dev = query.buffer.device_buffer().create_sub_buffer[native](
        0, dim
    )

    # Allocate output on device
    var out_dev = ctx.enqueue_create_buffer[native](n_vectors)

    # Launch kernel
    var num_blocks = math.ceildiv(n_vectors, BLOCK_SIZE)
    comptime kernel = _cosine_similarity_gpu_kernel[native]
    ctx.enqueue_function_experimental[kernel](
        vec_dev,
        query_dev,
        out_dev,
        n_vectors,
        dim,
        grid_dim=num_blocks,
        block_dim=BLOCK_SIZE,
    )

    # Build device-only result
    var bm = BitmapBuilder.alloc(n_vectors)
    bm.unsafe_range_set(0, n_vectors, True)
    var device_bytes = n_vectors * size_of[native]()
    var buf = Buffer[MemorySpace.DEVICE].device_only(
        out_dev.create_sub_buffer[DType.uint8](0, device_bytes), device_bytes
    )
    return PrimitiveArray[T, MemorySpace.DEVICE](
        length=n_vectors,
        offset=0,
        bitmap=bm^.freeze().to_device(ctx),
        buffer=buf^,
    )

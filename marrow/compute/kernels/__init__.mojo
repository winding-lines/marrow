"""Core arity helpers for compute kernels.

These generic loop functions handle buffer allocation, null propagation,
and iteration. Specific kernels (add, sum, etc.) are thin wrappers.

Null propagation model
----------------------
Null handling is separated from data computation:
1. `bitmap_and` computes the output validity bitmap upfront (bitwise AND of inputs).
2. The data kernel (elementwise, simd, or gpu) runs on ALL elements without
   per-element null checks. Null slots in the output have undefined data values
   but are correctly marked invalid by the precomputed bitmap.

This matches Arrow C++ and Arrow Rust's design and enables SIMD vectorization
of the data loop since there are no conditional branches.

Kernel tiers for binary array → array operations
-------------------------------------------------
  - `binary_simd` — SIMD-width-parameterized function. Since `Scalar[T]` is
    `SIMD[T, 1]`, all scalar functions satisfy this signature and the tail
    loop uses `func[1]` as the scalar fallback. This is the only CPU tier.
  - `binary_gpu` — GPU kernel launch via DeviceContext; not all kernels need this.

For reductions (array → scalar):
  - `reduce` — accumulates over valid (non-null) elements only.

TODO: Add scalar operand support — broadcasting a single scalar value to match
      an array's length. Currently all kernels require two equal-length array
      operands.
TODO: binary_gpu does not propagate null bitmaps; GPU-side bitmap_and is not
      yet implemented. All GPU kernel outputs are marked all-valid.
"""

import math
from sys import size_of, has_accelerator
from sys.info import simd_byte_width

from gpu import global_idx
from gpu.host import DeviceContext

from marrow.arrays import PrimitiveArray, Array
from marrow.buffers import Buffer, Bitmap, BitmapBuilder, BufferBuilder, MemorySpace
from marrow.builders import PrimitiveBuilder
from marrow.dtypes import DataType, all_numeric_dtypes, materialize


# ---------------------------------------------------------------------------
# Null bitmap kernel
# ---------------------------------------------------------------------------


fn bitmap_and(a: Bitmap, b: Bitmap, length: Int) -> Bitmap:
    """Compute the output validity bitmap as the bitwise AND of two input bitmaps.

    Output bit i is True iff both a[i] and b[i] are True (valid).
    SIMD-vectorized over bitmap bytes when both bitmaps are byte-aligned (offset == 0),
    which is the common case for freshly constructed arrays.

    Args:
        a: First input bitmap.
        b: Second input bitmap.
        length: Number of bits (array elements) to process.

    Returns:
        A new Bitmap where result[i] = a[i] AND b[i].
    """
    var byte_count = math.ceildiv(length, 8)
    var result = BitmapBuilder.alloc(length)
    var rp = result.buffer.ptr
    var ap = a.unsafe_ptr()
    var bp = b.unsafe_ptr()

    comptime width = simd_byte_width()
    var i = 0
    while i + width <= byte_count:
        (rp + i).store(
            (ap + i).load[width=width]() & (bp + i).load[width=width]()
        )
        i += width
    while i < byte_count:
        rp[i] = ap[i] & bp[i]
        i += 1

    return result^.freeze()


# ---------------------------------------------------------------------------
# Unary arity helper
# ---------------------------------------------------------------------------


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
    var result = PrimitiveBuilder[OutT](length)
    for i in range(length):
        if array.is_valid(i):
            result.unsafe_set(i, func(array.unsafe_get(i)))
    result.length = length
    return result^.freeze()


fn unary_simd[
    T: DataType,
    func: fn[W: Int](SIMD[T.native, W]) -> SIMD[T.native, W],
](array: PrimitiveArray[T]) -> PrimitiveArray[T]:
    """SIMD-vectorized unary kernel.

    Computes the output validity bitmap upfront (copy of input bitmap), then
    applies func element-wise using full SIMD vectors followed by a scalar tail.
    Null slots in the output have undefined data values but are correctly
    marked invalid by the copied bitmap.

    Parameters:
        T: Element DataType (same for input and output).
        func: The element-wise unary function parameterized by SIMD width W.
              Must accept and return SIMD[T.native, W] for any W >= 1.

    Args:
        array: The input array.

    Returns:
        A new PrimitiveArray with func applied element-wise using SIMD.
    """
    comptime native = T.native
    comptime width = simd_byte_width() // size_of[native]()
    var length = len(array)

    var bm = Bitmap(copy=array.bitmap)
    var buf = BufferBuilder.alloc[native](length)

    var ap = array.buffer.unsafe_ptr[native](array.offset)
    var op = buf.unsafe_ptr[native]()

    var i = 0
    while i + width <= length:
        (op + i).store(func[width]((ap + i).load[width=width]()))
        i += width
    while i < length:
        op[i] = func[1](ap[i])
        i += 1

    return PrimitiveArray[T](
        length=length,
        offset=0,
        bitmap=bm,
        buffer=buf^.freeze(),
    )


# ---------------------------------------------------------------------------
# Binary arity helpers
# ---------------------------------------------------------------------------


fn binary_simd[
    T: DataType,
    func: fn[W: Int](SIMD[T.native, W], SIMD[T.native, W]) -> SIMD[T.native, W],
    name: StringLiteral = "",
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
) raises -> PrimitiveArray[T]:
    """SIMD-vectorized binary kernel.

    Computes the output validity bitmap upfront via `bitmap_and`, then applies
    func element-wise using full SIMD vectors followed by a scalar tail.
    Null slots in the output have undefined data values but are correctly
    marked invalid by the precomputed bitmap.

    Parameters:
        T: Element DataType (same for both inputs and output).
        func: The element-wise binary function parameterized by SIMD width W.
              Must accept and return SIMD[T.native, W] for any W >= 1.
        name: Operation name used in length-mismatch error messages.

    Args:
        left: The left input array.
        right: The right input array.

    Returns:
        A new PrimitiveArray with func applied element-wise using SIMD.
    """
    if len(left) != len(right):
        raise Error(
            String(name)
            + ": arrays must have the same length, got "
            + String(len(left))
            + " and "
            + String(len(right))
        )
    var length = len(left)
    comptime native = T.native
    comptime width = simd_byte_width() // size_of[native]()

    var bm = bitmap_and(left.bitmap, right.bitmap, length)
    var buf = BufferBuilder.alloc[native](length)

    var lp = left.buffer.unsafe_ptr[native](left.offset)
    var rp = right.buffer.unsafe_ptr[native](right.offset)
    var op = buf.unsafe_ptr[native]()

    var i = 0
    while i + width <= length:
        (op + i).store(
            func[width](
                (lp + i).load[width=width](), (rp + i).load[width=width]()
            )
        )
        i += width
    while i < length:
        op[i] = func[1](lp[i], rp[i])
        i += 1

    return PrimitiveArray[T](
        length=length,
        offset=0,
        bitmap=bm,
        buffer=buf^.freeze(),
    )


# ---------------------------------------------------------------------------
# Reduction arity helper
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# GPU binary arity helpers
# ---------------------------------------------------------------------------


fn binary_gpu_kernel[
    dtype: DType,
    func: fn[W: Int](SIMD[dtype, W], SIMD[dtype, W]) -> SIMD[dtype, W],
](
    lhs: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    rhs: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    length: Int,
):
    """Generic GPU kernel: applies a binary SIMD function element-wise (W=1 per thread)."""
    var tid = global_idx.x
    if tid < UInt(length):
        result[tid] = func[1](lhs[tid], rhs[tid])


fn binary_gpu[
    T: DataType,
    func: fn[W: Int](SIMD[T.native, W], SIMD[T.native, W]) -> SIMD[T.native, W],
    name: StringLiteral = "",
](
    left: PrimitiveArray[T, MemorySpace.DEVICE],
    right: PrimitiveArray[T, MemorySpace.DEVICE],
    ctx: DeviceContext,
) raises -> PrimitiveArray[T, MemorySpace.DEVICE]:
    """GPU orchestrator: launches binary_gpu_kernel and returns a device array.

    Parameters:
        T: Element DataType.
        func: SIMD binary function (same signature as binary_simd). Called with W=1
              per GPU thread since `Scalar[T] = SIMD[T, 1]`.
        name: Operation name used in length-mismatch error messages.

    Args:
        left: Left operand (device-resident).
        right: Right operand (device-resident).
        ctx: GPU device context.

    Returns:
        A new device-resident PrimitiveArray with func applied element-wise.
    """
    comptime if not has_accelerator():
        raise Error(String(name) + ": no GPU accelerator available on this system")
    if len(left) != len(right):
        raise Error(
            String(name)
            + ": arrays must have the same length, got "
            + String(len(left))
            + " and "
            + String(len(right))
        )
    var length = len(left)
    comptime native = T.native
    comptime BLOCK_SIZE = 256

    var lhs_dev = left.buffer.device_buffer().create_sub_buffer[native](
        0, length
    )
    var rhs_dev = right.buffer.device_buffer().create_sub_buffer[native](
        0, length
    )

    var out_dev = ctx.enqueue_create_buffer[native](length)

    var num_blocks = math.ceildiv(length, BLOCK_SIZE)
    comptime kernel = binary_gpu_kernel[native, func]
    ctx.enqueue_function_experimental[kernel](
        lhs_dev,
        rhs_dev,
        out_dev,
        length,
        grid_dim=num_blocks,
        block_dim=BLOCK_SIZE,
    )

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
# Runtime-typed dispatch helper
# ---------------------------------------------------------------------------


fn binary_array_dispatch[
    name: StringLiteral,
    func: fn[T: DataType](PrimitiveArray[T], PrimitiveArray[T]) raises -> PrimitiveArray[T],
](
    left: Array[MemorySpace.CPU],
    right: Array[MemorySpace.CPU],
) raises -> Array[MemorySpace.CPU]:
    """Runtime-typed binary dispatch: checks dtype match, loops over numeric types.

    Parameters:
        name: Operation name used in error messages.
        func: The typed binary kernel to dispatch to.

    Args:
        left: Left operand (runtime-typed Array).
        right: Right operand (runtime-typed Array).

    Returns:
        A new Array with the element-wise result.
    """
    if left.dtype != right.dtype:
        raise Error(
            String(name)
            + ": dtype mismatch: "
            + String(left.dtype)
            + " vs "
            + String(right.dtype)
        )
    comptime for dtype in all_numeric_dtypes:
        if left.dtype == materialize[dtype]():
            return Array(
                func[dtype](
                    PrimitiveArray[dtype](data=left),
                    PrimitiveArray[dtype](data=right),
                )
            )
    raise Error(String(name) + ": unsupported dtype " + String(left.dtype))

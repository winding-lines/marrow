"""Arrow-compatible memory buffers: immutable types and mutable builders.

Lifecycle
---------
1. **Allocate** a mutable builder:  `BufferBuilder.alloc[T](n)`
2. **Write** through builder methods:  `builder.unsafe_set(i, v)`
3. **Freeze** into an immutable buffer:  `builder.finish()` → `Buffer`

`finish()` is a zero-cost type-level cast (via `rebind`) because builders and
immutable buffers share the same in-memory layout — only the pointer origin
differs.

Ownership
---------
Every Buffer/BufferBuilder holds an `ArcPointer[Allocation]` that manages
the lifetime of its memory. For Mojo-allocated buffers, the release callback
frees the raw pointer. For foreign memory (e.g. Arrow C Data Interface),
the release callback invokes the producer's cleanup.

Immutable Buffers are Copyable with O(1) shared semantics: copying bumps the
internal ArcPointer ref count. When the last copy is dropped, the release
callback fires. No data is ever deep-copied.

BufferBuilder is Movable but NOT Copyable — it represents unique ownership of
mutable memory.

Bitmap operations
-----------------
Validity bitmaps are stored as plain `Buffer` / `BufferBuilder` with bit-packed
bytes (1 bit per element, LSB first, matching the Arrow specification). Free
functions `bitmap_get`, `bitmap_set`, `bitmap_range_set`, `bitmap_extend`, and
`bitmap_count_ones` operate directly on raw `UnsafePointer[UInt8]`.
"""

from memory import (
    memset_zero,
    memcpy,
    memset,
    ArcPointer,
)
from sys.info import simd_byte_width
from sys import size_of
import math
from gpu.host import DeviceBuffer, DeviceContext


@fieldwise_init
struct MemorySpace(
    Equatable, ImplicitlyCopyable, Movable, Stringable, Writable
):
    """Identifies where a buffer's memory resides at runtime.

    Stored as a field on Buffer to enable device-aware dispatch without
    requiring type parameters on every array type.
    """

    var _value: UInt8

    comptime CPU = MemorySpace(0)
    """Standard CPU heap allocation (default)."""

    comptime DEVICE = MemorySpace(1)
    """GPU device memory (DeviceBuffer). Not CPU-accessible."""

    comptime PINNED = MemorySpace(2)
    """Pinned host memory (HostBuffer). CPU-accessible, fast GPU DMA. [reserved]"""

    @always_inline
    fn is_cpu_accessible(self) -> Bool:
        """Return True if this memory space can be read from the CPU."""
        return self != MemorySpace.DEVICE

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return self._value != other._value

    fn __str__(self) -> String:
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        if self._value == 0:
            writer.write("cpu")
        elif self._value == 1:
            writer.write("device")
        elif self._value == 2:
            writer.write("pinned")
        else:
            writer.write("unknown")


fn dynamic_size_of(dtype: DType) -> Int:
    """Get size of a dtype by dispatching to compile-time size_of."""
    if dtype == DType.bool:
        return size_of[DType.bool]()
    elif dtype == DType.int8:
        return size_of[DType.int8]()
    elif dtype == DType.int16:
        return size_of[DType.int16]()
    elif dtype == DType.int32:
        return size_of[DType.int32]()
    elif dtype == DType.int64:
        return size_of[DType.int64]()
    elif dtype == DType.uint8:
        return size_of[DType.uint8]()
    elif dtype == DType.uint16:
        return size_of[DType.uint16]()
    elif dtype == DType.uint32:
        return size_of[DType.uint32]()
    elif dtype == DType.uint64:
        return size_of[DType.uint64]()
    elif dtype == DType.float32:
        return size_of[DType.float32]()
    elif dtype == DType.float64:
        return size_of[DType.float64]()
    debug_assert(False, "Can't get the size of ", dtype)
    return 0


comptime simd_width = simd_byte_width()

comptime simd_widths = (simd_width, simd_width // 2, 1)


# ---------------------------------------------------------------------------
# Allocation — owns a memory region with a release callback
# ---------------------------------------------------------------------------


struct Allocation(Movable):
    """Owns a memory allocation with a typed release callback.

    When the last ArcPointer[Allocation] is dropped, __del__ fires and
    invokes the release function — e.g. freeing CPU memory or invoking the
    Arrow C Data Interface release callback.
    """

    var ptr: UnsafePointer[UInt8, MutAnyOrigin]
    var release: fn(UnsafePointer[UInt8, MutAnyOrigin]) -> None

    fn __init__(
        out self,
        ptr: UnsafePointer[UInt8, MutAnyOrigin],
        release: fn(UnsafePointer[UInt8, MutAnyOrigin]) -> None,
    ):
        self.ptr = ptr
        self.release = release

    fn __del__(deinit self):
        self.release(self.ptr)


fn _cpu_release(ptr: UnsafePointer[UInt8, MutAnyOrigin]) -> None:
    """Release callback for Mojo-allocated CPU buffers."""
    ptr.free()


# ---------------------------------------------------------------------------
# BufferBuilder — mutable buffer for building arrays
# ---------------------------------------------------------------------------


struct BufferBuilder(Movable):
    """Mutable contiguous memory region with 64-byte alignment.

    Use `BufferBuilder.alloc()` to allocate, write with `unsafe_set()`,
    then call `finish()` to obtain an immutable `Buffer`.
    """

    var ptr: UnsafePointer[UInt8, MutExternalOrigin]
    var size: Int

    fn __init__(
        out self,
        ptr: UnsafePointer[UInt8, MutExternalOrigin],
        size: Int,
    ):
        self.ptr = ptr
        self.size = size

    # TODO: remove this, use alloc[Dtype.bool] instead
    @staticmethod
    fn alloc_bits[I: Intable](n_bits: I) -> BufferBuilder:
        """Allocate a zeroed buffer large enough to hold n_bits bit-packed values.
        """
        return BufferBuilder.alloc(math.ceildiv(Int(n_bits), 8))

    @staticmethod
    fn alloc[
        I: Intable, //, T: DType = DType.uint8
    ](length: I) -> BufferBuilder:
        """Allocate a 64-byte-aligned buffer for `length` elements of type T."""
        var byte_size = math.align_up(Int(length) * size_of[T](), 64)
        var ptr = alloc[UInt8](byte_size, alignment=64)
        memset_zero(ptr, byte_size)
        return BufferBuilder(ptr, byte_size)

    fn finish(mut self) -> Buffer:
        """Snapshot the mutable builder into an immutable Buffer and reset state.

        The current allocation is transferred to the returned Buffer via an
        ArcPointer[Allocation]. A fresh zero-capacity allocation is installed
        on this builder so it can continue to be used after the call.
        """
        var old_ptr = self.ptr
        var old_size = self.size
        # Reset self to a fresh empty allocation; field assignment avoids
        # triggering __del__ on old_ptr (UnsafePointer has no destructor).
        var new_ptr = alloc[UInt8](0, alignment=64)
        self.ptr = rebind[UnsafePointer[UInt8, MutExternalOrigin]](new_ptr)
        self.size = 0
        var result = Buffer(
            rebind[UnsafePointer[UInt8, ImmutExternalOrigin]](old_ptr), old_size
        )
        result.dealloc = ArcPointer(
            Allocation(
                ptr=rebind[UnsafePointer[UInt8, MutAnyOrigin]](old_ptr),
                release=_cpu_release,
            )
        )
        return result^

    # TODO: remove it in favor of resize[Dtype.bool]
    fn resize_bits[I: Intable](mut self, bit_length: I, bit_start: Int = 0):
        """Resize as a bit-packed bitmap, shifting bits if bit_start is non-zero.
        """
        var byte_start = bit_start // 8
        var sub_bit = bit_start % 8
        var new_byte_count = math.ceildiv(Int(bit_length), 8)
        var new = BufferBuilder.alloc(new_byte_count)
        if sub_bit == 0:
            var available = self.size - byte_start
            memcpy(
                dest=new.ptr,
                src=self.ptr + byte_start,
                count=min(new_byte_count, available),
            )
        else:
            for i in range(new_byte_count):
                var src_idx = byte_start + i
                var lo = UInt8(0)
                var hi = UInt8(0)
                if src_idx < self.size:
                    lo = self.ptr[src_idx] >> UInt8(sub_bit)
                if src_idx + 1 < self.size:
                    hi = self.ptr[src_idx + 1] << UInt8(8 - sub_bit)
                new.ptr[i] = lo | hi
        swap(self, new)

    fn resize[
        I: Intable, //, T: DType = DType.uint8
    ](mut self, length: I, start: Int = 0):
        comptime elem_bytes = size_of[T]()
        var new = BufferBuilder.alloc[T](length)
        memcpy(
            dest=new.ptr,
            src=self.ptr + (start * elem_bytes),
            count=min(self.size - start * elem_bytes, Int(length) * elem_bytes),
        )
        swap(self, new)

    # TODO: add special case for Dtype.bool
    @always_inline
    fn length[T: DType = DType.uint8](self) -> Int:
        return self.size // size_of[T]()

    @always_inline
    fn unsafe_ptr[
        T: DType = DType.uint8
    ](self) -> UnsafePointer[Scalar[T], MutExternalOrigin]:
        return self.ptr.bitcast[Scalar[T]]()

    # TODO: would be nice to remove
    @always_inline
    fn unsafe_get[T: DType = DType.uint8](self, index: Int) -> Scalar[T]:
        comptime output = Scalar[T]
        return self.ptr.bitcast[output]()[index]

    @always_inline
    fn unsafe_set[
        T: DType = DType.uint8
    ](mut self, index: Int, value: Scalar[T]):
        comptime output = Scalar[T]
        self.ptr.bitcast[output]()[index] = value

    # TODO: would be nice to remove
    @always_inline
    fn simd_load[T: DType, W: Int](self, index: Int) -> SIMD[T, W]:
        """Load W elements of type T at element index `index`."""
        return (self.ptr.bitcast[Scalar[T]]() + index).load[width=W]()

    @always_inline
    fn simd_store[T: DType, W: Int](mut self, index: Int, value: SIMD[T, W]):
        """Store W elements of type T at element index `index`."""
        (self.ptr.bitcast[Scalar[T]]() + index).store(value)

    fn __del__(deinit self):
        self.ptr.free()


# ---------------------------------------------------------------------------
# Buffer — immutable buffer for read-only array data
# ---------------------------------------------------------------------------


struct Buffer(ImplicitlyCopyable, Movable):
    """Immutable contiguous memory region with 64-byte alignment.

    The `space` field identifies where the buffer's memory lives at runtime:
    - `MemorySpace.CPU` (default): heap-allocated, CPU-accessible
    - `MemorySpace.DEVICE`: GPU device memory, not CPU-accessible

    Calling `unsafe_get` on a device buffer raises a runtime error.
    Use `to_host()` to download device data before reading.

    Buffers are Copyable with O(1) shared semantics: copying bumps the
    internal ArcPointer ref count without copying any data.
    """

    var ptr: UnsafePointer[UInt8, ImmutExternalOrigin]
    var size: Int
    var space: MemorySpace
    var dealloc: Optional[ArcPointer[Allocation]]
    var _device: Optional[DeviceBuffer[DType.uint8]]

    fn __init__(
        out self,
        ptr: UnsafePointer[UInt8, ImmutExternalOrigin],
        size: Int,
    ):
        self.ptr = ptr
        self.size = size
        self.space = MemorySpace.CPU
        self.dealloc = None
        self._device = None

    fn __init__(out self, *, copy: Self):
        self.ptr = copy.ptr
        self.size = copy.size
        self.space = copy.space
        self.dealloc = copy.dealloc
        self._device = copy._device

    @staticmethod
    fn alloc_bits[I: Intable](n_bits: I) -> Buffer:
        """Allocate a zeroed buffer large enough to hold n_bits bit-packed values.
        """
        var b = BufferBuilder.alloc_bits(n_bits)
        return b.finish()

    @staticmethod
    fn foreign_view[
        I: Intable, //
    ](
        ptr: UnsafePointer[NoneType, MutAnyOrigin],
        length: I,
        dtype: DType,
        owner: ArcPointer[Allocation],
    ) -> Buffer:
        """Create an immutable view into foreign memory.

        The caller passes an ArcPointer[Allocation] that keeps the
        source allocation alive for as long as any buffer (or bitmap) derived
        from it exists.  When the last such buffer is dropped the owner's
        release callback fires automatically.
        """
        var result = Buffer(
            rebind[UnsafePointer[UInt8, ImmutExternalOrigin]](
                ptr.bitcast[UInt8]()
            ),
            math.align_up(Int(length) * dynamic_size_of(dtype), 64),
        )
        result.dealloc = Optional(owner)
        return result^

    @staticmethod
    fn device_only(dev: DeviceBuffer[DType.uint8], size: Int) -> Buffer:
        """Create a device-only buffer from a GPU DeviceBuffer handle.

        The resulting buffer is not CPU-accessible. Call `to_host()` to
        download to CPU memory.
        """
        var result = Buffer(UnsafePointer[UInt8, ImmutExternalOrigin](), size)
        result.space = MemorySpace.DEVICE
        result._device = dev
        return result^

    fn __del__(deinit self):
        pass

    @always_inline
    fn is_cpu(self) -> Bool:
        """Return True if the buffer is CPU-accessible."""
        return self.space != MemorySpace.DEVICE

    @always_inline
    fn is_device(self) -> Bool:
        """Return True if the buffer lives on a GPU device."""
        return self.space == MemorySpace.DEVICE

    @always_inline
    fn has_device(self) -> Bool:
        """Return True if the buffer lives on a GPU device."""
        return self.is_device()

    fn device_buffer(self) -> DeviceBuffer[DType.uint8]:
        """Return the DeviceBuffer handle. Only valid on device buffers."""
        debug_assert(self.space == MemorySpace.DEVICE, "not a device buffer")
        return self._device.value()

    fn to_device(self, ctx: DeviceContext) raises -> Buffer:
        """Upload buffer data to the GPU, returning a new device-only Buffer."""
        if self.space == MemorySpace.DEVICE:
            raise Error("to_device: buffer is already on device")
        var dev = ctx.enqueue_create_buffer[DType.uint8](self.size)
        ctx.enqueue_copy(dev, self.ptr)
        return Buffer.device_only(dev, self.size)

    fn to_host(self, ctx: DeviceContext) raises -> Buffer:
        """Download buffer data from the GPU, returning a new CPU Buffer."""
        if self.space != MemorySpace.DEVICE:
            raise Error("to_host: buffer is not on device")
        var builder = BufferBuilder.alloc(self.size)
        ctx.enqueue_copy(builder.ptr, self._device.value())
        ctx.synchronize()
        return builder.finish()

    # TODO: use Dtype.bool specialization
    @always_inline
    fn length[T: DType = DType.uint8](self) -> Int:
        return self.size // size_of[T]()

    @always_inline
    fn unsafe_ptr[
        T: DType = DType.uint8
    ](self, offset: Int = 0) -> UnsafePointer[Scalar[T], ImmutExternalOrigin]:
        """Return a typed pointer to the element at offset."""
        debug_assert(
            self.space != MemorySpace.DEVICE,
            "cannot read device buffer, call to_host() first",
        )
        return self.ptr.bitcast[Scalar[T]]() + offset

    @always_inline
    fn unsafe_get[T: DType = DType.uint8](self, index: Int) -> Scalar[T]:
        debug_assert(
            self.space != MemorySpace.DEVICE,
            "cannot read device buffer, call to_host() first",
        )
        comptime output = Scalar[T]
        return self.ptr.bitcast[output]()[index]

    @always_inline
    fn simd_load[T: DType, W: Int](self, index: Int) -> SIMD[T, W]:
        """Load W elements of type T at element index `index`."""
        debug_assert(
            self.space != MemorySpace.DEVICE,
            "cannot read device buffer, call to_host() first",
        )
        return (self.ptr.bitcast[Scalar[T]]() + index).load[width=W]()


# ---------------------------------------------------------------------------
# Bitmap free functions — operate on raw UnsafePointer[UInt8]
# ---------------------------------------------------------------------------


# TODO: move it to unsafe_get with Dtype.bool specialization
@always_inline
fn bitmap_get(
    ptr: UnsafePointer[UInt8, ImmutExternalOrigin], index: Int
) -> Bool:
    """Read the bit at `index` from a bit-packed bitmap byte array."""
    return Bool((ptr[index // 8] >> UInt8(index % 8)) & 1)


# TODO: move it to unsafe_set with Dtype.bool specialization
@always_inline
fn bitmap_set(
    ptr: UnsafePointer[UInt8, MutExternalOrigin], index: Int, value: Bool
):
    """Write `value` to bit `index` in a bit-packed bitmap byte array."""
    var byte_index = index // 8
    var bit_mask = UInt8(1 << (index % 8))
    if value:
        ptr[byte_index] = ptr[byte_index] | bit_mask
    else:
        ptr[byte_index] = ptr[byte_index] & ~bit_mask


# TODO: remove it
fn _bitmap_partial_byte_set(
    ptr: UnsafePointer[UInt8, MutExternalOrigin],
    byte_index: Int,
    bit_pos_start: Int,
    bit_pos_end: Int,
    value: Bool,
):
    """Set bits [bit_pos_start, bit_pos_end) within a single byte to `value`."""
    debug_assert(
        bit_pos_start >= 0
        and bit_pos_end <= 8
        and bit_pos_start <= bit_pos_end,
        "Invalid range: ",
        bit_pos_start,
        " to ",
        bit_pos_end,
    )
    var mask = UInt8((1 << (bit_pos_end - bit_pos_start)) - 1) << UInt8(
        bit_pos_start
    )
    if value:
        ptr[byte_index] = ptr[byte_index] | mask
    else:
        ptr[byte_index] = ptr[byte_index] & ~mask


# TODO: remove it
fn bitmap_range_set(
    ptr: UnsafePointer[UInt8, MutExternalOrigin],
    start: Int,
    length: Int,
    value: Bool,
):
    """Set `length` bits starting at `start` to `value` in a bit-packed bitmap.
    """
    var end = start + length
    var start_byte = start // 8
    var bit_pos_start = start % 8
    var end_byte = end // 8
    var bit_pos_end = end % 8

    if bit_pos_start != 0 or bit_pos_end != 0:
        if start_byte == end_byte:
            _bitmap_partial_byte_set(
                ptr, start_byte, bit_pos_start, bit_pos_end, value
            )
            return
        if bit_pos_start != 0:
            _bitmap_partial_byte_set(ptr, start_byte, bit_pos_start, 8, value)
            start_byte += 1
        if bit_pos_end != 0:
            _bitmap_partial_byte_set(ptr, end_byte, 0, bit_pos_end, value)

    if end_byte > start_byte:
        memset(
            ptr + start_byte,
            value=UInt8(255 if value else 0),
            count=end_byte - start_byte,
        )


# TODO: remove it
fn bitmap_extend(
    dst: UnsafePointer[UInt8, MutExternalOrigin],
    src: UnsafePointer[UInt8, ImmutExternalOrigin],
    dst_start: Int,
    length: Int,
):
    """Copy `length` bits from `src` (starting at bit 0) into `dst` at `dst_start`.
    """
    for i in range(length):
        bitmap_set(dst, dst_start + i, bitmap_get(src, i))


# TODO: move it to kernels
fn bitmap_count_ones(
    ptr: UnsafePointer[UInt8, ImmutExternalOrigin], byte_count: Int
) -> Int:
    """Count the number of set (1) bits across `byte_count` bytes."""
    comptime width = simd_byte_width()
    var count = 0
    var i = 0
    while i + width <= byte_count:
        count += Int((ptr + i).load[width=width]().reduce_bit_count())
        i += width
    while i < byte_count:
        count += Int((ptr + i).load[width=1]().reduce_bit_count())
        i += 1
    return count

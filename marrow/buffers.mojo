"""Arrow-compatible memory buffers: immutable types and mutable builders.

Lifecycle
---------
1. **Allocate** a mutable builder:  `BufferBuilder.alloc[T](n)`
2. **Write** through builder methods:  `builder.unsafe_set(i, v)`
3. **Freeze** into an immutable buffer:  `builder^.freeze()` → `Buffer`

`freeze()` is a zero-cost type-level cast (via `rebind`) because builders and
immutable buffers share the same in-memory layout — only the pointer origin
differs.

Ownership
---------
Every Buffer/BufferBuilder holds an `ArcPointer[Allocation]` that manages
the lifetime of its memory. For Mojo-allocated buffers, the release callback
frees the raw pointer. For foreign memory (e.g. Arrow C Data Interface),
the release callback invokes the producer's cleanup.

Immutable Buffers and Bitmaps are Copyable with O(1) shared semantics:
copying bumps the internal ArcPointer ref count. When the last copy is
dropped, the release callback fires. No data is ever deep-copied.

Builders (BufferBuilder, BitmapBuilder) are Movable but NOT Copyable — they
represent unique ownership of mutable memory.
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
from bit import pop_count, count_trailing_zeros
from gpu.host import DeviceBuffer, DeviceContext


@fieldwise_init
struct MemorySpace(ImplicitlyCopyable, Movable, Equatable, Stringable, Writable):
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
    then call `freeze()` to obtain an immutable `Buffer`.
    """

    var ptr: UnsafePointer[UInt8, MutExternalOrigin]
    var size: Int
    var dealloc: Optional[ArcPointer[Allocation]]

    fn __init__(
        out self,
        ptr: UnsafePointer[UInt8, MutExternalOrigin],
        size: Int,
    ):
        self.ptr = ptr
        self.size = size
        self.dealloc = None

    @staticmethod
    fn alloc[
        I: Intable, //, T: DType = DType.uint8
    ](length: I) -> BufferBuilder:
        """Allocate a 64-byte-aligned buffer for `length` elements of type T."""
        var byte_size = math.align_up(Int(length) * size_of[T](), 64)
        var ptr = alloc[UInt8](byte_size, alignment=64)
        memset_zero(ptr, byte_size)
        var result = BufferBuilder(ptr, byte_size)
        result.dealloc = ArcPointer(Allocation(ptr=ptr, release=_cpu_release))
        return result^

    fn freeze(deinit self) -> Buffer:
        """Consume the mutable builder and return an immutable Buffer."""
        var ptr = rebind[UnsafePointer[UInt8, ImmutExternalOrigin]](self.ptr)
        var result = Buffer(ptr, self.size)
        result.dealloc = self.dealloc^
        return result^

    fn resize[I: Intable, //, T: DType = DType.uint8](mut self, length: I, start: Int = 0):
        comptime elem_bytes = size_of[T]()
        var new = BufferBuilder.alloc[T](length)
        memcpy(
            dest=new.ptr,
            src=self.ptr + (start * elem_bytes),
            count=min(
                self.size - start * elem_bytes, Int(length) * elem_bytes
            ),
        )
        swap(self, new)

    fn __del__(deinit self):
        pass

    @always_inline
    fn length[T: DType = DType.uint8](self) -> Int:
        return self.size // size_of[T]()

    @always_inline
    fn unsafe_ptr[T: DType = DType.uint8](self) -> UnsafePointer[Scalar[T], MutExternalOrigin]:
        return self.ptr.bitcast[Scalar[T]]()

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

    @always_inline
    fn simd_load[T: DType, W: Int](self, index: Int) -> SIMD[T, W]:
        """Load W elements of type T at element index `index`."""
        return (self.ptr.bitcast[Scalar[T]]() + index).load[width=W]()

    @always_inline
    fn simd_store[T: DType, W: Int](mut self, index: Int, value: SIMD[T, W]):
        """Store W elements of type T at element index `index`."""
        (self.ptr.bitcast[Scalar[T]]() + index).store(value)


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

    @implicit
    fn __init__(out self, var bitmap: Bitmap):
        """Implicitly convert an immutable Bitmap to its underlying Buffer."""
        self.ptr = bitmap.buffer.ptr
        self.size = bitmap.buffer.size
        self.space = bitmap.buffer.space
        self.dealloc = None
        self._device = bitmap.buffer._device
        swap(self.dealloc, bitmap.buffer.dealloc)

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
    fn device_only(
        dev: DeviceBuffer[DType.uint8], size: Int
    ) -> Buffer:
        """Create a device-only buffer from a GPU DeviceBuffer handle.

        The resulting buffer is not CPU-accessible. Call `to_host()` to
        download to CPU memory.
        """
        var result = Buffer(
            UnsafePointer[UInt8, ImmutExternalOrigin](), size
        )
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
        return builder^.freeze()

    @always_inline
    fn length[T: DType = DType.uint8](self) -> Int:
        return self.size // size_of[T]()

    @always_inline
    fn unsafe_ptr[T: DType = DType.uint8](self, offset: Int = 0) -> UnsafePointer[Scalar[T], ImmutExternalOrigin]:
        """Return a typed pointer to the element at offset."""
        debug_assert(self.space != MemorySpace.DEVICE, "cannot read device buffer, call to_host() first")
        return self.ptr.bitcast[Scalar[T]]() + offset

    @always_inline
    fn unsafe_get[T: DType = DType.uint8](self, index: Int) -> Scalar[T]:
        debug_assert(self.space != MemorySpace.DEVICE, "cannot read device buffer, call to_host() first")
        comptime output = Scalar[T]
        return self.ptr.bitcast[output]()[index]

    @always_inline
    fn simd_load[T: DType, W: Int](self, index: Int) -> SIMD[T, W]:
        """Load W elements of type T at element index `index`."""
        debug_assert(self.space != MemorySpace.DEVICE, "cannot read device buffer, call to_host() first")
        return (self.ptr.bitcast[Scalar[T]]() + index).load[width=W]()


# ---------------------------------------------------------------------------
# BitmapBuilder — mutable bit-packed validity bitmap
# ---------------------------------------------------------------------------


struct BitmapBuilder(Movable, Stringable):
    """Mutable bit-packed validity bitmap backed by a BufferBuilder.

    Use `BitmapBuilder.alloc()` to allocate, write with `unsafe_set()`,
    then call `freeze()` to obtain an immutable `Bitmap`.
    """

    var buffer: BufferBuilder

    @staticmethod
    fn alloc[I: Intable](length: I) -> BitmapBuilder:
        var byte_length = math.ceildiv(Int(length), 8)
        return BitmapBuilder(BufferBuilder.alloc(byte_length))

    fn __init__(out self, var buffer: BufferBuilder):
        self.buffer = buffer^

    fn freeze(deinit self) -> Bitmap:
        """Consume the mutable builder and return an immutable Bitmap."""
        return Bitmap(self.buffer^.freeze())

    fn __str__(self) -> String:
        var output = String()
        for i in range(self.length()):
            var value = self.unsafe_get(i)
            if value:
                output += "T"
            else:
                output += "f"
            if i > 16:
                output += "..."
                break
        return output

    fn unsafe_get(self, index: Int) -> Bool:
        return Bool((self.buffer.ptr[index // 8] >> UInt8(index % 8)) & 1)

    fn unsafe_set(mut self, index: Int, value: Bool) -> None:
        var byte_index = index // 8
        var bit_mask = UInt8(1 << (index % 8))
        if value:
            self.buffer.ptr[byte_index] = self.buffer.ptr[byte_index] | bit_mask
        else:
            self.buffer.ptr[byte_index] = (
                self.buffer.ptr[byte_index] & ~bit_mask
            )

    @always_inline
    fn simd_load[W: Int](self, byte_offset: Int) -> SIMD[DType.uint8, W]:
        """Load W bytes from the bitmap at byte offset `byte_offset`."""
        return (self.buffer.ptr + byte_offset).load[width=W]()

    @always_inline
    fn simd_store[W: Int](mut self, byte_offset: Int, value: SIMD[DType.uint8, W]):
        """Store W bytes to the bitmap at byte offset `byte_offset`."""
        (self.buffer.ptr + byte_offset).store(value)

    @always_inline
    fn length(self) -> Int:
        return self.buffer.size * 8

    @always_inline
    fn size(self) -> Int:
        return self.buffer.size

    fn resize[I: Intable](mut self, length: I, start: Int = 0):
        var byte_start = start // 8
        var sub_bit = start % 8
        var new_byte_count = math.ceildiv(Int(length), 8)
        var new = BufferBuilder.alloc(new_byte_count)
        if sub_bit == 0:
            var available = self.buffer.size - byte_start
            memcpy(dest=new.ptr, src=self.buffer.ptr + byte_start, count=min(new_byte_count, available))
        else:
            for i in range(new_byte_count):
                var src_idx = byte_start + i
                var lo = UInt8(0)
                var hi = UInt8(0)
                if src_idx < self.buffer.size:
                    lo = self.buffer.ptr[src_idx] >> UInt8(sub_bit)
                if src_idx + 1 < self.buffer.size:
                    hi = self.buffer.ptr[src_idx + 1] << UInt8(8 - sub_bit)
                new.ptr[i] = lo | hi
        swap(self.buffer, new)

    fn extend(
        mut self,
        other: Bitmap,
        start: Int,
        length: Int,
    ) -> None:
        """Extends the bitmap with the other's array's bitmap.

        Args:
            other: The bitmap to take content from.
            start: The starting index in the destination array.
            length: The number of elements to copy from the source array.
        """
        self.buffer.resize(math.ceildiv(start + length, 8))

        for i in range(length):
            self.unsafe_set(i + start, other.unsafe_get(i))

    fn partial_byte_set(
        mut self,
        byte_index: Int,
        bit_pos_start: Int,
        bit_pos_end: Int,
        value: Bool,
    ) -> None:
        """Set a range of bits in one specific byte of the bitmap to the specified value.
        """

        debug_assert(
            bit_pos_start >= 0
            and bit_pos_end <= 8
            and bit_pos_start <= bit_pos_end,
            "Invalid range: ",
            bit_pos_start,
            " to ",
            bit_pos_end,
        )

        # Process the partial byte at the start, if appropriate.
        var mask = (1 << (bit_pos_end - bit_pos_start)) - 1
        mask = mask << bit_pos_start
        var initial_value = self.buffer.unsafe_get[DType.uint8](byte_index)
        var buffer_value = initial_value
        if value:
            buffer_value = buffer_value | UInt8(mask)
        else:
            buffer_value = buffer_value & ~UInt8(mask)
        self.buffer.unsafe_set[DType.uint8](byte_index, buffer_value)

    fn bit_count(self) -> Int:
        """The number of bits with value 1 in the BitmapBuilder."""
        var start = 0
        var count = 0
        while start < self.buffer.size:
            if self.buffer.size - start > simd_width:
                count += (
                    (self.buffer.ptr + start)
                    .load[width=simd_width]()
                    .reduce_bit_count()
                )
                start += simd_width
            else:
                count += (
                    (self.buffer.ptr + start).load[width=1]().reduce_bit_count()
                )
                start += 1
        return count

    fn count_leading_bits(self, start: Int = 0, value: Bool = False) -> Int:
        """Count the number of leading bits with the given value."""
        var count = 0
        var index = start // 8
        var bit_in_first_byte = start % 8

        if bit_in_first_byte != 0:
            var loaded = (self.buffer.ptr + index).load[width=1]()
            if value:
                loaded = ~loaded
            var mask = (1 << bit_in_first_byte) - 1
            loaded &= ~UInt8(mask)
            leading_zeros = Int(count_trailing_zeros(loaded))
            if leading_zeros == 0:
                return count
            count = leading_zeros - bit_in_first_byte
            if leading_zeros != 8:
                return count
            index += 1

        while index < self.size():
            comptime for width_index in range(len(simd_widths)):
                comptime width = simd_widths[width_index]
                if self.size() - index >= width:
                    var loaded = (self.buffer.ptr + index).load[width=width]()
                    if value:
                        loaded = ~loaded
                    var leading_zeros = count_trailing_zeros(loaded)
                    for i in range(width):
                        count += Int(leading_zeros[i])
                        if leading_zeros[i] != 8:
                            return count
                    index += width
                    break
        return count

    fn count_leading_zeros(self, start: Int = 0) -> Int:
        """Count the number of leading 0s in the bitmap."""
        return self.count_leading_bits(start, value=False)

    fn count_leading_ones(self, start: Int = 0) -> Int:
        """Count the number of leading 1s in the bitmap."""
        return self.count_leading_bits(start, value=True)

    fn unsafe_range_set[
        T: Intable, U: Intable, //
    ](mut self, start: T, length: U, value: Bool) -> None:
        """Set a range of bits in the bitmap to the specified value.

        Args:
            start: The starting index in the bitmap.
            length: The number of bits to set.
            value: The value to set the bits to.
        """

        # Process the partial byte at the ends.
        var start_int = Int(start)
        var end_int = start_int + Int(length)
        var start_index = start_int // 8
        var bit_pos_start = start_int % 8
        var end_index = end_int // 8
        var bit_pos_end = end_int % 8

        if bit_pos_start != 0 or bit_pos_end != 0:
            if start_index == end_index:
                self.partial_byte_set(
                    start_index, bit_pos_start, bit_pos_end, value
                )
            else:
                if bit_pos_start != 0:
                    self.partial_byte_set(start_index, bit_pos_start, 8, value)
                    start_index += 1
                if bit_pos_end != 0:
                    self.partial_byte_set(end_index, 0, bit_pos_end, value)

        # Now take care of the full bytes.
        if end_index > start_index:
            var byte_value = 255 if value else 0
            memset(
                self.buffer.ptr + start_index,
                value=UInt8(byte_value),
                count=end_index - start_index,
            )


# ---------------------------------------------------------------------------
# Bitmap — immutable bit-packed validity bitmap
# ---------------------------------------------------------------------------


struct Bitmap(
    ImplicitlyCopyable, Movable, Stringable
):
    """Immutable bit-packed validity bitmap backed by a Buffer.

    Tracks which elements of an Arrow array are valid (True) vs null (False).

    Copyable with O(1) shared semantics (delegates to Buffer's copy).
    """

    var buffer: Buffer

    @staticmethod
    fn foreign_view[
        I: Intable, //
    ](
        ptr: UnsafePointer[NoneType, MutAnyOrigin],
        length: I,
        owner: ArcPointer[Allocation],
    ) -> Bitmap:
        """Create an immutable view into foreign memory."""
        var byte_length = math.ceildiv(Int(length), 8)
        var buffer = Buffer.foreign_view(ptr, byte_length, DType.uint8, owner)
        return Bitmap(buffer^)

    fn __init__(out self, var buffer: Buffer):
        self.buffer = buffer^

    fn __init__(out self, *, copy: Self):
        self.buffer = Buffer(copy=copy.buffer)

    fn __str__(self) -> String:
        debug_assert(self.buffer.space != MemorySpace.DEVICE, "cannot stringify device bitmap")
        var output = String()
        for i in range(self.length()):
            var value = self.unsafe_get(i)
            if value:
                output += "T"
            else:
                output += "f"
            if i > 16:
                output += "..."
                break
        return output

    @always_inline
    fn unsafe_ptr(self, offset: Int = 0) -> UnsafePointer[UInt8, ImmutExternalOrigin]:
        """Return a byte pointer to the byte at bit offset.

        Requires offset to be byte-aligned (offset % 8 == 0).
        Use this for SIMD bitmap operations.
        """
        debug_assert(self.buffer.space != MemorySpace.DEVICE, "cannot read device bitmap, call to_host() first")
        debug_assert(offset % 8 == 0, "bitmap offset is not byte-aligned")
        return self.buffer.ptr + (offset // 8)

    fn unsafe_get(self, index: Int) -> Bool:
        debug_assert(self.buffer.space != MemorySpace.DEVICE, "cannot read device bitmap, call to_host() first")
        return Bool((self.buffer.ptr[index // 8] >> UInt8(index % 8)) & 1)

    @always_inline
    fn simd_load[W: Int](self, byte_offset: Int) -> SIMD[DType.uint8, W]:
        """Load W bytes from the bitmap at byte offset `byte_offset`."""
        debug_assert(self.buffer.space != MemorySpace.DEVICE, "cannot read device bitmap, call to_host() first")
        return (self.buffer.ptr + byte_offset).load[width=W]()

    @always_inline
    fn length(self) -> Int:
        return self.buffer.size * 8

    @always_inline
    fn size(self) -> Int:
        return self.buffer.size

    fn has_device(self) -> Bool:
        """Return True if the bitmap lives on a GPU device."""
        return self.buffer.has_device()

    fn to_device(self, ctx: DeviceContext) raises -> Bitmap:
        """Upload bitmap data to the GPU."""
        return Bitmap(self.buffer.to_device(ctx))

    fn to_host(self, ctx: DeviceContext) raises -> Bitmap:
        """Download bitmap data from the GPU."""
        return Bitmap(self.buffer.to_host(ctx))

    fn device_buffer(self) -> DeviceBuffer[DType.uint8]:
        """Return the underlying DeviceBuffer handle."""
        return self.buffer.device_buffer()

    fn bit_count(self) -> Int:
        """The number of bits with value 1 in the Bitmap."""
        debug_assert(self.buffer.space != MemorySpace.DEVICE, "cannot count bits on device bitmap")
        var start = 0
        var count = 0
        while start < self.buffer.size:
            if self.buffer.size - start > simd_width:
                count += (
                    (self.buffer.ptr + start)
                    .load[width=simd_width]()
                    .reduce_bit_count()
                )
                start += simd_width
            else:
                count += (
                    (self.buffer.ptr + start).load[width=1]().reduce_bit_count()
                )
                start += 1
        return count

    fn count_leading_bits(self, start: Int = 0, value: Bool = False) -> Int:
        """Count the number of leading bits with the given value in the bitmap."""
        debug_assert(self.buffer.space != MemorySpace.DEVICE, "cannot count bits on device bitmap")

        var count = 0
        var index = start // 8
        var bit_in_first_byte = start % 8

        if bit_in_first_byte != 0:
            var loaded = (self.buffer.ptr + index).load[width=1]()
            if value:
                loaded = ~loaded
            var mask = (1 << bit_in_first_byte) - 1
            loaded &= ~UInt8(mask)
            leading_zeros = Int(count_trailing_zeros(loaded))
            if leading_zeros == 0:
                return count
            count = leading_zeros - bit_in_first_byte
            if leading_zeros != 8:
                return count

            index += 1

        while index < self.size():
            comptime for width_index in range(len(simd_widths)):
                comptime width = simd_widths[width_index]
                if self.size() - index >= width:
                    var loaded = (self.buffer.ptr + index).load[width=width]()
                    if value:
                        loaded = ~loaded
                    var leading_zeros = count_trailing_zeros(loaded)
                    for i in range(width):
                        count += Int(leading_zeros[i])
                        if leading_zeros[i] != 8:
                            return count
                    index += width
                    break
        return count

    fn count_leading_zeros(self, start: Int = 0) -> Int:
        """Count the number of leading 0s in the bitmap."""
        return self.count_leading_bits(start, value=False)

    fn count_leading_ones(self, start: Int = 0) -> Int:
        """Count the number of leading 1s in the bitmap."""
        return self.count_leading_bits(start, value=True)

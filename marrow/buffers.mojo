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
    var offset: Int
    var dealloc: Optional[ArcPointer[Allocation]]
    var device: Optional[DeviceBuffer[DType.uint8]]

    fn __init__(
        out self,
        ptr: UnsafePointer[UInt8, MutExternalOrigin],
        size: Int,
        offset: Int = 0,
    ):
        self.ptr = ptr
        self.size = size
        self.offset = offset
        self.dealloc = None
        self.device = None

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
        var result = Buffer(ptr, self.size, self.offset)
        result.dealloc = self.dealloc^
        result.device = self.device^
        return result^

    fn resize[I: Intable, //, T: DType = DType.uint8](mut self, length: I):
        comptime elem_bytes = size_of[T]()
        var new = BufferBuilder.alloc[T](length)
        memcpy(
            dest=new.ptr,
            src=self.ptr + (self.offset * elem_bytes),
            count=min(
                self.size - self.offset * elem_bytes, Int(length) * elem_bytes
            ),
        )
        swap(self, new)

    fn __del__(deinit self):
        pass

    @always_inline
    fn length[T: DType = DType.uint8](self) -> Int:
        return self.size // size_of[T]()

    @always_inline
    fn unsafe_get[T: DType = DType.uint8](self, index: Int) -> Scalar[T]:
        comptime output = Scalar[T]
        return self.ptr.bitcast[output]()[index + self.offset]

    @always_inline
    fn unsafe_set[
        T: DType = DType.uint8
    ](mut self, index: Int, value: Scalar[T]):
        comptime output = Scalar[T]
        self.ptr.bitcast[output]()[index + self.offset] = value


# ---------------------------------------------------------------------------
# Buffer — immutable buffer for read-only array data
# ---------------------------------------------------------------------------


struct Buffer(ImplicitlyCopyable, Movable):
    """Immutable contiguous memory region with 64-byte alignment.

    Buffers are Copyable with O(1) shared semantics: copying bumps the
    internal ArcPointer ref count without copying any data. The last copy
    to be dropped triggers the release callback that frees the memory.

    Data can reside on host (CPU), device (GPU), or both. Use `to_device()`
    to upload and `to_host()` to download. Since buffers are immutable,
    both copies are always in sync.
    """

    var ptr: UnsafePointer[UInt8, ImmutExternalOrigin]
    var size: Int
    var offset: Int
    var dealloc: Optional[ArcPointer[Allocation]]
    var device: Optional[DeviceBuffer[DType.uint8]]

    fn __init__(
        out self,
        ptr: UnsafePointer[UInt8, ImmutExternalOrigin],
        size: Int,
        offset: Int = 0,
    ):
        self.ptr = ptr
        self.size = size
        self.offset = offset
        self.dealloc = None
        self.device = None

    fn __init__(out self, *, copy: Self):
        self.ptr = copy.ptr
        self.size = copy.size
        self.offset = copy.offset
        self.dealloc = copy.dealloc
        self.device = copy.device

    @implicit
    fn __init__(out self, var bitmap: Bitmap):
        """Implicitly convert an immutable Bitmap to its underlying Buffer."""
        self.ptr = bitmap.buffer.ptr
        self.size = bitmap.buffer.size
        self.offset = bitmap.buffer.offset
        self.dealloc = None
        self.device = bitmap.buffer.device
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

    fn __del__(deinit self):
        pass

    @always_inline
    fn has_device(self) -> Bool:
        """Return True if the buffer has a device (GPU) copy."""
        return Bool(self.device)

    fn to_device(self, ctx: DeviceContext) raises -> Buffer:
        """Upload buffer data to the GPU. Returns self if already on device."""
        if self.device:
            return self
        var dev = ctx.enqueue_create_buffer[DType.uint8](self.size)
        ctx.enqueue_copy(dev, self.ptr + self.offset)
        var result = Buffer(copy=self)
        result.device = dev
        result.offset = 0
        return result^

    fn to_host(self, ctx: DeviceContext) raises -> Buffer:
        """Download buffer data from the GPU. Returns self if already on host."""
        if self.dealloc:
            return self
        var builder = BufferBuilder.alloc(self.size)
        ctx.enqueue_copy(builder.ptr, self.device.value())
        ctx.synchronize()
        var result = builder^.freeze()
        result.device = self.device
        return result^

    @always_inline
    fn length[T: DType = DType.uint8](self) -> Int:
        return self.size // size_of[T]()

    @always_inline
    fn unsafe_get[T: DType = DType.uint8](self, index: Int) -> Scalar[T]:
        comptime output = Scalar[T]
        return self.ptr.bitcast[output]()[index + self.offset]


# ---------------------------------------------------------------------------
# BitmapBuilder — mutable bit-packed validity bitmap
# ---------------------------------------------------------------------------


struct BitmapBuilder(Movable, Stringable):
    """Mutable bit-packed validity bitmap backed by a BufferBuilder.

    Use `BitmapBuilder.alloc()` to allocate, write with `unsafe_set()`,
    then call `freeze()` to obtain an immutable `Bitmap`.
    """

    var buffer: BufferBuilder
    var offset: Int

    @staticmethod
    fn alloc[I: Intable](length: I) -> BitmapBuilder:
        var byte_length = math.ceildiv(Int(length), 8)
        return BitmapBuilder(BufferBuilder.alloc(byte_length))

    fn __init__(out self, var buffer: BufferBuilder, offset: Int = 0):
        self.buffer = buffer^
        self.offset = offset

    fn freeze(deinit self) -> Bitmap:
        """Consume the mutable builder and return an immutable Bitmap."""
        return Bitmap(self.buffer^.freeze(), self.offset)

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
        var adjusted = index + self.offset
        return Bool((self.buffer.ptr[adjusted // 8] >> UInt8(adjusted % 8)) & 1)

    fn unsafe_set(mut self, index: Int, value: Bool) -> None:
        var adjusted = index + self.offset
        var byte_index = adjusted // 8
        var bit_mask = UInt8(1 << (adjusted % 8))
        if value:
            self.buffer.ptr[byte_index] = self.buffer.ptr[byte_index] | bit_mask
        else:
            self.buffer.ptr[byte_index] = (
                self.buffer.ptr[byte_index] & ~bit_mask
            )

    @always_inline
    fn length(self) -> Int:
        return self.buffer.size * 8

    @always_inline
    fn size(self) -> Int:
        return self.buffer.size

    fn resize[I: Intable](mut self, length: I, start: Int = 0):
        var bit_start = start + self.offset
        self.buffer.offset = bit_start // 8
        self.buffer.resize(math.ceildiv(Int(length) + bit_start % 8, 8))
        self.offset = bit_start % 8

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


struct Bitmap(ImplicitlyCopyable, Movable, Stringable):
    """Immutable bit-packed validity bitmap backed by a Buffer.

    Tracks which elements of an Arrow array are valid (True) vs null (False).

    Copyable with O(1) shared semantics (delegates to Buffer's copy).
    """

    var buffer: Buffer
    var offset: Int

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

    fn __init__(out self, var buffer: Buffer, offset: Int = 0):
        self.buffer = buffer^
        self.offset = offset

    fn __init__(out self, *, copy: Self):
        self.buffer = Buffer(copy=copy.buffer)
        self.offset = copy.offset

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
        var adjusted = index + self.offset
        return Bool((self.buffer.ptr[adjusted // 8] >> UInt8(adjusted % 8)) & 1)

    @always_inline
    fn length(self) -> Int:
        return self.buffer.size * 8

    @always_inline
    fn size(self) -> Int:
        return self.buffer.size

    fn has_device(self) -> Bool:
        """Return True if the bitmap has a device (GPU) copy."""
        return self.buffer.has_device()

    fn to_device(self, ctx: DeviceContext) raises -> Bitmap:
        """Upload bitmap data to the GPU."""
        return Bitmap(self.buffer.to_device(ctx), offset=0)

    fn to_host(self, ctx: DeviceContext) raises -> Bitmap:
        """Download bitmap data from the GPU."""
        return Bitmap(self.buffer.to_host(ctx), self.offset)

    fn bit_count(self) -> Int:
        """The number of bits with value 1 in the Bitmap."""
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
        """Count the number of leading bits with the given value in the bitmap, starting at a given posiion.

        Note that index 0 in the bitmap translates to right most bit in the first byte of the buffer.
        So when we are looking for leading zeros from a bitmap standpoing we need to look at
        trailing zeros in the bitmap's associated buffer.

        The SIMD API available looks at leading zeros only, we negate the input when needed.

        Args:
          start: The position where we should start counting.
          value: The value of the bits we want to count.

        Returns:
          The number of leadinging bits with the given value in the bitmap.
        """

        var count = 0
        var index = start // 8
        var bit_in_first_byte = start % 8

        if bit_in_first_byte != 0:
            # Process the partial first byte by applying a mask.
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
                # The first byte has some bits of the other value, just return the count.
                return count

            index += 1

        # Process full bytes.
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
                    # break from the simd widths loop
                    break
        return count

    fn count_leading_zeros(self, start: Int = 0) -> Int:
        """Count the number of leading 0s in the given value in the bitmap, starting at a given posiion.

        Note that index 0 in the bitmap translates to right most bit in the first byte of the buffer.
        So when we are looking for leading zeros from a bitmap standpoing we need to look at
        trailing zeros in the bitmap's associated buffer.

        Args:
            start: The position where we should start counting.

        Returns:
             The number of leading zeros in the bitmap.
        """
        return self.count_leading_bits(start, value=False)

    fn count_leading_ones(self, start: Int = 0) -> Int:
        """Count the number of leading 1s in the given value in the bitmap, starting at a given posiion.

        Note that index 0 in the bitmap translates to right most bit in the first byte of the buffer.
        So when we are looking for leading zeros from a bitmap standpoing we need to look at
        trailing zeros in the bitmap's associated buffer.

        Args:
          start: The position where we should start counting.

        Returns:
          The number of leading ones in the bitmap.
        """
        return self.count_leading_bits(start, value=True)

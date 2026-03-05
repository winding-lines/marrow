"""Bitmap — immutable bit-packed validity buffer with SIMD operations.

Bitmap
------
`Bitmap` wraps an immutable `Buffer` plus `_offset: Int` and `_length: Int`.

  _offset
      Bit offset into `_buffer`.  Enables zero-copy slicing: `slice(n, m)`
      returns a new `Bitmap` that shares the same `Buffer` but starts at a
      later bit position.  Always 0 for freshly-built bitmaps.

  _length
      Number of valid bits.

Copying a `Bitmap` is O(1) — it bumps the `Buffer`'s `ArcPointer` ref-count.

BitmapBuilder
-------------
Mutable counterpart.  Wraps `BufferBuilder` for incremental construction.
Call `finish(length)` to freeze into an immutable `Bitmap`.
"""

from std.sys.info import simd_byte_width
from std.bit import pop_count
import std.math as math
from std.memory import memset

from .buffers import Buffer, BufferBuilder


# ---------------------------------------------------------------------------
# Bitmap — immutable, bit-packed
# ---------------------------------------------------------------------------


struct Bitmap(ImplicitlyCopyable, Movable, Sized, Writable):
    """Immutable bit-packed validity bitmap.

    Wraps a `Buffer` with a `_offset` (bit offset) and `_length` (bit count).
    Copying is O(1); the backing `Buffer` uses `ArcPointer` shared ownership.
    """

    var _buffer: Buffer
    var _offset: Int
    var _length: Int

    fn __init__(out self, buffer: Buffer, offset: Int, length: Int):
        self._buffer = buffer
        self._offset = offset
        self._length = length

    @always_inline
    fn __len__(self) -> Int:
        return self._length

    @always_inline
    fn is_valid(self, index: Int) -> Bool:
        """Return True if bit at (_offset + index) is set (value is valid)."""
        var bit_index = self._offset + index
        return Bool((self._buffer.ptr[bit_index >> 3] >> UInt8(bit_index & 7)) & 1)

    @always_inline
    fn is_null(self, index: Int) -> Bool:
        """Return True if bit at (_offset + index) is unset (value is null)."""
        return not self.is_valid(index)

    fn count_set_bits(self) -> Int:
        """Count set bits in [_offset, _offset + _length) using SIMD popcount."""
        if self._length == 0:
            return 0

        comptime width = simd_byte_width()
        var ptr = self._buffer.unsafe_ptr()

        # Fast path: non-sliced bitmap (_offset == 0).
        # Zero-padding invariant guarantees bits beyond _length are 0, so the
        # SIMD loop can overshoot byte_count into the padded region without
        # overcounting.  No lead or trail correction needed.
        if self._offset == 0:
            var byte_count = (self._length + 7) >> 3
            var count = 0
            var i = 0
            while i < byte_count:
                count += Int(pop_count((ptr + i).load[width=width]()).reduce_add())
                i += width
            return count

        # General path: sliced bitmap (_offset != 0).
        # Shadow bytes beyond byte_end_ceil may be non-zero (original buffer
        # data), so the SIMD loop cannot overshoot safely.  Scalar tail handles
        # remainder bytes; lead/trail subtractions correct boundary bytes.
        var bit_start = self._offset
        var bit_end = self._offset + self._length
        var byte_start = bit_start >> 3
        var byte_end = bit_end >> 3
        var lead = bit_start & 7
        var trail = bit_end & 7
        var byte_end_ceil = byte_end + Int(trail != 0)
        var count = 0
        var i = byte_start
        while i + width <= byte_end_ceil:
            count += Int(pop_count((ptr + i).load[width=width]()).reduce_add())
            i += width
        while i < byte_end_ceil:
            count += Int(pop_count(ptr[i]))
            i += 1
        if lead != 0:
            count -= Int(pop_count(ptr[byte_start] & UInt8((1 << lead) - 1)))
        if trail != 0:
            count -= Int(pop_count(ptr[byte_end] >> UInt8(trail)))
        return count

    fn slice(self, offset: Int, length: Int) -> Bitmap:
        """Return a zero-copy view of `length` bits starting at `offset`."""
        return Bitmap(self._buffer, self._offset + offset, length)

    @always_inline
    fn _prev_aligned_offset(self) -> Int:
        """Return the byte offset of the nearest preceding 64-byte aligned address.

        Backs up from this bitmap's first byte to the nearest 64-byte boundary
        within the buffer: `align_down(_offset >> 3, 64)`.  The number of lead
        bytes between that address and the bitmap's first byte is
        `(_offset >> 3) - _prev_aligned_offset()`.
        """
        return math.align_down(self._offset >> 3, 64)

    # --- Bulk SIMD operations ---
    #
    # Fast path: byte-aligned offsets → pure SIMD over all bytes.
    # Arrow buffers are 64-byte aligned and zero-padded to multiples of 64 B.
    # simd_byte_width() ∈ {16,32,64} all divide 64, so a
    # SIMD load/store never goes out of bounds — no scalar tail needed.
    # For binary ops, padding bytes are zero in both inputs (invariant upheld
    # at construction), so `0 OP 0 = 0` — no masking needed.  For __invert__,
    # padding inverts to 0xFF but count_set_bits/is_valid use _length to bound
    # all reads, so the unmasked shadow bytes are never observed.
    #
    # Fallback: non-byte-aligned offset → correct bit-by-bit loop.

    fn __and__(self, other: Bitmap) -> Bitmap:
        """Return the bitwise AND of self and other."""
        debug_assert(self._length == other._length, "Bitmap lengths must match")
        return self._binop[_and](other)

    fn __or__(self, other: Bitmap) -> Bitmap:
        """Return the bitwise OR of self and other."""
        debug_assert(self._length == other._length, "Bitmap lengths must match")
        return self._binop[_or](other)

    fn __xor__(self, other: Bitmap) -> Bitmap:
        """Return the bitwise XOR of self and other."""
        debug_assert(self._length == other._length, "Bitmap lengths must match")
        return self._binop[_xor](other)

    fn __invert__(self) -> Bitmap:
        """Return the bitwise NOT of this bitmap.

        Reads from the nearest preceding 64-byte aligned address (see
        `_prev_aligned_offset`) so every SIMD load is cache-line aligned.
        The result carries `_offset = lead_bytes * 8 + shift` where `lead_bytes`
        is the byte distance to that boundary (0–63) and `shift = _offset & 7`.

        When `result_offset == 0` (source was already cache-line AND byte aligned),
        `count_set_bits` uses the fast path which requires zero-padding past
        `_length`; the SIMD overshoot bytes are zeroed back.  For any non-zero
        `result_offset`, `count_set_bits` takes the general path (lead/trail
        subtraction), so no fixup is needed.
        """
        var shift = self._offset & 7
        var aligned_offset = self._prev_aligned_offset()
        var lead_bytes = (self._offset >> 3) - aligned_offset
        var src = self._buffer.unsafe_ptr() + aligned_offset
        var byte_count = math.ceildiv(shift + self._length, 8)
        var total_bytes = lead_bytes + byte_count
        var builder = BufferBuilder.alloc(total_bytes)
        var dst = builder.unsafe_ptr()
        comptime width = simd_byte_width()
        comptime assert 64 % width == 0
        var i = 0
        while i < total_bytes:
            (dst + i).store(~(src + i).load[width=width]())
            i += width
        var result_offset = (lead_bytes << 3) + shift
        if result_offset == 0:
            var trail = self._length & 7
            if trail != 0:
                dst[total_bytes - 1] &= UInt8((1 << trail) - 1)
            memset(dst + total_bytes, 0, i - total_bytes)
        return Bitmap(builder.finish(), result_offset, self._length)

    fn and_not(self, other: Bitmap) -> Bitmap:
        """Return self & ~other  (A AND NOT B).

        Useful for null propagation: combine validity where *both* must be valid,
        and exclude elements that are null in `other`.
        """
        debug_assert(self._length == other._length, "Bitmap lengths must match")
        return self._binop[_and_not](other)

    @always_inline
    fn _binop[
        op: fn[W: Int](SIMD[DType.uint8, W], SIMD[DType.uint8, W]) -> SIMD[DType.uint8, W]
    ](self, other: Bitmap) -> Bitmap:
        """Apply a byte-level binary operation across two bitmaps, returning a new Bitmap.

        Selects between two code paths based on the sub-byte offsets
        (`shift = offset & 7`) of both operands:

        **Same shift** (`shift_a == shift_b`, including the byte-aligned case `shift == 0`):
          Both source byte ranges are in the same relative bit alignment, so a
          direct SIMD op suffices — no bit shuffling needed.  The output carries
          `offset = shift_a` so the leading garbage bits are never observed.
          Measured throughput: ~360–370 GElems/s at 10 M bits (Apple M-series).

        **Different shifts** (`shift_a != shift_b`):
          Shifts only the higher-offset operand right by `delta = |shift_a - shift_b|`
          bits using a single overlapping SIMD load (the "shift-one" technique):

              aligned[i] = (hi[i] >> delta) | (hi[i+1] << (8 - delta))

          The lower-offset operand is read directly without any extra loads.
          The output carries `offset = min(shift_a, shift_b)`.
          Measured throughput: ~305 GElems/s at 10 M bits — ~24% faster than
          shifting both operands independently, and ~500× faster than the
          previous bit-by-bit fallback.

        Arrow buffers are 64-byte aligned and zero-padded so SIMD loads past the
        last valid byte are always safe within the padded region.
        """
        comptime width = simd_byte_width()
        comptime assert 64 % width == 0
        var shift_a = self._offset & 7
        var shift_b = other._offset & 7
        var a = self._buffer.unsafe_ptr() + (self._offset >> 3)
        var b = other._buffer.unsafe_ptr() + (other._offset >> 3)

        if shift_a == shift_b:
            # Most common path: identical sub-byte alignment (covers all freshly-built
            # bitmaps where both offsets are 0, and sliced bitmaps with the same offset).
            # Direct SIMD op; output carries the shared offset.
            var total_bits = shift_a + self._length
            var byte_count = math.ceildiv(total_bits, 8)
            var builder = BufferBuilder.alloc[DType.bool](total_bits)
            var dst = builder.ptr
            var i = 0
            while i < byte_count:
                (dst + i).store(op[width]((a + i).load[width=width](), (b + i).load[width=width]()))
                i += width
            return Bitmap(builder.finish(), shift_a, self._length)
        elif shift_a > shift_b:
            # Align `a` to `b`'s layout: shift `a` right by delta.
            # Output offset = shift_b = min(shift_a, shift_b).
            var delta = shift_a - shift_b
            var total_bits = shift_b + self._length
            var byte_count = math.ceildiv(total_bits, 8)
            var builder = BufferBuilder.alloc[DType.bool](total_bits)
            var dst = builder.ptr
            var i = 0
            while i < byte_count:
                var aligned_a = ((a + i).load[width=width]() >> UInt8(delta)) | (
                    (a + i + 1).load[width=width]() << UInt8(8 - delta)
                )
                (dst + i).store(op[width](aligned_a, (b + i).load[width=width]()))
                i += width
            return Bitmap(builder.finish(), shift_b, self._length)
        else:
            # Align `b` to `a`'s layout: shift `b` right by delta.
            # Output offset = shift_a = min(shift_a, shift_b).
            var delta = shift_b - shift_a
            var total_bits = shift_a + self._length
            var byte_count = math.ceildiv(total_bits, 8)
            var builder = BufferBuilder.alloc[DType.bool](total_bits)
            var dst = builder.ptr
            var i = 0
            while i < byte_count:
                var aligned_b = ((b + i).load[width=width]() >> UInt8(delta)) | (
                    (b + i + 1).load[width=width]() << UInt8(8 - delta)
                )
                (dst + i).store(op[width]((a + i).load[width=width](), aligned_b))
                i += width
            return Bitmap(builder.finish(), shift_a, self._length)

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(
            "Bitmap(offset=", self._offset, ", length=", self._length, ")"
        )


@always_inline
fn _and[W: Int](a: SIMD[DType.uint8, W], b: SIMD[DType.uint8, W]) -> SIMD[DType.uint8, W]:
    return a & b


@always_inline
fn _or[W: Int](a: SIMD[DType.uint8, W], b: SIMD[DType.uint8, W]) -> SIMD[DType.uint8, W]:
    return a | b


@always_inline
fn _xor[W: Int](a: SIMD[DType.uint8, W], b: SIMD[DType.uint8, W]) -> SIMD[DType.uint8, W]:
    return a ^ b


@always_inline
fn _and_not[W: Int](a: SIMD[DType.uint8, W], b: SIMD[DType.uint8, W]) -> SIMD[DType.uint8, W]:
    return a & ~b


# ---------------------------------------------------------------------------
# BitmapBuilder — mutable counterpart to Bitmap
# ---------------------------------------------------------------------------


struct BitmapBuilder(Movable):
    """Mutable bit-packed bitmap builder.

    Wraps `BufferBuilder` with bit-level write operations.  Call `finish(length)`
    to freeze into an immutable `Bitmap`.

    Example:
        var bm = BitmapBuilder.alloc(10)
        bm.set_bit(0, True)
        bm.set_bit(5, True)
        var bitmap = bm.finish(10)   # Bitmap of 10 bits, 2 set
    """

    var _builder: BufferBuilder

    fn __init__(out self, var builder: BufferBuilder):
        self._builder = builder^

    @staticmethod
    fn alloc(length: Int) -> BitmapBuilder:
        """Allocate a zero-filled builder for `length` bits."""
        return BitmapBuilder(BufferBuilder.alloc[DType.bool](length))

    @always_inline
    fn unsafe_ptr(self) -> UnsafePointer[UInt8, MutExternalOrigin]:
        """Return the raw mutable byte pointer (for low-level bit operations)."""
        return self._builder.ptr

    @always_inline
    fn set_bit(mut self, index: Int, value: Bool):
        """Set or clear the bit at `index`."""
        self._builder.unsafe_set[DType.bool](index, value)

    fn set_range(mut self, start: Int, length: Int, value: Bool):
        """Set `length` bits starting at `start` to `value`.

        Handles byte-aligned bulk fills via `memset` for the middle bytes,
        with partial masks at the boundaries.
        """
        if length == 0:
            return
        var ptr = self._builder.ptr
        var end = start + length
        var start_byte = start >> 3
        var start_bit = start & 7
        var end_byte = end >> 3
        var end_bit = end & 7

        var fill = UInt8(255 if value else 0)

        if start_byte == end_byte:
            # All bits in one byte
            var mask = UInt8((1 << end_bit) - 1) & (UInt8(0xFF) << UInt8(start_bit))
            if value:
                ptr[start_byte] = ptr[start_byte] | mask
            else:
                ptr[start_byte] = ptr[start_byte] & ~mask
            return

        # Leading partial byte
        if start_bit != 0:
            var mask = UInt8(0xFF) << UInt8(start_bit)
            if value:
                ptr[start_byte] = ptr[start_byte] | mask
            else:
                ptr[start_byte] = ptr[start_byte] & ~mask
            start_byte += 1

        # Trailing partial byte
        if end_bit != 0:
            var mask = UInt8((1 << end_bit) - 1)
            if value:
                ptr[end_byte] = ptr[end_byte] | mask
            else:
                ptr[end_byte] = ptr[end_byte] & ~mask

        # Full middle bytes
        if end_byte > start_byte:
            memset(ptr + start_byte, fill, end_byte - start_byte)

    fn extend(mut self, src: Bitmap, dst_start: Int, length: Int):
        """Copy `length` bits from `src` (from its `_offset`) into self at `dst_start`.

        Replaces the `bitmap_extend` free function.
        """
        var ptr = self._builder.ptr
        for i in range(length):
            var bit = src.is_valid(i)
            var pos = dst_start + i
            var byte_idx = pos >> 3
            var bit_mask = UInt8(1 << (pos & 7))
            if bit:
                ptr[byte_idx] = ptr[byte_idx] | bit_mask
            else:
                ptr[byte_idx] = ptr[byte_idx] & ~bit_mask

    fn resize(mut self, capacity: Int) raises:
        """Resize the underlying buffer to hold `capacity` bits."""
        self._builder.resize[DType.bool](capacity)

    fn finish(mut self, length: Int) -> Bitmap:
        """Freeze the builder into an immutable `Bitmap` of `length` bits.

        The builder is reset to empty and can be reused after this call.
        """
        return Bitmap(self._builder.finish(), 0, length)

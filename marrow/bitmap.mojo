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

from std.sys.info import simd_width_of
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
        """Count set bits in [_offset, _offset + _length) using SIMD popcount.

        Uses lead/trail corrections for boundary bytes so no zero-padding
        invariant is required — works correctly regardless of what lies in
        the buffer outside the bitmap's view.
        """
        if self._length == 0:
            return 0
        comptime width = simd_width_of[DType.uint8]()
        var ptr = self._buffer.unsafe_ptr()
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
    fn _simd_offset_range(self) -> Tuple[Int, Int]:
        """Return (start_byte, end_byte), both 64-byte-aligned, covering all bitmap data.

        start_byte = align_down(_offset >> 3, 64)  — previous cache-line boundary
        end_byte   = align_up(ceildiv(_offset + _length, 8), 64)  — next cache-line boundary

        The range [start_byte, end_byte) is always a multiple of 64 bytes, so a
        SIMD loop with width dividing 64 terminates exactly at end_byte with no
        overshoot.  Bits outside [_offset, _offset+_length) in the result are
        arbitrary and never observed by operations that respect _offset and _length.
        """
        var start = math.align_down(self._offset >> 3, 64)
        var end = math.align_up(math.ceildiv(self._offset + self._length, 8), 64)
        return (start, end)

    # --- Bulk SIMD operations ---
    #
    # Arrow buffers are 64-byte aligned and zero-padded to multiples of 64 B.
    # simd_width_of[DType.uint8]() ∈ {16,32,64} all divide 64, so SIMD loops over ranges
    # returned by `_simd_byte_offset_range` terminate exactly on a cache-line boundary
    # with no overshoot.  Bits outside [_offset, _offset+_length) in results
    # are arbitrary; all operations that consume bitmaps respect _offset and _length.

    fn __and__(self, other: Bitmap) raises -> Bitmap:
        """Return the bitwise AND of self and other."""
        debug_assert(self._length == other._length, "Bitmap lengths must match")
        return self._binop[_and](other)

    fn __or__(self, other: Bitmap) raises -> Bitmap:
        """Return the bitwise OR of self and other."""
        debug_assert(self._length == other._length, "Bitmap lengths must match")
        return self._binop[_or](other)

    fn __xor__(self, other: Bitmap) raises -> Bitmap:
        """Return the bitwise XOR of self and other."""
        debug_assert(self._length == other._length, "Bitmap lengths must match")
        return self._binop[_xor](other)

    fn __invert__(self) -> Bitmap:
        """Return the bitwise NOT of this bitmap.

        Operates over the cache-line-aligned byte range from `_simd_byte_offset_range`,
        so every SIMD load/store is aligned and the loop terminates exactly on a
        64-byte boundary with no overshoot.  The result carries
        `_offset = lead_bytes * 8 + (self._offset & 7)` where `lead_bytes` is
        the byte distance from the aligned start to the bitmap's first byte.
        """
        comptime width = simd_width_of[DType.uint8]()
        comptime assert 64 % width == 0

        var start, end = self._simd_offset_range()
        var total_bytes = end - start
        var builder = BufferBuilder.alloc_uninit(total_bytes)

        var src = self._buffer.unsafe_ptr() + start
        var dst = builder.unsafe_ptr()
        for i in range(0, total_bytes, width):
            (dst + i).store(~(src + i).load[width=width]())

        var new_bit_offset = self._offset - (start << 3)
        return Bitmap(builder.finish(), new_bit_offset, self._length)

    fn and_not(self, other: Bitmap) raises -> Bitmap:
        """Return self & ~other  (A AND NOT B).

        Useful for null propagation: combine validity where *both* must be valid,
        and exclude elements that are null in `other`.
        """
        debug_assert(self._length == other._length, "Bitmap lengths must match")
        return self._binop[_and_not](other)

    @always_inline
    fn _binop[
        op: fn[W: Int](SIMD[DType.uint8, W], SIMD[DType.uint8, W]) -> SIMD[DType.uint8, W]
    ](self, other: Bitmap) raises -> Bitmap:
        """Apply a byte-level binary operation across two bitmaps, returning a new Bitmap.

        Two code paths based on sub-byte alignment (`_offset & 7`):

        Same sub-byte offset (includes both-zero): direct SIMD op on
        corresponding bytes, no bit shifting needed.

        Different sub-byte offset: only `other` is bit-shifted to align
        with `self` via overlapping loads: `(lo >> delta) | (hi << (8 - delta))`.

        Cache-line alignment
        --------------------
        `src_a` is backed up to its previous 64-byte boundary (`lead_a = byte_a & 63`)
        so that `src_a` and `dst` are cache-line aligned.  `src_b` is shifted by the
        same `lead_a` bytes to maintain byte correspondence — it may not be aligned
        itself, but aligning 2 of 3 memory streams (src_a reads + dst writes) avoids
        the majority of cache-line splits.  This yields ~10% throughput gain at L2/L3
        sizes (1M bits) where alignment matters most; at 10M+ the workload is
        memory-bandwidth bound and alignment has negligible effect.

        Arrow buffers are 64-byte aligned at their base and zero-padded to 64-byte
        multiples, so backing up src_a and overreading at the tail are always safe.

        The result carries `new_bit_offset = lead_a * 8 + shift_a` to account for
        the extra leading bytes in the output buffer.
        """
        # width ∈ {16,32,64} bytes; all divide 64, so SIMD loops over
        # 64-byte-aligned ranges terminate exactly with no remainder.
        comptime width = simd_width_of[DType.uint8]()
        comptime assert 64 % width == 0
        # Process one 64-byte cache line per outer iteration, fully
        # unrolled into width-byte SIMD ops at compile time.
        comptime unroll = 64 // width

        # Sub-byte bit offset within the first byte of each operand.
        # Only values 0-7; the byte-level position is handled by the
        # pointer arithmetic below.
        var shift_a = self._offset & 7
        var shift_b = other._offset & 7
        var byte_a = self._offset >> 3
        var byte_b = other._offset >> 3

        # Back src_a up to its previous 64-byte (cache-line) boundary.
        # This makes src_a and dst aligned.  src_b is shifted by the
        # same lead_a bytes so that src_a[i] and src_b[i] still refer
        # to corresponding bitmap bytes.  src_b may not be cache-line
        # aligned itself, but we can only align both when
        # byte_a % 64 == byte_b % 64 — so we settle for 2 of 3 streams.
        var lead_a = byte_a & 63
        var src_a = self._buffer.unsafe_ptr() + (byte_a - lead_a)
        var src_b = other._buffer.unsafe_ptr() + (byte_b - lead_a)

        # total_bytes includes the lead_a prefix and is rounded up to a
        # 64-byte multiple so the SIMD loop terminates exactly.
        # alloc_uninit skips memset_zero — safe because every byte is
        # written by the SIMD loop before the buffer is read.
        var total_bytes = math.align_up(lead_a + math.ceildiv(self._length + shift_a, 8), 64)
        var builder = BufferBuilder.alloc_uninit(total_bytes)
        var dst = builder.unsafe_ptr()

        # The output buffer has lead_a extra bytes at the front, so the
        # result's bit offset must account for them.
        var new_bit_offset = lead_a * 8 + shift_a

        if shift_a == shift_b:
            # Fast path (~445 GElems/s at 10M): both operands share the
            # same sub-byte alignment, so bytes correspond 1:1 and we
            # can apply the op directly without any bit shifting.
            for i in range(0, total_bytes, 64):
                comptime for j in range(unroll):
                    comptime k = j * width
                    (dst + i + k).store(
                        op(
                            (src_a + i + k).load[width=width](),
                            (src_b + i + k).load[width=width](),
                        )
                    )
        else:
            # Shift path (~340 GElems/s at 10M): sub-byte offsets differ.
            # Only other is shifted to align with self — shifting one
            # operand instead of both saves ~25% vs normalizing both to
            # offset 0.  delta is the bit distance to rotate other's
            # bytes rightward; each output byte is assembled from two
            # overlapping source bytes: (lo >> delta) | (hi << (8-delta)).
            # The +1 overread in hi is safe due to Arrow's 64-byte padding.
            var delta = (shift_b - shift_a) & 7
            var rs = UInt8(delta)
            var ls = UInt8(8 - delta)
            for i in range(0, total_bytes, 64):
                comptime for j in range(unroll):
                    comptime k = j * width
                    var a = (src_a + i + k).load[width=width]()
                    var lo = (src_b + i + k).load[width=width]()
                    var hi = (src_b + i + k + 1).load[width=width]()
                    (dst + i + k).store(op(a, (lo >> rs) | (hi << ls)))

        return Bitmap(builder.finish(), new_bit_offset, self._length)


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

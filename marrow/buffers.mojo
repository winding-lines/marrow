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


struct ForeignMemoryOwner(Movable):
    """Keeps foreign memory alive by holding a typed release callback.

    When the last ArcPointer[ForeignMemoryOwner] is dropped, __del__ fires and
    invokes the release function — e.g. the Arrow C Data Interface release callback.
    This is equivalent to Rust's Deallocation::Custom(Arc<dyn Allocation>) pattern.
    """

    var ptr: UnsafePointer[NoneType, MutAnyOrigin]
    var release: fn(UnsafePointer[NoneType, MutAnyOrigin]) -> None

    fn __init__(
        out self,
        ptr: UnsafePointer[NoneType, MutAnyOrigin],
        release: fn(UnsafePointer[NoneType, MutAnyOrigin]) -> None,
    ):
        self.ptr = ptr
        self.release = release

    fn __del__(deinit self):
        self.release(self.ptr)


struct Buffer(Movable):
    var ptr: UnsafePointer[UInt8, MutAnyOrigin]
    var size: Int
    var offset: Int
    # None  → this buffer owns the memory; ptr.free() is called on drop.
    # Some  → a reference-counted ForeignMemoryOwner manages the lifetime;
    #         ptr is NOT freed directly (the owner's release callback handles it).
    var _owner: Optional[ArcPointer[ForeignMemoryOwner]]

    fn __init__(
        out self,
        ptr: UnsafePointer[UInt8, MutAnyOrigin],
        size: Int,
        offset: Int = 0,
        _owner: Optional[ArcPointer[ForeignMemoryOwner]] = None,
    ):
        self.ptr = ptr
        self.size = size
        self.offset = offset
        self._owner = _owner

    @staticmethod
    fn alloc[I: Intable, //, T: DType = DType.uint8](length: I) -> Buffer:
        var size = math.align_up(Int(length) * size_of[T](), 64)
        var ptr = alloc[UInt8](size, alignment=64)
        memset_zero(ptr, size)
        return Buffer(ptr, size)

    @staticmethod
    fn from_values[dtype: DType](*values: Scalar[dtype]) -> Buffer:
        """Build a buffer from a list of values."""
        var buffer = Self.alloc[dtype](len(values))

        for i in range(len(values)):
            buffer.unsafe_set[dtype](i, values[i])

        return buffer^

    @staticmethod
    fn foreign_view[
        I: Intable, //
    ](
        ptr: UnsafePointer[NoneType, MutAnyOrigin],
        length: I,
        dtype: DType,
        owner: ArcPointer[ForeignMemoryOwner],
    ) -> Buffer:
        """Create a non-owning view into foreign memory.

        The caller passes an ArcPointer[ForeignMemoryOwner] that keeps the
        source allocation alive for as long as any buffer (or bitmap) derived
        from it exists.  When the last such buffer is dropped the owner's
        release callback fires automatically.
        """
        return Buffer(
            ptr.bitcast[UInt8](),
            math.align_up(Int(length) * dynamic_size_of(dtype), 64),
            _owner=Optional(owner),
        )

    @always_inline
    fn get_ptr_at(self, index: Int) -> UnsafePointer[UInt8, MutAnyOrigin]:
        return self.ptr + index

    fn resize[I: Intable, //, T: DType = DType.uint8](mut self, length: I):
        comptime elem_bytes = size_of[T]()
        var new = Buffer.alloc[T](length)
        memcpy(
            dest=new.ptr,
            src=self.get_ptr_at(self.offset * elem_bytes),
            count=min(
                self.size - self.offset * elem_bytes, Int(length) * elem_bytes
            ),
        )
        swap(self.ptr, new.ptr)
        swap(self.size, new.size)
        swap(self._owner, new._owner)
        self.offset = 0

    fn __del__(deinit self):
        if not self._owner:
            self.ptr.free()

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


struct Bitmap(Movable, Stringable):

    """Hold information about the null records in an array."""

    var buffer: Buffer
    var offset: Int

    @staticmethod
    fn alloc[I: Intable](length: I) -> Bitmap:
        var byte_length = math.ceildiv(Int(length), 8)
        return Bitmap(Buffer.alloc(byte_length))

    @staticmethod
    fn foreign_view[
        I: Intable, //
    ](
        ptr: UnsafePointer[NoneType, MutAnyOrigin],
        length: I,
        owner: ArcPointer[ForeignMemoryOwner],
    ) -> Bitmap:
        var byte_length = math.ceildiv(Int(length), 8)
        var buffer = Buffer.foreign_view(ptr, byte_length, DType.uint8, owner)
        return Bitmap(buffer^)

    fn as_buffer(deinit self) -> Buffer:
        return self.buffer^

    fn __init__(out self, var buffer: Buffer, offset: Int = 0):
        self.buffer = buffer^
        self.offset = offset

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
        var new = Bitmap.alloc(length)
        for i in range(Int(length)):
            new.unsafe_set(i, self.unsafe_get(i + start))
        swap(self.buffer, new.buffer)
        self.offset = 0

    fn bit_count(self) -> Int:
        """The number of bits with value 1 in the Bitmap."""
        var start = 0
        var count = 0
        while start < self.buffer.size:
            if self.buffer.size - start > simd_width:
                count += (
                    self.buffer.get_ptr_at(start)
                    .load[width=simd_width]()
                    .reduce_bit_count()
                )
                start += simd_width
            else:
                count += (
                    self.buffer.get_ptr_at(start)
                    .load[width=1]()
                    .reduce_bit_count()
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
            var loaded = self.buffer.get_ptr_at(index).load[width=1]()
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

            @parameter
            for width_index in range(len(simd_widths)):
                comptime width = simd_widths[width_index]
                if self.size() - index >= width:
                    var loaded = self.buffer.get_ptr_at(index).load[
                        width=width
                    ]()
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
                self.buffer.get_ptr_at(start_index),
                value=UInt8(byte_value),
                count=end_index - start_index,
            )

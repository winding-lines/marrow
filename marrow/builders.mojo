"""Array builders for constructing Arrow arrays incrementally.

Each builder type parallels an immutable array type:
- `BoolBuilder`       → `BoolArray`
- `PrimitiveBuilder`  → `PrimitiveArray`
- `StringBuilder`     → `StringArray`
- `ListBuilder`       → `ListArray`
- `StructBuilder`     → `StructArray`

Builders own their buffers directly (no ArcPointer wrapping), making
mutation straightforward.  When building is complete, call `freeze()`
to consume the builder and return the corresponding immutable array.

Example
-------
    var b = PrimitiveBuilder[int64](capacity=1024)
    b.append(42)
    b.unsafe_append_null()
    var arr = b^.freeze()   # PrimitiveArray[int64]
"""

from memory import memcpy
from sys import size_of
from .buffers import Buffer, BufferBuilder, Bitmap, BitmapBuilder, MemorySpace
from .dtypes import *
from .arrays import (
    Array,
    BoolArray,
    PrimitiveArray,
    StringArray,
    ListArray,
    FixedSizeListArray,
    StructArray,
)


struct BoolBuilder(Movable, Sized):
    """Builder for `BoolArray`.  Owns bitmap and values directly."""

    var length: Int
    var offset: Int
    var capacity: Int
    var bitmap: BitmapBuilder
    var values: BitmapBuilder

    fn __init__(out self, capacity: Int = 0, offset: Int = 0):
        self.capacity = capacity
        self.length = 0
        self.offset = offset
        self.bitmap = BitmapBuilder.alloc(capacity)
        self.values = BitmapBuilder.alloc(capacity)

    fn resize(mut self, capacity: Int):
        self.bitmap.resize(capacity)
        self.values.resize(capacity)
        self.capacity = capacity

    @always_inline
    fn __len__(self) -> Int:
        return self.length

    @always_inline
    fn unsafe_set(mut self, index: Int, value: Bool):
        self.bitmap.unsafe_set(index + self.offset, True)
        self.values.unsafe_set(index + self.offset, value)

    @always_inline
    fn unsafe_append(mut self, value: Bool):
        self.unsafe_set(self.length, value)
        self.length += 1

    @always_inline
    fn unsafe_append_null(mut self):
        self.bitmap.unsafe_set(self.length + self.offset, False)
        self.length += 1

    fn __setitem__(mut self, index: Int, value: Bool) raises:
        if index < 0 or index >= self.length:
            raise Error(
                "index "
                + String(index)
                + " out of bounds for length "
                + String(self.length)
            )
        self.unsafe_set(index, value)

    fn shrink_to_fit(mut self):
        """Shrink buffers to exact length and normalize offset to 0 in place."""
        if self.length == self.capacity and self.offset == 0:
            return

        self.values.resize(self.length, self.offset)
        self.bitmap.resize(self.length, self.offset)
        self.offset = 0
        self.capacity = self.length

    fn freeze(deinit self) -> BoolArray:
        """Shrink buffers to exact length and return as an immutable array."""
        self.shrink_to_fit()
        return BoolArray(
            length=self.length,
            offset=0,
            bitmap=self.bitmap^.freeze(),
            values=self.values^.freeze(),
        )

    fn append(mut self, value: Bool):
        if self.length >= self.capacity:
            self.resize(max(self.capacity * 2, self.length + 1))
        self.unsafe_append(value)

    fn extend(mut self, values: List[Bool]):
        if self.__len__() + len(values) >= self.capacity:
            self.resize(self.capacity + len(values))
        for value in values:
            self.unsafe_append(value)


struct PrimitiveBuilder[T: DataType](Movable, Sized):
    """Builder for `PrimitiveArray[T]`.  Owns bitmap and buffer directly."""

    comptime dtype = Self.T
    comptime scalar = Scalar[Self.T.native]

    var length: Int
    var offset: Int
    var capacity: Int
    var bitmap: BitmapBuilder
    var buffer: BufferBuilder

    fn __init__(out self, capacity: Int = 0, offset: Int = 0):
        self.capacity = capacity
        self.length = 0
        self.offset = offset
        self.bitmap = BitmapBuilder.alloc(capacity)
        self.buffer = BufferBuilder.alloc[Self.T.native](capacity)

    fn resize(mut self, capacity: Int):
        self.bitmap.resize(capacity)
        self.buffer.resize[Self.T.native](capacity)
        self.capacity = capacity

    @always_inline
    fn __len__(self) -> Int:
        return self.length

    @always_inline
    fn unsafe_set(mut self, index: Int, value: Self.scalar):
        self.bitmap.unsafe_set(index + self.offset, True)
        self.buffer.unsafe_set[Self.T.native](index + self.offset, value)

    @always_inline
    fn unsafe_append(mut self, value: Self.scalar):
        self.unsafe_set(self.length, value)
        self.length += 1

    @always_inline
    fn unsafe_append_null(mut self):
        self.bitmap.unsafe_set(self.length + self.offset, False)
        self.length += 1

    fn __setitem__(mut self, index: Int, value: Self.scalar) raises:
        if index < 0 or index >= self.length:
            raise Error(
                "index "
                + String(index)
                + " out of bounds for length "
                + String(self.length)
            )
        self.unsafe_set(index, value)

    fn shrink_to_fit(mut self):
        """Shrink buffers to exact length and normalize offset to 0 in place."""
        if self.length == self.capacity and self.offset == 0:
            return

        self.buffer.offset = self.offset
        self.buffer.resize[Self.T.native](self.length)
        self.bitmap.resize(self.length, self.offset)
        self.offset = 0
        self.capacity = self.length

    fn freeze(deinit self) -> PrimitiveArray[Self.T]:
        """Shrink buffers to exact length and return as an immutable array."""
        self.shrink_to_fit()
        return PrimitiveArray[Self.T](
            length=self.length,
            offset=0,
            bitmap=self.bitmap^.freeze(),
            buffer=self.buffer^.freeze(),
        )

    fn append(mut self, value: Self.scalar):
        if self.length >= self.capacity:
            self.resize(max(self.capacity * 2, self.length + 1))
        self.unsafe_append(value)

    fn extend(mut self, values: List[self.scalar]):
        if self.__len__() + len(values) >= self.capacity:
            self.resize(self.capacity + len(values))
        for value in values:
            self.unsafe_append(value)


struct StringBuilder(Movable, Sized):
    """Builder for `StringArray`.  Owns bitmap, offsets, and values directly."""

    var length: Int
    var offset: Int
    var capacity: Int
    var bitmap: BitmapBuilder
    var offsets: BufferBuilder
    var values: BufferBuilder

    fn __init__(out self, capacity: Int = 0):
        """Create a StringBuilder with the given capacity."""
        self.capacity = capacity
        self.length = 0
        self.offset = 0
        self.bitmap = BitmapBuilder.alloc(capacity)
        self.offsets = BufferBuilder.alloc[DType.uint32](capacity + 1)
        self.values = BufferBuilder.alloc[DType.uint8](capacity)
        self.offsets.unsafe_set[DType.uint32](0, 0)

    fn __len__(self) -> Int:
        return self.length

    fn resize(mut self, capacity: Int):
        """Resize the bitmap and offsets buffers to the given capacity."""
        self.bitmap.resize(capacity)
        self.offsets.resize[DType.uint32](capacity + 1)
        self.capacity = capacity

    fn unsafe_append(mut self, value: String):
        """Append a string value without bounds checking."""
        var index = self.length
        var last_offset = self.offsets.ptr.bitcast[UInt32]()[index + self.offsets.offset]
        var next_offset = last_offset + UInt32(len(value))
        self.length += 1
        self.bitmap.unsafe_set(index, True)
        self.offsets.unsafe_set[DType.uint32](index + 1, next_offset)
        self.values.resize[DType.uint8](next_offset)
        var dst_address = self.values.ptr + Int(last_offset)
        var src_address = value.unsafe_ptr()
        memcpy(dest=dst_address, src=src_address, count=len(value))

    fn shrink_to_fit(mut self):
        """Shrink buffers to exact length and normalize offset to 0 in place."""
        if self.length == self.capacity and self.offset == 0:
            return

        var start_byte = Int(self.offsets.unsafe_get[DType.uint32](self.offset))
        var end_byte = Int(
            self.offsets.unsafe_get[DType.uint32](self.offset + self.length)
        )

        # Compact offsets, then rebase by subtracting start_byte
        self.offsets.offset = self.offset
        self.offsets.resize[DType.uint32](self.length + 1)
        for i in range(self.length + 1):
            var off = Int(self.offsets.unsafe_get[DType.uint32](i))
            self.offsets.unsafe_set[DType.uint32](i, UInt32(off - start_byte))

        self.values.offset = start_byte
        self.values.resize(end_byte - start_byte)
        self.bitmap.resize(self.length, self.offset)
        self.offset = 0
        self.capacity = self.length

    fn freeze(deinit self) -> StringArray:
        """Shrink buffers to exact length and return as an immutable array."""
        self.shrink_to_fit()
        return StringArray(
            length=self.length,
            offset=0,
            bitmap=self.bitmap^.freeze(),
            offsets=self.offsets^.freeze(),
            values=self.values^.freeze(),
        )


struct ListBuilder(Movable, Sized):
    """Builder for `ListArray`.  Owns bitmap and offsets directly."""

    var dtype: DataType
    var length: Int
    var offset: Int
    var capacity: Int
    var bitmap: BitmapBuilder
    var offsets: BufferBuilder
    var values: Array[MemorySpace.CPU]

    fn __init__(
        out self,
        var dtype: DataType,
        length: Int,
        offset: Int,
        capacity: Int,
        var bitmap: BitmapBuilder,
        var offsets: BufferBuilder,
        var values: Array[MemorySpace.CPU],
    ):
        """Builder constructor for creating a mutable ListBuilder."""
        self.dtype = dtype^
        self.length = length
        self.offset = offset
        self.capacity = capacity
        self.bitmap = bitmap^
        self.offsets = offsets^
        self.values = values^

    @staticmethod
    fn from_values(
        var values: Array[MemorySpace.CPU], capacity: Int = 1
    ) raises -> ListBuilder:
        """Create a ListBuilder wrapping the given values.

        Args:
            values: Array to use as the first element in the ListArray.
            capacity: The capacity of the ListArray.
        """
        var length = values.length

        var bitmap = BitmapBuilder.alloc(capacity)
        bitmap.unsafe_set(0, True)
        var offsets = BufferBuilder.alloc[DType.uint32](capacity + 1)
        offsets.unsafe_set[DType.uint32](0, 0)
        offsets.unsafe_set[DType.uint32](1, UInt32(length))

        var list_dtype = list_(values.dtype.copy())
        return ListBuilder(
            dtype=list_dtype^,
            length=1,
            offset=0,
            capacity=capacity,
            bitmap=bitmap^,
            offsets=offsets^,
            values=values^,
        )

    fn __len__(self) -> Int:
        return self.length

    fn unsafe_append(mut self, is_valid: Bool):
        self.bitmap.unsafe_set(self.length, is_valid)
        self.offsets.unsafe_set[DType.uint32](
            self.length + 1, UInt32(self.values.length)
        )
        self.length += 1

    fn shrink_to_fit(mut self):
        """Shrink buffers to exact length and normalize offset to 0 in place."""
        if self.length == self.capacity and self.offset == 0:
            return

        self.offsets.offset = self.offset
        self.offsets.resize[DType.uint32](self.length + 1)
        self.bitmap.resize(self.length, self.offset)
        self.offset = 0
        self.capacity = self.length

    fn freeze(deinit self) -> ListArray:
        """Shrink buffers to exact length and return as an immutable array."""
        self.shrink_to_fit()
        return ListArray(
            dtype=self.dtype^,
            length=self.length,
            offset=0,
            bitmap=self.bitmap^.freeze(),
            offsets=self.offsets^.freeze(),
            values=self.values^,
        )


struct FixedSizeListBuilder(Movable, Sized):
    """Builder for `FixedSizeListArray`.  Owns bitmap directly."""

    var dtype: DataType
    var length: Int
    var offset: Int
    var capacity: Int
    var bitmap: BitmapBuilder
    var values: Array[MemorySpace.CPU]

    fn __init__(
        out self,
        var dtype: DataType,
        length: Int,
        offset: Int,
        capacity: Int,
        var bitmap: BitmapBuilder,
        var values: Array[MemorySpace.CPU],
    ):
        self.dtype = dtype^
        self.length = length
        self.offset = offset
        self.capacity = capacity
        self.bitmap = bitmap^
        self.values = values^

    @staticmethod
    fn from_values(
        var values: Array[MemorySpace.CPU], list_size: Int, capacity: Int = 1
    ) raises -> FixedSizeListBuilder:
        """Create a FixedSizeListBuilder wrapping the given values.

        Args:
            values: Array to use as the child values.
            list_size: Fixed number of elements per list.
            capacity: The capacity of the builder.
        """
        if values.length % list_size != 0:
            raise Error(
                "values length {} not divisible by list_size {}".format(
                    values.length, list_size
                )
            )
        var n_lists = values.length // list_size

        var bitmap = BitmapBuilder.alloc(max(n_lists, capacity))
        bitmap.unsafe_range_set(0, n_lists, True)

        var fsl_dtype = fixed_size_list_(values.dtype.copy(), list_size)
        return FixedSizeListBuilder(
            dtype=fsl_dtype^,
            length=n_lists,
            offset=0,
            capacity=max(n_lists, capacity),
            bitmap=bitmap^,
            values=values^,
        )

    fn __len__(self) -> Int:
        return self.length

    fn unsafe_append(mut self, is_valid: Bool):
        self.bitmap.unsafe_set(self.length, is_valid)
        self.length += 1

    fn shrink_to_fit(mut self):
        if self.length == self.capacity and self.offset == 0:
            return
        self.bitmap.resize(self.length, self.offset)
        self.offset = 0
        self.capacity = self.length

    fn freeze(deinit self) -> FixedSizeListArray:
        """Shrink buffers to exact length and return as an immutable array."""
        self.shrink_to_fit()
        return FixedSizeListArray(
            dtype=self.dtype^,
            length=self.length,
            offset=0,
            bitmap=self.bitmap^.freeze(),
            values=self.values^,
        )


struct StructBuilder(Movable, Sized):
    """Builder for `StructArray`.  Owns bitmap directly."""

    var dtype: DataType
    var length: Int
    var capacity: Int
    var bitmap: BitmapBuilder
    var children: List[Array[MemorySpace.CPU]]

    fn __init__(
        out self,
        var fields: List[Field],
        capacity: Int = 0,
    ):
        var bitmap = BitmapBuilder.alloc(capacity)
        bitmap.unsafe_range_set(0, capacity, True)

        self.dtype = struct_(fields)
        self.capacity = capacity
        self.length = 0
        self.bitmap = bitmap^
        self.children = []

    fn __len__(self) -> Int:
        return self.length

    fn shrink_to_fit(mut self):
        """Shrink bitmap to exact length in place."""
        if self.length == self.capacity:
            return

        self.bitmap.resize(self.length)
        self.capacity = self.length

    fn freeze(deinit self) -> StructArray:
        """Shrink bitmap to exact length and return as an immutable array."""
        self.shrink_to_fit()
        return StructArray(
            dtype=self.dtype^,
            length=self.length,
            bitmap=self.bitmap^.freeze(),
            children=self.children^,
        )


# --- Factory functions ---


fn array[T: DataType]() -> PrimitiveArray[T]:
    """Create an empty immutable primitive array."""
    return PrimitiveBuilder[T](0).freeze()


fn array[T: DataType](values: List[Optional[Int]]) -> PrimitiveArray[T]:
    """Create an immutable primitive array from a list of values.

    `None` entries become null.
    """
    var a = PrimitiveBuilder[T](len(values))
    for value in values:
        if value:
            a.unsafe_append(Scalar[T.native](value.value()))
        else:
            a.unsafe_append_null()
    return a^.freeze()


fn array(values: List[Optional[Bool]]) -> BoolArray:
    """Create an immutable bool array from a list of values.

    `None` entries become null.
    """
    var a = BoolBuilder(len(values))
    for value in values:
        if value:
            a.unsafe_append(value.value())
        else:
            a.unsafe_append_null()
    return a^.freeze()


fn nulls[T: DataType](size: Int) -> PrimitiveArray[T]:
    """Create an immutable PrimitiveArray where all values are null.

    Parameters:
        T: The DataType of the array elements.

    Args:
        size: The number of null elements.

    Returns:
        A PrimitiveArray[T] with all values null.
    """
    var a = PrimitiveBuilder[T](capacity=size)
    a.length = size
    return a^.freeze()


fn arange[T: DataType](start: Int, end: Int) -> PrimitiveArray[T]:
    """Create an immutable integer array from start to end (exclusive).

    Parameters:
        T: An integer DataType.

    Args:
        start: The starting value (inclusive).
        end: The ending value (exclusive).

    Returns:
        A PrimitiveArray[T] with values [start, start+1, ..., end-1].
    """
    comptime assert T.is_integer(), "arange() only supports integer DataTypes"
    var a = PrimitiveBuilder[T](end - start)
    for i in range(start, end):
        a.unsafe_append(Scalar[T.native](i))
    return a^.freeze()

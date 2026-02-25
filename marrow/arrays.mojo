from memory import ArcPointer, memcpy
from .buffers import Buffer, Bitmap
from .dtypes import *


@fieldwise_init
struct Array(Copyable, Movable, Stringable):
    """Array is the lower level abstraction directly usable by the library consumer.

    Equivalent with https://github.com/apache/arrow/blob/7184439dea96cd285e6de00e07c5114e4919a465/cpp/src/arrow/array/data.h#L62-L84.
    """

    var dtype: DataType
    var length: Int
    var bitmap: ArcPointer[Bitmap]
    var buffers: List[ArcPointer[Buffer]]
    var children: List[ArcPointer[Array]]
    var offset: Int

    @staticmethod
    fn from_buffer[dtype: DataType](var buffer: Buffer, length: Int) -> Array:
        """Build an Array from a buffer where all the values are not null."""
        var bitmap = Bitmap.alloc(length)
        bitmap.unsafe_range_set(0, length, True)
        return Array(
            dtype=materialize[dtype](),
            length=length,
            bitmap=ArcPointer(bitmap^),
            buffers=[ArcPointer(buffer^)],
            children=[],
            offset=0,
        )

    @implicit
    fn __init__[T: DataType](out self, array: PrimitiveArray[T]):
        self.dtype = materialize[T]()
        self.length = array.length
        self.offset = array.offset
        self.bitmap = array.bitmap
        self.buffers = [array.buffer]
        self.children = []

    @implicit
    fn __init__(out self, array: StringArray):
        self.dtype = materialize[string]()
        self.length = array.length
        self.offset = array.offset
        self.bitmap = array.bitmap
        self.buffers = [array.offsets, array.values]
        self.children = []

    @implicit
    fn __init__(out self, array: ListArray):
        self.dtype = array.dtype.copy()
        self.length = array.length
        self.offset = array.offset
        self.bitmap = array.bitmap
        self.buffers = [array.offsets]
        self.children = [array.values]

    @implicit
    fn __init__(out self, array: StructArray):
        self.dtype = array.dtype.copy()
        self.length = array.length
        self.offset = 0
        self.bitmap = array.bitmap
        self.buffers = []
        self.children = array.children.copy()

    fn __init__(out self, *, copy: Self):
        self.dtype = copy.dtype.copy()
        self.length = copy.length
        self.bitmap = copy.bitmap
        self.buffers = copy.buffers.copy()
        self.children = copy.children.copy()
        self.offset = copy.offset

    fn __init__(out self, *, deinit take: Self):
        self.dtype = take.dtype^
        self.length = take.length
        self.bitmap = take.bitmap^
        self.buffers = take.buffers^
        self.children = take.children^
        self.offset = take.offset

    fn is_valid(self, index: Int) -> Bool:
        return self.bitmap[].unsafe_get(index + self.offset)

    fn __str__(self) -> String:
        from .pretty import ArrayPrinter

        var printer = ArrayPrinter()
        try:
            printer.visit(self)
        except:
            pass
        return printer^.finish()

    fn as_primitive[T: DataType](self) raises -> PrimitiveArray[T]:
        return PrimitiveArray[T](self)

    fn as_int8(self) raises -> Int8Array:
        return Int8Array(self)

    fn as_int16(self) raises -> Int16Array:
        return Int16Array(self)

    fn as_int32(self) raises -> Int32Array:
        return Int32Array(self)

    fn as_int64(self) raises -> Int64Array:
        return Int64Array(self)

    fn as_uint8(self) raises -> UInt8Array:
        return UInt8Array(self)

    fn as_uint16(self) raises -> UInt16Array:
        return UInt16Array(self)

    fn as_uint32(self) raises -> UInt32Array:
        return UInt32Array(self)

    fn as_uint64(self) raises -> UInt64Array:
        return UInt64Array(self)

    fn as_float32(self) raises -> Float32Array:
        return Float32Array(self)

    fn as_float64(self) raises -> Float64Array:
        return Float64Array(self)

    fn as_string(self) raises -> StringArray:
        return StringArray(self)

    fn as_list(self) raises -> ListArray:
        return ListArray(self)

    fn as_struct(self) raises -> StructArray:
        return StructArray(data=self)

    fn append_to_array(
        deinit self: Array, mut combined: Array, start: Int
    ) -> Int:
        """Append the content self to the combined array, consumes self.

        Args:
            combined: Array to append to.
            start: Position where to append.

        Returns:
            The new start position.
        """
        combined.bitmap[].extend(self.bitmap[], start, self.length)
        combined.buffers.extend(self.buffers^)
        combined.children.extend(self.children^)
        return start + self.length


struct PrimitiveArray[T: DataType](Movable, Sized):
    """An Arrow array of primitive types."""

    comptime dtype = Self.T
    comptime scalar = Scalar[Self.T.native]
    var length: Int
    var offset: Int
    var capacity: Int
    var bitmap: ArcPointer[Bitmap]
    var buffer: ArcPointer[Buffer]

    fn __init__(out self, ref data: Array, offset: Int = 0) raises:
        # TODO(kszucs): put a dtype constraint here
        if data.dtype != materialize[Self.T]():
            raise Error(
                "Unexpected dtype '"
                + String(data.dtype)
                + "' instead of '"
                + String(materialize[Self.T]())
                + "'."
            )
        elif len(data.buffers) != 1:
            raise Error("PrimitiveArray requires exactly one buffer")

        self.offset = data.offset + offset
        self.length = data.length
        self.capacity = data.length
        self.bitmap = data.bitmap
        self.buffer = data.buffers[0]

    fn __init__(out self, capacity: Int = 0, offset: Int = 0):
        self.capacity = capacity
        self.length = 0
        self.offset = offset
        self.bitmap = ArcPointer(Bitmap.alloc(capacity))
        self.buffer = ArcPointer(Buffer.alloc[Self.T.native](capacity))

    fn grow(mut self, capacity: Int):
        self.bitmap[].grow(capacity)
        self.buffer[].grow[Self.T.native](capacity)
        self.capacity = capacity

    @always_inline
    fn __len__(self) -> Int:
        return self.length

    @always_inline
    fn is_valid(self, index: Int) -> Bool:
        return self.bitmap[].unsafe_get(index + self.offset)

    @always_inline
    fn unsafe_get(self, index: Int) -> Self.scalar:
        return self.buffer[].unsafe_get[Self.T.native](index + self.offset)

    @always_inline
    fn unsafe_set(mut self, index: Int, value: Self.scalar):
        self.bitmap[].unsafe_set(index + self.offset, True)
        self.buffer[].unsafe_set[Self.T.native](index + self.offset, value)

    @always_inline
    fn unsafe_append(mut self, value: Self.scalar):
        self.unsafe_set(self.length, value)
        self.length += 1

    @always_inline
    fn unsafe_append_null(mut self):
        self.bitmap[].unsafe_set(self.length + self.offset, False)
        self.length += 1

    @staticmethod
    fn nulls(size: Int) raises -> PrimitiveArray[Self.T]:
        """Creates a new PrimitiveArray filled with null values."""
        var bitmap = Bitmap.alloc(size)
        bitmap.unsafe_range_set(0, size, False)
        var buffer = Buffer.alloc[Self.T.native](size)
        return PrimitiveArray[Self.T](
            data=Array(
                dtype=materialize[Self.T](),
                length=size,
                bitmap=ArcPointer(bitmap^),
                buffers=[ArcPointer(buffer^)],
                children=[],
                offset=0,
            ),
        )

    fn append(mut self, value: Self.scalar):
        if self.length >= self.capacity:
            self.grow(max(self.capacity * 2, self.length + 1))
        self.unsafe_append(value)

    # fn append(mut self, value: Optional[Self.scalar]):

    fn extend(mut self, values: List[self.scalar]):
        if self.__len__() + len(values) >= self.capacity:
            self.grow(self.capacity + len(values))
        for value in values:
            self.unsafe_append(value)

    fn null_count(self) -> Int:
        """Returns the number of null values in the array."""
        var valid_count = self.bitmap[].buffer.bit_count()
        return self.length - valid_count


comptime BoolArray = PrimitiveArray[bool_]
comptime Int8Array = PrimitiveArray[int8]
comptime Int16Array = PrimitiveArray[int16]
comptime Int32Array = PrimitiveArray[int32]
comptime Int64Array = PrimitiveArray[int64]
comptime UInt8Array = PrimitiveArray[uint8]
comptime UInt16Array = PrimitiveArray[uint16]
comptime UInt32Array = PrimitiveArray[uint32]
comptime UInt64Array = PrimitiveArray[uint64]
comptime Float32Array = PrimitiveArray[float32]
comptime Float64Array = PrimitiveArray[float64]


struct StringArray(Movable, Sized):
    var length: Int
    var offset: Int
    var capacity: Int
    var bitmap: ArcPointer[Bitmap]
    var offsets: ArcPointer[Buffer]
    var values: ArcPointer[Buffer]

    fn __init__(out self, ref data: Array) raises:
        if data.dtype != materialize[string]():
            raise Error(
                "Unexpected dtype '"
                + String(data.dtype)
                + "' instead of 'string'."
            )
        elif len(data.buffers) != 2:
            raise Error("StringArray requires exactly two buffers")

        self.length = data.length
        self.offset = data.offset
        self.capacity = data.length
        self.bitmap = data.bitmap
        self.offsets = data.buffers[0]
        self.values = data.buffers[1]

    fn __init__(out self, capacity: Int = 0):
        # TODO(kszucs): initial values capacity should be either 0 or some value received from the user
        self.capacity = capacity
        self.length = 0
        self.offset = 0
        self.bitmap = ArcPointer(Bitmap.alloc(capacity))
        self.offsets = ArcPointer(Buffer.alloc[DType.uint32](capacity + 1))
        self.values = ArcPointer(Buffer.alloc[DType.uint8](capacity))
        self.offsets[].unsafe_set[DType.uint32](0, 0)

    fn __len__(self) -> Int:
        return self.length

    fn grow(mut self, capacity: Int):
        self.bitmap[].grow(capacity)
        self.offsets[].grow[DType.uint32](capacity + 1)
        self.capacity = capacity

    # fn shrink_to_fit(out self):

    fn is_valid(self, index: Int) -> Bool:
        return self.bitmap[].unsafe_get(index)

    fn unsafe_append(mut self, value: String):
        # todo(kszucs): use unsafe set
        var index = self.length
        var last_offset = self.offsets[].unsafe_get[DType.uint32](index)
        var next_offset = last_offset + UInt32(len(value))
        self.length += 1
        self.bitmap[].unsafe_set(index, True)
        self.offsets[].unsafe_set[DType.uint32](index + 1, next_offset)
        self.values[].grow[DType.uint8](next_offset)
        var dst_address = self.values[].get_ptr_at(Int(last_offset))
        var src_address = value.unsafe_ptr()
        memcpy(dest=dst_address, src=src_address, count=len(value))

    fn unsafe_get(self, index: UInt) -> StringSlice[ImmutAnyOrigin]:
        var offset_idx = Int(index) + self.offset
        var start_offset = self.offsets[].unsafe_get[DType.uint32](offset_idx)
        var end_offset = self.offsets[].unsafe_get[DType.uint32](offset_idx + 1)
        var address = self.values[].get_ptr_at(Int(start_offset))
        var length = Int(end_offset) - Int(start_offset)
        return StringSlice(
            unsafe_from_utf8=Span[Byte](
                ptr=address.mut_cast[False](), length=length
            )
        )

    fn unsafe_set(mut self, index: Int, value: String) raises:
        var start_offset = self.offsets[].unsafe_get[DType.int32](index)
        var end_offset = self.offsets[].unsafe_get[DType.int32](index + 1)
        var length = Int(end_offset - start_offset)

        if length != len(value):
            raise Error(
                "String length mismatch, inplace update must have the same"
                " length"
            )

        var dst_address = self.values[].get_ptr_at(Int(start_offset))
        var src_address = value.unsafe_ptr()
        memcpy(dest=dst_address, src=src_address, count=length)


struct ListArray(Movable, Sized):
    var dtype: DataType
    var length: Int
    var offset: Int
    var capacity: Int
    var bitmap: ArcPointer[Bitmap]
    var offsets: ArcPointer[Buffer]
    var values: ArcPointer[Array]

    fn __init__(out self, ref data: Array) raises:
        if not data.dtype.is_list():
            raise Error(
                "Unexpected dtype " + String(data.dtype) + " instead of 'list'"
            )
        elif len(data.buffers) != 1:
            raise Error("ListArray requires exactly one buffer")
        elif len(data.children) != 1:
            raise Error("ListArray requires exactly one child array")

        self.dtype = data.dtype.copy()
        self.length = data.length
        self.offset = data.offset
        self.capacity = data.length
        self.bitmap = data.bitmap
        self.offsets = data.buffers[0]
        self.values = data.children[0]

    @staticmethod
    fn from_values(var values: Array, capacity: Int = 1) raises -> ListArray:
        """Create a ListArray wrapping the given values as its first element.

        Default capacity is at least 1 to accomodate the values.

        Args:
            values: Array to use as the first element in the ListArray.
            capacity: The capacity of the ListArray.
        """
        var length = values.length

        var bitmap = Bitmap.alloc(capacity)
        bitmap.unsafe_set(0, True)
        var offsets = Buffer.alloc[DType.uint32](capacity + 1)
        offsets.unsafe_set[DType.uint32](0, 0)
        offsets.unsafe_set[DType.uint32](1, UInt32(length))

        var list_dtype = list_(values.dtype.copy())
        var data = Array(
            dtype=list_dtype^,
            length=1,
            bitmap=ArcPointer(bitmap^),
            buffers=[ArcPointer(offsets^)],
            children=[ArcPointer(values^)],
            offset=0,
        )
        return ListArray(data^)

    fn __len__(self) -> Int:
        return self.length

    fn is_valid(self, index: Int) -> Bool:
        return self.bitmap[].unsafe_get(index)

    fn unsafe_append(mut self, is_valid: Bool):
        self.bitmap[].unsafe_set(self.length, is_valid)
        self.offsets[].unsafe_set[DType.uint32](
            self.length + 1, UInt32(self.values[].length)
        )
        self.length += 1

    fn unsafe_get(self, index: Int, out array_data: Array) raises:
        """Access the value at a given index in the list array.

        Use an out argument to allow the caller to re-use memory while iterating over a pyarrow structure.
        """
        var start = Int(
            self.offsets[].unsafe_get[DType.int32](self.offset + index)
        )
        var end = Int(
            self.offsets[].unsafe_get[DType.int32](self.offset + index + 1)
        )
        ref first_child = self.values[]
        return Array(
            dtype=first_child.dtype.copy(),
            bitmap=first_child.bitmap,
            buffers=first_child.buffers.copy(),
            offset=start,
            length=end - start,
            children=first_child.children.copy(),
        )


struct StructArray(Movable, Sized):
    var dtype: DataType
    var length: Int
    var capacity: Int
    var bitmap: ArcPointer[Bitmap]
    var children: List[ArcPointer[Array]]

    fn __init__(
        out self,
        var fields: List[Field],
        capacity: Int = 0,
    ):
        var bitmap = Bitmap.alloc(capacity)
        bitmap.unsafe_range_set(0, capacity, True)

        self.dtype = struct_(fields)
        self.capacity = capacity
        self.length = 0
        self.bitmap = ArcPointer(bitmap^)
        self.children = []

    fn __init__(out self, *, ref data: Array):
        self.dtype = data.dtype.copy()
        self.length = data.length
        self.capacity = data.length
        self.bitmap = data.bitmap
        self.children = data.children.copy()

    fn __len__(self) -> Int:
        return self.length

    fn _index_for_field_name(self, name: StringSlice) raises -> Int:
        for idx, ref field in enumerate(self.dtype.fields):
            if field.name == name:
                return idx

        raise Error("Field {} does not exist in this StructArray.".format(name))

    fn unsafe_get(
        self, name: StringSlice
    ) raises -> ref[self.children[0]] Array:
        """Access the field with the given name in the struct."""
        return self.children[self._index_for_field_name(name)][]


struct ChunkedArray(Stringable):
    """An array-like composed from a (possibly empty) collection of pyarrow.Arrays.

    [Reference](https://arrow.apache.org/docs/python/generated/pyarrow.ChunkedArray.html#pyarrow-chunkedarray).
    """

    var dtype: DataType
    var length: Int
    var chunks: List[Array]

    fn _compute_length(mut self) -> None:
        """Update the length of the array from the length of its chunks."""
        var total_length = 0
        for chunk in self.chunks:
            total_length += chunk.length
        self.length = total_length

    fn __init__(out self, var dtype: DataType, var chunks: List[Array]):
        self.dtype = dtype^
        self.chunks = chunks^
        self.length = 0
        self._compute_length()

    fn __str__(self) -> String:
        from .pretty import ArrayPrinter

        var printer = ArrayPrinter()
        try:
            printer.visit(self)
        except:
            pass
        return printer^.finish()

    fn chunk(self, index: Int) -> ref[self.chunks] Array:
        """Returns the chunk at the given index.

        Args:
          index: The desired index.

        Returns:
          A reference to the chunk at the given index.
        """
        return self.chunks[index]

    fn combine_chunks(var self, out combined: Array):
        """Combines all chunks into a single array."""
        var bitmap = ArcPointer(Bitmap.alloc(self.length))
        combined = Array(
            dtype=self.dtype.copy(),
            length=self.length,
            bitmap=bitmap,
            buffers=[],
            children=[],
            offset=0,
        )
        var start = 0
        while self.chunks:
            var chunk = self.chunks.pop(0)
            start += chunk^.append_to_array(combined, start)
        return combined^


fn array[T: DataType]() -> PrimitiveArray[T]:
    """Create an empty primitive array."""
    return PrimitiveArray[T]()


fn array[T: DataType](values: List[Optional[Int]]) -> PrimitiveArray[T]:
    """Create a primitive array from a list of values, where None becomes null."""
    var a = PrimitiveArray[T](len(values))
    for value in values:
        if value:
            a.unsafe_append(Scalar[T.native](value.value()))
        else:
            a.unsafe_append_null()
    return a^


# TODO(stdlib): Bool literals (True/False) coerce to Optional[Bool], so a
# single Optional overload covers both nullable and non-nullable bool arrays.
fn array(values: List[Optional[Bool]]) -> BoolArray:
    """Create a bool array from a list of values, where None becomes null."""
    var a = BoolArray(len(values))
    for value in values:
        if value:
            a.unsafe_append(BoolArray.scalar(value.value()))
        else:
            a.unsafe_append_null()
    return a^


fn arange[T: DataType](start: Int, end: Int) -> PrimitiveArray[T]:
    """Create an integer array from start to end (exclusive), similar to numpy.arange.

    Parameters:
        T: An integer DataType (int8, int16, int32, int64, uint8, uint16, uint32, uint64).

    Args:
        start: The starting value (inclusive).
        end: The ending value (exclusive).

    Returns:
        A PrimitiveArray[T] with values [start, start+1, ..., end-1].
    """
    comptime assert T.is_integer(), "arange() only supports integer DataTypes"
    var a = PrimitiveArray[T](end - start)
    for i in range(start, end):
        a.unsafe_append(Scalar[T.native](i))
    return a^

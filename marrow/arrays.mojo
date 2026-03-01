"""Arrow columnar arrays — always immutable.

Every typed array (`BoolArray`, `PrimitiveArray`, `StringArray`, `ListArray`,
`StructArray`) is immutable.  To *build* an array incrementally, use the
corresponding builder from `marrow.builders` and call `freeze()`.

Array — the generic container
-----------------------------
`Array` is the low-level, type-erased container used for storage, exchange
(C Data Interface), and visitor dispatch.  It holds immutable bitmaps,
buffers, and child arrays directly (no ArcPointer wrapping — sharing is
handled inside Buffer via its internal ArcPointer[Allocation]).  Typed arrays
convert to/from `Array` via implicit constructors and `as_*()` accessors.
"""

from memory import memcpy
from sys import size_of
from gpu.host import DeviceContext
from .buffers import Buffer, Bitmap, BitmapBuilder, MemorySpace
from .dtypes import *


@fieldwise_init
struct Array[space: MemorySpace = MemorySpace.CPU](Copyable, Movable, Stringable):
    """Array is the lower level abstraction directly usable by the library consumer.

    Equivalent with https://github.com/apache/arrow/blob/7184439dea96cd285e6de00e07c5114e4919a465/cpp/src/arrow/array/data.h#L62-L84.

    Array holds immutable bitmap and buffers. Use typed array builders
    (e.g. PrimitiveBuilder[T]) to construct data, then convert to Array
    for storage/exchange.
    """

    var dtype: DataType
    var length: Int
    var bitmap: Bitmap[Self.space]
    var buffers: List[Buffer[Self.space]]
    var children: List[Array[Self.space]]
    var offset: Int

    @staticmethod
    fn from_buffer[dtype: DataType](
        buffer: Buffer[MemorySpace.CPU], length: Int
    ) -> Array[MemorySpace.CPU]:
        """Build an Array from a buffer where all the values are not null."""
        var bitmap = BitmapBuilder.alloc(length)
        bitmap.unsafe_range_set(0, length, True)
        var buffers = List[Buffer[MemorySpace.CPU]]()
        buffers.append(buffer)
        return Array[MemorySpace.CPU](
            dtype=materialize[dtype](),
            length=length,
            bitmap=bitmap^.freeze(),
            buffers=buffers^,
            children=List[Array[MemorySpace.CPU]](),
            offset=0,
        )

    @implicit
    fn __init__[T: DataType](out self, array: PrimitiveArray[T, Self.space]):
        self.dtype = materialize[T]()
        self.length = array.length
        self.offset = array.offset
        self.bitmap = array.bitmap
        self.buffers = [array.buffer]
        self.children = []

    @implicit
    fn __init__(out self, array: BoolArray[Self.space]):
        self.dtype = materialize[bool_]()
        self.length = array.length
        self.offset = array.offset
        self.bitmap = array.bitmap
        self.buffers = [array.values.buffer]
        self.children = []

    @implicit
    fn __init__(out self, array: StringArray[Self.space]):
        self.dtype = materialize[string]()
        self.length = array.length
        self.offset = array.offset
        self.bitmap = array.bitmap
        self.buffers = [array.offsets, array.values]
        self.children = []

    @implicit
    fn __init__(out self, array: ListArray[Self.space]):
        self.dtype = array.dtype.copy()
        self.length = array.length
        self.offset = array.offset
        self.bitmap = array.bitmap
        self.buffers = [array.offsets]
        self.children = [array.values.copy()]

    @implicit
    fn __init__(out self, array: FixedSizeListArray[Self.space]):
        self.dtype = array.dtype.copy()
        self.length = array.length
        self.offset = array.offset
        self.bitmap = array.bitmap
        self.buffers = []
        self.children = [array.values.copy()]

    @implicit
    fn __init__(out self, array: StructArray[Self.space]):
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
        comptime assert Self.space != MemorySpace.DEVICE
        return self.bitmap.unsafe_get(index + self.offset)

    fn __str__(self) -> String:
        comptime assert Self.space != MemorySpace.DEVICE
        from .pretty import ArrayPrinter

        var printer = ArrayPrinter()
        try:
            printer.visit(self)
        except:
            pass
        return printer^.finish()

    fn as_primitive[T: DataType](self) raises -> PrimitiveArray[T, Self.space]:
        return PrimitiveArray[T, Self.space](self)

    fn as_bool(self) raises -> BoolArray[Self.space]:
        return BoolArray[Self.space](self)

    fn as_int8(self) raises -> PrimitiveArray[int8, Self.space]:
        return PrimitiveArray[int8, Self.space](self)

    fn as_int16(self) raises -> PrimitiveArray[int16, Self.space]:
        return PrimitiveArray[int16, Self.space](self)

    fn as_int32(self) raises -> PrimitiveArray[int32, Self.space]:
        return PrimitiveArray[int32, Self.space](self)

    fn as_int64(self) raises -> PrimitiveArray[int64, Self.space]:
        return PrimitiveArray[int64, Self.space](self)

    fn as_uint8(self) raises -> PrimitiveArray[uint8, Self.space]:
        return PrimitiveArray[uint8, Self.space](self)

    fn as_uint16(self) raises -> PrimitiveArray[uint16, Self.space]:
        return PrimitiveArray[uint16, Self.space](self)

    fn as_uint32(self) raises -> PrimitiveArray[uint32, Self.space]:
        return PrimitiveArray[uint32, Self.space](self)

    fn as_uint64(self) raises -> PrimitiveArray[uint64, Self.space]:
        return PrimitiveArray[uint64, Self.space](self)

    fn as_float32(self) raises -> PrimitiveArray[float32, Self.space]:
        return PrimitiveArray[float32, Self.space](self)

    fn as_float64(self) raises -> PrimitiveArray[float64, Self.space]:
        return PrimitiveArray[float64, Self.space](self)

    fn as_string(self) raises -> StringArray[Self.space]:
        return StringArray[Self.space](self)

    fn as_list(self) raises -> ListArray[Self.space]:
        return ListArray[Self.space](self)

    fn as_fixed_size_list(self) raises -> FixedSizeListArray[Self.space]:
        return FixedSizeListArray[Self.space](self)

    fn as_struct(self) raises -> StructArray[Self.space]:
        return StructArray[Self.space](data=self)


@fieldwise_init
struct BoolArray[space: MemorySpace = MemorySpace.CPU](Movable, Sized):
    """An immutable Arrow array of boolean values stored as a bit-packed buffer.
    """

    var length: Int
    var offset: Int
    var bitmap: Bitmap[Self.space]
    var values: Bitmap[Self.space]

    fn __init__(out self, ref data: Array[Self.space], offset: Int = 0) raises:
        if data.dtype != materialize[bool_]():
            raise Error(
                "Unexpected dtype '"
                + String(data.dtype)
                + "' instead of 'bool'."
            )
        elif len(data.buffers) != 1:
            raise Error("BoolArray requires exactly one buffer")
        self.offset = data.offset + offset
        self.length = data.length
        self.bitmap = data.bitmap
        self.values = Bitmap(data.buffers[0])

    @always_inline
    fn __len__(self) -> Int:
        return self.length

    @always_inline
    fn is_valid(self, index: Int) -> Bool:
        return self.bitmap.unsafe_get(index + self.offset)

    @always_inline
    fn unsafe_get(self, index: Int) -> Bool:
        return self.values.unsafe_get(index + self.offset)

    fn __getitem__(self, index: Int) raises -> Bool:
        if index < 0 or index >= self.length:
            raise Error(
                "index "
                + String(index)
                + " out of bounds for length "
                + String(self.length)
            )
        return self.unsafe_get(index)

    fn null_count(self) -> Int:
        """Returns the number of null values in the array."""
        var valid_count = self.bitmap.bit_count()
        return self.length - valid_count


@fieldwise_init
struct PrimitiveArray[T: DataType, space: MemorySpace = MemorySpace.CPU](
    Movable, Sized
):
    """An immutable Arrow array of fixed-size primitive values (integers, floats, etc.).
    """

    comptime dtype = Self.T
    comptime scalar = Scalar[Self.T.native]

    var length: Int
    var offset: Int
    var bitmap: Bitmap[Self.space]
    var buffer: Buffer[Self.space]

    fn __init__(out self, ref data: Array[Self.space], offset: Int = 0) raises:
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
        self.bitmap = data.bitmap
        self.buffer = data.buffers[0]

    @always_inline
    fn __len__(self) -> Int:
        return self.length

    @always_inline
    fn is_valid(self, index: Int) -> Bool:
        return self.bitmap.unsafe_get(index + self.offset)

    @always_inline
    fn unsafe_get(self, index: Int) -> Self.scalar:
        comptime assert Self.space != MemorySpace.DEVICE, "cannot read device array, call to_host() first"
        return self.buffer.unsafe_get[Self.T.native](index + self.offset)

    fn __getitem__(self, index: Int) raises -> Self.scalar:
        if index < 0 or index >= self.length:
            raise Error(
                "index "
                + String(index)
                + " out of bounds for length "
                + String(self.length)
            )
        return self.unsafe_get(index)

    fn null_count(self) -> Int:
        """Returns the number of null values in the array."""
        comptime assert Self.space != MemorySpace.DEVICE
        var valid_count = self.bitmap.bit_count()
        return self.length - valid_count

    fn to_device(
        self, ctx: DeviceContext
    ) raises -> PrimitiveArray[Self.T, MemorySpace.DEVICE]:
        """Upload array data to the GPU."""
        comptime assert Self.space == MemorySpace.CPU
        return PrimitiveArray[Self.T, MemorySpace.DEVICE](
            length=self.length,
            offset=0,
            bitmap=self.bitmap.to_device(ctx),
            buffer=self.buffer.to_device(ctx),
        )

    fn to_host(
        self, ctx: DeviceContext
    ) raises -> PrimitiveArray[Self.T, MemorySpace.CPU]:
        """Download array data from the GPU."""
        comptime assert Self.space == MemorySpace.DEVICE
        return PrimitiveArray[Self.T, MemorySpace.CPU](
            length=self.length,
            offset=0,
            bitmap=self.bitmap.to_host(ctx),
            buffer=self.buffer.to_host(ctx),
        )


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


@fieldwise_init
struct StringArray[space: MemorySpace = MemorySpace.CPU](Movable, Sized):
    """An immutable Arrow array of variable-length UTF-8 strings."""

    var length: Int
    var offset: Int
    var bitmap: Bitmap[Self.space]
    var offsets: Buffer[Self.space]
    var values: Buffer[Self.space]

    fn __init__(out self, ref data: Array[Self.space]) raises:
        """Construct a StringArray from a generic Array.

        Raises:
            If the dtype is not string or the buffer count is wrong.
        """
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
        self.bitmap = data.bitmap
        self.offsets = data.buffers[0]
        self.values = data.buffers[1]

    fn __len__(self) -> Int:
        """Return the number of elements in the array."""
        return self.length

    fn is_valid(self, index: Int) -> Bool:
        """Return True if the element at the given index is not null."""
        return self.bitmap.unsafe_get(index)

    fn unsafe_get[
        self_origin: Origin[mut=False], //
    ](ref[self_origin] self, index: UInt) -> StringSlice[self_origin]:
        """Return a StringSlice for the element at the given index without bounds checking.
        """
        var offset_idx = Int(index) + self.offset
        var start_offset = self.offsets.unsafe_get[DType.uint32](offset_idx)
        var end_offset = self.offsets.unsafe_get[DType.uint32](offset_idx + 1)
        var length = Int(end_offset) - Int(start_offset)
        var ptr = (
            (self.values.ptr + Int(start_offset))
            .mut_cast[False]()
            .unsafe_origin_cast[self_origin]()
        )
        return StringSlice[self_origin](ptr=ptr, length=length)

    fn __getitem__[
        self_origin: Origin[mut=False], //
    ](ref[self_origin] self, index: Int) raises -> StringSlice[self_origin]:
        """Return a StringSlice for the element at the given index.

        Raises:
            If the index is out of bounds.
        """
        if index < 0 or index >= self.length:
            raise Error(
                "index "
                + String(index)
                + " out of bounds for length "
                + String(self.length)
            )
        return self.unsafe_get(UInt(index))


@fieldwise_init
struct ListArray[space: MemorySpace = MemorySpace.CPU](Movable, Sized):
    """An immutable Arrow array of variable-length lists (each element is a sub-array).
    """

    var dtype: DataType
    var length: Int
    var offset: Int
    var bitmap: Bitmap[Self.space]
    var offsets: Buffer[Self.space]
    var values: Array[Self.space]

    fn __init__(out self, ref data: Array[Self.space]) raises:
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
        self.bitmap = data.bitmap
        self.offsets = data.buffers[0]
        self.values = data.children[0].copy()

    fn __len__(self) -> Int:
        return self.length

    fn is_valid(self, index: Int) -> Bool:
        return self.bitmap.unsafe_get(index)

    fn unsafe_get(self, index: Int, out array_data: Array[Self.space]) raises:
        """Access the value at a given index in the list array.

        Use an out argument to allow the caller to re-use memory while iterating over a pyarrow structure.
        """
        var start = Int(
            self.offsets.unsafe_get[DType.int32](self.offset + index)
        )
        var end = Int(
            self.offsets.unsafe_get[DType.int32](self.offset + index + 1)
        )
        return Array[Self.space](
            dtype=self.values.dtype.copy(),
            bitmap=self.values.bitmap,
            buffers=self.values.buffers.copy(),
            offset=start,
            length=end - start,
            children=self.values.children.copy(),
        )


@fieldwise_init
struct FixedSizeListArray[space: MemorySpace = MemorySpace.CPU](Movable, Sized):
    """An immutable Arrow array of fixed-size lists (each element is a sub-array of the same length).
    """

    var dtype: DataType
    var length: Int
    var offset: Int
    var bitmap: Bitmap[Self.space]
    var values: Array[Self.space]

    fn __init__(out self, ref data: Array[Self.space]) raises:
        if not data.dtype.is_fixed_size_list():
            raise Error(
                "Unexpected dtype "
                + String(data.dtype)
                + " instead of 'fixed_size_list'"
            )
        elif len(data.buffers) != 0:
            raise Error("FixedSizeListArray requires zero buffers")
        elif len(data.children) != 1:
            raise Error("FixedSizeListArray requires exactly one child array")

        self.dtype = data.dtype.copy()
        self.length = data.length
        self.offset = data.offset
        self.bitmap = data.bitmap
        self.values = data.children[0].copy()

    fn __len__(self) -> Int:
        return self.length

    fn is_valid(self, index: Int) -> Bool:
        return self.bitmap.unsafe_get(index)

    fn unsafe_get(self, index: Int, out array_data: Array[Self.space]) raises:
        var list_size = self.dtype.size
        var start = (self.offset + index) * list_size
        return Array[Self.space](
            dtype=self.values.dtype.copy(),
            bitmap=self.values.bitmap,
            buffers=self.values.buffers.copy(),
            offset=start,
            length=list_size,
            children=self.values.children.copy(),
        )

    fn to_device(
        self, ctx: DeviceContext
    ) raises -> FixedSizeListArray[MemorySpace.DEVICE]:
        """Upload child values to the GPU."""
        comptime assert Self.space == MemorySpace.CPU
        var new_buffers = List[Buffer[MemorySpace.DEVICE]]()
        for i in range(len(self.values.buffers)):
            new_buffers.append(self.values.buffers[i].to_device(ctx))
        var new_child = Array[MemorySpace.DEVICE](
            dtype=self.values.dtype.copy(),
            bitmap=self.values.bitmap.to_device(ctx),
            buffers=new_buffers^,
            offset=self.values.offset,
            length=self.values.length,
            children=List[Array[MemorySpace.DEVICE]](),
        )
        return FixedSizeListArray[MemorySpace.DEVICE](
            dtype=self.dtype.copy(),
            length=self.length,
            offset=self.offset,
            bitmap=self.bitmap.to_device(ctx),
            values=new_child^,
        )


@fieldwise_init
struct StructArray[space: MemorySpace = MemorySpace.CPU](Movable, Sized):
    """An immutable Arrow array of structs (each element is a collection of named fields).
    """

    var dtype: DataType
    var length: Int
    var bitmap: Bitmap[Self.space]
    var children: List[Array[Self.space]]

    fn __init__(out self, *, ref data: Array[Self.space]):
        self.dtype = data.dtype.copy()
        self.length = data.length
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
    ) raises -> ref[self.children[0]] Array[Self.space]:
        """Access the field with the given name in the struct."""
        return self.children[self._index_for_field_name(name)]


struct ChunkedArray[space: MemorySpace = MemorySpace.CPU](Stringable):
    """An array-like composed from a (possibly empty) collection of pyarrow.Arrays.

    [Reference](https://arrow.apache.org/docs/python/generated/pyarrow.ChunkedArray.html#pyarrow-chunkedarray).
    """

    var dtype: DataType
    var length: Int
    var chunks: List[Array[Self.space]]

    fn _compute_length(mut self) -> None:
        """Update the length of the array from the length of its chunks."""
        var total_length = 0
        for chunk in self.chunks:
            total_length += chunk.length
        self.length = total_length

    fn __init__(out self, var dtype: DataType, var chunks: List[Array[Self.space]]):
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

    fn chunk(self, index: Int) -> ref[self.chunks] Array[Self.space]:
        """Returns the chunk at the given index.

        Args:
          index: The desired index.

        Returns:
          A reference to the chunk at the given index.
        """
        return self.chunks[index]

    fn combine_chunks(var self, out combined: Array[Self.space]):
        """Combines all chunks into a single array."""
        comptime assert Self.space == MemorySpace.CPU, "combine_chunks requires CPU arrays"
        var bitmap = BitmapBuilder.alloc(self.length)
        var buffers = List[Buffer[Self.space]]()
        var children = List[Array[Self.space]]()
        var start = 0
        while self.chunks:
            var chunk = self.chunks.pop(0)
            var chunk_length = chunk.length
            bitmap.extend(
                rebind[Bitmap[MemorySpace.CPU]](chunk.bitmap),
                start,
                chunk_length,
            )
            for i in range(len(chunk.buffers)):
                buffers.append(chunk.buffers[i])
            for i in range(len(chunk.children)):
                children.append(chunk.children[i].copy())
            start += chunk_length
        combined = Array[Self.space](
            dtype=self.dtype.copy(),
            length=self.length,
            bitmap=rebind[Bitmap[Self.space]](bitmap^.freeze()),
            buffers=buffers^,
            children=children^,
            offset=0,
        )


from .builders import array, nulls, arange

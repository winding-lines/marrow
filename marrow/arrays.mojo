"""Arrow columnar arrays — always immutable.

Every typed array (`PrimitiveArray`, `StringArray`, `ListArray`, `StructArray`)
is immutable.  To *build* an array incrementally, use the corresponding builder
from `marrow.builders` and call `finish()`.

`BoolArray` is an alias for `PrimitiveArray[bool_]`.

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
from .buffers import (
    Buffer,
    BufferBuilder,
    bitmap_extend,
    bitmap_count_ones,
)
from .dtypes import *


@fieldwise_init
struct Array(Copyable, Movable, Stringable):
    """Array is the lower level abstraction directly usable by the library consumer.

    Equivalent with https://github.com/apache/arrow/blob/7184439dea96cd285e6de00e07c5114e4919a465/cpp/src/arrow/array/data.h#L62-L84.

    Array holds immutable bitmap and buffers. Use typed array builders
    (e.g. PrimitiveBuilder[T]) to construct data, then convert to Array
    for storage/exchange.
    """

    var dtype: DataType
    var length: Int
    var nulls: Int
    var bitmap: Buffer
    var buffers: List[Buffer]
    var children: List[Array]
    var offset: Int

    @implicit
    fn __init__[T: DataType](out self, array: PrimitiveArray[T]):
        self.dtype = materialize[T]()
        self.length = array.length
        self.nulls = array.nulls
        self.offset = array.offset
        self.bitmap = array.bitmap
        self.buffers = [array.buffer]
        self.children = []

    @implicit
    fn __init__(out self, array: StringArray):
        self.dtype = materialize[string]()
        self.length = array.length
        self.nulls = array.nulls
        self.offset = array.offset
        self.bitmap = array.bitmap
        self.buffers = [array.offsets, array.values]
        self.children = []

    @implicit
    fn __init__(out self, array: ListArray):
        self.dtype = array.dtype.copy()
        self.length = array.length
        self.nulls = array.nulls
        self.offset = array.offset
        self.bitmap = array.bitmap
        self.buffers = [array.offsets]
        self.children = [array.values.copy()]

    @implicit
    fn __init__(out self, array: FixedSizeListArray):
        self.dtype = array.dtype.copy()
        self.length = array.length
        self.nulls = array.nulls
        self.offset = array.offset
        self.bitmap = array.bitmap
        self.buffers = []
        self.children = [array.values.copy()]

    @implicit
    fn __init__(out self, array: StructArray):
        self.dtype = array.dtype.copy()
        self.length = array.length
        self.nulls = array.nulls
        self.offset = 0
        self.bitmap = array.bitmap
        self.buffers = []
        self.children = array.children.copy()

    fn __init__(out self, *, copy: Self):
        self.dtype = copy.dtype.copy()
        self.length = copy.length
        self.nulls = copy.nulls
        self.bitmap = copy.bitmap
        self.buffers = copy.buffers.copy()
        self.children = copy.children.copy()
        self.offset = copy.offset

    fn __init__(out self, *, deinit take: Self):
        self.dtype = take.dtype^
        self.length = take.length
        self.nulls = take.nulls
        self.bitmap = take.bitmap^
        self.buffers = take.buffers^
        self.children = take.children^
        self.offset = take.offset

    fn is_valid(self, index: Int) -> Bool:
        return self.bitmap.unsafe_get[DType.bool](index + self.offset)

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

    fn as_bool(self) raises -> BoolArray:
        return self.as_primitive[bool_]()

    fn as_int8(self) raises -> PrimitiveArray[int8]:
        return PrimitiveArray[int8](self)

    fn as_int16(self) raises -> PrimitiveArray[int16]:
        return PrimitiveArray[int16](self)

    fn as_int32(self) raises -> PrimitiveArray[int32]:
        return PrimitiveArray[int32](self)

    fn as_int64(self) raises -> PrimitiveArray[int64]:
        return PrimitiveArray[int64](self)

    fn as_uint8(self) raises -> PrimitiveArray[uint8]:
        return PrimitiveArray[uint8](self)

    fn as_uint16(self) raises -> PrimitiveArray[uint16]:
        return PrimitiveArray[uint16](self)

    fn as_uint32(self) raises -> PrimitiveArray[uint32]:
        return PrimitiveArray[uint32](self)

    fn as_uint64(self) raises -> PrimitiveArray[uint64]:
        return PrimitiveArray[uint64](self)

    fn as_float32(self) raises -> PrimitiveArray[float32]:
        return PrimitiveArray[float32](self)

    fn as_float64(self) raises -> PrimitiveArray[float64]:
        return PrimitiveArray[float64](self)

    fn as_string(self) raises -> StringArray:
        return StringArray(self)

    fn as_list(self) raises -> ListArray:
        return ListArray(self)

    fn as_fixed_size_list(self) raises -> FixedSizeListArray:
        return FixedSizeListArray(self)

    fn as_struct(self) raises -> StructArray:
        return StructArray(data=self)


@fieldwise_init
struct PrimitiveArray[T: DataType](Movable, Sized):
    """An immutable Arrow array of fixed-size primitive values (integers, floats, etc.).
    """

    comptime dtype = Self.T
    comptime scalar = Scalar[Self.T.native]

    var length: Int
    var nulls: Int
    var offset: Int
    var bitmap: Buffer
    var buffer: Buffer

    fn __init__(out self, ref data: Array, offset: Int = 0) raises:
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

        self.length = data.length
        self.nulls = data.nulls
        self.offset = data.offset + offset
        self.bitmap = data.bitmap
        self.buffer = data.buffers[0]

    @always_inline
    fn __len__(self) -> Int:
        return self.length

    @always_inline
    fn is_valid(self, index: Int) -> Bool:
        return self.bitmap.unsafe_get[DType.bool](index + self.offset)

    @always_inline
    fn unsafe_get(self, index: Int) -> Self.scalar:
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
        return self.nulls

    fn to_device(self, ctx: DeviceContext) raises -> PrimitiveArray[Self.T]:
        """Upload array data to the GPU."""
        return PrimitiveArray[Self.T](
            length=self.length,
            nulls=self.nulls,
            offset=0,
            bitmap=self.bitmap.to_device(ctx),
            buffer=self.buffer.to_device(ctx),
        )

    fn to_cpu(self, ctx: DeviceContext) raises -> PrimitiveArray[Self.T]:
        """Download array data from the GPU to owned CPU heap buffers."""
        return PrimitiveArray[Self.T](
            length=self.length,
            nulls=self.nulls,
            offset=0,
            bitmap=self.bitmap.to_cpu(ctx),
            buffer=self.buffer.to_cpu(ctx),
        )


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


@fieldwise_init
struct StringArray(Movable, Sized):
    """An immutable Arrow array of variable-length UTF-8 strings."""

    var length: Int
    var nulls: Int
    var offset: Int
    var bitmap: Buffer
    var offsets: Buffer
    var values: Buffer

    fn __init__(out self, ref data: Array) raises:
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
        self.nulls = data.nulls
        self.offset = data.offset
        self.bitmap = data.bitmap
        self.offsets = data.buffers[0]
        self.values = data.buffers[1]

    fn __len__(self) -> Int:
        """Return the number of elements in the array."""
        return self.length

    fn is_valid(self, index: Int) -> Bool:
        """Return True if the element at the given index is not null."""
        return self.bitmap.unsafe_get[DType.bool](index + self.offset)

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
struct ListArray(Movable, Sized):
    """An immutable Arrow array of variable-length lists (each element is a sub-array).
    """

    var dtype: DataType
    var length: Int
    var nulls: Int
    var offset: Int
    var bitmap: Buffer
    var offsets: Buffer
    var values: Array

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
        self.nulls = data.nulls
        self.offset = data.offset
        self.bitmap = data.bitmap
        self.offsets = data.buffers[0]
        self.values = data.children[0].copy()

    fn __len__(self) -> Int:
        return self.length

    fn is_valid(self, index: Int) -> Bool:
        return self.bitmap.unsafe_get[DType.bool](index + self.offset)

    fn unsafe_get(self, index: Int, out array_data: Array) raises:
        """Access the value at a given index in the list array.

        Use an out argument to allow the caller to re-use memory while iterating over a pyarrow structure.
        """
        var start = Int(
            self.offsets.unsafe_get[DType.int32](self.offset + index)
        )
        var end = Int(
            self.offsets.unsafe_get[DType.int32](self.offset + index + 1)
        )
        return Array(
            dtype=self.values.dtype.copy(),
            length=end - start,
            nulls=-1,
            bitmap=self.values.bitmap,
            buffers=self.values.buffers.copy(),
            children=self.values.children.copy(),
            offset=start,
        )


@fieldwise_init
struct FixedSizeListArray(Movable, Sized):
    """An immutable Arrow array of fixed-size lists (each element is a sub-array of the same length).
    """

    var dtype: DataType
    var length: Int
    var nulls: Int
    var offset: Int
    var bitmap: Buffer
    var values: Array

    fn __init__(out self, ref data: Array) raises:
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
        self.nulls = data.nulls
        self.offset = data.offset
        self.bitmap = data.bitmap
        self.values = data.children[0].copy()

    fn __len__(self) -> Int:
        return self.length

    fn is_valid(self, index: Int) -> Bool:
        return self.bitmap.unsafe_get[DType.bool](index + self.offset)

    fn unsafe_get(self, index: Int, out array_data: Array) raises:
        var list_size = self.dtype.size
        var start = (self.offset + index) * list_size
        return Array(
            dtype=self.values.dtype.copy(),
            length=list_size,
            nulls=-1,
            bitmap=self.values.bitmap,
            buffers=self.values.buffers.copy(),
            children=self.values.children.copy(),
            offset=start,
        )

    fn to_device(self, ctx: DeviceContext) raises -> FixedSizeListArray:
        """Upload child values to the GPU."""
        var new_buffers = List[Buffer]()
        for i in range(len(self.values.buffers)):
            new_buffers.append(self.values.buffers[i].to_device(ctx))
        var new_child = Array(
            dtype=self.values.dtype.copy(),
            length=self.values.length,
            nulls=self.values.nulls,
            bitmap=self.values.bitmap.to_device(ctx),
            buffers=new_buffers^,
            children=List[Array](),
            offset=self.values.offset,
        )
        return FixedSizeListArray(
            dtype=self.dtype.copy(),
            length=self.length,
            nulls=self.nulls,
            offset=self.offset,
            bitmap=self.bitmap.to_device(ctx),
            values=new_child^,
        )


@fieldwise_init
struct StructArray(Movable, Sized):
    """An immutable Arrow array of structs (each element is a collection of named fields).
    """

    var dtype: DataType
    var length: Int
    var nulls: Int
    var bitmap: Buffer
    var children: List[Array]

    fn __init__(out self, *, ref data: Array):
        self.dtype = data.dtype.copy()
        self.length = data.length
        self.nulls = data.nulls
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
        return self.children[self._index_for_field_name(name)]


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
        var bitmap = BufferBuilder.alloc[DType.bool](self.length)
        var buffers = List[Buffer]()
        var children = List[Array]()
        var start = 0
        while self.chunks:
            var chunk = self.chunks.pop(0)
            var chunk_length = chunk.length
            bitmap_extend(
                bitmap.ptr, chunk.bitmap.unsafe_ptr(), start, chunk_length
            )
            for i in range(len(chunk.buffers)):
                buffers.append(chunk.buffers[i])
            for i in range(len(chunk.children)):
                children.append(chunk.children[i].copy())
            start += chunk_length
        var frozen_bitmap = bitmap.finish()
        var nulls = self.length - bitmap_count_ones(frozen_bitmap, frozen_bitmap.size)
        combined = Array(
            dtype=self.dtype.copy(),
            length=self.length,
            nulls=nulls,
            bitmap=frozen_bitmap^,
            buffers=buffers^,
            children=children^,
            offset=0,
        )


from .builders import array, nulls, arange

"""Arrow columnar arrays with compile-time mutability control.

Every typed array (`BoolArray`, `PrimitiveArray`, `StringArray`, `ListArray`,
`StructArray`) carries a `mut: Bool` parameter that defaults to `False`
(immutable).  This means **the common read path requires no annotation** — a
bare `PrimitiveArray[int64]` is immutable, matching the fact that most code
*consumes* arrays rather than building them.

Lifecycle — building an array
-----------------------------
1. **Create a builder** with explicit `mut=True`:

       var builder = PrimitiveArray[int64, mut=True](capacity=1024)

2. **Append / set values** through gated methods (only available on `mut=True`):

       builder.append(42)
       builder.unsafe_append_null()

3. **Freeze** the builder into an immutable array (zero-cost `rebind`):

       var arr = builder^.freeze()   # PrimitiveArray[int64]

   `freeze()` shrinks buffers to exact length (normalises offset to 0) and
   returns the default (immutable) type.  After freezing, the builder is
   consumed — no mutable handle remains.

Convenience factory functions (`array()`, `arange()`, `nulls()`) encapsulate
this lifecycle: they build with `mut=True` internally and return a frozen,
immutable array.

Array — the generic container
-----------------------------
`Array` is the low-level, type-erased container used for storage, exchange
(C Data Interface), and visitor dispatch.  It always holds **immutable**
bitmaps and buffers (`ArcPointer[Bitmap]`, `ArcPointer[Buffer]`) — there is
no mutable variant of `Array`.  Typed arrays convert to/from `Array` via
implicit constructors and `as_*()` accessors.
"""

from memory import ArcPointer, memcpy
from sys import size_of
from .buffers import Buffer, Bitmap
from .dtypes import *


@fieldwise_init
struct Array(Copyable, Movable, Stringable):
    """Array is the lower level abstraction directly usable by the library consumer.

    Equivalent with https://github.com/apache/arrow/blob/7184439dea96cd285e6de00e07c5114e4919a465/cpp/src/arrow/array/data.h#L62-L84.

    Array holds immutable bitmap and buffers. Use typed array builders
    (e.g. PrimitiveArray[Self.T, mut=True]) to construct data, then convert
    to Array for storage/exchange.
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
        var buffers = List[ArcPointer[Buffer]]()
        buffers.append(rebind[ArcPointer[Buffer]](ArcPointer(buffer^)))
        return Array(
            dtype=materialize[dtype](),
            length=length,
            bitmap=ArcPointer(bitmap^.freeze()),
            buffers=buffers^,
            children=[],
            offset=0,
        )

    @implicit
    fn __init__[
        T: DataType, is_mut: Bool
    ](out self, array: PrimitiveArray[T, mut=is_mut]):
        self.dtype = materialize[T]()
        self.length = array.length
        self.offset = array.offset
        self.bitmap = rebind[ArcPointer[Bitmap]](array.bitmap)
        self.buffers = [rebind[ArcPointer[Buffer]](array.buffer)]
        self.children = []

    @implicit
    fn __init__[is_mut: Bool](out self, array: BoolArray[mut=is_mut]):
        self.dtype = materialize[bool_]()
        self.length = array.length
        self.offset = array.offset
        self.bitmap = rebind[ArcPointer[Bitmap]](array.bitmap)
        var buffers = List[ArcPointer[Buffer]]()
        buffers.append(rebind[ArcPointer[Buffer]](array.values))
        self.buffers = buffers^
        self.children = []

    @implicit
    fn __init__[is_mut: Bool](out self, array: StringArray[mut=is_mut]):
        self.dtype = materialize[string]()
        self.length = array.length
        self.offset = array.offset
        self.bitmap = rebind[ArcPointer[Bitmap]](array.bitmap)
        self.buffers = [
            rebind[ArcPointer[Buffer]](array.offsets),
            rebind[ArcPointer[Buffer]](array.values),
        ]
        self.children = []

    @implicit
    fn __init__[is_mut: Bool](out self, array: ListArray[mut=is_mut]):
        self.dtype = array.dtype.copy()
        self.length = array.length
        self.offset = array.offset
        self.bitmap = rebind[ArcPointer[Bitmap]](array.bitmap)
        self.buffers = [rebind[ArcPointer[Buffer]](array.offsets)]
        self.children = [array.values]

    @implicit
    fn __init__[is_mut: Bool](out self, array: StructArray[mut=is_mut]):
        self.dtype = array.dtype.copy()
        self.length = array.length
        self.offset = 0
        self.bitmap = rebind[ArcPointer[Bitmap]](array.bitmap)
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

    fn as_bool(self) raises -> BoolArray:
        return BoolArray(self)

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

    fn as_struct(self) raises -> StructArray:
        return StructArray(data=self)


struct BoolArray[*, mut: Bool = False](Movable, Sized):
    """An Arrow array of boolean values stored as a bit-packed buffer.

    Immutable by default (`mut=False`).  Use `BoolArray[mut=True](capacity=n)`
    to create a mutable builder, then call `freeze()` to obtain an immutable
    `BoolArray`.  Constructing from an `Array` always yields an immutable view.
    """

    var length: Int
    var offset: Int
    var capacity: Int
    var bitmap: ArcPointer[Bitmap[mut=Self.mut]]
    var values: ArcPointer[Bitmap[mut=Self.mut]]

    fn __init__(out self: BoolArray, ref data: Array, offset: Int = 0) raises:
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
        self.capacity = data.length
        self.bitmap = data.bitmap
        self.values = rebind[ArcPointer[Bitmap]](data.buffers[0])

    fn __init__(out self: BoolArray[mut=True], capacity: Int = 0, offset: Int = 0):
        self.capacity = capacity
        self.length = 0
        self.offset = offset
        self.bitmap = ArcPointer(Bitmap.alloc(capacity))
        self.values = ArcPointer(Bitmap.alloc(capacity))

    fn __init__[
        other_mut: Bool, //
    ](out self: BoolArray, deinit other: BoolArray[mut=other_mut]):
        self.length = other.length
        self.offset = other.offset
        self.capacity = other.capacity
        self.bitmap = rebind[ArcPointer[Bitmap]](other.bitmap^)
        self.values = rebind[ArcPointer[Bitmap]](other.values^)

    fn resize(mut self: BoolArray[mut=True], capacity: Int):
        self.bitmap[].resize(capacity)
        self.values[].resize(capacity)
        self.capacity = capacity

    @always_inline
    fn __len__(self) -> Int:
        return self.length

    @always_inline
    fn is_valid(self, index: Int) -> Bool:
        return self.bitmap[].unsafe_get(index + self.offset)

    @always_inline
    fn unsafe_get(self, index: Int) -> Bool:
        return self.values[].unsafe_get(index + self.offset)

    @always_inline
    fn unsafe_set(mut self: BoolArray[mut=True], index: Int, value: Bool):
        self.bitmap[].unsafe_set(index + self.offset, True)
        self.values[].unsafe_set(index + self.offset, value)

    @always_inline
    fn unsafe_append(mut self: BoolArray[mut=True], value: Bool):
        self.unsafe_set(self.length, value)
        self.length += 1

    @always_inline
    fn unsafe_append_null(mut self: BoolArray[mut=True]):
        self.bitmap[].unsafe_set(self.length + self.offset, False)
        self.length += 1

    fn __getitem__(self, index: Int) raises -> Bool:
        if index < 0 or index >= self.length:
            raise Error(
                "index "
                + String(index)
                + " out of bounds for length "
                + String(self.length)
            )
        return self.unsafe_get(index)

    fn __setitem__(mut self: BoolArray[mut=True], index: Int, value: Bool) raises:
        if index < 0 or index >= self.length:
            raise Error(
                "index "
                + String(index)
                + " out of bounds for length "
                + String(self.length)
            )
        self.unsafe_set(index, value)

    fn shrink_to_fit(mut self: BoolArray[mut=True]):
        """Shrink buffers to exact length and normalize offset to 0 in place."""
        if self.length == self.capacity and self.offset == 0:
            return

        self.values[].resize(self.length, self.offset)
        self.bitmap[].resize(self.length, self.offset)
        self.offset = 0
        self.capacity = self.length

    fn freeze(deinit self: BoolArray[mut=True]) -> BoolArray:
        """Shrink buffers to exact length and return as an immutable array."""
        self.shrink_to_fit()
        return BoolArray(self^)

    fn append(mut self: BoolArray[mut=True], value: Bool):
        if self.length >= self.capacity:
            self.resize(max(self.capacity * 2, self.length + 1))
        self.unsafe_append(value)

    fn extend(mut self: BoolArray[mut=True], values: List[Bool]):
        if self.__len__() + len(values) >= self.capacity:
            self.resize(self.capacity + len(values))
        for value in values:
            self.unsafe_append(value)

    fn null_count(self) -> Int:
        """Returns the number of null values in the array."""
        var valid_count = self.bitmap[].bit_count()
        return self.length - valid_count


struct PrimitiveArray[T: DataType, *, mut: Bool = False](Movable, Sized):
    """An Arrow array of fixed-size primitive values (integers, floats, etc.).

    Immutable by default (`mut=False`).  Use
    `PrimitiveArray[T, mut=True](capacity=n)` to create a mutable builder,
    then call `freeze()` to obtain an immutable `PrimitiveArray[T]`.
    """

    # comptime assert T.is_primitive(), "PrimitiveArray requires a primitive data type"
    comptime dtype = Self.T
    comptime scalar = Scalar[Self.T.native]

    var length: Int
    var offset: Int
    var capacity: Int
    var bitmap: ArcPointer[Bitmap[mut=Self.mut]]
    var buffer: ArcPointer[Buffer[mut=Self.mut]]

    fn __init__(out self: PrimitiveArray[Self.T], ref data: Array, offset: Int = 0) raises:
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

    fn __init__(out self: PrimitiveArray[Self.T, mut=True], capacity: Int = 0, offset: Int = 0):
        self.capacity = capacity
        self.length = 0
        self.offset = offset
        self.bitmap = ArcPointer(Bitmap.alloc(capacity))
        self.buffer = ArcPointer(Buffer.alloc[Self.T.native](capacity))

    fn __init__[
        other_mut: Bool, //
    ](out self: PrimitiveArray[Self.T], deinit other: PrimitiveArray[Self.T, mut=other_mut]):
        self.length = other.length
        self.offset = other.offset
        self.capacity = other.capacity
        self.bitmap = rebind[ArcPointer[Bitmap]](other.bitmap^)
        self.buffer = rebind[ArcPointer[Buffer]](other.buffer^)

    fn resize(mut self: PrimitiveArray[Self.T, mut=True], capacity: Int):
        self.bitmap[].resize(capacity)
        self.buffer[].resize[Self.T.native](capacity)
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
    fn unsafe_set(mut self: PrimitiveArray[Self.T, mut=True], index: Int, value: Self.scalar):
        self.bitmap[].unsafe_set(index + self.offset, True)
        self.buffer[].unsafe_set[Self.T.native](index + self.offset, value)

    @always_inline
    fn unsafe_append(mut self: PrimitiveArray[Self.T, mut=True], value: Self.scalar):
        self.unsafe_set(self.length, value)
        self.length += 1

    @always_inline
    fn unsafe_append_null(mut self: PrimitiveArray[Self.T, mut=True]):
        self.bitmap[].unsafe_set(self.length + self.offset, False)
        self.length += 1

    fn __getitem__(self, index: Int) raises -> Self.scalar:
        if index < 0 or index >= self.length:
            raise Error(
                "index "
                + String(index)
                + " out of bounds for length "
                + String(self.length)
            )
        return self.unsafe_get(index)

    fn __setitem__(mut self: PrimitiveArray[Self.T, mut=True], index: Int, value: Self.scalar) raises:
        if index < 0 or index >= self.length:
            raise Error(
                "index "
                + String(index)
                + " out of bounds for length "
                + String(self.length)
            )
        self.unsafe_set(index, value)

    fn shrink_to_fit(mut self: PrimitiveArray[Self.T, mut=True]):
        """Shrink buffers to exact length and normalize offset to 0 in place."""
        if self.length == self.capacity and self.offset == 0:
            return

        self.buffer[].offset = self.offset
        self.buffer[].resize[Self.T.native](self.length)
        self.bitmap[].resize(self.length, self.offset)
        self.offset = 0
        self.capacity = self.length

    fn freeze(deinit self: PrimitiveArray[Self.T, mut=True]) -> PrimitiveArray[Self.T]:
        """Shrink buffers to exact length and return as an immutable array."""
        self.shrink_to_fit()
        return PrimitiveArray[Self.T](self^)

    fn append(mut self: PrimitiveArray[Self.T, mut=True], value: Self.scalar):
        if self.length >= self.capacity:
            self.resize(max(self.capacity * 2, self.length + 1))
        self.unsafe_append(value)

    fn extend(mut self: PrimitiveArray[Self.T, mut=True], values: List[self.scalar]):
        if self.__len__() + len(values) >= self.capacity:
            self.resize(self.capacity + len(values))
        for value in values:
            self.unsafe_append(value)

    fn null_count(self) -> Int:
        """Returns the number of null values in the array."""
        var valid_count = self.bitmap[].bit_count()
        return self.length - valid_count


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


struct StringArray[*, mut: Bool = False](Movable, Sized):
    """An Arrow array of variable-length UTF-8 strings.

    Immutable by default (`mut=False`).  Use `StringArray[mut=True](capacity=n)`
    to create a mutable builder, then call `freeze()` to obtain an immutable
    `StringArray`.
    """
    var length: Int
    var offset: Int
    var capacity: Int
    var bitmap: ArcPointer[Bitmap[mut=Self.mut]]
    var offsets: ArcPointer[Buffer[mut=Self.mut]]
    var values: ArcPointer[Buffer[mut=Self.mut]]

    fn __init__(out self: StringArray, ref data: Array) raises:
        """Construct an immutable StringArray from a generic Array.

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
        self.capacity = data.length
        self.bitmap = data.bitmap
        self.offsets = data.buffers[0]
        self.values = data.buffers[1]

    fn __init__(out self: StringArray[mut=True], capacity: Int = 0):
        """Create a mutable StringArray builder with the given capacity."""
        self.capacity = capacity
        self.length = 0
        self.offset = 0
        self.bitmap = ArcPointer(Bitmap.alloc(capacity))
        self.offsets = ArcPointer(Buffer.alloc[DType.uint32](capacity + 1))
        self.values = ArcPointer(Buffer.alloc[DType.uint8](capacity))
        self.offsets[].unsafe_set[DType.uint32](0, 0)

    fn __init__[
        other_mut: Bool, //
    ](out self: StringArray, deinit other: StringArray[mut=other_mut]):
        """Convert a StringArray of any mutability to an immutable StringArray."""
        self.length = other.length
        self.offset = other.offset
        self.capacity = other.capacity
        self.bitmap = rebind[ArcPointer[Bitmap]](other.bitmap^)
        self.offsets = rebind[ArcPointer[Buffer]](other.offsets^)
        self.values = rebind[ArcPointer[Buffer]](other.values^)

    fn __len__(self) -> Int:
        """Return the number of elements in the array."""
        return self.length

    fn resize(mut self: StringArray[mut=True], capacity: Int):
        """Resize the bitmap and offsets buffers to the given capacity."""
        self.bitmap[].resize(capacity)
        self.offsets[].resize[DType.uint32](capacity + 1)
        self.capacity = capacity

    fn is_valid(self, index: Int) -> Bool:
        """Return True if the element at the given index is not null."""
        return self.bitmap[].unsafe_get(index)

    fn unsafe_append(mut self: StringArray[mut=True], value: String):
        """Append a string value without bounds checking."""
        var index = self.length
        var last_offset = self.offsets[].unsafe_get[DType.uint32](index)
        var next_offset = last_offset + UInt32(len(value))
        self.length += 1
        self.bitmap[].unsafe_set(index, True)
        self.offsets[].unsafe_set[DType.uint32](index + 1, next_offset)
        self.values[].resize[DType.uint8](next_offset)
        var dst_address = self.values[].ptr + Int(last_offset)
        var src_address = value.unsafe_ptr()
        memcpy(dest=dst_address, src=src_address, count=len(value))

    fn unsafe_get[
        self_origin: Origin[mut=False], //
    ](ref [self_origin] self, index: UInt) -> StringSlice[self_origin]:
        """Return a StringSlice for the element at the given index without bounds checking."""
        var offset_idx = Int(index) + self.offset
        var start_offset = self.offsets[].unsafe_get[DType.uint32](offset_idx)
        var end_offset = self.offsets[].unsafe_get[DType.uint32](offset_idx + 1)
        var length = Int(end_offset) - Int(start_offset)
        var ptr = (self.values[].ptr + Int(start_offset))
            .mut_cast[False]()
            .unsafe_origin_cast[self_origin]()
        return StringSlice[self_origin](ptr=ptr, length=length)

    fn unsafe_set(mut self: StringArray[mut=True], index: Int, value: String) raises:
        """Replace the string at the given index in place.

        Raises:
            If the new string length differs from the existing one.
        """
        var start_offset = self.offsets[].unsafe_get[DType.int32](index)
        var end_offset = self.offsets[].unsafe_get[DType.int32](index + 1)
        var length = Int(end_offset - start_offset)

        if length != len(value):
            raise Error(
                "String length mismatch, inplace update must have the same"
                " length"
            )

        var dst_address = self.values[].ptr + Int(start_offset)
        var src_address = value.unsafe_ptr()
        memcpy(dest=dst_address, src=src_address, count=length)

    fn __getitem__[
        self_origin: Origin[mut=False], //
    ](ref [self_origin] self, index: Int) raises -> StringSlice[self_origin]:
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

    fn shrink_to_fit(mut self: StringArray[mut=True]):
        """Shrink buffers to exact length and normalize offset to 0 in place."""
        if self.length == self.capacity and self.offset == 0:
            return

        var start_byte = Int(
            self.offsets[].unsafe_get[DType.uint32](self.offset)
        )
        var end_byte = Int(
            self.offsets[].unsafe_get[DType.uint32](self.offset + self.length)
        )

        # Compact offsets, then rebase by subtracting start_byte
        self.offsets[].offset = self.offset
        self.offsets[].resize[DType.uint32](self.length + 1)
        for i in range(self.length + 1):
            var off = Int(self.offsets[].unsafe_get[DType.uint32](i))
            self.offsets[].unsafe_set[DType.uint32](i, UInt32(off - start_byte))

        self.values[].offset = start_byte
        self.values[].resize(end_byte - start_byte)
        self.bitmap[].resize(self.length, self.offset)
        self.offset = 0
        self.capacity = self.length

    fn freeze(deinit self: StringArray[mut=True]) -> StringArray:
        """Shrink buffers to exact length and return as an immutable array."""
        self.shrink_to_fit()
        return StringArray(self^)


struct ListArray[*, mut: Bool = False](Movable, Sized):
    """An Arrow array of variable-length lists (each element is a sub-array).

    Immutable by default (`mut=False`).  Use `ListArray.from_values()` to
    create a mutable builder, then call `freeze()` to obtain an immutable
    `ListArray`.
    """
    var dtype: DataType
    var length: Int
    var offset: Int
    var capacity: Int
    var bitmap: ArcPointer[Bitmap[mut=Self.mut]]
    var offsets: ArcPointer[Buffer[mut=Self.mut]]
    var values: ArcPointer[Array]

    fn __init__(out self: ListArray, ref data: Array) raises:
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

    fn __init__[
        other_mut: Bool, //
    ](out self: ListArray, deinit other: ListArray[mut=other_mut]):
        self.dtype = other.dtype^
        self.length = other.length
        self.offset = other.offset
        self.capacity = other.capacity
        self.bitmap = rebind[ArcPointer[Bitmap]](other.bitmap^)
        self.offsets = rebind[ArcPointer[Buffer]](other.offsets^)
        self.values = other.values^

    fn __init__(
        out self: ListArray[mut=True],
        var dtype: DataType,
        length: Int,
        offset: Int,
        capacity: Int,
        var bitmap: ArcPointer[Bitmap[mut=True]],
        var offsets: ArcPointer[Buffer[mut=True]],
        var values: ArcPointer[Array],
    ):
        """Builder constructor for creating a mutable ListArray."""
        self.dtype = dtype^
        self.length = length
        self.offset = offset
        self.capacity = capacity
        self.bitmap = bitmap^
        self.offsets = offsets^
        self.values = values^

    @staticmethod
    fn from_values(var values: Array, capacity: Int = 1) raises -> ListArray[mut=True]:
        """Create a mutable ListArray builder wrapping the given values.

        Returns a mutable `ListArray[mut=True]` so that additional entries can
        be appended.  Call `freeze()` when done to obtain an immutable
        `ListArray`.

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
        return ListArray[mut=True](
            dtype=list_dtype^,
            length=1,
            offset=0,
            capacity=capacity,
            bitmap=ArcPointer(bitmap^),
            offsets=ArcPointer(offsets^),
            values=ArcPointer(values^),
        )

    fn __len__(self) -> Int:
        return self.length

    fn is_valid(self, index: Int) -> Bool:
        return self.bitmap[].unsafe_get(index)

    fn unsafe_append(mut self: ListArray[mut=True], is_valid: Bool):
        self.bitmap[].unsafe_set(self.length, is_valid)
        self.offsets[].unsafe_set[DType.uint32](
            self.length + 1, UInt32(self.values[].length)
        )
        self.length += 1

    fn shrink_to_fit(mut self: ListArray[mut=True]):
        """Shrink buffers to exact length and normalize offset to 0 in place."""
        if self.length == self.capacity and self.offset == 0:
            return

        self.offsets[].offset = self.offset
        self.offsets[].resize[DType.uint32](self.length + 1)
        self.bitmap[].resize(self.length, self.offset)
        self.offset = 0
        self.capacity = self.length

    fn freeze(deinit self: ListArray[mut=True]) -> ListArray:
        """Shrink buffers to exact length and return as an immutable array."""
        self.shrink_to_fit()
        return ListArray(self^)

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


struct StructArray[*, mut: Bool = False](Movable, Sized):
    """An Arrow array of structs (each element is a collection of named fields).

    Immutable by default (`mut=False`).  Use
    `StructArray[mut=True](fields, capacity=n)` to create a mutable builder,
    then call `freeze()` to obtain an immutable `StructArray`.
    """
    var dtype: DataType
    var length: Int
    var capacity: Int
    var bitmap: ArcPointer[Bitmap[mut=Self.mut]]
    var children: List[ArcPointer[Array]]

    fn __init__(
        out self: StructArray[mut=True],
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

    fn __init__(out self: StructArray, *, ref data: Array):
        self.dtype = data.dtype.copy()
        self.length = data.length
        self.capacity = data.length
        self.bitmap = data.bitmap
        self.children = data.children.copy()

    fn __init__[
        other_mut: Bool, //
    ](out self: StructArray, deinit other: StructArray[mut=other_mut]):
        self.dtype = other.dtype^
        self.length = other.length
        self.capacity = other.capacity
        self.bitmap = rebind[ArcPointer[Bitmap]](other.bitmap^)
        self.children = other.children^

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

    fn shrink_to_fit(mut self: StructArray[mut=True]):
        """Shrink bitmap to exact length in place."""
        if self.length == self.capacity:
            return

        self.bitmap[].resize(self.length)
        self.capacity = self.length

    fn freeze(deinit self: StructArray[mut=True]) -> StructArray:
        """Shrink bitmap to exact length and return as an immutable array."""
        self.shrink_to_fit()
        return StructArray(self^)


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
        var bitmap = Bitmap.alloc(self.length)
        var buffers = List[ArcPointer[Buffer]]()
        var children = List[ArcPointer[Array]]()
        var start = 0
        while self.chunks:
            var chunk = self.chunks.pop(0)
            var chunk_length = chunk.length
            bitmap.extend(chunk.bitmap[], start, chunk_length)
            for i in range(len(chunk.buffers)):
                buffers.append(chunk.buffers[i])
            for i in range(len(chunk.children)):
                children.append(chunk.children[i])
            start += chunk_length
        combined = Array(
            dtype=self.dtype.copy(),
            length=self.length,
            bitmap=ArcPointer(bitmap^.freeze()),
            buffers=buffers^,
            children=children^,
            offset=0,
        )


fn array[T: DataType]() -> PrimitiveArray[T]:
    """Create an empty immutable primitive array."""
    return PrimitiveArray[T, mut=True](0).freeze()


fn array[T: DataType](values: List[Optional[Int]]) -> PrimitiveArray[T]:
    """Create an immutable primitive array from a list of values.

    `None` entries become null.
    """
    var a = PrimitiveArray[T, mut=True](len(values))
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
    var a = BoolArray[mut=True](len(values))
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
    var a = PrimitiveArray[T, mut=True](capacity=size)
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
    var a = PrimitiveArray[T, mut=True](end - start)
    for i in range(start, end):
        a.unsafe_append(Scalar[T.native](i))
    return a^.freeze()

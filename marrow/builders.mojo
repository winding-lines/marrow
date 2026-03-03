"""Array builders for constructing Arrow arrays incrementally.

`Builder` is the type-erased, ref-counted mutable handle — the mutable counterpart
of `Array`.  Typed builders (`PrimitiveBuilder[T]`, `StringBuilder`,
`ListBuilder`, `FixedSizeListBuilder`, `StructBuilder`) each hold an
`ArcPointer[BuilderData]` and expose a type-safe `append` / `finish` API
modelled after Arrow C++'s builder hierarchy.

`PrimitiveBuilder[T]`, and `StringBuilder` implicitly convert to `Builder`,
enabling them to be passed directly as child builders to composite types.
The conversion clones the `ArcPointer`, so the typed builder remains usable after
passing it to a composite builder.

`BoolBuilder` is an alias for `PrimitiveBuilder[bool_]`.

Example
-------
    var b = PrimitiveBuilder[int64](capacity=1024)
    b.append(42)
    b.append_null()
    var arr = b.finish()    # PrimitiveArray[int64]

    # PrimitiveBuilder implicitly converts to Builder
    var child = PrimitiveBuilder[float32](capacity=64)
    var list_b = ListBuilder(child, capacity=10)
    child.append(1.0)   # child still usable — shared ArcPointer
"""

from memory import memcpy, ArcPointer
from sys import size_of
from .buffers import Buffer, BufferBuilder, bitmap_count_ones
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


# ---------------------------------------------------------------------------
# BuilderData — internal mutable core
# ---------------------------------------------------------------------------


struct BuilderData(Movable):
    """Internal mutable builder data — the mutable counterpart of `Array`.

    Layout mirrors `Array`:
      - `bitmap`   — null-validity bit-buffer (always directly owned)
      - `buffers`  — data buffers, each ref-counted via `ArcPointer[BufferBuilder]`
      - `children` — child builders for nested types, each an `ArcPointer[BuilderData]`

    Wrap in `ArcPointer[BuilderData]` (or use the `Builder` wrapper) for shared ownership.
    Call `finish()` to consume and produce an immutable `Array`.
    """

    var dtype: DataType
    var length: Int
    var capacity: Int
    var bitmap: BufferBuilder
    # TODO: only required arcpointer for bufferbuilders because List requires copyable elements
    var buffers: List[ArcPointer[BufferBuilder]]
    var children: List[ArcPointer[BuilderData]]

    fn __init__(
        out self,
        var dtype: DataType,
        length: Int,
        capacity: Int,
        var bitmap: BufferBuilder,
        var buffers: List[ArcPointer[BufferBuilder]],
        var children: List[ArcPointer[BuilderData]],
    ):
        self.dtype = dtype^
        self.length = length
        self.capacity = capacity
        self.bitmap = bitmap^
        self.buffers = buffers^
        self.children = children^

    fn __len__(self) -> Int:
        return self.length

    # TODO: must do shrink_to_fit!
    # TODO: maintain null count and remove bitmap count ones
    fn finish(mut self) -> Array:
        """Snapshot the builder into an immutable `Array` and reset state.

        After this call the builder is reset to empty (length=0, capacity=0).
        Each buffer and bitmap is transferred to the returned Array; fresh
        empty allocations are installed so the builder can be reused.
        """
        var frozen_bitmap = self.bitmap.finish()
        var nulls = self.length - bitmap_count_ones(frozen_bitmap, frozen_bitmap.size)
        var result = Array(
            dtype=self.dtype.copy(),
            length=self.length,
            nulls=nulls,
            bitmap=frozen_bitmap^,
            buffers=self._finish_buffers(),
            children=self._finish_children(),
            offset=0,
        )
        self.length = 0
        self.capacity = 0
        return result^

    fn _finish_buffers(mut self) -> List[Buffer]:
        # only to align with _finish_children which is required to avoid recursion warning
        var frozen_buffers = List[Buffer]()
        for i in range(len(self.buffers)):
            frozen_buffers.append(self.buffers[i][].finish())
        return frozen_buffers^

    fn _finish_children(mut self) -> List[Array]:
        # indirect call to avoid "self recursive call will cause an infinite loop" warning
        var result = List[Array]()
        for i in range(len(self.children)):
            result.append(self.children[i][].finish())
        return result^


# ---------------------------------------------------------------------------
# Builder — type-erased handle
# ---------------------------------------------------------------------------


struct Builder(Copyable, Movable):
    """Type-erased ref-counted handle to `BuilderData` — the mutable counterpart of `Array`.

    `PrimitiveBuilder[T]` and `StringBuilder` implicitly convert to `Builder` by
    cloning the shared `ArcPointer`, so the original typed builder remains usable
    after the conversion.  (`BoolBuilder` is an alias for `PrimitiveBuilder[bool_]`
    and therefore converts the same way.)

    Used as the child builder type for `ListBuilder`, `FixedSizeListBuilder`, and `StructBuilder`.
    """

    var data: ArcPointer[BuilderData]

    fn __init__(out self, var data: ArcPointer[BuilderData]):
        self.data = data^

    fn __copyinit__(out self, copy: Self):
        self.data = copy.data

    @implicit
    fn __init__[T: DataType](out self, value: PrimitiveBuilder[T]):
        self.data = value.data

    @implicit
    fn __init__(out self, value: StringBuilder):
        self.data = value.data

    @implicit
    fn __init__(out self, value: ListBuilder):
        self.data = value.data


# ---------------------------------------------------------------------------
# PrimitiveBuilder
# ---------------------------------------------------------------------------


struct PrimitiveBuilder[T: DataType](Movable, Sized):
    """Builder for fixed-size primitive arrays (integers, floats).

    buffers[0] — element data
    """

    comptime dtype = Self.T
    comptime scalar = Scalar[Self.T.native]

    var data: ArcPointer[BuilderData]

    fn __init__(out self, capacity: Int = 0):
        self.data = ArcPointer(
            BuilderData(
                dtype=materialize[Self.T](),
                length=0,
                capacity=capacity,
                bitmap=BufferBuilder.alloc[DType.bool](capacity),
                buffers=[
                    ArcPointer(BufferBuilder.alloc[Self.T.native](capacity))
                ],
                children=List[ArcPointer[BuilderData]](),
            )
        )

    fn __len__(self) -> Int:
        return self.data[].length

    fn grow(mut self, capacity: Int) raises:
        self.data[].bitmap.resize[DType.bool](capacity)
        self.data[].buffers[0][].resize[Self.T.native](capacity)
        self.data[].capacity = capacity

    # @always_inline
    fn append(mut self, value: Self.scalar) raises:
        if self.data[].length >= self.data[].capacity:
            self.grow(max(self.data[].capacity * 2, self.data[].length + 1))
        self.data[].bitmap.unsafe_set[DType.bool](self.data[].length, True)
        self.data[].buffers[0][].unsafe_set[Self.T.native](
            self.data[].length, value
        )
        self.data[].length += 1

    fn append(mut self, value: Bool) raises:
        comptime assert (
            Self.T == bool_
        ), "append(Bool) only supported for PrimitiveBuilder[bool_]"
        self.append(Self.scalar(value))

    @always_inline
    fn append_null(mut self) raises:
        if self.data[].length >= self.data[].capacity:
            self.grow(max(self.data[].capacity * 2, self.data[].length + 1))
        self.data[].bitmap.unsafe_set[DType.bool](self.data[].length, False)
        self.data[].length += 1

    fn extend(mut self, values: List[Self.scalar]) raises:
        var new_len = self.data[].length + len(values)
        if new_len >= self.data[].capacity:
            self.grow(max(self.data[].capacity * 2, new_len))
        for value in values:
            self.append(value)

    fn extend(mut self, values: List[Self.scalar], valid: List[Bool]) raises:
        for i in range(len(values)):
            if valid[i]:
                self.append(values[i])
            else:
                self.append_null()

    fn append_nulls(mut self, n: Int) raises:
        for _ in range(n):
            self.append_null()

    fn reserve(mut self, additional: Int) raises:
        var needed = self.data[].length + additional
        if needed > self.data[].capacity:
            self.grow(needed)

    fn finish(mut self) raises -> PrimitiveArray[Self.T]:
        return self.data[].finish().as_primitive[Self.T]()


# ---------------------------------------------------------------------------
# StringBuilder
# ---------------------------------------------------------------------------


struct StringBuilder(Movable, Sized):
    """Builder for variable-length UTF-8 string arrays.

    buffers[0] — uint32 offsets
    buffers[1] — utf-8 byte data (grown on demand)
    """

    var data: ArcPointer[BuilderData]

    fn __init__(out self, capacity: Int = 0):
        var offsets = BufferBuilder.alloc[DType.uint32](capacity + 1)
        offsets.unsafe_set[DType.uint32](0, 0)
        self.data = ArcPointer(
            BuilderData(
                dtype=materialize[string](),
                length=0,
                capacity=capacity,
                bitmap=BufferBuilder.alloc[DType.bool](capacity),
                buffers=[
                    ArcPointer(offsets^),
                    ArcPointer(BufferBuilder.alloc[DType.uint8](capacity)),
                ],
                children=List[ArcPointer[BuilderData]](),
            )
        )

    fn __len__(self) -> Int:
        return self.data[].length

    fn grow(mut self, capacity: Int) raises:
        self.data[].bitmap.resize[DType.bool](capacity)
        self.data[].buffers[0][].resize[DType.uint32](capacity + 1)
        self.data[].capacity = capacity

    fn append(mut self, value: String) raises:
        if self.data[].length >= self.data[].capacity:
            self.grow(max(self.data[].capacity * 2, self.data[].length + 1))
        var index = self.data[].length
        var last_offset = self.data[].buffers[0][].ptr.bitcast[UInt32]()[index]
        var next_offset = last_offset + UInt32(len(value))
        self.data[].length += 1
        self.data[].bitmap.unsafe_set[DType.bool](index, True)
        self.data[].buffers[0][].unsafe_set[DType.uint32](
            index + 1, next_offset
        )
        self.data[].buffers[1][].resize[DType.uint8](next_offset)
        memcpy(
            dest=self.data[].buffers[1][].ptr + Int(last_offset),
            src=value.unsafe_ptr(),
            count=len(value),
        )

    fn append_null(mut self) raises:
        if self.data[].length >= self.data[].capacity:
            self.grow(max(self.data[].capacity * 2, self.data[].length + 1))
        var index = self.data[].length
        var last_offset = self.data[].buffers[0][].ptr.bitcast[UInt32]()[index]
        self.data[].length += 1
        self.data[].bitmap.unsafe_set[DType.bool](index, False)
        self.data[].buffers[0][].unsafe_set[DType.uint32](
            index + 1, last_offset
        )

    fn extend(mut self, values: List[String], valid: List[Bool]) raises:
        for i in range(len(values)):
            if valid[i]:
                self.append(values[i])
            else:
                self.append_null()

    fn append_nulls(mut self, n: Int) raises:
        for _ in range(n):
            self.append_null()

    fn reserve(mut self, additional: Int) raises:
        var needed = self.data[].length + additional
        if needed > self.data[].capacity:
            self.grow(needed)

    fn finish(mut self) raises -> StringArray:
        return self.data[].finish().as_string()


# ---------------------------------------------------------------------------
# ListBuilder
# ---------------------------------------------------------------------------


struct ListBuilder(Movable, Sized):
    """Builder for variable-length list arrays.

    buffers[0]  — uint32 offsets
    children[0] — child element builder (Builder)
    """

    var data: ArcPointer[BuilderData]

    fn __init__(out self, var child: Builder, capacity: Int = 0):
        var offsets = BufferBuilder.alloc[DType.uint32](capacity + 1)
        offsets.unsafe_set[DType.uint32](0, 0)
        var child_dtype = child.data[].dtype.copy()
        self.data = ArcPointer(
            BuilderData(
                dtype=list_(child_dtype^),
                length=0,
                capacity=capacity,
                bitmap=BufferBuilder.alloc[DType.bool](capacity),
                buffers=[ArcPointer(offsets^)],
                children=[child.data],
            )
        )

    fn __len__(self) -> Int:
        return self.data[].length

    fn child(self) -> Builder:
        return Builder(self.data[].children[0])

    fn grow(mut self, capacity: Int) raises:
        self.data[].bitmap.resize[DType.bool](capacity)
        self.data[].buffers[0][].resize[DType.uint32](capacity + 1)
        self.data[].capacity = capacity

    fn append(mut self, is_valid: Bool) raises:
        if self.data[].length >= self.data[].capacity:
            self.grow(max(self.data[].capacity * 2, self.data[].length + 1))
        self.data[].bitmap.unsafe_set[DType.bool](self.data[].length, is_valid)
        var child_length = self.data[].children[0][].length
        self.data[].buffers[0][].unsafe_set[DType.uint32](
            self.data[].length + 1, UInt32(child_length)
        )
        self.data[].length += 1

    fn append_null(mut self) raises:
        self.append(False)

    fn append_nulls(mut self, n: Int) raises:
        for _ in range(n):
            self.append_null()

    fn reserve(mut self, additional: Int) raises:
        var needed = self.data[].length + additional
        if needed > self.data[].capacity:
            self.grow(needed)

    fn finish(mut self) raises -> ListArray:
        return self.data[].finish().as_list()


# ---------------------------------------------------------------------------
# FixedSizeListBuilder
# ---------------------------------------------------------------------------


struct FixedSizeListBuilder(Movable, Sized):
    """Builder for fixed-size list arrays.

    children[0] — child element builder (Builder)
    """

    var data: ArcPointer[BuilderData]

    fn __init__(
        out self, var child: Builder, list_size: Int, capacity: Int = 0
    ):
        var child_dtype = child.data[].dtype.copy()
        self.data = ArcPointer(
            BuilderData(
                dtype=fixed_size_list_(child_dtype^, list_size),
                length=0,
                capacity=capacity,
                bitmap=BufferBuilder.alloc[DType.bool](capacity),
                buffers=List[ArcPointer[BufferBuilder]](),
                children=[child.data],
            )
        )

    fn __len__(self) -> Int:
        return self.data[].length

    fn child(self) -> Builder:
        return Builder(self.data[].children[0])

    fn grow(mut self, capacity: Int) raises:
        self.data[].bitmap.resize[DType.bool](capacity)
        self.data[].capacity = capacity

    fn append(mut self, is_valid: Bool) raises:
        if self.data[].length >= self.data[].capacity:
            self.grow(max(self.data[].capacity * 2, self.data[].length + 1))
        self.data[].bitmap.unsafe_set[DType.bool](self.data[].length, is_valid)
        self.data[].length += 1

    fn append_null(mut self) raises:
        self.append(False)

    fn append_nulls(mut self, n: Int) raises:
        for _ in range(n):
            self.append_null()

    fn reserve(mut self, additional: Int) raises:
        var needed = self.data[].length + additional
        if needed > self.data[].capacity:
            self.grow(needed)

    fn finish(mut self) raises -> FixedSizeListArray:
        return self.data[].finish().as_fixed_size_list()


# ---------------------------------------------------------------------------
# StructBuilder
# ---------------------------------------------------------------------------


struct StructBuilder(Movable, Sized):
    """Builder for struct arrays.

    children[i] — field builder for field i (Builder)
    """

    var data: ArcPointer[BuilderData]

    fn __init__(
        out self,
        var fields: List[Field],
        var field_builders: List[Builder],
        capacity: Int = 0,
    ):
        var children = List[ArcPointer[BuilderData]]()
        for i in range(len(field_builders)):
            children.append(field_builders[i].data)
        self.data = ArcPointer(
            BuilderData(
                dtype=struct_(fields),
                length=0,
                capacity=capacity,
                bitmap=BufferBuilder.alloc[DType.bool](capacity),
                buffers=List[ArcPointer[BufferBuilder]](),
                children=children^,
            )
        )

    fn __len__(self) -> Int:
        return self.data[].length

    fn child(self, index: Int) -> Builder:
        return Builder(self.data[].children[index])

    fn grow(mut self, capacity: Int) raises:
        self.data[].bitmap.resize[DType.bool](capacity)
        self.data[].capacity = capacity

    fn append(mut self, is_valid: Bool) raises:
        if self.data[].length >= self.data[].capacity:
            self.grow(max(self.data[].capacity * 2, self.data[].length + 1))
        self.data[].bitmap.unsafe_set[DType.bool](self.data[].length, is_valid)
        self.data[].length += 1

    fn append_null(mut self) raises:
        self.append(False)

    fn append_nulls(mut self, n: Int) raises:
        for _ in range(n):
            self.append_null()

    fn reserve(mut self, additional: Int) raises:
        var needed = self.data[].length + additional
        if needed > self.data[].capacity:
            self.grow(needed)

    fn finish(mut self) raises -> StructArray:
        return StructArray(data=self.data[].finish())


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

comptime BoolBuilder = PrimitiveBuilder[bool_]
comptime Int8Builder = PrimitiveBuilder[int8]
comptime Int16Builder = PrimitiveBuilder[int16]
comptime Int32Builder = PrimitiveBuilder[int32]
comptime Int64Builder = PrimitiveBuilder[int64]
comptime UInt8Builder = PrimitiveBuilder[uint8]
comptime UInt16Builder = PrimitiveBuilder[uint16]
comptime UInt32Builder = PrimitiveBuilder[uint32]
comptime UInt64Builder = PrimitiveBuilder[uint64]
comptime Float32Builder = PrimitiveBuilder[float32]
comptime Float64Builder = PrimitiveBuilder[float64]


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


fn array[T: DataType]() raises -> PrimitiveArray[T]:
    """Create an empty primitive array."""
    var b = PrimitiveBuilder[T](0)
    return b.finish()


fn array[T: DataType](values: List[Optional[Int]]) raises -> PrimitiveArray[T]:
    """Create a primitive array from optional ints (`None` → null)."""
    var b = PrimitiveBuilder[T](len(values))
    for value in values:
        if value:
            b.append(Scalar[T.native](value.value()))
        else:
            b.append_null()
    return b.finish()


fn array(values: List[Optional[Bool]]) raises -> BoolArray:
    """Create a boolean array from optional bools (`None` → null)."""
    var b = BoolBuilder(len(values))
    for value in values:
        if value:
            b.append(Scalar[bool_.native](value.value()))
        else:
            b.append_null()
    return b.finish()


fn nulls[T: DataType](size: Int) raises -> PrimitiveArray[T]:
    """Create a primitive array of `size` null values."""
    var b = PrimitiveBuilder[T](capacity=size)
    b.data[].length = size
    return b.finish()


fn arange[T: DataType](start: Int, end: Int) raises -> PrimitiveArray[T]:
    """Create an integer array with values [start, end)."""
    comptime assert T.is_integer(), "arange() only supports integer DataTypes"
    var b = PrimitiveBuilder[T](end - start)
    for i in range(start, end):
        b.append(Scalar[T.native](i))
    return b.finish()

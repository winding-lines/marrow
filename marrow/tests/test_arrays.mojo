from testing import assert_equal, assert_true, assert_false, TestSuite
from marrow.arrays import *
from marrow.builders import (
    BoolBuilder,
    PrimitiveBuilder,
    StringBuilder,
    ListBuilder,
    FixedSizeListBuilder,
    StructBuilder,
)
from marrow.dtypes import *
from marrow.buffers import Buffer, BufferBuilder, bitmap_set, bitmap_range_set
from marrow.compute.filter import drop_nulls
from marrow.test_fixtures.arrays import (
    assert_bitmap_set,
    buffer_from,
    build_list_of_list,
    build_struct,
)


# --- Array (base) tests ---


def test_array_data_with_offset():
    """Test ArrayData with offset functionality."""
    # Create ArrayData with offset
    var bitmap = BufferBuilder.alloc_bits(10)
    var buffer = BufferBuilder.alloc[int8.native](10)

    # Set some data in the buffer
    buffer.unsafe_set[int8.native](2, 100)
    buffer.unsafe_set[int8.native](3, 200)
    buffer.unsafe_set[int8.native](4, 300)

    # Set validity bits
    bitmap_set(bitmap.ptr, 2, True)
    bitmap_set(bitmap.ptr, 3, True)
    bitmap_set(bitmap.ptr, 4, True)

    # Create ArrayData with offset=2
    var buffers = List[Buffer]()
    buffers.append(buffer^.freeze())
    var array_data = Array(
        dtype=materialize[int8](),
        length=3,
        bitmap=bitmap^.freeze(),
        buffers=buffers^,
        children=List[Array](),
        offset=2,
    )

    assert_equal(array_data.offset, 2)

    # Test is_valid with offset
    assert_true(array_data.is_valid(0))  # Should check bitmap[2]
    assert_true(array_data.is_valid(1))  # Should check bitmap[3]
    assert_true(array_data.is_valid(2))  # Should check bitmap[4]


def test_array_data_fieldwise_init():
    """Test that @fieldwise_init decorator works with offset field."""
    var bitmap = BufferBuilder.alloc_bits(5).freeze()
    var buffer = BufferBuilder.alloc[int8.native](5).freeze()

    # Test creating ArrayData with all fields specified including offset
    var buffers = List[Buffer]()
    buffers.append(buffer)
    var array_data = Array(
        dtype=materialize[int8](),
        length=5,
        bitmap=bitmap,
        buffers=buffers^,
        children=List[Array](),
        offset=3,
    )

    assert_equal(array_data.dtype, materialize[int8]())
    assert_equal(array_data.length, 5)
    assert_equal(array_data.offset, 3)


def test_array_from_primitive():
    var prim = array[int32]([1, 2, 3])
    var a = Array(prim^)
    assert_equal(a.length, 3)
    assert_equal(a.dtype, materialize[int32]())


def test_array_from_string():
    var s = StringBuilder()
    s.unsafe_append("hello")
    s.unsafe_append("world")
    var a = Array(s^.freeze())
    assert_equal(a.length, 2)
    assert_true(a.dtype.is_string())


def test_array_from_list():
    var b = PrimitiveBuilder[int64]()
    var l = ListBuilder.from_values(b^.freeze())
    var a = Array(l^.freeze())
    assert_true(a.dtype.is_list())


def test_array_from_struct():
    var fields = [Field("x", materialize[int32]())]
    var s = StructBuilder(fields^, capacity=5)
    var a = Array(s^.freeze())
    assert_true(a.dtype.is_struct())


def test_array_copy():
    var src_buffers = List[Buffer]()
    src_buffers.append(BufferBuilder.alloc[int8.native](3).freeze())
    var src = Array(
        dtype=materialize[int8](),
        length=3,
        bitmap=BufferBuilder.alloc_bits(3).freeze(),
        buffers=src_buffers^,
        children=List[Array](),
        offset=0,
    )
    var copy = src.copy()
    assert_equal(copy.length, src.length)
    assert_equal(copy.dtype, src.dtype)
    assert_equal(copy.offset, src.offset)
    # Mutating copy's length does not affect src
    copy.length = 99
    assert_equal(src.length, 3)


def test_array_move():
    var a_buffers = List[Buffer]()
    a_buffers.append(BufferBuilder.alloc[int8.native](5).freeze())
    var a = Array(
        dtype=materialize[int8](),
        length=5,
        bitmap=BufferBuilder.alloc_bits(5).freeze(),
        buffers=a_buffers^,
        children=List[Array](),
        offset=0,
    )
    var b = a^
    assert_equal(b.length, 5)
    assert_equal(b.dtype, materialize[int8]())


# --- PrimitiveArray tests ---


def test_boolean_array():
    var a = BoolBuilder()
    assert_equal(len(a), 0)
    assert_equal(a.capacity, 0)

    a.resize(3)
    assert_equal(len(a), 0)
    assert_equal(a.capacity, 3)

    a.append(True)
    a.append(False)
    a.append(True)
    assert_equal(len(a), 3)
    assert_equal(a.capacity, 3)

    a.append(True)
    assert_equal(len(a), 4)
    assert_equal(a.capacity, 6)

    var frozen = a^.freeze()
    assert_true(frozen.is_valid(0))
    assert_true(frozen.is_valid(1))
    assert_true(frozen.is_valid(2))
    assert_true(frozen.is_valid(3))

    var d = Array(frozen^)
    assert_equal(d.length, 4)

    var b = d^.as_bool()


def test_append():
    var a = PrimitiveBuilder[int8]()
    assert_equal(len(a), 0)
    assert_equal(a.capacity, 0)
    a.append(1)
    a.append(2)
    a.append(3)
    assert_equal(len(a), 3)
    assert_true(a.capacity >= len(a))


def test_array_empty():
    var a = array[int32]()
    assert_equal(len(a), 0)


def test_array_from_ints():
    var g = array[int8]([1, 2])
    assert_equal(len(g), 2)
    assert_equal(materialize[g.dtype](), materialize[int8]())
    assert_equal(g.unsafe_get(0), 1)
    assert_equal(g.unsafe_get(1), 2)

    var b = array([True, False, True])
    assert_equal(len(b), 3)
    assert_equal(b.unsafe_get(0), True)
    assert_equal(b.unsafe_get(1), False)
    assert_equal(b.unsafe_get(2), True)


def test_array_with_nulls():
    var a = array[int32]([1, None, 3])
    assert_equal(len(a), 3)
    assert_equal(a.null_count(), 1)
    assert_true(a.is_valid(0))
    assert_false(a.is_valid(1))
    assert_true(a.is_valid(2))
    assert_equal(a.unsafe_get(0), 1)
    assert_equal(a.unsafe_get(2), 3)

    var b = array([True, None, False])
    assert_equal(len(b), 3)
    assert_true(b.is_valid(0))
    assert_false(b.is_valid(1))
    assert_true(b.is_valid(2))


def test_arange():
    var a = arange[int32](1, 5)
    assert_equal(len(a), 4)
    assert_equal(a.unsafe_get(0), 1)
    assert_equal(a.unsafe_get(1), 2)
    assert_equal(a.unsafe_get(2), 3)
    assert_equal(a.unsafe_get(3), 4)

    var b = arange[uint8](0, 3)
    assert_equal(len(b), 3)
    assert_equal(b.unsafe_get(0), 0)
    assert_equal(b.unsafe_get(2), 2)


def test_arange_empty():
    var a = arange[int32](5, 5)
    assert_equal(len(a), 0)


def test_arange_single():
    var a = arange[int64](7, 8)
    assert_equal(len(a), 1)
    assert_equal(a.unsafe_get(0), 7)


def test_arange_validity():
    var a = arange[int16](0, 4)
    for i in range(4):
        assert_true(a.is_valid(i))


def test_arange_int8():
    var a = arange[int8](10, 15)
    assert_equal(len(a), 5)
    assert_equal(a.unsafe_get(0), 10)
    assert_equal(a.unsafe_get(4), 14)


def test_arange_uint64():
    var a = arange[uint64](100, 103)
    assert_equal(len(a), 3)
    assert_equal(a.unsafe_get(0), 100)
    assert_equal(a.unsafe_get(2), 102)


def test_drop_null() -> None:
    """Test the drop null function via the compute module."""
    from marrow.compute.filter import drop_nulls

    var primitive_array = array[uint8](
        [None, 1, None, 3, None, 5, None, 7, None, 9]
    )
    # Check the setup.
    assert_equal(primitive_array.null_count(), 5)
    assert_bitmap_set(primitive_array.bitmap, primitive_array.length, [1, 3, 5, 7, 9], "check setup")

    var result = drop_nulls[uint8](primitive_array)
    assert_equal(result.unsafe_get(0), 1)
    assert_equal(result.unsafe_get(1), 3)
    assert_equal(result.null_count(), 0)


def test_primitive_array_with_offset():
    """Test PrimitiveArray with offset functionality."""
    # Create a builder, populate, freeze, then create offset view
    var b = PrimitiveBuilder[int32](10)
    b.unsafe_append(100)
    b.unsafe_append(200)
    b.unsafe_append(300)
    b.unsafe_append(400)
    b.unsafe_append(500)
    var arr = b^.freeze()

    # Default offset should be 0
    assert_equal(arr.offset, 0)
    assert_equal(arr.unsafe_get(0), 100)
    assert_equal(arr.unsafe_get(1), 200)

    # Create a copy of array with offset, should point to the same buffers.
    var arr_buffers = List[Buffer]()
    arr_buffers.append(arr.buffer)
    var arr_data = Array(
        dtype=materialize[int32](),
        length=arr.length,
        bitmap=arr.bitmap,
        buffers=arr_buffers^,
        children=List[Array](),
        offset=arr.offset,
    )
    var arr_with_offset = PrimitiveArray[int32](arr_data, offset=2)
    assert_equal(arr_with_offset.offset, 2)

    # Test that offset affects get operations
    assert_equal(arr_with_offset.unsafe_get(0), 300)  # Should get arr[2]
    assert_equal(arr_with_offset.unsafe_get(1), 400)  # Should get arr[3]
    assert_equal(arr_with_offset.unsafe_get(2), 500)  # Should get arr[4]


def test_primitive_builder_moveinit_with_offset():
    """Test __moveinit__ preserves offset on builders."""
    var b = PrimitiveBuilder[int16](5, offset=3)
    b.unsafe_append(123)

    var moved = b^
    assert_equal(moved.offset, 3)
    var frozen = moved^.freeze()
    assert_equal(frozen.unsafe_get(0), 123)


def test_primitive_builder_constructor_with_offset():
    """Test PrimitiveBuilder constructor with offset parameter."""
    var b1 = PrimitiveBuilder[int8](10)  # Default offset=0
    assert_equal(b1.offset, 0)

    var b2 = PrimitiveBuilder[int8](10, offset=5)  # Explicit offset
    assert_equal(b2.offset, 5)


def test_primitive_builder_offset_with_validity():
    """Test that offset works correctly with validity bitmap."""
    var b = PrimitiveBuilder[uint8](10, offset=1)

    # Set some values with validity
    b.unsafe_append(42)  # This should set buffer[1] and bitmap[1]
    b.unsafe_append(43)  # This should set buffer[2] and bitmap[2]

    # Verify values are accessible through the frozen array
    var frozen = b^.freeze()
    assert_equal(frozen.unsafe_get(0), 42)
    assert_equal(frozen.unsafe_get(1), 43)

    # Verify bitmap is correct
    assert_true(frozen.is_valid(0))
    assert_true(frozen.is_valid(1))


def test_primitive_array_nulls_with_offset():
    """Test nulls() creates an array with all null values and default offset."""
    var null_arr = nulls[int64](5)
    assert_equal(null_arr.offset, 0)

    # All elements should be invalid (null)
    for i in range(5):
        assert_false(null_arr.is_valid(i))


# --- StringArray tests ---


def test_string_builder():
    var a = StringBuilder()
    assert_equal(len(a), 0)
    assert_equal(a.capacity, 0)

    a.resize(2)
    assert_equal(len(a), 0)
    assert_equal(a.capacity, 2)

    a.unsafe_append("hello")
    a.unsafe_append("world")
    assert_equal(len(a), 2)
    assert_equal(a.capacity, 2)

    var frozen = a^.freeze()
    assert_equal(String(frozen.unsafe_get(0)), "hello")
    assert_equal(String(frozen.unsafe_get(1)), "world")


# --- ListArray / StructArray tests ---



def test_list_bool_array():
    var bools = array([True, False, True])

    var builder = ListBuilder.from_values(bools^)
    var lists = builder^.freeze()
    assert_equal(len(lists), 1)
    var first_value = lists.unsafe_get(0)
    var bool_array = BoolArray(first_value)
    assert_equal(bool_array.unsafe_get(0), True)
    assert_equal(bool_array.unsafe_get(1), False)
    assert_equal(bool_array.unsafe_get(2), True)


def test_list_str():
    var strings = StringBuilder()
    strings.unsafe_append("hello")
    strings.unsafe_append("world")

    var builder = ListBuilder.from_values(strings^.freeze())
    var lists = builder^.freeze()
    var first_value = StringArray(lists.unsafe_get(0))

    assert_equal(first_value.unsafe_get(0), "hello")
    assert_equal(first_value.unsafe_get(1), "world")


def test_list_of_list():
    list2 = build_list_of_list[int64]()
    top = ListArray(list2.unsafe_get(0))
    middle_0 = top.unsafe_get(0)
    bottom = PrimitiveArray[int64](middle_0^)
    assert_equal(bottom.unsafe_get(1), 2)
    assert_equal(bottom.unsafe_get(0), 1)
    middle_1 = top.unsafe_get(1)
    bottom = PrimitiveArray[int64](middle_1^)
    assert_equal(bottom.unsafe_get(0), 3)
    assert_equal(bottom.unsafe_get(1), 4)


def test_fixed_size_list_int_array():
    """Construct a FixedSizeListArray of int64 lists, size=3."""
    var ints = array[int64]([1, 2, 3, 4, 5, 6])
    var builder = FixedSizeListBuilder.from_values(Array(ints^), list_size=3)
    assert_equal(builder.dtype, fixed_size_list_(materialize[int64](), 3))
    assert_equal(len(builder), 2)
    var fsl = builder^.freeze()
    assert_equal(len(fsl), 2)
    assert_equal(fsl.dtype.size, 3)

    # First list: [1, 2, 3]
    var first = fsl.unsafe_get(0).as_int64()
    assert_equal(len(first), 3)
    assert_equal(first.unsafe_get(0), 1)
    assert_equal(first.unsafe_get(1), 2)
    assert_equal(first.unsafe_get(2), 3)

    # Second list: [4, 5, 6]
    var second = fsl.unsafe_get(1).as_int64()
    assert_equal(second.unsafe_get(0), 4)
    assert_equal(second.unsafe_get(1), 5)
    assert_equal(second.unsafe_get(2), 6)


def test_fixed_size_list_roundtrip():
    """FixedSizeListArray -> Array -> as_fixed_size_list() roundtrip."""
    var ints = array[int32]([10, 20, 30, 40])
    var builder = FixedSizeListBuilder.from_values(Array(ints^), list_size=2)
    var fsl = builder^.freeze()

    # Convert to Array and back
    var data = Array(fsl^)
    assert_true(data.dtype.is_fixed_size_list())
    assert_equal(data.dtype.size, 2)
    assert_equal(len(data.buffers), 0)
    assert_equal(len(data.children), 1)

    var fsl2 = data^.as_fixed_size_list()
    assert_equal(len(fsl2), 2)
    var first = fsl2.unsafe_get(0).as_int32()
    assert_equal(first.unsafe_get(0), 10)
    assert_equal(first.unsafe_get(1), 20)


def test_fixed_size_list_with_nulls():
    """FixedSizeListArray with null lists."""
    var ints = array[int64]([1, 2, 3, 4, 5, 6])
    var builder = FixedSizeListBuilder.from_values(
        Array(ints^), list_size=3, capacity=3
    )
    # Append a null list (need to extend values first)
    builder.unsafe_append(False)
    assert_equal(len(builder), 3)

    var fsl = builder^.freeze()
    assert_true(fsl.is_valid(0))
    assert_true(fsl.is_valid(1))
    assert_false(fsl.is_valid(2))


def test_fixed_size_list_pretty_print():
    """Pretty printing FixedSizeListArray."""
    var ints = array[int64]([1, 2, 3, 4])
    var builder = FixedSizeListBuilder.from_values(Array(ints^), list_size=2)
    var fsl = builder^.freeze()
    var s = String(Array(fsl^))
    assert_true("FixedSizeListArray" in s)


def test_struct_array():
    var fields = [
        Field("id", materialize[int64]()),
        Field("name", materialize[string]()),
        Field("active", materialize[bool_]()),
    ]

    var struct_builder = StructBuilder(fields^, capacity=10)
    assert_equal(len(struct_builder), 0)
    assert_equal(struct_builder.capacity, 10)

    var data = Array(struct_builder^.freeze())
    assert_equal(data.length, 0)
    assert_true(data.dtype.is_struct())
    assert_equal(len(data.dtype.fields), 3)
    assert_equal(data.dtype.fields[0].name, "id")
    assert_equal(data.dtype.fields[1].name, "name")
    assert_equal(data.dtype.fields[2].name, "active")


def test_struct_array_unsafe_get():
    var struct_array = build_struct()
    ref int_data_a = struct_array.unsafe_get("int_data_a")
    var int_a = PrimitiveArray[int32](int_data_a.copy())
    assert_equal(int_a.unsafe_get(0), 1)
    assert_equal(int_a.unsafe_get(4), 5)
    ref int_data_b = struct_array.unsafe_get("int_data_b")
    var int_b = PrimitiveArray[int32](int_data_b.copy())
    assert_equal(int_b.unsafe_get(0), 10)
    assert_equal(int_b.unsafe_get(2), 30)


# --- ChunkedArray tests ---


def test_chunked_array():
    var arrays = List[Array]()
    arrays.append(array[uint8]([0]))
    arrays.append(array[uint8]([0, 1]))

    var chunked_array = ChunkedArray(materialize[int8](), arrays^)
    assert_equal(chunked_array.length, 3)

    assert_equal(chunked_array.chunk(0).length, 1)
    var second_chunk = chunked_array.chunk(1).copy().as_uint8()
    assert_equal(second_chunk.length, 2)
    assert_equal(second_chunk.unsafe_get(0), 0)
    assert_equal(second_chunk.unsafe_get(1), 1)


def test_combine_chunked_array():
    var arrays = List[Array]()
    arrays.append(array[uint8]([0]))
    arrays.append(array[uint8]([0, 1]))

    var chunked_array = ChunkedArray(materialize[int8](), arrays^)
    assert_equal(chunked_array.length, 3)
    assert_equal(len(chunked_array.chunks), 2)
    assert_equal(chunked_array.chunk(1).copy().as_uint8().unsafe_get(1), 1)

    var combined_array = chunked_array^.combine_chunks()
    assert_equal(combined_array.length, 3)
    assert_equal(combined_array.dtype, materialize[int8]())
    # Ensure that the last element of the last buffer has the expected value.
    assert_equal(combined_array.buffers[1].unsafe_get(1), 1)


def test_primitive_freeze_zero_copy():
    """Freeze() on an exact-size builder moves buffers without allocation."""
    var a = PrimitiveBuilder[int64](capacity=3)
    a.unsafe_append(10)
    a.unsafe_append(20)
    a.unsafe_append(30)
    var frozen = a^.freeze()
    assert_equal(frozen.length, 3)
    assert_equal(frozen.offset, 0)
    assert_equal(frozen.unsafe_get(0), 10)
    assert_equal(frozen.unsafe_get(1), 20)
    assert_equal(frozen.unsafe_get(2), 30)


def test_primitive_freeze_shrinks():
    """Freeze() on an over-allocated builder trims capacity to length."""
    var a = PrimitiveBuilder[int64](capacity=100)
    a.unsafe_append(42)
    a.unsafe_append(99)
    var frozen = a^.freeze()
    assert_equal(frozen.length, 2)
    assert_equal(frozen.unsafe_get(0), 42)
    assert_equal(frozen.unsafe_get(1), 99)


def test_primitive_freeze_via_append():
    """Freeze() works on a builder built with append() (auto-grow capacity)."""
    var a = PrimitiveBuilder[int64]()
    a.append(1)
    a.append(2)
    a.append(3)
    var frozen = a^.freeze()
    assert_equal(frozen.length, 3)
    assert_equal(frozen.unsafe_get(0), 1)
    assert_equal(frozen.unsafe_get(2), 3)


def test_primitive_freeze_preserves_nulls():
    """Freeze() preserves null validity information."""
    var a = PrimitiveBuilder[int64](capacity=3)
    a.unsafe_append(1)
    a.unsafe_append_null()
    a.unsafe_append(3)
    var frozen = a^.freeze()
    assert_equal(frozen.length, 3)
    assert_true(frozen.is_valid(0))
    assert_false(frozen.is_valid(1))
    assert_true(frozen.is_valid(2))


def test_primitive_freeze_converts_to_array():
    """A frozen PrimitiveArray still converts to the Array base implicitly."""
    var a = PrimitiveBuilder[int64]()
    a.append(7)
    a.append(8)
    var frozen = a^.freeze()
    var base: Array = frozen
    assert_equal(base.length, 2)
    assert_equal(base.dtype, materialize[int64]())


def test_primitive_freeze_with_offset():
    """Array with offset views into frozen data correctly."""
    var b = PrimitiveBuilder[int64](capacity=5)
    b.unsafe_append(0)
    b.unsafe_append(10)
    b.unsafe_append(20)
    b.unsafe_append(30)
    b.unsafe_append(40)
    var a = b^.freeze()
    var data_buffers = List[Buffer]()
    data_buffers.append(a.buffer)
    var data = Array(
        dtype=materialize[int64](),
        length=3,
        bitmap=a.bitmap,
        buffers=data_buffers^,
        children=List[Array](),
        offset=1,
    )
    var sliced = PrimitiveArray[int64](data)
    assert_equal(sliced.length, 3)
    assert_equal(sliced.unsafe_get(0), 10)
    assert_equal(sliced.unsafe_get(1), 20)
    assert_equal(sliced.unsafe_get(2), 30)


def test_getitem_bounds_check():
    """__getitem__ raises on out-of-bounds access."""
    var b = PrimitiveBuilder[int64]()
    b.append(1)
    b.append(2)
    var a = b^.freeze()
    try:
        _ = a[5]
        assert_true(False, "should have raised")
    except:
        pass
    try:
        _ = a[-1]
        assert_true(False, "should have raised")
    except:
        pass
    assert_equal(a[0], 1)
    assert_equal(a[1], 2)


def test_setitem_bounds_check():
    """Builder __setitem__ raises on out-of-bounds and works in bounds."""
    var a = PrimitiveBuilder[int64]()
    a.append(10)
    a[0] = 99
    var frozen = a^.freeze()
    assert_equal(frozen[0], 99)
    # Verify out-of-bounds write raises on a new builder
    var b = PrimitiveBuilder[int64]()
    b.append(10)
    try:
        b[1] = 0
        assert_true(False, "should have raised")
    except:
        pass


def test_string_freeze_zero_copy():
    """Freeze() on an exact-size StringBuilder moves buffers."""
    var s = StringBuilder(capacity=2)
    s.unsafe_append("hello")
    s.unsafe_append("world")
    var frozen = s^.freeze()
    assert_equal(frozen.length, 2)
    assert_equal(String(frozen.unsafe_get(0)), "hello")
    assert_equal(String(frozen.unsafe_get(1)), "world")


def test_string_freeze_shrinks():
    """Freeze() on an over-allocated StringBuilder trims to exact size."""
    var s = StringBuilder(capacity=100)
    s.unsafe_append("hi")
    var frozen = s^.freeze()
    assert_equal(frozen.length, 1)
    assert_equal(String(frozen.unsafe_get(0)), "hi")


def test_string_getitem_bounds_check():
    """StringArray __getitem__ raises on out-of-bounds."""
    var s = StringBuilder()
    s.unsafe_append("a")
    var frozen = s^.freeze()
    assert_equal(String(frozen[0]), "a")
    try:
        _ = frozen[1]
        assert_true(False, "should have raised")
    except:
        pass


def test_primitive_shrink_to_fit_with_offset():
    """Shrink_to_fit() with non-zero offset copies the correct data slice."""
    var a = PrimitiveBuilder[int64](capacity=8)
    a.unsafe_append(0)
    a.unsafe_append(10)
    a.unsafe_append(20)
    a.unsafe_append(30)
    a.unsafe_append(40)
    a.unsafe_append(50)
    a.unsafe_append(60)
    a.unsafe_append(70)

    # Simulate a slice: elements [3, 4, 5, 6] = [30, 40, 50, 60]
    a.offset = 3
    a.length = 4

    a.shrink_to_fit()

    assert_equal(a.offset, 0)
    assert_equal(a.length, 4)
    assert_equal(a.capacity, 4)
    var frozen = a^.freeze()
    assert_equal(frozen.unsafe_get(0), 30)
    assert_equal(frozen.unsafe_get(1), 40)
    assert_equal(frozen.unsafe_get(2), 50)
    assert_equal(frozen.unsafe_get(3), 60)
    for i in range(4):
        assert_true(frozen.is_valid(i))


def test_primitive_shrink_to_fit_preserves_nulls():
    """Shrink_to_fit() with offset preserves the null bitmap correctly."""
    var a = PrimitiveBuilder[int32](capacity=6)
    a.unsafe_append(0)
    a.unsafe_append_null()
    a.unsafe_append(20)
    a.unsafe_append_null()
    a.unsafe_append(40)
    a.unsafe_append(50)
    # values: [0, null, 20, null, 40, 50], take slice [1..4] = [null, 20, null]
    a.offset = 1
    a.length = 3

    a.shrink_to_fit()

    assert_equal(a.offset, 0)
    assert_equal(a.length, 3)
    var frozen = a^.freeze()
    assert_false(frozen.is_valid(0))
    assert_true(frozen.is_valid(1))
    assert_false(frozen.is_valid(2))
    assert_equal(frozen.unsafe_get(1), 20)


def test_string_shrink_to_fit_with_offset():
    """Shrink_to_fit() with non-zero offset extracts the correct string slice.
    """
    var s = StringBuilder()
    s.unsafe_append("alpha")
    s.unsafe_append("beta")
    s.unsafe_append("gamma")
    s.unsafe_append("delta")
    s.unsafe_append("epsilon")

    # Simulate a slice: elements [2, 3, 4] = ["gamma", "delta", "epsilon"]
    s.offset = 2
    s.length = 3

    s.shrink_to_fit()

    assert_equal(s.offset, 0)
    assert_equal(s.length, 3)
    assert_equal(s.capacity, 3)
    var frozen = s^.freeze()
    assert_equal(String(frozen.unsafe_get(0)), "gamma")
    assert_equal(String(frozen.unsafe_get(1)), "delta")
    assert_equal(String(frozen.unsafe_get(2)), "epsilon")


def test_list_shrink_to_fit_with_offset():
    """Shrink_to_fit() with non-zero offset copies the correct offsets slice."""
    var ints = array[int64]([1, 2, 3])
    var lists = ListBuilder.from_values(Array(ints^), capacity=5)
    # Append 4 more empty list entries so length=5, capacity=5
    for _ in range(4):
        lists.unsafe_append(True)
    assert_equal(len(lists), 5)

    # Simulate a slice: elements [1, 2, 3] (offset=1, length=3)
    lists.offset = 1
    lists.length = 3

    lists.shrink_to_fit()

    assert_equal(lists.offset, 0)
    assert_equal(lists.length, 3)
    assert_equal(lists.capacity, 3)
    # The raw offsets buffer should hold the 4 values from original positions 1..4
    assert_equal(
        lists.offsets.unsafe_get[DType.uint32](0),
        UInt32(3),  # original offsets[1] = 3 (first element has 3 child items)
    )
    assert_equal(
        lists.offsets.unsafe_get[DType.uint32](1),
        UInt32(3),  # original offsets[2] = 3 (empty list)
    )
    assert_equal(
        lists.offsets.unsafe_get[DType.uint32](3),
        UInt32(3),  # original offsets[4] = 3 (empty list)
    )


fn test_builder_freeze_types() raises:
    """Freeze() returns the correct immutable type — compile-time check."""
    var a = PrimitiveBuilder[int64]()
    a.append(1)
    var _: PrimitiveArray[int64] = a^.freeze()


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()

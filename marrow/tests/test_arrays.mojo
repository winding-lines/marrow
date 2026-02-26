from testing import assert_equal, assert_true, assert_false, TestSuite
from memory import ArcPointer
from marrow.arrays import *
from marrow.dtypes import *
from marrow.buffers import Buffer, Bitmap
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
    var bitmap = ArcPointer(Bitmap.alloc(10))
    var buffer = ArcPointer(Buffer.alloc[int8.native](10))

    # Set some data in the buffer
    buffer[].unsafe_set[int8.native](2, 100)
    buffer[].unsafe_set[int8.native](3, 200)
    buffer[].unsafe_set[int8.native](4, 300)

    # Set validity bits
    bitmap[].unsafe_set(2, True)
    bitmap[].unsafe_set(3, True)
    bitmap[].unsafe_set(4, True)

    # Create ArrayData with offset=2
    var array_data = Array(
        dtype=materialize[int8](),
        length=3,
        bitmap=bitmap,
        buffers=[buffer],
        children=[],
        offset=2,
    )

    assert_equal(array_data.offset, 2)

    # Test is_valid with offset
    assert_true(array_data.is_valid(0))  # Should check bitmap[2]
    assert_true(array_data.is_valid(1))  # Should check bitmap[3]
    assert_true(array_data.is_valid(2))  # Should check bitmap[4]


def test_array_data_fieldwise_init():
    """Test that @fieldwise_init decorator works with offset field."""
    var bitmap = ArcPointer(Bitmap.alloc(5))
    var buffer = ArcPointer(Buffer.alloc[int8.native](5))

    # Test creating ArrayData with all fields specified including offset
    var array_data = Array(
        dtype=materialize[int8](),
        length=5,
        bitmap=bitmap,
        buffers=[buffer],
        children=[],
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
    var s = StringArray()
    s.unsafe_append("hello")
    s.unsafe_append("world")
    var a = Array(s^)
    assert_equal(a.length, 2)
    assert_true(a.dtype.is_string())


def test_array_from_list():
    var ints = Int64Array()
    var l = ListArray.from_values(ints^)
    var a = Array(l^)
    assert_true(a.dtype.is_list())


def test_array_from_struct():
    var fields = [Field("x", materialize[int32]())]
    var s = StructArray(fields^, capacity=5)
    var a = Array(s^)
    assert_true(a.dtype.is_struct())


def test_array_copy():
    var src = Array(
        dtype=materialize[int8](),
        length=3,
        bitmap=ArcPointer(Bitmap.alloc(3)),
        buffers=[ArcPointer(Buffer.alloc[int8.native](3))],
        children=[],
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
    var a = Array(
        dtype=materialize[int8](),
        length=5,
        bitmap=ArcPointer(Bitmap.alloc(5)),
        buffers=[ArcPointer(Buffer.alloc[int8.native](5))],
        children=[],
        offset=0,
    )
    var b = a^
    assert_equal(b.length, 5)
    assert_equal(b.dtype, materialize[int8]())


# --- PrimitiveArray tests ---


def test_boolean_array():
    var a = BoolArray()
    assert_equal(len(a), 0)
    assert_equal(a.capacity, 0)

    a.grow(3)
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
    assert_true(a.is_valid(0))
    assert_true(a.is_valid(1))
    assert_true(a.is_valid(2))
    assert_true(a.is_valid(3))

    var d = Array(a^)
    assert_equal(d.length, 4)

    var b = d^.as_bool()


def test_append():
    var a = Int8Array()
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
    assert_bitmap_set(primitive_array.bitmap[], [1, 3, 5, 7, 9], "check setup")

    var result = drop_nulls[uint8](primitive_array)
    assert_equal(result.unsafe_get(0), 1)
    assert_equal(result.unsafe_get(1), 3)
    assert_equal(result.null_count(), 0)


def test_primitive_array_with_offset():
    """Test PrimitiveArray with offset functionality."""
    # Create a regular array first
    var arr = Int32Array(10)
    arr.unsafe_set(0, 100)
    arr.unsafe_set(1, 200)
    arr.unsafe_set(2, 300)
    arr.unsafe_set(3, 400)
    arr.unsafe_set(4, 500)

    # Default offset should be 0
    assert_equal(arr.offset, 0)
    assert_equal(arr.unsafe_get(0), 100)
    assert_equal(arr.unsafe_get(1), 200)

    # Create a copy of array with offset, should point to the same buffers.
    var arr_data = Array(
        dtype=materialize[int32](),
        length=arr.length,
        bitmap=arr.bitmap,
        buffers=[arr.buffer],
        children=[],
        offset=arr.offset,
    )
    var arr_with_offset = PrimitiveArray[int32](arr_data, offset=2)
    assert_equal(arr_with_offset.offset, 2)

    # Test that offset affects get operations
    assert_equal(arr_with_offset.unsafe_get(0), 300)  # Should get arr[2]
    assert_equal(arr_with_offset.unsafe_get(1), 400)  # Should get arr[3]
    assert_equal(arr_with_offset.unsafe_get(2), 500)  # Should get arr[4]

    # Test that offset affects set operations
    arr_with_offset.unsafe_set(3, 999)  # Should set arr[5]
    assert_equal(arr.unsafe_get(5), 999)


def test_primitive_array_moveinit_with_offset():
    """Test __moveinit__ preserves offset."""
    var arr = Int16Array(5, offset=3)
    arr.unsafe_set(0, 123)

    var moved_arr = arr^
    assert_equal(moved_arr.offset, 3)
    assert_equal(moved_arr.unsafe_get(0), 123)


def test_primitive_array_constructor_with_offset():
    """Test PrimitiveArray constructor with offset parameter."""
    var arr1 = Int8Array(10)  # Default offset=0
    assert_equal(arr1.offset, 0)

    var arr2 = Int8Array(10, offset=5)  # Explicit offset
    assert_equal(arr2.offset, 5)

    # Test that offset is also set correctly
    assert_equal(arr2.offset, 5)


def test_primitive_array_offset_with_validity():
    """Test that offset works correctly with validity bitmap."""
    var arr = UInt8Array(10, offset=1)

    # Set some values with validity
    arr.unsafe_set(0, 42)  # This should set buffer[1] and bitmap[1]
    arr.unsafe_set(1, 43)  # This should set buffer[2] and bitmap[2]

    # Verify values are accessible through offset
    assert_equal(arr.unsafe_get(0), 42)
    assert_equal(arr.unsafe_get(1), 43)

    # Verify bitmap is also offset correctly
    assert_true(arr.is_valid(0))  # Should check bitmap[1]
    assert_true(arr.is_valid(1))  # Should check bitmap[2]


def test_primitive_array_nulls_with_offset():
    """Test nulls() creates an array with all null values and default offset."""
    var null_arr = nulls[int64](5)
    assert_equal(null_arr.offset, 0)

    # All elements should be invalid (null)
    for i in range(5):
        assert_false(null_arr.is_valid(i))


# --- StringArray tests ---


def test_string_builder():
    var a = StringArray()
    assert_equal(len(a), 0)
    assert_equal(a.capacity, 0)

    a.grow(2)
    assert_equal(len(a), 0)
    assert_equal(a.capacity, 2)

    a.unsafe_append("hello")
    a.unsafe_append("world")
    assert_equal(len(a), 2)
    assert_equal(a.capacity, 2)

    assert_equal(String(a.unsafe_get(0)), "hello")
    assert_equal(String(a.unsafe_get(1)), "world")


# --- ListArray / StructArray tests ---


def test_list_int_array():
    var ints = Int64Array(
        Array.from_buffer[int64](buffer_from[DType.int64](1, 2, 3), 3)
    )
    var lists = ListArray.from_values(ints^)
    assert_equal(lists.dtype, list_(materialize[int64]()))

    var first_value = lists.unsafe_get(0)
    assert_equal(first_value.__str__(), "PrimitiveArray[int64]([1, 2, 3])")

    assert_equal(len(lists), 1)

    var data = Array(lists^)
    assert_equal(data.length, 1)

    var arr = data^.as_list()
    assert_equal(len(arr), 1)


def test_list_bool_array():
    var bools = array([True, False, True])

    var lists = ListArray.from_values(bools^)
    assert_equal(len(lists), 1)
    var first_value = lists.unsafe_get(0)
    var bool_array = BoolArray(first_value)
    assert_equal(bool_array.unsafe_get(0), True)
    assert_equal(bool_array.unsafe_get(1), False)
    assert_equal(bool_array.unsafe_get(2), True)


def test_list_str():
    var strings = StringArray()
    strings.unsafe_append("hello")
    strings.unsafe_append("world")

    var lists = ListArray.from_values(strings^)
    var first_value = StringArray(lists.unsafe_get(0))

    assert_equal(first_value.unsafe_get(0), "hello")
    assert_equal(first_value.unsafe_get(1), "world")


def test_list_of_list():
    list2 = build_list_of_list[int64]()
    top = ListArray(list2.unsafe_get(0))
    middle_0 = top.unsafe_get(0)
    bottom = Int64Array(middle_0^)
    assert_equal(bottom.unsafe_get(1), 2)
    assert_equal(bottom.unsafe_get(0), 1)
    middle_1 = top.unsafe_get(1)
    bottom = Int64Array(middle_1^)
    assert_equal(bottom.unsafe_get(0), 3)
    assert_equal(bottom.unsafe_get(1), 4)


def test_struct_array():
    var fields = [
        Field("id", materialize[int64]()),
        Field("name", materialize[string]()),
        Field("active", materialize[bool_]()),
    ]

    var struct_arr = StructArray(fields^, capacity=10)
    assert_equal(len(struct_arr), 0)
    assert_equal(struct_arr.capacity, 10)

    var data = Array(struct_arr^)
    assert_equal(data.length, 0)
    assert_true(data.dtype.is_struct())
    assert_equal(len(data.dtype.fields), 3)
    assert_equal(data.dtype.fields[0].name, "id")
    assert_equal(data.dtype.fields[1].name, "name")
    assert_equal(data.dtype.fields[2].name, "active")


def test_struct_array_unsafe_get():
    var struct_array = build_struct()
    ref int_data_a = struct_array.unsafe_get("int_data_a")
    var int_a = Int32Array(int_data_a.copy())
    assert_equal(int_a.unsafe_get(0), 1)
    assert_equal(int_a.unsafe_get(4), 5)
    ref int_data_b = struct_array.unsafe_get("int_data_b")
    var int_b = Int32Array(int_data_b.copy())
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
    assert_equal(combined_array.buffers[1][].unsafe_get(1), 1)


def test_primitive_freeze_zero_copy():
    """Freeze() on an exact-size array moves ArcPointers without allocation."""
    var a = Int64Array(capacity=3)
    a.unsafe_append(10)
    a.unsafe_append(20)
    a.unsafe_append(30)
    var frozen = a^.freeze()
    assert_equal(frozen.length, 3)
    assert_equal(frozen.capacity, 3)
    assert_equal(frozen.offset, 0)
    assert_equal(frozen.unsafe_get(0), 10)
    assert_equal(frozen.unsafe_get(1), 20)
    assert_equal(frozen.unsafe_get(2), 30)


def test_primitive_freeze_shrinks():
    """Freeze() on an over-allocated array trims capacity to length."""
    var a = Int64Array(capacity=100)
    a.unsafe_append(42)
    a.unsafe_append(99)
    var frozen = a^.freeze()
    assert_equal(frozen.length, 2)
    assert_equal(frozen.capacity, 2)
    assert_equal(frozen.unsafe_get(0), 42)
    assert_equal(frozen.unsafe_get(1), 99)


def test_primitive_freeze_via_append():
    """Freeze() works on an array built with append() (auto-grow capacity)."""
    var a = Int64Array()
    a.append(1)
    a.append(2)
    a.append(3)
    var frozen = a^.freeze()
    assert_equal(frozen.length, 3)
    assert_equal(frozen.capacity, 3)
    assert_equal(frozen.unsafe_get(0), 1)
    assert_equal(frozen.unsafe_get(2), 3)


def test_primitive_freeze_preserves_nulls():
    """Freeze() preserves null validity information."""
    var a = Int64Array(capacity=3)
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
    var a = Int64Array()
    a.append(7)
    a.append(8)
    var frozen = a^.freeze()
    var base: Array = frozen
    assert_equal(base.length, 2)
    assert_equal(base.dtype, materialize[int64]())


def test_primitive_freeze_with_offset():
    """Freeze() with non-zero offset normalizes the data (offset becomes 0)."""
    var a = Int64Array(capacity=5)
    a.unsafe_append(0)
    a.unsafe_append(10)
    a.unsafe_append(20)
    a.unsafe_append(30)
    a.unsafe_append(40)
    var data = Array(
        dtype=materialize[int64](),
        length=3,
        bitmap=a.bitmap,
        buffers=[a.buffer],
        children=[],
        offset=1,
    )
    var sliced = Int64Array(data)
    var frozen = sliced^.freeze()
    assert_equal(frozen.offset, 0)
    assert_equal(frozen.length, 3)
    assert_equal(frozen.unsafe_get(0), 10)
    assert_equal(frozen.unsafe_get(1), 20)
    assert_equal(frozen.unsafe_get(2), 30)


def test_getitem_bounds_check():
    """__getitem__ raises on out-of-bounds access."""
    var a = Int64Array()
    a.append(1)
    a.append(2)
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
    """__setitem__ raises on out-of-bounds and works in bounds."""
    var a = Int64Array()
    a.append(10)
    a[0] = 99
    assert_equal(a[0], 99)
    try:
        a[1] = 0
        assert_true(False, "should have raised")
    except:
        pass


def test_string_freeze_zero_copy():
    """Freeze() on an exact-size StringArray moves ArcPointers."""
    var s = StringArray(capacity=2)
    s.unsafe_append("hello")
    s.unsafe_append("world")
    var frozen = s^.freeze()
    assert_equal(frozen.length, 2)
    assert_equal(frozen.capacity, 2)
    assert_equal(String(frozen.unsafe_get(0)), "hello")
    assert_equal(String(frozen.unsafe_get(1)), "world")


def test_string_freeze_shrinks():
    """Freeze() on an over-allocated StringArray trims to exact size."""
    var s = StringArray(capacity=100)
    s.unsafe_append("hi")
    var frozen = s^.freeze()
    assert_equal(frozen.length, 1)
    assert_equal(frozen.capacity, 1)
    assert_equal(String(frozen.unsafe_get(0)), "hi")


def test_string_getitem_bounds_check():
    """StringArray __getitem__ raises on out-of-bounds."""
    var s = StringArray()
    s.unsafe_append("a")
    assert_equal(String(s[0]), "a")
    try:
        _ = s[1]
        assert_true(False, "should have raised")
    except:
        pass


def test_primitive_shrink_to_fit_with_offset():
    """Shrink_to_fit() with non-zero offset copies the correct data slice."""
    var a = Int64Array(capacity=8)
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
    assert_equal(a.unsafe_get(0), 30)
    assert_equal(a.unsafe_get(1), 40)
    assert_equal(a.unsafe_get(2), 50)
    assert_equal(a.unsafe_get(3), 60)
    for i in range(4):
        assert_true(a.is_valid(i))


def test_primitive_shrink_to_fit_preserves_nulls():
    """Shrink_to_fit() with offset preserves the null bitmap correctly."""
    var a = Int32Array(capacity=6)
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
    assert_false(a.is_valid(0))
    assert_true(a.is_valid(1))
    assert_false(a.is_valid(2))
    assert_equal(a.unsafe_get(1), 20)


def test_string_shrink_to_fit_with_offset():
    """Shrink_to_fit() with non-zero offset extracts the correct string slice.
    """
    var s = StringArray()
    s.unsafe_append("alpha")
    s.unsafe_append("beta")
    s.unsafe_append("gamma")
    s.unsafe_append("delta")
    s.unsafe_append("epsilon")

    # Simulate a slice: elements [2, 3, 4] = ["gamma", "delta", "epsilon"]
    s.offset = 2
    s.length = 3
    assert_equal(String(s.unsafe_get(0)), "gamma")

    s.shrink_to_fit()

    assert_equal(s.offset, 0)
    assert_equal(s.length, 3)
    assert_equal(s.capacity, 3)
    assert_equal(String(s.unsafe_get(0)), "gamma")
    assert_equal(String(s.unsafe_get(1)), "delta")
    assert_equal(String(s.unsafe_get(2)), "epsilon")


def test_list_shrink_to_fit_with_offset():
    """Shrink_to_fit() with non-zero offset copies the correct offsets slice."""
    var ints = array[int64]([1, 2, 3])
    var lists = ListArray.from_values(Array(ints^), capacity=5)
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
        lists.offsets[].unsafe_get[DType.uint32](0),
        UInt32(3),  # original offsets[1] = 3 (first element has 3 child items)
    )
    assert_equal(
        lists.offsets[].unsafe_get[DType.uint32](1),
        UInt32(3),  # original offsets[2] = 3 (empty list)
    )
    assert_equal(
        lists.offsets[].unsafe_get[DType.uint32](3),
        UInt32(3),  # original offsets[4] = 3 (empty list)
    )


fn test_mut_parameter_compile_time() raises:
    """Compile-time checks for the mut type parameter on all typed arrays."""
    # PrimitiveArray: mutable by default, frozen when mut=False
    comptime assert PrimitiveArray[int64].mut
    comptime assert not PrimitiveArray[int64, False].mut

    # StringArray
    comptime assert StringArray[].mut
    comptime assert not StringArray[False].mut

    # ListArray
    comptime assert ListArray[].mut
    comptime assert not ListArray[False].mut

    # StructArray
    comptime assert StructArray[].mut
    comptime assert not StructArray[False].mut

    # freeze() returns the same type with mut=False; the type annotation is itself
    # a compile-time check — if freeze() returned the wrong type, this won't compile.
    var a = Int64Array()
    a.append(1)
    var _: PrimitiveArray[int64, False] = a^.freeze()
    comptime assert not PrimitiveArray[int64, False].mut


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()

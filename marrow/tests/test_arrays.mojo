from testing import assert_equal, assert_true, assert_false, TestSuite
from memory import ArcPointer
from marrow.arrays import *
from marrow.dtypes import *
from marrow.buffers import Buffer, Bitmap
from marrow.test_fixtures.arrays import (
    build_array_data,
    assert_bitmap_set,
    build_list_of_list,
    build_struct,
)
from marrow.test_fixtures.bool_array import as_bool_array_scalar


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
    var prim = array[int32](1, 2, 3)
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

    a.append(as_bool_array_scalar(True))
    a.append(as_bool_array_scalar(False))
    a.append(as_bool_array_scalar(True))
    assert_equal(len(a), 3)
    assert_equal(a.capacity, 3)

    a.append(as_bool_array_scalar(True))
    assert_equal(len(a), 4)
    assert_equal(a.capacity, 6)
    assert_true(a.is_valid(0))
    assert_true(a.is_valid(1))
    assert_true(a.is_valid(2))
    assert_true(a.is_valid(3))

    var d = Array(a^)
    assert_equal(d.length, 4)

    var b = d^.as_primitive[bool_]()


def test_append():
    var a = Int8Array()
    assert_equal(len(a), 0)
    assert_equal(a.capacity, 0)
    a.append(1)
    a.append(2)
    a.append(3)
    assert_equal(len(a), 3)
    assert_true(a.capacity >= len(a))


def test_array_from_ints():
    var g = array[int8](1, 2)
    assert_equal(len(g), 2)
    assert_equal(materialize[g.dtype](), materialize[int8]())
    assert_equal(g.unsafe_get(0), 1)
    assert_equal(g.unsafe_get(1), 2)


def test_drop_null() -> None:
    """Test the drop null function."""
    var array_data = build_array_data(10, 5)

    var primitive_array = PrimitiveArray[uint8](array_data^)
    #
    # Check the setup.
    assert_equal(primitive_array.null_count(), 5)
    assert_bitmap_set(primitive_array.bitmap[], [1, 3, 5, 7, 9], "check setup")

    primitive_array.drop_nulls[DType.uint8]()
    assert_equal(primitive_array.unsafe_get(0), 1)
    assert_equal(primitive_array.unsafe_get(1), 3)
    assert_equal(primitive_array.null_count(), 0)
    assert_bitmap_set(primitive_array.bitmap[], [0, 1, 2, 3, 4], "after drop")


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
    var arr_with_offset = PrimitiveArray[int32](arr.data.copy(), offset=2)
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

    # Test that data.offset is also set correctly
    assert_equal(arr2.data.offset, 5)


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
    """Test PrimitiveArray.nulls static method creates array with default offset.
    """
    var null_arr = Int64Array.nulls(5)
    assert_equal(null_arr.offset, 0)
    assert_equal(null_arr.data.offset, 0)

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
        Array.from_buffer[int64](Buffer.from_values[DType.int64](1, 2, 3), 3)
    )
    var lists = ListArray.from_values(ints^)
    assert_equal(lists.data.dtype, list_(materialize[int64]()))

    var first_value = lists.unsafe_get(0)
    assert_equal(
        first_value.__str__(), "PrimitiveArray[DataType(code=int64)]([1, 2, 3])"
    )

    assert_equal(len(lists), 1)

    var data = Array(lists^)
    assert_equal(data.length, 1)

    var arr = data^.as_list()
    assert_equal(len(arr), 1)


def test_list_bool_array():
    var bools = BoolArray()

    bools.append(as_bool_array_scalar(True))
    bools.append(as_bool_array_scalar(False))
    bools.append(as_bool_array_scalar(True))

    var lists = ListArray.from_values(bools^)
    assert_equal(len(lists), 1)
    var first_value = lists.unsafe_get(0)
    var buffer = first_value.buffers[0]

    fn get(index: Int) -> Bool:
        return buffer[].unsafe_get[DType.bool](index)

    assert_equal(get(0), True)
    assert_equal(get(1), False)
    assert_equal(get(2), True)


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
    var first_array_data = build_array_data(1, 0)
    var arrays = List[Array]()
    arrays.append(first_array_data^)

    var second_array_data = build_array_data(2, 0)
    arrays.append(second_array_data^)

    var chunked_array = ChunkedArray(materialize[int8](), arrays^)
    assert_equal(chunked_array.length, 3)

    assert_equal(chunked_array.chunk(0).length, 1)
    var second_chunk = chunked_array.chunk(1).copy().as_uint8()
    assert_equal(second_chunk.data.length, 2)
    assert_equal(second_chunk.unsafe_get(0), 0)
    assert_equal(second_chunk.unsafe_get(1), 1)


def test_combine_chunked_array():
    var first_array_data = build_array_data(1, 0)
    var arrays = List[Array]()
    arrays.append(first_array_data^)

    var second_array_data = build_array_data(2, 0)
    arrays.append(second_array_data^)

    var chunked_array = ChunkedArray(materialize[int8](), arrays^)
    assert_equal(chunked_array.length, 3)
    assert_equal(len(chunked_array.chunks), 2)
    assert_equal(chunked_array.chunk(1).copy().as_uint8().unsafe_get(1), 1)

    var combined_array = chunked_array^.combine_chunks()
    assert_equal(combined_array.length, 3)
    assert_equal(combined_array.dtype, materialize[int8]())
    # Ensure that the last element of the last buffer has the expected value.
    assert_equal(combined_array.buffers[1][].unsafe_get(1), 1)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()

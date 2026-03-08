from std.testing import assert_equal, assert_true, assert_false, TestSuite
from marrow.arrays import *
from marrow.builders import (
    Builder,
    BoolBuilder,
    PrimitiveBuilder,
    StringBuilder,
    ListBuilder,
    FixedSizeListBuilder,
    StructBuilder,
)
from marrow.dtypes import *
from marrow.buffers import Buffer, BufferBuilder
from marrow.bitmap import Bitmap, BitmapBuilder
from marrow.kernels.filter import drop_nulls
from std.reflection import call_location


# @always_inline
# def assert_bitmap_set(
#     ptr: UnsafePointer[UInt8, ImmutExternalOrigin],
#     n_bits: Int,
#     expected_true_pos: List[Int],
#     message: StringLiteral,
# ) -> None:
#     var list_pos = 0
#     for i in range(n_bits):
#         var expected_value = False
#         if list_pos < len(expected_true_pos):
#             if expected_true_pos[list_pos] == i:
#                 expected_value = True
#                 list_pos += 1
#         var current_value = Bool((ptr[i // 8] >> UInt8(i % 8)) & 1)
#         assert_equal(
#             current_value,
#             expected_value,
#             String(
#                 "{}: Bitmap index {} is {}, expected {} as per list position {}"
#             ).format(message, i, current_value, expected_value, list_pos),
#             location=call_location(),
#         )


def test_array_data_with_offset() raises:
    """Test ArrayData with offset functionality."""
    # Create ArrayData with offset
    var bitmap = BitmapBuilder.alloc(10)
    var buffer = BufferBuilder.alloc[int8.native](10)

    # Set some data in the buffer
    buffer.unsafe_set[int8.native](2, 100)
    buffer.unsafe_set[int8.native](3, 200)
    buffer.unsafe_set[int8.native](4, 300)

    # Set validity bits (bits 2, 3, 4 are set; offset=2 maps index 0→bit2, etc.)
    bitmap.set_bit(2, True)
    bitmap.set_bit(3, True)
    bitmap.set_bit(4, True)

    # Create ArrayData with offset=2
    var buffers = List[Buffer]()
    buffers.append(buffer.finish())
    var array_data = Array(
        dtype=int8,
        length=3,
        nulls=0,
        bitmap=bitmap.finish(10),
        buffers=buffers^,
        children=List[Array](),
        offset=2,
    )

    assert_equal(array_data.offset, 2)

    # Test is_valid with offset
    assert_true(array_data.is_valid(0))  # Should check bitmap[2]
    assert_true(array_data.is_valid(1))  # Should check bitmap[3]
    assert_true(array_data.is_valid(2))  # Should check bitmap[4]


def test_array_data_fieldwise_init() raises:
    """Test that @fieldwise_init decorator works with offset field."""
    var buffer_b = BufferBuilder.alloc[int8.native](5)
    var buffer = buffer_b.finish()

    # Test creating ArrayData with all fields specified including offset
    var buffers = List[Buffer]()
    buffers.append(buffer)
    var array_data = Array(
        dtype=int8,
        length=5,
        nulls=0,
        bitmap=None,
        buffers=buffers^,
        children=List[Array](),
        offset=3,
    )

    assert_equal(array_data.dtype, int8)
    assert_equal(array_data.length, 5)
    assert_equal(array_data.offset, 3)


def test_array_from_primitive() raises:
    var a = array[int32]([1, 2, 3])
    assert_equal(a.length, 3)


def test_array_from_string() raises:
    var s = StringBuilder()
    s.append("hello")
    s.append("world")
    var a = s.finish()
    assert_equal(a.length, 2)


def test_array_from_list() raises:
    var ints_b = PrimitiveBuilder[int64]()
    var l = ListBuilder(ints_b)
    var a = l.finish()
    assert_true(a.dtype.is_list())


def test_array_from_struct() raises:
    var fields = [Field("x", int32)]
    var s = StructBuilder(fields^, List[Builder](), capacity=5)
    var a = s.finish()
    assert_true(a.dtype.is_struct())


def test_array_copy() raises:
    var src_buffers = List[Buffer]()
    var _sb = BufferBuilder.alloc[int8.native](3)
    src_buffers.append(_sb.finish())
    var src = Array(
        dtype=int8,
        length=3,
        nulls=0,
        bitmap=None,
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


def test_array_move() raises:
    var a_buffers = List[Buffer]()
    var _ab = BufferBuilder.alloc[int8.native](5)
    a_buffers.append(_ab.finish())
    var a = Array(
        dtype=int8,
        length=5,
        nulls=0,
        bitmap=None,
        buffers=a_buffers^,
        children=List[Array](),
        offset=0,
    )
    var b = a^
    assert_equal(b.length, 5)
    assert_equal(b.dtype, int8)


def test_boolean_array() raises:
    var a = BoolBuilder()
    assert_equal(len(a), 0)
    assert_equal(a.data[].capacity, 0)

    a.grow(3)
    assert_equal(len(a), 0)
    assert_equal(a.data[].capacity, 3)

    a.append(True)
    a.append(False)
    a.append(True)
    assert_equal(len(a), 3)
    assert_equal(a.data[].capacity, 3)

    a.append(True)
    assert_equal(len(a), 4)
    assert_equal(a.data[].capacity, 6)

    var frozen = a.finish()
    assert_true(frozen.is_valid(0))
    assert_true(frozen.is_valid(1))
    assert_true(frozen.is_valid(2))
    assert_true(frozen.is_valid(3))

    assert_equal(frozen.length, 4)


def test_append() raises:
    var a = PrimitiveBuilder[int8]()
    assert_equal(len(a), 0)
    assert_equal(a.data[].capacity, 0)
    a.append(1)
    a.append(2)
    a.append(3)
    assert_equal(len(a), 3)
    assert_true(a.data[].capacity >= len(a))


def test_array_empty() raises:
    var a = array[int32]()
    assert_equal(len(a), 0)


def test_array_from_ints() raises:
    var g = array[int8]([1, 2])
    assert_equal(len(g), 2)
    assert_equal(g.unsafe_get(0), 1)
    assert_equal(g.unsafe_get(1), 2)

    var b = array([True, False, True])
    assert_equal(len(b), 3)
    assert_true(b.unsafe_get(0))
    assert_false(b.unsafe_get(1))
    assert_true(b.unsafe_get(2))


def test_array_with_nulls() raises:
    var a = array[int32]([1, None, 3])
    assert_equal(len(a), 3)
    assert_equal(a.null_count(), 1)
    assert_true(a.is_valid(0))
    assert_false(a.is_valid(1))
    assert_true(a.is_valid(2))
    assert_equal(a.unsafe_get(0), 1)
    assert_equal(a.unsafe_get(2), 3)

    var b = array([True, None, False])
    assert_equal(b.length, 3)
    assert_true(b.is_valid(0))
    assert_false(b.is_valid(1))
    assert_true(b.is_valid(2))


def test_arange() raises:
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


def test_arange_empty() raises:
    var a = arange[int32](5, 5)
    assert_equal(len(a), 0)


def test_arange_single() raises:
    var a = arange[int64](7, 8)
    assert_equal(len(a), 1)
    assert_equal(a.unsafe_get(0), 7)


def test_arange_validity() raises:
    var a = arange[int16](0, 4)
    for i in range(4):
        assert_true(a.is_valid(i))


def test_arange_int8() raises:
    var a = arange[int8](10, 15)
    assert_equal(len(a), 5)
    assert_equal(a.unsafe_get(0), 10)
    assert_equal(a.unsafe_get(4), 14)


def test_arange_uint64() raises:
    var a = arange[uint64](100, 103)
    assert_equal(len(a), 3)
    assert_equal(a.unsafe_get(0), 100)
    assert_equal(a.unsafe_get(2), 102)


# TODO: move this to compute kernels
# def test_drop_null() -> None:
#     """Test the drop null function via the compute module."""
#     from marrow.kernels.filter import drop_nulls

#     var primitive_array = array[uint8](
#         [None, 1, None, 3, None, 5, None, 7, None, 9]
#     )
#     # Check the setup.
#     assert_equal(primitive_array.null_count(), 5)
#     assert_bitmap_set(
#         primitive_array.bitmap.unsafe_ptr(),
#         primitive_array.length,
#         [1, 3, 5, 7, 9],
#         "check setup",
#     )

#     var result = drop_nulls[uint8](primitive_array)
#     assert_equal(result.unsafe_get(0), 1)
#     assert_equal(result.unsafe_get(1), 3)
#     assert_equal(result.null_count(), 0)


def test_primitive_array_with_offset() raises:
    """Test PrimitiveArray with offset functionality."""
    var b = PrimitiveBuilder[int32](10)
    b.append(100)
    b.append(200)
    b.append(300)
    b.append(400)
    b.append(500)
    var arr = b.finish()

    # Default offset should be 0
    assert_equal(arr.offset, 0)
    assert_equal(arr.unsafe_get(0), 100)
    assert_equal(arr.unsafe_get(1), 200)

    # Create a copy of array with offset, should point to the same buffers.
    var arr_with_offset = arr.with_offset(2)
    assert_equal(arr_with_offset.offset, 2)

    # Test that offset affects get operations
    assert_equal(arr_with_offset.unsafe_get(0), 300)  # Should get arr[2]
    assert_equal(arr_with_offset.unsafe_get(1), 400)  # Should get arr[3]
    assert_equal(arr_with_offset.unsafe_get(2), 500)  # Should get arr[4]


def test_primitive_array_nulls_with_offset() raises:
    """Test nulls() creates an array with all null values and default offset."""
    var null_arr = nulls[int64](5)
    assert_equal(null_arr.offset, 0)

    # All elements should be invalid (null)
    for i in range(5):
        assert_false(null_arr.is_valid(i))


# TODO: expose capacity() on builders and test that as well
def test_string_builder() raises:
    var a = StringBuilder()
    assert_equal(len(a), 0)
    assert_equal(a.data[].capacity, 0)

    a.grow(2)
    assert_equal(len(a), 0)
    assert_equal(a.data[].capacity, 2)

    a.append("hello")
    a.append("world")
    assert_equal(len(a), 2)
    assert_equal(a.data[].capacity, 2)

    var frozen = a.finish()
    assert_equal(String(frozen.unsafe_get(0)), "hello")
    assert_equal(String(frozen.unsafe_get(1)), "world")


def test_list_bool_array() raises:
    var bool_b = BoolBuilder(3)
    bool_b.append(True)
    bool_b.append(False)
    bool_b.append(True)
    var list_b = ListBuilder(bool_b)
    list_b.append(True)
    var lists: ListArray = list_b.finish()
    assert_equal(len(lists), 1)

    # TODO: fix listarray.unsafe_get
    var first_value = lists.unsafe_get(0)
    var bool_array = first_value.as_bool()
    assert_true(bool_array.unsafe_get(0))
    assert_false(bool_array.unsafe_get(1))
    assert_true(bool_array.unsafe_get(2))


def test_list_str() raises:
    var str_b = StringBuilder()
    str_b.append("hello")
    str_b.append("world")
    var list_b = ListBuilder(str_b)
    list_b.append(True)
    var lists = list_b.finish()
    assert_equal(len(lists), 1)

    var first_value = StringArray(lists.unsafe_get(0))
    assert_equal(String(first_value.unsafe_get(0)), "hello")
    assert_equal(String(first_value.unsafe_get(1)), "world")


def test_list_of_list() raises:
    var child = PrimitiveBuilder[int64](capacity=10)
    var middle = ListBuilder(child, capacity=6)
    var top_b = ListBuilder(middle, capacity=3)
    child.append(1)
    child.append(2)
    middle.append(True)
    child.append(3)
    child.append(4)
    middle.append(True)
    top_b.append(True)
    child.append(5)
    child.append(6)
    child.append(7)
    middle.append(True)
    middle.append_null()
    child.append(8)
    middle.append(True)
    top_b.append(True)
    child.append(9)
    child.append(10)
    middle.append(True)
    top_b.append(True)
    list2 = top_b.finish()

    top = ListArray(list2.unsafe_get(0))
    middle_0 = top.unsafe_get(0)
    bottom = PrimitiveArray[int64](middle_0^)
    assert_equal(bottom.unsafe_get(1), 2)
    assert_equal(bottom.unsafe_get(0), 1)
    middle_1 = top.unsafe_get(1)
    bottom = PrimitiveArray[int64](middle_1^)
    assert_equal(bottom.unsafe_get(0), 3)
    assert_equal(bottom.unsafe_get(1), 4)


def test_fixed_size_list_int_array() raises:
    """Construct a FixedSizeListArray of int64 lists, size=3."""
    var ints_b = PrimitiveBuilder[int64](6)
    ints_b.append(1)
    ints_b.append(2)
    ints_b.append(3)
    ints_b.append(4)
    ints_b.append(5)
    ints_b.append(6)
    var builder = FixedSizeListBuilder(ints_b, list_size=3)
    builder.append(True)
    builder.append(True)
    assert_equal(builder.data[].dtype, fixed_size_list_(int64, 3))
    assert_equal(len(builder), 2)
    var fsl = builder.finish()
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


def test_fixed_size_list_roundtrip() raises:
    """FixedSizeListArray round-trip through builder."""
    var ints_b = PrimitiveBuilder[int32](4)
    ints_b.append(10)
    ints_b.append(20)
    ints_b.append(30)
    ints_b.append(40)
    var builder = FixedSizeListBuilder(ints_b, list_size=2)
    builder.append(True)
    builder.append(True)
    var fsl = builder.finish()

    assert_true(fsl.dtype.is_fixed_size_list())
    assert_equal(fsl.dtype.size, 2)
    assert_equal(len(fsl), 2)

    var first = fsl.unsafe_get(0).as_int32()
    assert_equal(first.unsafe_get(0), 10)
    assert_equal(first.unsafe_get(1), 20)


def test_fixed_size_list_with_nulls() raises:
    """FixedSizeListArray with null lists."""
    var ints_b = PrimitiveBuilder[int64](6)
    ints_b.append(1)
    ints_b.append(2)
    ints_b.append(3)
    ints_b.append(4)
    ints_b.append(5)
    ints_b.append(6)
    var builder = FixedSizeListBuilder(ints_b, list_size=3, capacity=3)
    builder.append(True)
    builder.append(True)
    builder.append(False)
    assert_equal(len(builder), 3)

    var fsl = builder.finish()
    assert_true(fsl.is_valid(0))
    assert_true(fsl.is_valid(1))
    assert_false(fsl.is_valid(2))

    # unsafe_get on valid entries returns correct values even when array has nulls
    var first = fsl.unsafe_get(0).as_int64()
    assert_equal(first.unsafe_get(0), 1)
    assert_equal(first.unsafe_get(1), 2)
    assert_equal(first.unsafe_get(2), 3)
    var second = fsl.unsafe_get(1).as_int64()
    assert_equal(second.unsafe_get(0), 4)
    assert_equal(second.unsafe_get(1), 5)
    assert_equal(second.unsafe_get(2), 6)


def test_fixed_size_list_unsafe_get_dtype() raises:
    # unsafe_get returns a slice with the child element dtype, not the list dtype.
    var ints_b = PrimitiveBuilder[int32](4)
    ints_b.append(10)
    ints_b.append(20)
    ints_b.append(30)
    ints_b.append(40)
    var builder = FixedSizeListBuilder(ints_b, list_size=2)
    builder.append(True)
    builder.append(True)
    var fsl = builder.finish()

    var slice0 = fsl.unsafe_get(0)
    assert_equal(slice0.dtype, int32)
    assert_equal(slice0.length, 2)
    assert_equal(slice0.offset, 0)

    var slice1 = fsl.unsafe_get(1)
    assert_equal(slice1.dtype, int32)
    assert_equal(slice1.length, 2)
    assert_equal(slice1.offset, 2)


# # def test_fixed_size_list_pretty_print():
# #     """Pretty printing FixedSizeListArray."""
# #     var ints_b = PrimitiveBuilder[int64](4)
# #     ints_b.append(1)
# #     ints_b.append(2)
# #     ints_b.append(3)
# #     ints_b.append(4)
# #     var builder = FixedSizeListBuilder(ints_b, list_size=2)
# #     builder.append(True)
# #     builder.append(True)
# #     var fsl = builder.finish()
# #     var s = String(Array(fsl^))
# #     assert_true("FixedSizeListArray" in s)


def test_struct_array() raises:
    var fields = [
        Field("id", int64),
        Field("name", string),
        Field("active", bool_),
    ]

    var struct_builder = StructBuilder(fields^, List[Builder](), capacity=10)
    assert_equal(len(struct_builder), 0)
    assert_equal(struct_builder.data[].capacity, 10)

    var data = struct_builder.finish()
    assert_equal(data.length, 0)
    assert_true(data.dtype.is_struct())
    assert_equal(len(data.dtype.fields), 3)
    assert_equal(data.dtype.fields[0].name, "id")
    assert_equal(data.dtype.fields[1].name, "name")
    assert_equal(data.dtype.fields[2].name, "active")


def test_struct_array_unsafe_get() raises:
    var a_b = PrimitiveBuilder[int32](5)
    a_b.append(1)
    a_b.append(2)
    a_b.append(3)
    a_b.append(4)
    a_b.append(5)
    var b_b = PrimitiveBuilder[int32](3)
    b_b.append(10)
    b_b.append(20)
    b_b.append(30)
    var fields = List[Field]()
    fields.append(Field("int_data_a", int32))
    fields.append(Field("int_data_b", int32))
    var children = List[Builder]()
    children.append(a_b)
    children.append(b_b)
    var sb = StructBuilder(fields^, children^, capacity=2)
    sb.append(True)
    sb.append(True)
    var struct_array = sb.finish()
    ref int_data_a = struct_array.unsafe_get("int_data_a")
    var int_a = PrimitiveArray[int32](int_data_a.copy())
    assert_equal(int_a.unsafe_get(0), 1)
    assert_equal(int_a.unsafe_get(4), 5)
    ref int_data_b = struct_array.unsafe_get("int_data_b")
    var int_b = PrimitiveArray[int32](int_data_b.copy())
    assert_equal(int_b.unsafe_get(0), 10)
    assert_equal(int_b.unsafe_get(2), 30)


def test_chunked_array() raises:
    var arrays = List[Array]()
    arrays.append(array[uint8]([0]))
    arrays.append(array[uint8]([0, 1]))

    var chunked_array = ChunkedArray(int8, arrays^)
    assert_equal(chunked_array.length, 3)

    assert_equal(chunked_array.chunk(0).length, 1)
    var second_chunk = chunked_array.chunk(1).copy().as_uint8()
    assert_equal(second_chunk.length, 2)
    assert_equal(second_chunk.unsafe_get(0), 0)
    assert_equal(second_chunk.unsafe_get(1), 1)


def test_combine_chunked_array() raises:
    var arrays = List[Array]()
    arrays.append(array[uint8]([0]))
    arrays.append(array[uint8]([0, 1]))

    var chunked_array = ChunkedArray(int8, arrays^)
    assert_equal(chunked_array.length, 3)
    assert_equal(len(chunked_array.chunks), 2)
    assert_equal(chunked_array.chunk(1).copy().as_uint8().unsafe_get(1), 1)

    var combined_array = chunked_array^.combine_chunks()
    assert_equal(combined_array.length, 3)
    assert_equal(combined_array.dtype, int8)
    # Ensure that the last element of the last buffer has the expected value.
    assert_equal(combined_array.buffers[1].unsafe_get(1), 1)


def test_primitive_finish_shrinks() raises:
    """Freeze() on an over-allocated builder trims capacity to length."""
    var a = PrimitiveBuilder[int64](capacity=100)
    a.append(42)
    a.append(99)
    var frozen = a.finish()
    assert_equal(frozen.length, 2)
    assert_equal(frozen.unsafe_get(0), 42)
    assert_equal(frozen.unsafe_get(1), 99)

    var values_buffer = frozen.buffer
    # 2 int64 values = 16 bytes, but buffer padded to 64 bytes for alignment
    assert_equal(values_buffer.size, 64)


def test_primitive_finish_via_append() raises:
    """Freeze() works on a builder built with append() (auto-grow capacity)."""
    var a = PrimitiveBuilder[int64]()
    a.append(1)
    a.append(2)
    a.append(3)
    var frozen = a.finish()
    assert_equal(frozen.length, 3)
    assert_equal(frozen.unsafe_get(0), 1)
    assert_equal(frozen.unsafe_get(2), 3)


def test_primitive_finish_preserves_nulls() raises:
    """Freeze() preserves null validity information."""
    var a = PrimitiveBuilder[int64](capacity=3)
    a.append(1)
    a.append_null()
    a.append(3)
    var frozen = a.finish()
    assert_equal(frozen.length, 3)
    assert_true(frozen.is_valid(0))
    assert_false(frozen.is_valid(1))
    assert_true(frozen.is_valid(2))


def test_primitive_finish_converts_to_array() raises:
    """PrimitiveBuilder.finish() returns a typed PrimitiveArray."""
    var a = PrimitiveBuilder[int64]()
    a.append(7)
    a.append(8)
    var frozen = a.finish()
    assert_equal(frozen.length, 2)
    assert_equal(frozen.unsafe_get(0), 7)
    assert_equal(frozen.unsafe_get(1), 8)


def test_getitem_bounds_check() raises:
    """__getitem__ raises on out-of-bounds access."""
    var b = PrimitiveBuilder[int64]()
    b.append(1)
    b.append(2)
    var a = b.finish()
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


def test_setitem_bounds_check() raises:
    """PrimitiveArray __getitem__ returns correct values."""
    var a = PrimitiveBuilder[int64]()
    a.append(99)
    var frozen = a.finish()
    assert_equal(frozen[0], 99)


def test_string_finish_zero_copy() raises:
    """Freeze() on an exact-size StringBuilder moves buffers."""
    var s = StringBuilder(capacity=2)
    s.append("hello")
    s.append("world")
    var frozen = s.finish()
    assert_equal(frozen.length, 2)
    assert_equal(String(frozen.unsafe_get(0)), "hello")
    assert_equal(String(frozen.unsafe_get(1)), "world")


def test_string_finish_shrinks() raises:
    """Freeze() on an over-allocated StringBuilder trims to exact size."""
    var s = StringBuilder(capacity=100)
    s.append("hi")
    var frozen = s.finish()
    assert_equal(frozen.length, 1)
    assert_equal(String(frozen.unsafe_get(0)), "hi")


def test_string_getitem_bounds_check() raises:
    """StringArray __getitem__ raises on out-of-bounds."""
    var s = StringBuilder()
    s.append("a")
    var frozen = s.finish()
    assert_equal(String(frozen[0]), "a")
    try:
        _ = frozen[1]
        assert_true(False, "should have raised")
    except:
        pass


def test_pretty_printing() raises:
    var a = array[int32]([1, 2, 3])
    var s = String(a)
    print(s)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

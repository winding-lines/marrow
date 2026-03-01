from marrow.arrays import (
    Array,
    ListArray,
    StructArray,
)
from marrow.buffers import Buffer, Bitmap, BufferBuilder, BitmapBuilder
from marrow.dtypes import uint8, DataType, list_, int32, Field, struct_
from testing import assert_equal
from reflection import call_location


fn buffer_from[dtype: DType](*values: Scalar[dtype]) -> Buffer:
    """Test helper: build a Buffer from literal values."""
    var buffer = BufferBuilder.alloc[dtype](len(values))
    for i in range(len(values)):
        buffer.unsafe_set[dtype](i, values[i])
    return buffer^.freeze()


@always_inline
def assert_bitmap_set(
    bitmap: Bitmap, expected_true_pos: List[Int], message: StringLiteral
) -> None:
    var list_pos = 0
    for i in range(bitmap.length()):
        var expected_value = False
        if list_pos < len(expected_true_pos):
            if expected_true_pos[list_pos] == i:
                expected_value = True
                list_pos += 1
        var current_value = bitmap.unsafe_get(i)
        assert_equal(
            current_value,
            expected_value,
            String(
                "{}: Bitmap index {} is {}, expected {} as per list position {}"
            ).format(message, i, current_value, expected_value, list_pos),
            location=call_location(),
        )


@always_inline
def assert_bitmap_set(
    bitmap: BitmapBuilder, expected_true_pos: List[Int], message: StringLiteral
) -> None:
    var list_pos = 0
    for i in range(bitmap.length()):
        var expected_value = False
        if list_pos < len(expected_true_pos):
            if expected_true_pos[list_pos] == i:
                expected_value = True
                list_pos += 1
        var current_value = bitmap.unsafe_get(i)
        assert_equal(
            current_value,
            expected_value,
            String(
                "{}: Bitmap index {} is {}, expected {} as per list position {}"
            ).format(message, i, current_value, expected_value, list_pos),
            location=call_location(),
        )


fn build_list_of_int[data_type: DataType]() raises -> ListArray:
    """Build a test ListArray that itself contains a ListArray of IntArrays."""
    # Define all the values.
    var bitmap = BitmapBuilder.alloc(10)
    bitmap.unsafe_range_set(0, 10, True)
    var buffer = BufferBuilder.alloc[data_type.native](10)
    for i in range(10):
        buffer.unsafe_set[data_type.native](i, Scalar[data_type.native](i + 1))

    var value_buffers = List[Buffer]()
    value_buffers.append(buffer^.freeze())
    var value_data = Array(
        dtype=materialize[data_type](),
        length=10,
        bitmap=bitmap^.freeze(),
        buffers=value_buffers^,
        children=List[Array](),
        offset=0,
    )

    # Define the PrimitiveArrays.
    var value_offset = buffer_from[DType.int32](0, 2, 4, 7, 7, 8, 10)

    var list_bitmap = BitmapBuilder.alloc(6)
    list_bitmap.unsafe_range_set(0, 6, True)
    list_bitmap.unsafe_set(3, False)
    var list_buffers = List[Buffer]()
    list_buffers.append(value_offset^)
    var list_children = List[Array]()
    list_children.append(value_data^)
    var list_data = Array(
        dtype=list_(materialize[data_type]()),
        length=6,
        buffers=list_buffers^,
        children=list_children^,
        bitmap=list_bitmap^.freeze(),
        offset=0,
    )
    return ListArray(list_data^)


fn build_list_of_list[data_type: DataType]() raises -> ListArray:
    """Build a test ListArray that itself contains a ListArray of IntArrays.

    See: https://elferherrera.github.io/arrow_guide/arrays_nested.html
    """

    # Define all the values.
    var bitmap = BitmapBuilder.alloc(10)
    bitmap.unsafe_range_set(0, 10, True)
    var buffer = BufferBuilder.alloc[data_type.native](10)
    for i in range(10):
        buffer.unsafe_set[data_type.native](i, Scalar[data_type.native](i + 1))

    var value_buffers = List[Buffer]()
    value_buffers.append(buffer^.freeze())
    var value_data = Array(
        dtype=materialize[data_type](),
        length=10,
        bitmap=bitmap^.freeze(),
        buffers=value_buffers^,
        children=List[Array](),
        offset=0,
    )

    # Define the PrimitiveArrays.
    var value_offset = buffer_from[DType.int32](0, 2, 4, 7, 7, 8, 10)

    var list_bitmap = BitmapBuilder.alloc(6)
    list_bitmap.unsafe_range_set(0, 6, True)
    list_bitmap.unsafe_set(3, False)
    var list_buffers = List[Buffer]()
    list_buffers.append(value_offset^)
    var list_children = List[Array]()
    list_children.append(value_data^)
    var list_data = Array(
        dtype=list_(materialize[data_type]()),
        length=6,
        buffers=list_buffers^,
        children=list_children^,
        bitmap=list_bitmap^.freeze(),
        offset=0,
    )

    # Now define the master array data.
    var top_offsets = buffer_from[DType.int32](0, 2, 5, 6)
    var top_bitmap = BitmapBuilder.alloc(4)
    top_bitmap.unsafe_range_set(0, 4, True)
    var top_buffers = List[Buffer]()
    top_buffers.append(top_offsets^)
    var top_children = List[Array]()
    top_children.append(list_data^)
    return ListArray(
        Array(
            dtype=list_(list_(materialize[data_type]())),
            length=4,
            buffers=top_buffers^,
            children=top_children^,
            bitmap=top_bitmap^.freeze(),
            offset=0,
        )
    )


def build_struct() -> StructArray:
    var int_data_a = Array.from_buffer[int32](
        buffer_from[DType.int32](1, 2, 3, 4, 5), 5
    )
    var field_1 = Field("int_data_a", materialize[int32]())

    var int_data_b = Array.from_buffer[int32](
        buffer_from[DType.int32](10, 20, 30), 3
    )
    var field_2 = Field("int_data_b", materialize[int32]())
    bitmap = BitmapBuilder.alloc(2)
    bitmap.unsafe_range_set(0, 2, True)
    var struct_children = List[Array]()
    struct_children.append(int_data_a^)
    struct_children.append(int_data_b^)
    var struct_array_data = Array(
        dtype=struct_([field_1^, field_2^]),
        length=2,
        bitmap=bitmap^.freeze(),
        offset=0,
        buffers=List[Buffer](),
        children=struct_children^,
    )
    return StructArray(data=struct_array_data^)

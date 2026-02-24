from testing import assert_equal, assert_true, TestSuite

from marrow.arrays import *
from marrow.dtypes import *
from marrow.pretty import ArrayPrinter
from marrow.test_fixtures.arrays import (
    build_array_data,
    build_list_of_int,
    build_list_of_list,
    build_struct,
    bool_array,
)


def _fmt(array: Array, limit: Int = 3) -> String:
    var printer = ArrayPrinter(limit=limit)
    printer.visit(array)
    return printer.output^


def _fmt_chunked(array: ChunkedArray, limit: Int = 3) -> String:
    var printer = ArrayPrinter(limit=limit)
    printer.visit(array)
    return printer.output^


def test_format_primitive():
    var a = array[int32](1, 2, 3)
    assert_equal(
        str(Array(a^)),
        "PrimitiveArray[DataType(code=int32)]([1, 2, 3])",
    )


def test_format_primitive_with_limit():
    var a = array[int32](1, 2, 3, 4, 5)
    assert_equal(
        _fmt(Array(a^), limit=3),
        "PrimitiveArray[DataType(code=int32)]([1, 2, 3, ...])",
    )


def test_format_bool():
    var a = bool_array(True, False)
    assert_equal(
        str(Array(a^)),
        "PrimitiveArray[DataType(code=bool)]([True, False])",
    )


def test_format_string():
    var s = StringArray()
    s.unsafe_append("hello")
    s.unsafe_append("world")
    assert_equal(
        str(Array(s^)),
        "StringArray([hello, world])",
    )


def test_format_list():
    var arr = build_list_of_int[int64]()
    assert_equal(
        _fmt(Array(arr^), limit=3),
        (
            "ListArray([PrimitiveArray[DataType(code=int64)]([1, 2]),"
            " PrimitiveArray[DataType(code=int64)]([3, 4]),"
            " PrimitiveArray[DataType(code=int64)]([5, 6, 7]), ...])"
        ),
    )


def test_format_list_of_list():
    var arr = build_list_of_list[int16]()
    assert_equal(
        str(Array(arr^)),
        (
            "ListArray([ListArray([PrimitiveArray[DataType(code=int16)]([1,"
            " 2]), PrimitiveArray[DataType(code=int16)]([3, 4])]),"
            " ListArray([PrimitiveArray[DataType(code=int16)]([5, 6, 7]),"
            " PrimitiveArray[DataType(code=int16)]([]),"
            " PrimitiveArray[DataType(code=int16)]([8])]),"
            " ListArray([PrimitiveArray[DataType(code=int16)]([9, 10])]), ...])"
        ),
    )


def test_format_struct():
    var struct_arr = build_struct()
    assert_equal(
        _fmt(Array(struct_arr^), limit=3),
        (
            "StructArray({'int_data_a': "
            "PrimitiveArray[DataType(code=int32)]([1, 2, 3, ...]), "
            "'int_data_b': PrimitiveArray[DataType(code=int32)]([10, 20, 30])})"
        ),
    )


def test_format_empty_struct():
    var fields = [
        Field("id", materialize[int64]()),
        Field("name", materialize[string]()),
        Field("active", materialize[bool_]()),
    ]
    var s = StructArray(fields^, capacity=10)
    assert_equal(str(Array(s^)), "StructArray({})")


def test_format_chunked():
    var first = build_array_data(2, 0)
    var second = build_array_data(3, 0)
    var chunks = List[Array]()
    chunks.append(first^)
    chunks.append(second^)
    var chunked = ChunkedArray(materialize[uint8](), chunks^)
    assert_equal(
        str(chunked),
        (
            "ChunkedArray([PrimitiveArray[DataType(code=uint8)]([0, 1]),"
            " PrimitiveArray[DataType(code=uint8)]([0, 1, 2])])"
        ),
    )


def test_format_limits():
    assert_equal(
        _fmt(Array(array[int32](1, 2, 3, 4, 5, 6, 7, 8, 9, 10)), limit=0),
        "PrimitiveArray[DataType(code=int32)]([...])",
    )
    assert_equal(
        _fmt(Array(array[int32](1, 2, 3, 4, 5, 6, 7, 8, 9, 10)), limit=1),
        "PrimitiveArray[DataType(code=int32)]([1, ...])",
    )
    assert_equal(
        _fmt(Array(array[int32](1, 2, 3, 4, 5, 6, 7, 8, 9, 10)), limit=10),
        "PrimitiveArray[DataType(code=int32)]([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])",
    )


def test_format_empty_array():
    var arr = Int32Array(0)
    assert_equal(
        str(Array(arr^)),
        "PrimitiveArray[DataType(code=int32)]([])",
    )


def test_format_all_nulls():
    var arr = Int32Array(3)
    arr.data.length = 3
    arr.data.bitmap[].unsafe_range_set(0, 3, False)
    assert_equal(
        str(Array(arr^)),
        "PrimitiveArray[DataType(code=int32)]([NULL, NULL, NULL])",
    )


def test_format_mixed_nulls():
    var arr = Int32Array(5)
    arr.append(1)
    arr.append(2)
    arr.data.bitmap[].unsafe_set(2, False)
    arr.data.length = 3
    arr.append(4)
    assert_equal(
        str(Array(arr^)),
        "PrimitiveArray[DataType(code=int32)]([1, 2, NULL, ...])",
    )


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()

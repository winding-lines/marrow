from testing import assert_equal, TestSuite

from marrow.arrays import *
from marrow.buffers import bitmap_range_set, bitmap_set
from marrow.builders import PrimitiveBuilder, StringBuilder, StructBuilder
from marrow.dtypes import *
from marrow.pretty import ArrayPrinter
from marrow.test_fixtures.arrays import (
    build_list_of_int,
    build_list_of_list,
    build_struct,
)


def _fmt(arr: Array, limit: Int = 3) -> String:
    var printer = ArrayPrinter(limit=limit)
    printer.visit(arr)
    return printer^.finish()


def _fmt_chunked(arr: ChunkedArray, limit: Int = 3) -> String:
    var printer = ArrayPrinter(limit=limit)
    printer.visit(arr)
    return printer^.finish()


def test_format_primitive():
    var a = array[int32]([1, 2, 3])
    assert_equal(
        _fmt(Array(a^)),
        "PrimitiveArray[int32]([1, 2, 3])",
    )


def test_format_primitive_with_limit():
    var a = array[int32]([1, 2, 3, 4, 5])
    assert_equal(
        _fmt(Array(a^), limit=3),
        "PrimitiveArray[int32]([1, 2, 3, ...])",
    )


def test_format_bool():
    var a = array([True, False])
    assert_equal(
        _fmt(Array(a^)),
        "PrimitiveArray[bool]([True, False])",
    )


def test_format_string():
    var s = StringBuilder(capacity=2)
    s.unsafe_append("hello")
    s.unsafe_append("world")
    assert_equal(
        _fmt(Array(s^.freeze())),
        "StringArray([hello, world])",
    )


def test_format_list():
    var arr = build_list_of_int[int64]()
    assert_equal(
        _fmt(Array(arr^), limit=3),
        (
            "ListArray([PrimitiveArray[int64]([1, 2]),"
            " PrimitiveArray[int64]([3, 4]),"
            " PrimitiveArray[int64]([5, 6, 7]), ...])"
        ),
    )


def test_format_list_of_list():
    var arr = build_list_of_list[int16]()
    assert_equal(
        _fmt(Array(arr^)),
        (
            "ListArray([ListArray([PrimitiveArray[int16]([1,"
            " 2]), PrimitiveArray[int16]([3, 4])]),"
            " ListArray([PrimitiveArray[int16]([5, 6, 7]),"
            " NULL,"
            " PrimitiveArray[int16]([8])]),"
            " ListArray([PrimitiveArray[int16]([9, 10])]), ...])"
        ),
    )


def test_format_struct():
    var struct_arr = build_struct()
    assert_equal(
        _fmt(Array(struct_arr^), limit=3),
        (
            "StructArray({'int_data_a': "
            "PrimitiveArray[int32]([1, 2, 3, ...]), "
            "'int_data_b': PrimitiveArray[int32]([10, 20, 30])})"
        ),
    )


def test_format_empty_struct():
    var fields = [
        Field("id", materialize[int64]()),
        Field("name", materialize[string]()),
        Field("active", materialize[bool_]()),
    ]
    var s = StructBuilder(fields^, capacity=10)
    assert_equal(_fmt(Array(s^.freeze())), "StructArray({})")


def test_format_chunked():
    var first = array[uint8]([0, 1])
    var second = array[uint8]([0, 1, 2])
    var chunks = List[Array]()
    chunks.append(first^)
    chunks.append(second^)
    var chunked = ChunkedArray(materialize[uint8](), chunks^)
    assert_equal(
        _fmt_chunked(chunked^),
        (
            "ChunkedArray([PrimitiveArray[uint8]([0, 1]),"
            " PrimitiveArray[uint8]([0, 1, 2])])"
        ),
    )


def test_format_limits():
    assert_equal(
        _fmt(Array(array[int32]([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])), limit=0),
        "PrimitiveArray[int32]([...])",
    )
    assert_equal(
        _fmt(Array(array[int32]([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])), limit=1),
        "PrimitiveArray[int32]([1, ...])",
    )
    assert_equal(
        _fmt(Array(array[int32]([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])), limit=10),
        "PrimitiveArray[int32]([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])",
    )


def test_format_empty_array():
    var arr = PrimitiveBuilder[int32](0).freeze()
    assert_equal(
        _fmt(Array(arr^)),
        "PrimitiveArray[int32]([])",
    )


def test_format_all_nulls():
    var b = PrimitiveBuilder[int32](3)
    b.length = 3
    bitmap_range_set(b.bitmap.ptr, 0, 3, False)
    assert_equal(
        _fmt(Array(b^.freeze())),
        "PrimitiveArray[int32]([NULL, NULL, NULL])",
    )


def test_format_mixed_nulls():
    var b = PrimitiveBuilder[int32](5)
    b.append(1)
    b.append(2)
    bitmap_set(b.bitmap.ptr, 2, False)
    b.length = 3
    b.append(4)
    assert_equal(
        _fmt(Array(b^.freeze())),
        "PrimitiveArray[int32]([1, 2, NULL, ...])",
    )


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()

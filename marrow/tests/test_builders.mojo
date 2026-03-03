"""Tests for array builders (BoolBuilder, PrimitiveBuilder, StringBuilder,
ListBuilder, FixedSizeListBuilder, StructBuilder) and factory functions."""

from testing import assert_equal, assert_true, assert_false, TestSuite
from marrow.arrays import (
    Array,
    BoolArray,
    PrimitiveArray,
    StringArray,
    ListArray,
    FixedSizeListArray,
    StructArray,
)
from marrow.builders import (
    Builder,
    BoolBuilder,
    PrimitiveBuilder,
    StringBuilder,
    ListBuilder,
    FixedSizeListBuilder,
    StructBuilder,
    array,
    nulls,
    arange,
)
from marrow.dtypes import *


# ---------------------------------------------------------------------------
# BoolBuilder
# ---------------------------------------------------------------------------


def test_bool_builder_zero_length():
    var b = BoolBuilder()
    assert_equal(len(b), 0)
    var frozen = b.finish()
    assert_equal(frozen.length, 0)


def test_bool_builder_append_null():
    var b = BoolBuilder()
    b.append(True)
    b.append_null()
    b.append(False)
    var frozen = b.finish()
    assert_equal(frozen.length, 3)
    assert_true(frozen.is_valid(0))
    assert_false(frozen.is_valid(1))
    assert_true(frozen.is_valid(2))
    assert_true(frozen.unsafe_get(0))
    assert_false(frozen.unsafe_get(2))


def test_bool_builder_null_count():
    var b = BoolBuilder(4)
    b.append(True)
    b.append_null()
    b.append_null()
    b.append(False)
    var frozen = b.finish()
    assert_equal(frozen.null_count(), 2)


def test_bool_builder_all_nulls():
    var b = BoolBuilder(3)
    b.append_null()
    b.append_null()
    b.append_null()
    var frozen = b.finish()
    assert_equal(frozen.length, 3)
    assert_equal(frozen.null_count(), 3)
    for i in range(3):
        assert_false(frozen.is_valid(i))


def test_bool_builder_all_false():
    var b = BoolBuilder(3)
    b.append(False)
    b.append(False)
    b.append(False)
    var frozen = b.finish()
    assert_equal(frozen.length, 3)
    assert_equal(frozen.null_count(), 0)
    for i in range(3):
        assert_true(frozen.is_valid(i))
        assert_false(frozen.unsafe_get(i))


def test_bool_builder_as_builder():
    var b = BoolBuilder(2)
    b.append(True)
    b.append(False)
    var builder: Builder = b^
    assert_equal(builder.data[].length, 2)


# ---------------------------------------------------------------------------
# PrimitiveBuilder — type coverage
# ---------------------------------------------------------------------------


def test_primitive_builder_int16():
    var b = PrimitiveBuilder[int16](2)
    b.append(32767)
    b.append(-32768)
    var frozen = b.finish()
    assert_equal(frozen.unsafe_get(0), 32767)
    assert_equal(frozen.unsafe_get(1), -32768)


def test_primitive_builder_uint32():
    var b = PrimitiveBuilder[uint32](2)
    b.append(0)
    b.append(42)
    var frozen = b.finish()
    assert_equal(frozen.unsafe_get(0), 0)
    assert_equal(frozen.unsafe_get(1), 42)


def test_primitive_builder_float32():
    var b = PrimitiveBuilder[float32](3)
    b.append(1.5)
    b.append(0.0)
    b.append(-3.14)
    var frozen = b.finish()
    assert_equal(frozen.length, 3)
    assert_true(frozen.is_valid(0))
    assert_true(frozen.is_valid(1))
    assert_true(frozen.is_valid(2))


def test_primitive_builder_float64():
    var b = PrimitiveBuilder[float64](2)
    b.append(1.0)
    b.append_null()
    var frozen = b.finish()
    assert_equal(frozen.null_count(), 1)
    assert_true(frozen.is_valid(0))
    assert_false(frozen.is_valid(1))


def test_primitive_builder_capacity_doubling():
    """Builder doubles capacity starting from zero capacity."""
    var b = PrimitiveBuilder[int32]()
    b.append(0)
    b.append(1)
    b.append(2)
    b.append(3)
    b.append(4)
    b.append(5)
    b.append(6)
    b.append(7)
    b.append(8)
    b.append(9)
    assert_equal(len(b), 10)
    var frozen = b.finish()
    assert_equal(frozen.unsafe_get(0), 0)
    assert_equal(frozen.unsafe_get(4), 4)
    assert_equal(frozen.unsafe_get(9), 9)


def test_primitive_builder_as_builder():
    var b = PrimitiveBuilder[int64](3)
    b.append(1)
    b.append(2)
    b.append(3)
    var builder: Builder = b^
    assert_equal(builder.data[].length, 3)
    assert_equal(builder.data[].dtype, materialize[int64]())


def test_primitive_builder_null_count():
    var b = PrimitiveBuilder[int32](5)
    b.append(1)
    b.append_null()
    b.append_null()
    b.append(4)
    b.append_null()
    var frozen = b.finish()
    assert_equal(frozen.null_count(), 3)


def test_primitive_builder_all_nulls():
    var b = PrimitiveBuilder[int64](4)
    for _ in range(4):
        b.append_null()
    var frozen = b.finish()
    assert_equal(frozen.length, 4)
    assert_equal(frozen.null_count(), 4)
    for i in range(4):
        assert_false(frozen.is_valid(i))


# ---------------------------------------------------------------------------
# StringBuilder
# ---------------------------------------------------------------------------


def test_string_builder_append_null():
    var b = StringBuilder(3)
    b.append("hello")
    b.append_null()
    b.append("world")
    var frozen = b.finish()
    assert_equal(frozen.length, 3)
    assert_true(frozen.is_valid(0))
    assert_false(frozen.is_valid(1))
    assert_true(frozen.is_valid(2))
    assert_equal(String(frozen.unsafe_get(0)), "hello")
    assert_equal(String(frozen.unsafe_get(2)), "world")


def test_string_builder_null_count():
    var b = StringBuilder()
    b.append("a")
    b.append_null()
    b.append_null()
    b.append("b")
    var frozen = b.finish()
    assert_equal(frozen.length, 4)
    assert_true(frozen.is_valid(0))
    assert_false(frozen.is_valid(1))
    assert_false(frozen.is_valid(2))
    assert_true(frozen.is_valid(3))


def test_string_builder_empty_string():
    var b = StringBuilder(3)
    b.append("")
    b.append("x")
    b.append("")
    var frozen = b.finish()
    assert_equal(frozen.length, 3)
    assert_equal(String(frozen.unsafe_get(0)), "")
    assert_equal(String(frozen.unsafe_get(1)), "x")
    assert_equal(String(frozen.unsafe_get(2)), "")


def test_string_builder_offsets_correct():
    """Null entry must not advance the offset; valid entries advance by len."""
    var b = StringBuilder(4)
    b.append("ab")  # bytes [0..2)
    b.append("cde")  # bytes [2..5)
    b.append_null()  # bytes [5..5) — offset stays at 5
    b.append("f")  # bytes [5..6)
    var frozen = b.finish()
    assert_equal(frozen.length, 4)
    assert_equal(String(frozen.unsafe_get(0)), "ab")
    assert_equal(String(frozen.unsafe_get(1)), "cde")
    assert_equal(String(frozen.unsafe_get(3)), "f")


def test_string_builder_all_nulls():
    var b = StringBuilder(2)
    b.append_null()
    b.append_null()
    var frozen = b.finish()
    assert_equal(frozen.length, 2)
    assert_false(frozen.is_valid(0))
    assert_false(frozen.is_valid(1))


def test_string_builder_capacity_growth():
    """Builder grows when appending beyond initial capacity."""
    var b = StringBuilder(1)
    b.append("first")
    b.append("second")
    b.append("third")
    var frozen = b.finish()
    assert_equal(frozen.length, 3)
    assert_equal(String(frozen.unsafe_get(0)), "first")
    assert_equal(String(frozen.unsafe_get(1)), "second")
    assert_equal(String(frozen.unsafe_get(2)), "third")


def test_string_builder_as_builder():
    var b = StringBuilder(1)
    b.append("test")
    var builder: Builder = b^
    assert_equal(builder.data[].length, 1)
    assert_equal(builder.data[].dtype, materialize[string]())


# ---------------------------------------------------------------------------
# ListBuilder
# ---------------------------------------------------------------------------


def test_list_builder_empty():
    var child = PrimitiveBuilder[int32]()
    var b = ListBuilder(child^)
    var frozen = b.finish()
    assert_equal(frozen.length, 0)
    assert_true(frozen.dtype.is_list())


def test_list_builder_append_null():
    var child = PrimitiveBuilder[int64]()
    var b = ListBuilder(child^)
    b.append(True)
    b.append_null()
    b.append(True)
    var frozen = b.finish()
    assert_equal(frozen.length, 3)
    assert_true(frozen.is_valid(0))
    assert_false(frozen.is_valid(1))
    assert_true(frozen.is_valid(2))


def test_list_builder_null_count():
    var child = PrimitiveBuilder[int32]()
    var b = ListBuilder(child^)
    b.append(True)
    b.append_null()
    b.append_null()
    b.append(True)
    var frozen = b.finish()
    assert_equal(frozen.length, 4)
    assert_true(frozen.is_valid(0))
    assert_false(frozen.is_valid(1))
    assert_false(frozen.is_valid(2))
    assert_true(frozen.is_valid(3))


def test_list_builder_empty_list():
    """A valid but empty list (zero child elements) is valid."""
    var child = PrimitiveBuilder[int64]()
    var b = ListBuilder(child^)
    b.append(True)
    var frozen = b.finish()
    assert_equal(frozen.length, 1)
    assert_true(frozen.is_valid(0))
    var inner = PrimitiveArray[int64](frozen.unsafe_get(0))
    assert_equal(inner.length, 0)


def test_list_builder_dtype():
    var child = PrimitiveBuilder[int64]()
    var b = ListBuilder(child^)
    assert_equal(b.data[].dtype, list_(materialize[int64]()))


def test_list_builder_child_accessor():
    var child = PrimitiveBuilder[int32](4)
    child.append(10)
    var b = ListBuilder(child^)
    var child_arc = b.child()
    assert_equal(child_arc.data[].length, 1)


def test_list_builder_multiple_nulls_offsets():
    """Multiple null entries must not advance child offsets."""
    var child = PrimitiveBuilder[int32]()
    child.append(1)
    child.append(2)
    var b = ListBuilder(child^)
    b.append(True)  # list with child elements [1, 2]
    b.append_null()  # null — child length unchanged
    b.append_null()  # null — child length unchanged
    var frozen = b.finish()
    assert_equal(frozen.length, 3)
    assert_true(frozen.is_valid(0))
    assert_false(frozen.is_valid(1))
    assert_false(frozen.is_valid(2))
    var first = PrimitiveArray[int32](frozen.unsafe_get(0))
    assert_equal(first.length, 2)
    assert_equal(first.unsafe_get(0), 1)
    assert_equal(first.unsafe_get(1), 2)


def test_list_builder_string_child():
    var str_b = StringBuilder()
    str_b.append("hello")
    str_b.append("world")
    var b = ListBuilder(str_b^)
    b.append(True)
    var frozen = b.finish()
    assert_equal(frozen.length, 1)
    var inner = StringArray(frozen.unsafe_get(0))
    assert_equal(String(inner.unsafe_get(0)), "hello")
    assert_equal(String(inner.unsafe_get(1)), "world")


# ---------------------------------------------------------------------------
# FixedSizeListBuilder
# ---------------------------------------------------------------------------


def test_fixed_size_list_builder_zero_length():
    var child = PrimitiveBuilder[int32]()
    var b = FixedSizeListBuilder(child, list_size=4)
    var frozen = b.finish()
    assert_equal(frozen.length, 0)
    assert_true(frozen.dtype.is_fixed_size_list())
    assert_equal(frozen.dtype.size, 4)


def test_fixed_size_list_builder_float32():
    var child = PrimitiveBuilder[float32](4)
    child.append(1.0)
    child.append(2.0)
    child.append(3.0)
    child.append(4.0)
    var b = FixedSizeListBuilder(child, list_size=2)
    b.append(True)
    b.append(True)
    var frozen = b.finish()
    assert_equal(frozen.length, 2)
    var first = frozen.unsafe_get(0).as_float32()
    assert_equal(first.length, 2)
    var second = frozen.unsafe_get(1).as_float32()
    assert_equal(second.length, 2)


def test_fixed_size_list_builder_with_nulls():
    var child = PrimitiveBuilder[int64](6)
    child.append(0)
    child.append(1)
    child.append(2)
    child.append(3)
    child.append(4)
    child.append(5)
    var b = FixedSizeListBuilder(child, list_size=2, capacity=3)
    b.append(True)
    b.append(False)
    b.append(True)
    var frozen = b.finish()
    assert_equal(frozen.length, 3)
    assert_true(frozen.is_valid(0))
    assert_false(frozen.is_valid(1))
    assert_true(frozen.is_valid(2))


def test_fixed_size_list_builder_dtype():
    var child = PrimitiveBuilder[int32]()
    var b = FixedSizeListBuilder(child, list_size=3)
    assert_equal(b.data[].dtype, fixed_size_list_(materialize[int32](), 3))


def test_fixed_size_list_builder_child_accessor():
    var child = PrimitiveBuilder[int64](2)
    child.append(100)
    child.append(200)
    var b = FixedSizeListBuilder(child, list_size=2)
    var child_arc = b.child()
    assert_equal(child_arc.data[].length, 2)


def test_fixed_size_list_builder_size1():
    """FixedSizeList of size 1 — each entry is a single-element list."""
    var child = PrimitiveBuilder[int32](3)
    child.append(7)
    child.append(8)
    child.append(9)
    var b = FixedSizeListBuilder(child, list_size=1)
    b.append(True)
    b.append(True)
    b.append(True)
    var frozen = b.finish()
    assert_equal(frozen.length, 3)
    var first = frozen.unsafe_get(0).as_int32()
    assert_equal(first.length, 1)
    assert_equal(first.unsafe_get(0), 7)
    var third = frozen.unsafe_get(2).as_int32()
    assert_equal(third.unsafe_get(0), 9)


# ---------------------------------------------------------------------------
# StructBuilder
# ---------------------------------------------------------------------------


def test_struct_builder_zero_length():
    var fields = List[Field]()
    var children = List[Builder]()
    var sb = StructBuilder(fields^, children^)
    var frozen = sb.finish()
    assert_equal(frozen.length, 0)
    assert_true(frozen.dtype.is_struct())


def test_struct_builder_append_valid():
    """Struct validity tracks correct; child builders drive field values."""
    var id_b = PrimitiveBuilder[int64](3)
    id_b.append(1)
    id_b.append(2)
    id_b.append(3)
    var score_b = PrimitiveBuilder[float64](3)
    score_b.append(0.1)
    score_b.append(0.2)
    score_b.append(0.3)

    var fields = List[Field]()
    fields.append(Field("id", materialize[int64]()))
    fields.append(Field("score", materialize[float64]()))
    var children = List[Builder]()
    children.append(id_b)
    children.append(score_b)
    var sb = StructBuilder(fields^, children^, capacity=3)
    sb.append(True)
    sb.append(True)
    sb.append(True)
    assert_equal(len(sb), 3)
    var frozen = sb.finish()
    assert_equal(frozen.length, 3)
    assert_equal(len(frozen.dtype.fields), 2)
    assert_equal(frozen.dtype.fields[0].name, "id")
    assert_equal(frozen.dtype.fields[1].name, "score")
    # All entries valid
    for i in range(3):
        assert_true(frozen.bitmap.unsafe_get[DType.bool](i))


def test_struct_builder_append_null():
    """Null struct entries — validity bitmap reflects nulls."""
    var id_b = PrimitiveBuilder[int32](2)
    id_b.append(10)
    id_b.append(20)

    var fields = List[Field]()
    fields.append(Field("id", materialize[int32]()))
    var children = List[Builder]()
    children.append(id_b)
    var sb = StructBuilder(fields^, children^, capacity=2)
    sb.append(True)
    sb.append_null()
    var frozen = sb.finish()
    assert_equal(frozen.length, 2)
    assert_true(frozen.bitmap.unsafe_get[DType.bool](0))
    assert_false(frozen.bitmap.unsafe_get[DType.bool](1))


def test_struct_builder_field_values_accessible():
    """Child field values are accessible after finish."""
    var x_b = PrimitiveBuilder[int32](2)
    x_b.append(42)
    x_b.append(99)

    var fields = List[Field]()
    fields.append(Field("x", materialize[int32]()))
    var children = List[Builder]()
    children.append(x_b)
    var sb = StructBuilder(fields^, children^, capacity=2)
    sb.append(True)
    sb.append(True)
    var frozen = sb.finish()

    ref field_data = frozen.unsafe_get("x")
    var x_arr = PrimitiveArray[int32](field_data.copy())
    assert_equal(x_arr.unsafe_get(0), 42)
    assert_equal(x_arr.unsafe_get(1), 99)


def test_struct_builder_multi_type_fields():
    """Struct with primitive, string, and bool fields."""
    var id_b = PrimitiveBuilder[int64](2)
    id_b.append(1)
    id_b.append(2)
    var name_b = StringBuilder(2)
    name_b.append("alice")
    name_b.append("bob")
    var active_b = BoolBuilder(2)
    active_b.append(True)
    active_b.append(False)

    var fields = List[Field]()
    fields.append(Field("id", materialize[int64]()))
    fields.append(Field("name", materialize[string]()))
    fields.append(Field("active", materialize[bool_]()))
    var children = List[Builder]()
    children.append(id_b)
    children.append(name_b)
    children.append(active_b)
    var sb = StructBuilder(fields^, children^, capacity=2)
    sb.append(True)
    sb.append(True)

    var frozen = sb.finish()
    assert_equal(frozen.length, 2)
    assert_equal(len(frozen.dtype.fields), 3)
    assert_equal(frozen.dtype.fields[0].name, "id")
    assert_equal(frozen.dtype.fields[1].name, "name")
    assert_equal(frozen.dtype.fields[2].name, "active")


def test_struct_builder_child_accessor():
    var x_b = PrimitiveBuilder[int32](1)
    x_b.append(7)
    var y_b = PrimitiveBuilder[int32](1)
    y_b.append(8)

    var fields = List[Field]()
    fields.append(Field("x", materialize[int32]()))
    fields.append(Field("y", materialize[int32]()))
    var children = List[Builder]()
    children.append(x_b)
    children.append(y_b)
    var sb = StructBuilder(fields^, children^)
    sb.append(True)

    assert_equal(sb.child(0).data[].length, 1)
    assert_equal(sb.child(1).data[].length, 1)


def test_struct_builder_capacity_growth():
    var id_b = PrimitiveBuilder[int32]()
    id_b.append(0)
    id_b.append(1)
    id_b.append(2)
    id_b.append(3)
    id_b.append(4)

    var fields = List[Field]()
    fields.append(Field("id", materialize[int32]()))
    var children = List[Builder]()
    children.append(id_b)
    var sb = StructBuilder(fields^, children^)
    for _ in range(5):
        sb.append(True)
    var frozen = sb.finish()
    assert_equal(frozen.length, 5)


def test_struct_builder_field_names_preserved():
    """Field names survive builder → finish cycle."""
    var a_b = PrimitiveBuilder[int8](1)
    a_b.append(1)
    var b_b = PrimitiveBuilder[int8](1)
    b_b.append(2)

    var fields = List[Field]()
    fields.append(Field("alpha", materialize[int8]()))
    fields.append(Field("beta", materialize[int8]()))
    var children = List[Builder]()
    children.append(a_b)
    children.append(b_b)
    var sb = StructBuilder(fields^, children^)
    sb.append(True)
    var frozen = sb.finish()
    assert_equal(frozen.dtype.fields[0].name, "alpha")
    assert_equal(frozen.dtype.fields[1].name, "beta")


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def test_factory_array_empty():
    var a = array[int32]()
    assert_equal(len(a), 0)
    assert_equal(a.null_count(), 0)


def test_factory_array_bool_with_nulls():
    var a = array([True, None, False, None, True])
    assert_equal(len(a), 5)
    assert_equal(a.null_count(), 2)
    assert_true(a.is_valid(0))
    assert_false(a.is_valid(1))
    assert_true(a.is_valid(2))
    assert_false(a.is_valid(3))
    assert_true(a.is_valid(4))
    assert_true(a.unsafe_get(0))
    assert_false(a.unsafe_get(2))
    assert_true(a.unsafe_get(4))


def test_factory_array_bool_all_nulls():
    var a = array([None, None, None])
    assert_equal(len(a), 3)
    assert_equal(a.null_count(), 3)


def test_factory_array_int_with_nulls():
    var a = array[int32]([1, None, 3, None, 5])
    assert_equal(len(a), 5)
    assert_equal(a.null_count(), 2)
    assert_equal(a.unsafe_get(0), 1)
    assert_equal(a.unsafe_get(2), 3)
    assert_equal(a.unsafe_get(4), 5)
    assert_true(a.is_valid(0))
    assert_false(a.is_valid(1))
    assert_false(a.is_valid(3))


def test_factory_array_int_all_valid():
    var a = array[int64]([10, 20, 30])
    assert_equal(len(a), 3)
    assert_equal(a.null_count(), 0)
    assert_equal(a.unsafe_get(0), 10)
    assert_equal(a.unsafe_get(1), 20)
    assert_equal(a.unsafe_get(2), 30)


def test_factory_nulls_all_invalid():
    var a = nulls[int64](5)
    assert_equal(len(a), 5)
    assert_equal(a.null_count(), 5)
    for i in range(5):
        assert_false(a.is_valid(i))


def test_factory_nulls_zero():
    var a = nulls[int32](0)
    assert_equal(len(a), 0)


def test_factory_nulls_one():
    var a = nulls[int8](1)
    assert_equal(len(a), 1)
    assert_false(a.is_valid(0))


def test_factory_arange_validity():
    var a = arange[int32](0, 5)
    assert_equal(len(a), 5)
    assert_equal(a.null_count(), 0)
    for i in range(5):
        assert_true(a.is_valid(i))
    assert_equal(a.unsafe_get(0), 0)
    assert_equal(a.unsafe_get(4), 4)


def test_factory_arange_non_zero_start():
    var a = arange[int64](10, 15)
    assert_equal(len(a), 5)
    assert_equal(a.unsafe_get(0), 10)
    assert_equal(a.unsafe_get(4), 14)


def test_factory_arange_single():
    var a = arange[int32](7, 8)
    assert_equal(len(a), 1)
    assert_equal(a.unsafe_get(0), 7)


def test_factory_arange_empty():
    var a = arange[int32](3, 3)
    assert_equal(len(a), 0)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()

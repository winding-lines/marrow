from testing import assert_equal, assert_true, TestSuite

from marrow.arrays import (
    array,
    Array,
    PrimitiveArray,
    nulls,
)
from marrow.builders import PrimitiveBuilder
from marrow.dtypes import int32, int64, uint8
from marrow.compute.filter import drop_nulls


def test_drop_nulls_typed():
    """Drop nulls removes null elements and compacts valid ones."""
    var a = PrimitiveBuilder[int32](4)
    a.append(10)
    a.append_null()
    a.append(30)
    a.append_null()
    # [10, null, 30, null]
    var result = drop_nulls[int32](a.finish())
    assert_equal(len(result), 2)
    assert_equal(result.unsafe_get(0), 10)
    assert_equal(result.unsafe_get(1), 30)


def test_drop_nulls_no_nulls():
    var a = array[int64]([1, 2, 3])
    var result = drop_nulls[int64](a)
    assert_equal(len(result), 3)
    assert_equal(result.unsafe_get(0), 1)
    assert_equal(result.unsafe_get(1), 2)
    assert_equal(result.unsafe_get(2), 3)


def test_drop_nulls_all_nulls():
    var a = nulls[int64](5)
    var result = drop_nulls[int64](a)
    assert_equal(len(result), 0)


def test_drop_nulls_empty():
    var a = array[int32]()
    var result = drop_nulls[int32](a)
    assert_equal(len(result), 0)


def test_drop_nulls_untyped():
    var result = drop_nulls(
        array[uint8]([None, 1, None, 3, None, 5, None, 7, None, 9])
    )
    assert_equal(result.length, 5)


def test_drop_nulls_matches_old_behavior():
    """Verify that new drop_nulls produces same results as old in-place version.
    """
    var primitive = array[uint8]([None, 1, None, 3, None, 5, None, 7, None, 9])
    var result = drop_nulls[uint8](primitive)
    assert_equal(len(result), 5)
    # The fixture sets values to i % 256, with odd indices valid
    assert_equal(result.unsafe_get(0), 1)
    assert_equal(result.unsafe_get(1), 3)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()

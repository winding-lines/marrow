from testing import assert_equal, assert_true, TestSuite

from marrow.arrays import array, Array, Int32Array, Int64Array, PrimitiveArray
from marrow.dtypes import int32, int64, uint8
from marrow.compute.filter import drop_nulls


def test_drop_nulls_typed():
    """Drop nulls removes null elements and compacts valid ones."""
    var a = Int32Array(4)
    a.unsafe_append(10)
    # index 1 is null
    a.unsafe_append(30)
    # Now length=2 with indices 0,1 valid. We need a gap.
    # Build manually: append 2 values, set length to 4, set index 3
    a.length = 4
    a.unsafe_set(3, 40)
    # Now: [10, 30, null, 40]
    var result = drop_nulls[int32](a)
    assert_equal(len(result), 3)
    assert_equal(result.unsafe_get(0), 10)
    assert_equal(result.unsafe_get(1), 30)
    assert_equal(result.unsafe_get(2), 40)


def test_drop_nulls_no_nulls():
    var a = array[int64]([1, 2, 3])
    var result = drop_nulls[int64](a)
    assert_equal(len(result), 3)
    assert_equal(result.unsafe_get(0), 1)
    assert_equal(result.unsafe_get(1), 2)
    assert_equal(result.unsafe_get(2), 3)


def test_drop_nulls_all_nulls():
    var a = Int64Array.nulls(5)
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

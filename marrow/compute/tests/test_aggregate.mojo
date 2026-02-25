from testing import assert_equal, TestSuite

from marrow.arrays import array, Array, Int32Array, Int64Array
from marrow.dtypes import int32, int64
from marrow.compute.aggregate import sum


def test_sum_typed():
    var a = array[int64]([1, 2, 3, 4, 5])
    var result = sum[int64](a)
    assert_equal(result, 15)


def test_sum_with_nulls():
    """Sum skips null values."""
    var a = Int32Array(3)
    a.unsafe_append(10)
    a.unsafe_append(20)
    # index 2 is null
    a.length = 3
    var result = sum[int32](a)
    assert_equal(result, 30)


def test_sum_all_nulls():
    var a = Int64Array.nulls(5)
    var result = sum[int64](a)
    assert_equal(result, 0)


def test_sum_empty():
    var a = array[int32]([])
    var result = sum[int32](a)
    assert_equal(result, 0)


def test_sum_untyped():
    var a = Array(array[int64]([1, 2, 3]))
    var result = sum(a)
    assert_equal(result, 6.0)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()

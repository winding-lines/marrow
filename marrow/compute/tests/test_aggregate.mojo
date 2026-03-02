from testing import assert_equal, TestSuite

from marrow.arrays import array, Array, PrimitiveArray, nulls
from marrow.builders import PrimitiveBuilder
from marrow.dtypes import int32, int64
from marrow.compute.kernels.sum import sum


def test_sum_typed():
    var a = array[int64]([1, 2, 3, 4, 5])
    var result = sum[int64](a)
    assert_equal(result, 15)


def test_sum_with_nulls():
    """Sum skips null values."""
    var a = PrimitiveBuilder[int32](3)
    a.append(10)
    a.append(20)
    a.append_null()  # index 2 is null
    var result = sum[int32](a.finish())
    assert_equal(result, 30)


def test_sum_all_nulls():
    var a = nulls[int64](5)
    var result = sum[int64](a)
    assert_equal(result, 0)


def test_sum_empty():
    var a = array[int32]()
    var result = sum[int32](a)
    assert_equal(result, 0)


def test_sum_untyped():
    var a = Array(array[int64]([1, 2, 3]))
    var result = sum(a)
    assert_equal(result, 6.0)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()

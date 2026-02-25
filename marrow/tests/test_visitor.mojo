from testing import assert_equal, TestSuite

from marrow.arrays import *
from marrow.dtypes import *
from marrow.visitor import ArrayVisitor


struct ElementCounter(ArrayVisitor):
    """Demonstrates custom visitor: counts valid elements across array kinds."""

    var count: Int

    fn __init__(out self):
        self.count = 0

    fn visit[T: DataType](mut self, array: PrimitiveArray[T]) raises:
        self.count += array.null_count() * -1 + array.length

    fn visit(mut self, array: StringArray) raises:
        self.count += array.length

    fn visit(mut self, array: ListArray) raises:
        self.count += array.length

    fn visit(mut self, array: StructArray) raises:
        self.count += array.length


def test_custom_visitor():
    var a = array[int64]([10, 20, 30, 40])
    var counter = ElementCounter()
    counter.visit(Array(a^))
    assert_equal(counter.count, 4)


def test_chunked_array_default_dispatch():
    """ChunkedArray.visit default delegates to visit(Array) for each chunk."""
    var chunks = List[Array]()
    chunks.append(Array(array[int64]([1, 2, 3])))
    chunks.append(Array(array[int64]([4, 5])))
    var chunked = ChunkedArray(materialize[int64](), chunks^)
    var counter = ElementCounter()
    counter.visit(chunked)
    assert_equal(counter.count, 5)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()

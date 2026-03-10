from std.testing import assert_equal, assert_true, TestSuite
from marrow.dyn import AnyArray, PrimitiveArray, ListArray, StructArray
from marrow.dtypes import int32, int64, float64


def test_primitive_dispatch() raises:
    var arr: AnyArray = PrimitiveArray[int64]([1, 2, 3, 4, 5])
    assert_equal(arr.length(), 5)
    assert_equal(arr.null_count(), 0)
    assert_true(arr.is_valid(0))
    assert_true(arr.is_valid(4))


def test_primitive_with_nulls() raises:
    var arr: AnyArray = PrimitiveArray[int32]([1, 0, 3], null_count=1)
    assert_equal(arr.length(), 3)
    assert_equal(arr.null_count(), 1)


def test_primitive_different_dtypes() raises:
    """Different parameterizations of PrimitiveArray all erase into AnyArray."""
    var i32: AnyArray = PrimitiveArray[int32]([1, 2, 3])
    var i64: AnyArray = PrimitiveArray[int64]([10, 20, 30, 40])
    var f64: AnyArray = PrimitiveArray[float64]([1.5, 2.5])

    assert_equal(i32.length(), 3)
    assert_equal(i64.length(), 4)
    assert_equal(f64.length(), 2)

    # heterogeneous list of different primitive dtypes
    var arrays = List[AnyArray]()
    arrays.append(i32)
    arrays.append(i64)
    arrays.append(f64)
    assert_equal(arrays[0].length(), 3)
    assert_equal(arrays[1].length(), 4)
    assert_equal(arrays[2].length(), 2)


def test_list_dispatch() raises:
    var child: AnyArray = PrimitiveArray[int64]([10, 20, 30, 40, 50])
    # offsets [0, 3, 5] means two sublists: elements [0..3) and [3..5)
    var arr: AnyArray = ListArray([0, 3, 5], child)
    assert_equal(arr.length(), 2)
    assert_equal(arr.null_count(), 0)


def test_struct_dispatch() raises:
    var f1: AnyArray = PrimitiveArray[int32]([1, 2, 3])
    var f2: AnyArray = PrimitiveArray[float64]([4.0, 5.0, 6.0])
    var children = List[AnyArray]()
    children.append(f1)
    children.append(f2)
    var arr: AnyArray = StructArray(3, children^)
    assert_equal(arr.length(), 3)
    assert_equal(arr.null_count(), 0)


def test_heterogeneous_list() raises:
    """A List[AnyArray] can hold primitive, list, and struct arrays together."""
    var prim: AnyArray = PrimitiveArray[int64]([1, 2, 3])

    var list_child: AnyArray = PrimitiveArray[int32]([10, 20, 30])
    var lst: AnyArray = ListArray([0, 2, 3], list_child)

    var f1: AnyArray = PrimitiveArray[int64]([1, 2])
    var f2: AnyArray = PrimitiveArray[int64]([3, 4])
    var ch = List[AnyArray]()
    ch.append(f1)
    ch.append(f2)
    var st: AnyArray = StructArray(2, ch^)

    var arrays = List[AnyArray]()
    arrays.append(prim)
    arrays.append(lst)
    arrays.append(st)

    assert_equal(arrays[0].length(), 3)
    assert_equal(arrays[1].length(), 2)
    assert_equal(arrays[2].length(), 2)


def test_nested_struct_of_lists() raises:
    """Struct where one field is a list array — deeply nested dispatch."""
    var list_child: AnyArray = PrimitiveArray[int64]([1, 2, 3, 4, 5, 6])
    var list_arr: AnyArray = ListArray([0, 3, 6], list_child)
    var scalar_arr: AnyArray = PrimitiveArray[int32]([100, 200])

    var children = List[AnyArray]()
    children.append(list_arr)
    children.append(scalar_arr)
    var st: AnyArray = StructArray(2, children^)

    assert_equal(st.length(), 2)
    assert_equal(st.null_count(), 0)


def test_copy_semantics() raises:
    """Copies share the underlying data via ArcPointer ref-counting."""
    var original: AnyArray = PrimitiveArray[int64]([1, 2, 3])
    var copy = original
    assert_equal(copy.length(), 3)
    assert_equal(original.length(), 3)


def test_multiple_copies_no_early_destruct() raises:
    """Multiple copies of a AnyArray must keep the data alive until the last one drops."""
    var a: AnyArray = PrimitiveArray[int64]([10, 20, 30])
    var b = a
    var c = b
    var d = a

    # all four point to the same underlying data
    assert_equal(a.length(), 3)
    assert_equal(b.length(), 3)
    assert_equal(c.length(), 3)
    assert_equal(d.length(), 3)

    # put copies into a list, then drop the list — originals must survive
    var lst = List[AnyArray]()
    lst.append(a)
    lst.append(b)
    lst.append(c)
    lst.append(d)
    assert_equal(len(lst), 4)
    _ = lst^  # drop all list elements

    # the four local copies must still dispatch correctly
    assert_equal(a.length(), 3)
    assert_equal(b.length(), 3)
    assert_equal(c.length(), 3)
    assert_equal(d.length(), 3)
    assert_true(a.is_valid(0))
    assert_true(d.is_valid(2))


def test_upcast_downcast_primitive() raises:
    var dyn = PrimitiveArray[int64]([1, 2, 3]).as_any()
    var arc = dyn.downcast[PrimitiveArray[int64]]()
    assert_equal(arc[].length(), 3)
    assert_equal(arc[].values[0], 1)
    assert_equal(arc[].values[2], 3)


def test_upcast_downcast_list() raises:
    var child: AnyArray = PrimitiveArray[int64]([10, 20, 30])
    var dyn = ListArray([0, 2, 3], child).as_any()
    var arc = dyn.downcast[ListArray]()
    assert_equal(arc[].length(), 2)
    assert_equal(arc[].offsets[1], 2)


def test_upcast_downcast_struct() raises:
    var f1: AnyArray = PrimitiveArray[int32]([1, 2])
    var f2: AnyArray = PrimitiveArray[float64]([3.0, 4.0])
    var ch = List[AnyArray]()
    ch.append(f1)
    ch.append(f2)
    var dyn = StructArray(2, ch^).as_any()
    var arc = dyn.downcast[StructArray]()
    assert_equal(arc[].length(), 2)


def test_downcast_outlives_dynarray() raises:
    """Downcast ArcPointer keeps data alive after AnyArray is dropped."""
    var dyn: AnyArray = PrimitiveArray[int64]([42, 99])
    var arc = dyn.downcast[PrimitiveArray[int64]]()
    _ = dyn^
    assert_equal(arc[].values[0], 42)
    assert_equal(arc[].values[1], 99)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

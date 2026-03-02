"""Vector (shape-changing) kernels."""

from marrow.arrays import PrimitiveArray, Array

from marrow.builders import PrimitiveBuilder
from marrow.dtypes import DataType, bool_, all_numeric_dtypes, materialize


fn drop_nulls[
    T: DataType
](array: PrimitiveArray[T]) raises -> PrimitiveArray[T]:
    """Create a new array containing only the valid (non-null) elements.

    Args:
        array: The input array.

    Returns:
        A new PrimitiveArray containing only valid elements with a
        fully-valid bitmap.
    """
    var valid_count = len(array) - array.null_count()
    var result = PrimitiveBuilder[T](valid_count)
    for i in range(len(array)):
        if array.is_valid(i):
            result.append(array.unsafe_get(i))
    return result.finish()


fn drop_nulls(array: Array) raises -> Array:
    """Runtime-typed drop_nulls: dispatches to the correct typed version.

    Args:
        array: The input array (runtime-typed).

    Returns:
        A new Array with null elements removed.
    """
    if array.dtype == materialize[bool_]():
        return Array(drop_nulls[bool_](PrimitiveArray[bool_](data=array)))

    comptime for dtype in all_numeric_dtypes:
        if array.dtype == materialize[dtype]():
            return Array(drop_nulls[dtype](PrimitiveArray[dtype](data=array)))

    raise Error("drop_nulls: unsupported dtype " + String(array.dtype))

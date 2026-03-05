"""Boolean and bitwise kernels."""

from marrow.arrays import PrimitiveArray
from marrow.bitmap import Bitmap
from marrow.dtypes import bool_ as bool_dt


fn count_true(array: PrimitiveArray[bool_dt]) raises -> Int:
    """Count True values in a bit-packed boolean array.

    Note: Arrow booleans are bit-packed — each buffer byte holds 8 elements.
    `reduce_simd` cannot be used here because it iterates by element count
    and treats each byte as one element, which is semantically wrong for
    bit-packed data.

    Assumes array.offset == 0.

    Args:
        array: A bit-packed boolean array.

    Returns:
        Number of True (and non-null) elements.
    """
    var n = len(array)
    var data_bm = Bitmap(array.buffer, array.offset, n)
    if array.nulls > 0:
        var combined = data_bm & array.bitmap.value()
        return combined.count_set_bits()
    return data_bm.count_set_bits()

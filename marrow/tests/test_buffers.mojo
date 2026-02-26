from testing import assert_equal, assert_true, assert_false, TestSuite
from marrow.test_fixtures.arrays import assert_bitmap_set

from marrow.buffers import *


def is_aligned[
    T: AnyType
](ptr: UnsafePointer[T, MutAnyOrigin], alignment: Int) -> Bool:
    return (Int(ptr) % alignment) == 0


def test_buffer_init():
    var b = Buffer.alloc(10)
    assert_equal(b.size, 64)
    assert_true(is_aligned(b.ptr, 64))


def test_bitmap_alloc_sizes():
    # 10 bits → ceildiv(10,8)=2 bytes → aligned to 64
    var b1 = Bitmap.alloc(10)
    assert_equal(b1.size(), 64)

    # 64*8+1 bits → ceildiv(513,8)=65 bytes → aligned to 128
    var b2 = Bitmap.alloc(64 * 8 + 1)
    assert_equal(b2.size(), 128)


def test_buffer_grow():
    var b = Buffer.alloc(10)
    b.unsafe_set(0, 111)
    assert_equal(b.size, 64)
    b.resize(20)
    assert_equal(b.size, 64)
    assert_equal(b.unsafe_get(0), 111)
    b.resize(80)
    assert_equal(b.size, 128)
    assert_equal(b.unsafe_get(0), 111)


def test_buffer_set_get():
    var buf = Buffer.alloc(10)
    assert_equal(buf.size, 64)

    buf.unsafe_set(0, 42)
    buf.unsafe_set(1, 43)
    buf.unsafe_set(2, 44)
    assert_equal(buf.unsafe_get(0), 42)
    assert_equal(buf.unsafe_get(1), 43)
    assert_equal(buf.unsafe_get(2), 44)

    assert_equal(buf.size, 64)
    assert_equal(
        buf.length[DType.uint16](), 32
    )  # 64 bytes / 2 bytes per element
    # reinterpreting the underlying bits as uint16
    assert_equal(buf.unsafe_get[DType.uint16](0), 42 + (43 << 8))
    assert_equal(buf.unsafe_get[DType.uint16](1), 44)


def test_buffer_swap():
    var one = Buffer.alloc(10)
    one.unsafe_set(0, 111)
    var two = Buffer.alloc(10)
    two.unsafe_set(0, 222)

    swap(one, two)

    assert_equal(one.unsafe_get(0), 222)
    assert_equal(two.unsafe_get(0), 111)


def test_bitmap():
    var b = Bitmap.alloc(10)
    assert_equal(b.size(), 64)
    assert_equal(b.length(), 64 * 8)
    assert_equal(b.bit_count(), 0)

    assert_false(b.unsafe_get(0))
    b.unsafe_set(0, True)
    assert_true(b.unsafe_get(0))
    assert_equal(b.bit_count(), 1)
    assert_false(b.unsafe_get(1))
    b.unsafe_set(1, True)
    assert_true(b.unsafe_get(1))
    assert_equal(b.bit_count(), 2)


def test_count_leading_zeros():
    var b = Bitmap.alloc(10)
    var expected_bits = b.length()
    assert_equal(b.count_leading_zeros(), expected_bits)
    assert_equal(b.count_leading_zeros(10), expected_bits - 10)

    b.unsafe_set(0, True)
    assert_equal(b.count_leading_zeros(), 0)
    assert_equal(b.count_leading_zeros(1), expected_bits - 1)
    b.unsafe_set(0, False)

    var to_test = [0, 1, 7, 8, 10, 16, 31]
    for i in range(len(to_test)):
        bit_position = to_test[i]
        b.unsafe_set(bit_position, True)
        assert_equal(b.count_leading_zeros(), bit_position)
        if bit_position > 4:
            # Count with start position.
            assert_equal(b.count_leading_bits(4), bit_position - 4)
        b.unsafe_set(bit_position, False)


def test_count_leading_ones():
    var b = Bitmap.alloc(10)
    assert_equal(b.count_leading_ones(), 0)
    b.unsafe_set(0, True)
    assert_equal(b.count_leading_ones(), 1)
    assert_equal(b.count_leading_ones(1), 0)

    b.unsafe_set(1, True)
    assert_equal(b.count_leading_ones(), 2)
    assert_equal(b.count_leading_ones(1), 1)


def _reset(mut bitmap: Bitmap[mut=True]):
    bitmap.unsafe_range_set(0, bitmap.length(), False)
    assert_bitmap_set(bitmap, [], "after _reset")


def test_unsafe_range_set():
    var bitmap = Bitmap.alloc(16)

    bitmap.unsafe_range_set(0, 10, True)
    assert_bitmap_set(bitmap, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "range 0-10")
    bitmap.unsafe_range_set(0, 10, False)
    assert_bitmap_set(bitmap, [], "reset")

    bitmap.unsafe_range_set(0, 0, True)
    assert_bitmap_set(bitmap, [], "range 0")

    var to_test = [0, 1, 7, 8, 15]
    for pos in range(len(to_test)):
        _reset(bitmap)
        var start_bit = to_test[pos]
        bitmap.unsafe_range_set(start_bit, 1, True)
        assert_bitmap_set(bitmap, [start_bit], "range  1")
        if to_test[pos] < bitmap.length() - 1:
            _reset(bitmap)
            bitmap.unsafe_range_set(start_bit, 2, True)
            assert_bitmap_set(bitmap, [start_bit, start_bit + 1], "range 2")


def test_partial_byte_set():
    var bitmap = Bitmap.alloc(16)

    bitmap.unsafe_range_set(0, 0, True)
    assert_bitmap_set(bitmap, [], "range 0")

    # Set one bit to True.
    bitmap.partial_byte_set(0, 0, 1, True)
    assert_bitmap_set(bitmap, [0], "set bit 0")

    # Set one bit to False.
    bitmap.partial_byte_set(0, 0, 1, False)
    assert_bitmap_set(bitmap, [], "reset bit 0")

    # Set multiple bits to True.
    bitmap.partial_byte_set(1, 2, 5, True)
    assert_bitmap_set(bitmap, [10, 11, 12], "set multiple bits")

    # Set multiple bits to False.
    bitmap.partial_byte_set(1, 3, 5, False)
    assert_bitmap_set(bitmap, [10], "reset multiple bits")


def test_expand_bitmap() -> None:
    var bitmap = Bitmap.alloc(6)
    bitmap.unsafe_set(0, True)
    bitmap.unsafe_set(5, True)
    assert_bitmap_set(bitmap, [0, 5], "initial setup")

    # Create a new bitmap with 2 bits
    var new_bitmap = Bitmap.alloc(2)
    new_bitmap.unsafe_set(0, True)

    # Expand the bitmap
    bitmap.extend(new_bitmap, 6, 2)
    assert_bitmap_set(bitmap, [0, 5, 6], "after expand")


def test_buffer_with_offset():
    # Test Buffer with offset functionality
    var buf = Buffer.alloc(10)
    assert_equal(buf.offset, 0)  # Default offset should be 0

    # Set values in buffer without offset
    buf.unsafe_set(0, 42)
    buf.unsafe_set(1, 43)
    buf.unsafe_set(2, 44)
    buf.unsafe_set(3, 55)

    # Shift the offset and test that get/set are adjusted
    buf.offset = 2
    assert_equal(buf.offset, 2)
    assert_equal(buf.unsafe_get(0), 44)  # reads buf[2]
    buf.unsafe_set(1, 99)  # writes buf[3]
    buf.offset = 0
    assert_equal(buf.unsafe_get(3), 99)


def test_buffer_moveinit_with_offset():
    # Test __moveinit__ preserves offset
    var buf = Buffer.alloc(5)
    buf.offset = 3
    buf.unsafe_set(0, 123)

    var moved_buf = buf^
    assert_equal(moved_buf.offset, 3)
    assert_equal(moved_buf.unsafe_get(0), 123)


def test_buffer_swap_with_offset():
    # Test swap preserves offsets correctly
    var buf1 = Buffer.alloc(5)
    buf1.offset = 2
    buf1.unsafe_set(0, 111)

    var buf2 = Buffer.alloc(5)
    buf2.offset = 4
    buf2.unsafe_set(0, 222)

    swap(buf1, buf2)

    # After swap, buf1 should have buf2's original offset and data
    assert_equal(buf1.offset, 4)
    assert_equal(buf1.unsafe_get(0), 222)

    # And buf2 should have buf1's original offset and data
    assert_equal(buf2.offset, 2)
    assert_equal(buf2.unsafe_get(0), 111)


def test_bitmap_with_offset():
    # Populate a Bitmap with known bits then test offset arithmetic in place.
    var bm = Bitmap.alloc(16)
    bm.unsafe_set(3, True)
    bm.unsafe_set(5, True)
    bm.unsafe_set(6, True)

    # Apply an offset directly and verify reads are shifted.
    bm.offset = 3
    assert_equal(bm.offset, 3)
    assert_true(bm.unsafe_get(0))  # bit 3
    assert_false(bm.unsafe_get(1))  # bit 4
    assert_true(bm.unsafe_get(2))  # bit 5
    assert_true(bm.unsafe_get(3))  # bit 6

    # Writes are also shifted by the offset.
    bm.unsafe_set(4, True)  # should set bit 7
    bm.offset = 0
    assert_true(bm.unsafe_get(7))


def test_bitmap_moveinit_with_offset():
    # Test __moveinit__ preserves offset
    var bitmap = Bitmap(Buffer.alloc(1), offset=2)
    bitmap.unsafe_set(0, True)

    var moved_bitmap = bitmap^
    assert_equal(moved_bitmap.offset, 2)
    assert_true(moved_bitmap.unsafe_get(0))


def test_buffer_freeze():
    var buf = Buffer.alloc(10)
    buf.unsafe_set(0, 42)
    buf.unsafe_set(1, 99)

    var frozen = buf^.freeze()
    # Reads still work on the frozen buffer.
    assert_equal(frozen.unsafe_get(0), 42)
    assert_equal(frozen.unsafe_get(1), 99)
    assert_equal(frozen.size, 64)
    assert_equal(frozen.length(), 64)


def test_buffer_freeze_preserves_offset():
    var buf = Buffer.alloc(10)
    buf.unsafe_set(2, 77)
    buf.offset = 2

    var frozen = buf^.freeze()
    assert_equal(frozen.offset, 2)
    assert_equal(frozen.unsafe_get(0), 77)


def test_bitmap_freeze():
    var bm = Bitmap.alloc(16)
    bm.unsafe_set(0, True)
    bm.unsafe_set(5, True)
    bm.unsafe_set(7, True)

    var frozen = bm^.freeze()
    # Reads still work on the frozen bitmap.
    assert_true(frozen.unsafe_get(0))
    assert_false(frozen.unsafe_get(1))
    assert_true(frozen.unsafe_get(5))
    assert_true(frozen.unsafe_get(7))
    assert_equal(frozen.bit_count(), 3)


def test_bitmap_freeze_preserves_offset():
    var bm = Bitmap.alloc(16)
    bm.unsafe_set(3, True)
    bm.offset = 3

    var frozen = bm^.freeze()
    assert_equal(frozen.offset, 3)
    assert_true(frozen.unsafe_get(0))  # bit 3


def test_bitmap_to_buffer_implicit():
    # Bitmap implicitly converts to Buffer when passed where a Buffer is expected.
    var bm = Bitmap.alloc(8)
    bm.unsafe_set(0, True)
    bm.unsafe_set(7, True)
    var expected_size = bm.size()

    # The implicit conversion consumes the bitmap and yields its underlying buffer.
    var buf: Buffer[mut=True] = bm^
    assert_equal(buf.size, expected_size)
    # Bit 0 set and bit 7 set → byte 0 should be 0b10000001 = 129
    assert_equal(buf.unsafe_get(0), 129)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()

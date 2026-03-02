import math
from testing import assert_equal, assert_true, assert_false, TestSuite
from reflection import call_location

from marrow.buffers import *


@always_inline
def assert_bitmap_set(
    ptr: UnsafePointer[UInt8, MutAnyOrigin],
    n_bits: Int,
    expected_true_pos: List[Int],
    message: StringLiteral,
) -> None:
    var list_pos = 0
    for i in range(n_bits):
        var expected_value = False
        if list_pos < len(expected_true_pos):
            if expected_true_pos[list_pos] == i:
                expected_value = True
                list_pos += 1
        var current_value = Bool((ptr[i // 8] >> UInt8(i % 8)) & 1)
        assert_equal(
            current_value,
            expected_value,
            String(
                "{}: Bitmap index {} is {}, expected {} as per list position {}"
            ).format(message, i, current_value, expected_value, list_pos),
            location=call_location(),
        )


def is_aligned[
    T: AnyType
](ptr: UnsafePointer[T, MutAnyOrigin], alignment: Int) -> Bool:
    return (Int(ptr) % alignment) == 0


def test_buffer_init():
    var b = BufferBuilder.alloc(10)
    assert_equal(b.size, 64)
    assert_true(is_aligned(b.ptr, 64))


def test_alloc_bits():
    # 10 bits → ceildiv(10,8)=2 bytes → aligned to 64
    var b1 = BufferBuilder.alloc_bits(10)
    assert_equal(b1.size, 64)

    # 64*8+1 bits → ceildiv(513,8)=65 bytes → aligned to 128
    var b2 = BufferBuilder.alloc_bits(64 * 8 + 1)
    assert_equal(b2.size, 128)


def test_buffer_grow():
    var b = BufferBuilder.alloc(10)
    b.unsafe_set(0, 111)
    assert_equal(b.size, 64)
    b.resize(20)
    assert_equal(b.size, 64)
    assert_equal(b.unsafe_get(0), 111)
    b.resize(80)
    assert_equal(b.size, 128)
    assert_equal(b.unsafe_get(0), 111)


def test_buffer_set_get():
    var buf = BufferBuilder.alloc(10)
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
    var one = BufferBuilder.alloc(10)
    one.unsafe_set(0, 111)
    var two = BufferBuilder.alloc(10)
    two.unsafe_set(0, 222)

    swap(one, two)

    assert_equal(one.unsafe_get(0), 222)
    assert_equal(two.unsafe_get(0), 111)


def test_bitmap_get_set():
    var b = BufferBuilder.alloc_bits(10)
    assert_equal(b.size, 64)

    assert_false(Bool((b.ptr[0] >> UInt8(0)) & 1))
    bitmap_set(b.ptr, 0, True)
    assert_true(Bool((b.ptr[0] >> UInt8(0)) & 1))
    bitmap_set(b.ptr, 1, True)
    assert_true(Bool((b.ptr[0] >> UInt8(1)) & 1))

    var frozen = b.finish()
    assert_true(bitmap_get(frozen.unsafe_ptr(), 0))
    assert_true(bitmap_get(frozen.unsafe_ptr(), 1))
    assert_false(bitmap_get(frozen.unsafe_ptr(), 2))
    # 10-bit bitmap → 2 bytes of actual data
    assert_equal(bitmap_count_ones(frozen.unsafe_ptr(), math.ceildiv(10, 8)), 2)


def _reset(mut bitmap: BufferBuilder, n_bits: Int):
    bitmap_range_set(bitmap.ptr, 0, n_bits, False)
    assert_bitmap_set(bitmap.ptr, n_bits, [], "after _reset")


def test_bitmap_range_set():
    var bitmap = BufferBuilder.alloc_bits(16)
    var n_bits = 16

    bitmap_range_set(bitmap.ptr, 0, 10, True)
    assert_bitmap_set(
        bitmap.ptr, n_bits, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "range 0-10"
    )
    bitmap_range_set(bitmap.ptr, 0, 10, False)
    assert_bitmap_set(bitmap.ptr, n_bits, [], "reset")

    bitmap_range_set(bitmap.ptr, 0, 0, True)
    assert_bitmap_set(bitmap.ptr, n_bits, [], "range 0")

    var to_test = [0, 1, 7, 8, 15]
    for pos in range(len(to_test)):
        _reset(bitmap, n_bits)
        var start_bit = to_test[pos]
        bitmap_range_set(bitmap.ptr, start_bit, 1, True)
        assert_bitmap_set(bitmap.ptr, n_bits, [start_bit], "range  1")
        if to_test[pos] < n_bits - 1:
            _reset(bitmap, n_bits)
            bitmap_range_set(bitmap.ptr, start_bit, 2, True)
            assert_bitmap_set(
                bitmap.ptr, n_bits, [start_bit, start_bit + 1], "range 2"
            )


def test_bitmap_extend():
    var src = BufferBuilder.alloc_bits(6)
    bitmap_set(src.ptr, 0, True)
    bitmap_set(src.ptr, 5, True)

    var dst = BufferBuilder.alloc_bits(8)
    bitmap_range_set(dst.ptr, 0, 8, False)
    bitmap_extend(dst.ptr, src.finish().unsafe_ptr(), 0, 6)
    assert_bitmap_set(dst.ptr, 8, [0, 5], "after extend")

    # extend into offset position
    var dst2 = BufferBuilder.alloc_bits(8)
    bitmap_range_set(dst2.ptr, 0, 8, False)
    var src2 = BufferBuilder.alloc_bits(2)
    bitmap_set(src2.ptr, 0, True)
    bitmap_extend(dst2.ptr, src2.finish().unsafe_ptr(), 6, 2)
    assert_bitmap_set(dst2.ptr, 8, [6], "extend at offset 6")


def test_buffer_finish():
    var buf = BufferBuilder.alloc(10)
    buf.unsafe_set(0, 42)
    buf.unsafe_set(1, 99)

    var frozen = buf.finish()
    # Reads still work on the frozen buffer.
    assert_equal(frozen.unsafe_get(0), 42)
    assert_equal(frozen.unsafe_get(1), 99)
    assert_equal(frozen.size, 64)
    assert_equal(frozen.length(), 64)


def test_buffer_no_device():
    # CPU-allocated buffers have no device buffer
    var buf = BufferBuilder.alloc(10)
    var frozen = buf.finish()
    assert_true(not frozen.has_device())


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()

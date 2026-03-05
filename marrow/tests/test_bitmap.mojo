"""Tests for Bitmap and BitmapBuilder."""

from std.testing import assert_equal, assert_true, assert_false, TestSuite

from marrow.bitmap import Bitmap, BitmapBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


fn _make(n_bits: Int, set_bits: List[Int]) -> Bitmap:
    """Build a Bitmap with exactly the specified bits set."""
    var b = BitmapBuilder.alloc(n_bits)
    for i in range(len(set_bits)):
        b.set_bit(set_bits[i], True)
    return b.finish(n_bits)


# ---------------------------------------------------------------------------
# BitmapBuilder
# ---------------------------------------------------------------------------


def test_builder_alloc_zero_fills():
    """Freshly-allocated builder has all bits cleared."""
    var b = BitmapBuilder.alloc(10)
    var bm = b.finish(10)
    for i in range(10):
        assert_false(bm.is_valid(i))


def test_builder_set_bit_true():
    var b = BitmapBuilder.alloc(8)
    b.set_bit(0, True)
    b.set_bit(3, True)
    b.set_bit(7, True)
    var bm = b.finish(8)
    assert_true(bm.is_valid(0))
    assert_false(bm.is_valid(1))
    assert_false(bm.is_valid(2))
    assert_true(bm.is_valid(3))
    assert_false(bm.is_valid(4))
    assert_false(bm.is_valid(5))
    assert_false(bm.is_valid(6))
    assert_true(bm.is_valid(7))


def test_builder_set_bit_false_clears():
    var b = BitmapBuilder.alloc(8)
    b.set_bit(0, True)
    b.set_bit(1, True)
    b.set_bit(1, False)
    var bm = b.finish(8)
    assert_true(bm.is_valid(0))
    assert_false(bm.is_valid(1))


def test_builder_set_range_all_true():
    var b = BitmapBuilder.alloc(16)
    b.set_range(0, 16, True)
    var bm = b.finish(16)
    for i in range(16):
        assert_true(bm.is_valid(i))


def test_builder_set_range_partial():
    # set_range sets exactly the requested range, leaving the rest unchanged
    var b = BitmapBuilder.alloc(16)
    b.set_range(4, 8, True)  # bits 4-11 set
    var bm = b.finish(16)
    for i in range(4):
        assert_false(bm.is_valid(i))
    for i in range(4, 12):
        assert_true(bm.is_valid(i))
    for i in range(12, 16):
        assert_false(bm.is_valid(i))


def test_builder_set_range_clear():
    var b = BitmapBuilder.alloc(16)
    b.set_range(0, 16, True)
    b.set_range(3, 5, False)  # clear bits 3-7
    var bm = b.finish(16)
    for i in range(3):
        assert_true(bm.is_valid(i))
    for i in range(3, 8):
        assert_false(bm.is_valid(i))
    for i in range(8, 16):
        assert_true(bm.is_valid(i))


def test_builder_set_range_zero_length():
    var b = BitmapBuilder.alloc(8)
    b.set_range(0, 0, True)
    var bm = b.finish(8)
    for i in range(8):
        assert_false(bm.is_valid(i))


def test_builder_extend():
    # extend copies bits from a Bitmap into the builder at dst_start
    var src_b = BitmapBuilder.alloc(6)
    src_b.set_bit(0, True)
    src_b.set_bit(5, True)
    var src = src_b.finish(6)

    var dst = BitmapBuilder.alloc(8)
    dst.extend(src, 0, 6)
    var bm = dst.finish(8)
    assert_true(bm.is_valid(0))
    assert_false(bm.is_valid(1))
    assert_false(bm.is_valid(4))
    assert_true(bm.is_valid(5))
    assert_false(bm.is_valid(6))
    assert_false(bm.is_valid(7))


def test_builder_extend_with_offset():
    # extend into a non-zero dst_start position
    var src_b = BitmapBuilder.alloc(2)
    src_b.set_bit(0, True)
    var src = src_b.finish(2)

    var dst = BitmapBuilder.alloc(8)
    dst.extend(src, 6, 2)  # copy 2 bits starting at dst bit 6
    var bm = dst.finish(8)
    for i in range(6):
        assert_false(bm.is_valid(i))
    assert_true(bm.is_valid(6))
    assert_false(bm.is_valid(7))


def test_builder_finish_length():
    var b = BitmapBuilder.alloc(20)
    b.set_range(0, 20, True)
    var bm = b.finish(15)  # finish with fewer bits than allocated
    assert_equal(len(bm), 15)


# ---------------------------------------------------------------------------
# Bitmap — basic queries
# ---------------------------------------------------------------------------


def test_len():
    var bm = _make(13, [])
    assert_equal(len(bm), 13)


def test_is_valid_and_is_null():
    var bm = _make(8, [1, 4, 6])
    assert_false(bm.is_valid(0))
    assert_true(bm.is_valid(1))
    assert_false(bm.is_valid(2))
    assert_false(bm.is_valid(3))
    assert_true(bm.is_valid(4))
    assert_false(bm.is_valid(5))
    assert_true(bm.is_valid(6))
    assert_false(bm.is_valid(7))
    # is_null is the inverse
    assert_true(bm.is_null(0))
    assert_false(bm.is_null(1))


# ---------------------------------------------------------------------------
# count_set_bits
# ---------------------------------------------------------------------------


def test_count_set_bits_empty():
    var b = BitmapBuilder.alloc(0)
    var bm = b.finish(0)
    assert_equal(bm.count_set_bits(), 0)


def test_count_set_bits_none_set():
    var bm = _make(16, [])
    assert_equal(bm.count_set_bits(), 0)


def test_count_set_bits_all_set():
    var b = BitmapBuilder.alloc(16)
    b.set_range(0, 16, True)
    var bm = b.finish(16)
    assert_equal(bm.count_set_bits(), 16)


def test_count_set_bits_known_pattern():
    # bits 0, 3, 7, 8, 15 → 5 set bits
    var bm = _make(16, [0, 3, 7, 8, 15])
    assert_equal(bm.count_set_bits(), 5)


def test_count_set_bits_partial_last_byte():
    # 10-bit bitmap: bits 0-7 all set, bits 8-9 both set
    var b = BitmapBuilder.alloc(10)
    b.set_range(0, 10, True)
    var bm = b.finish(10)
    assert_equal(bm.count_set_bits(), 10)


def test_count_set_bits_with_offset():
    # count_set_bits on a sliced bitmap respects _offset
    var b = BitmapBuilder.alloc(16)
    b.set_range(0, 16, True)
    var full = b.finish(16)
    # slice starting at bit 4, length 8 → all 8 bits set
    var sliced = full.slice(4, 8)
    assert_equal(sliced.count_set_bits(), 8)


def test_count_set_bits_large():
    # exercises the SIMD loop for multi-SIMD-width bitmaps
    var b = BitmapBuilder.alloc(1024)
    b.set_range(0, 512, True)  # first half set
    var bm = b.finish(1024)
    assert_equal(bm.count_set_bits(), 512)


# ---------------------------------------------------------------------------
# slice
# ---------------------------------------------------------------------------


def test_slice_shares_buffer():
    # slice returns a zero-copy view with correct offset and length
    var bm = _make(16, [0, 5, 10, 15])
    var s = bm.slice(4, 8)  # bits 4-11 of original
    assert_equal(len(s), 8)
    # bit 5 of original = index 1 in sliced view
    assert_true(s.is_valid(1))
    # bit 10 of original = index 6 in sliced view
    assert_true(s.is_valid(6))
    # bit 4 of original = index 0, not set
    assert_false(s.is_valid(0))


def test_slice_single_bit():
    var bm = _make(8, [3])
    var s = bm.slice(3, 1)
    assert_equal(len(s), 1)
    assert_true(s.is_valid(0))


def test_slice_count_set_bits():
    var bm = _make(16, [2, 3, 4, 5, 6])
    var s = bm.slice(2, 5)  # bits 2-6 → all 5 set
    assert_equal(s.count_set_bits(), 5)


# ---------------------------------------------------------------------------
# __invert__
# ---------------------------------------------------------------------------


def test_invert_all_zeros():
    var bm = _make(8, [])
    var inv = ~bm
    assert_equal(len(inv), 8)
    for i in range(8):
        assert_true(inv.is_valid(i))


def test_invert_all_ones():
    var b = BitmapBuilder.alloc(8)
    b.set_range(0, 8, True)
    var bm = b.finish(8)
    var inv = ~bm
    for i in range(8):
        assert_false(inv.is_valid(i))


def test_invert_pattern():
    # bits 1, 3, 5 set → inverted: 0, 2, 4, 6, 7 set
    var bm = _make(8, [1, 3, 5])
    var inv = ~bm
    assert_true(inv.is_valid(0))
    assert_false(inv.is_valid(1))
    assert_true(inv.is_valid(2))
    assert_false(inv.is_valid(3))
    assert_true(inv.is_valid(4))
    assert_false(inv.is_valid(5))
    assert_true(inv.is_valid(6))
    assert_true(inv.is_valid(7))


def test_invert_does_not_bleed_past_length():
    """Bits beyond _length must be 0 in the result (no spurious set bits)."""
    var bm = _make(10, [])  # 10 bits, all clear
    var inv = ~bm
    # only bits 0-9 are inverted; bits 10-15 of last byte must stay 0
    assert_equal(inv.count_set_bits(), 10)


# ---------------------------------------------------------------------------
# __and__
# ---------------------------------------------------------------------------


def test_and_basic():
    # [1,0,1,0,1,0,1,0] & [1,1,0,0,1,1,0,0] = [1,0,0,0,1,0,0,0]
    var a = _make(8, [0, 2, 4, 6])
    var b = _make(8, [0, 1, 4, 5])
    var r = a & b
    assert_equal(len(r), 8)
    assert_true(r.is_valid(0))
    assert_false(r.is_valid(1))
    assert_false(r.is_valid(2))
    assert_false(r.is_valid(3))
    assert_true(r.is_valid(4))
    assert_false(r.is_valid(5))
    assert_false(r.is_valid(6))
    assert_false(r.is_valid(7))


def test_and_identity():
    # a & all-ones == a
    var a = _make(16, [1, 5, 9, 13])
    var ones_b = BitmapBuilder.alloc(16)
    ones_b.set_range(0, 16, True)
    var ones = ones_b.finish(16)
    var r = a & ones
    for i in range(16):
        assert_equal(r.is_valid(i), a.is_valid(i))


def test_and_annihilator():
    # a & all-zeros == all-zeros
    var a = _make(16, [1, 5, 9, 13])
    var zeros = _make(16, [])
    var r = a & zeros
    for i in range(16):
        assert_false(r.is_valid(i))


def test_and_large():
    # exercises the SIMD loop
    var b1 = BitmapBuilder.alloc(1024)
    b1.set_range(0, 512, True)
    var a = b1.finish(1024)

    var b2 = BitmapBuilder.alloc(1024)
    b2.set_range(256, 512, True)
    var b = b2.finish(1024)

    var r = a & b
    assert_equal(r.count_set_bits(), 256)  # overlap in bits 256-511


# ---------------------------------------------------------------------------
# __or__
# ---------------------------------------------------------------------------


def test_or_basic():
    var a = _make(8, [0, 2])
    var b = _make(8, [1, 2])
    var r = a | b
    assert_true(r.is_valid(0))
    assert_true(r.is_valid(1))
    assert_true(r.is_valid(2))
    assert_false(r.is_valid(3))


def test_or_idempotent():
    # a | a == a
    var a = _make(16, [0, 3, 7, 10])
    var r = a | a
    for i in range(16):
        assert_equal(r.is_valid(i), a.is_valid(i))


# ---------------------------------------------------------------------------
# __xor__
# ---------------------------------------------------------------------------


def test_xor_basic():
    # [1,0,1,0] ^ [1,1,0,0] = [0,1,1,0]
    var a = _make(4, [0, 2])
    var b = _make(4, [0, 1])
    var r = a ^ b
    assert_false(r.is_valid(0))
    assert_true(r.is_valid(1))
    assert_true(r.is_valid(2))
    assert_false(r.is_valid(3))


def test_xor_self_is_zero():
    # a ^ a == all-zeros
    var a = _make(16, [1, 3, 5, 7])
    var r = a ^ a
    for i in range(16):
        assert_false(r.is_valid(i))


# ---------------------------------------------------------------------------
# and_not
# ---------------------------------------------------------------------------


def test_and_not_basic():
    # [1,0,1,0] & ~[1,1,0,0] = [1,0,1,0] & [0,0,1,1] = [0,0,1,0]
    var a = _make(4, [0, 2])
    var b = _make(4, [0, 1])
    var r = a.and_not(b)
    assert_false(r.is_valid(0))
    assert_false(r.is_valid(1))
    assert_true(r.is_valid(2))
    assert_false(r.is_valid(3))


def test_and_not_with_none_mask():
    # a.and_not(all-zeros) == a
    var a = _make(8, [0, 3, 7])
    var zeros = _make(8, [])
    var r = a.and_not(zeros)
    for i in range(8):
        assert_equal(r.is_valid(i), a.is_valid(i))


# ---------------------------------------------------------------------------
# Fallback path (non-zero offset)
# ---------------------------------------------------------------------------


def test_and_with_same_nonzero_offset():
    """Binary ops on sliced bitmaps sharing the same non-zero sub-byte offset."""
    var full = _make(16, [2, 3, 4, 6, 10, 11, 12, 14])
    var a = full.slice(2, 8)  # bits 2-9 of original: [1,1,1,0,1,0,0,0]
    var b = full.slice(2, 8)  # same slice
    var r = a & b
    # a & a == a
    for i in range(8):
        assert_equal(r.is_valid(i), a.is_valid(i))


def test_and_same_shift_fast_path():
    """Bitmaps with identical non-zero sub-byte offsets use the same-shift SIMD path."""
    # Build two 12-bit bitmaps with known patterns; slice both at offset=3
    # so both have sub-byte shift = 3 (same shift, non-zero).
    var fa = _make(16, [3, 5, 7, 9, 11])  # bits 3,5,7,9,11 set
    var fb = _make(16, [3, 4, 7, 8, 11])  # bits 3,4,7,8,11 set
    # slice(3, 9): 9 bits starting at offset 3 → indices 0-8 in sliced view
    # a slice indices: orig bits 3,5,7,9,11 → slice indices 0,2,4,6,8
    # b slice indices: orig bits 3,4,7,8,11 → slice indices 0,1,4,5,8
    var a = fa.slice(3, 9)
    var b = fb.slice(3, 9)
    var r = a & b
    assert_equal(len(r), 9)
    # AND: intersection at slice indices 0,4,8
    assert_true(r.is_valid(0))
    assert_false(r.is_valid(1))
    assert_false(r.is_valid(2))
    assert_false(r.is_valid(3))
    assert_true(r.is_valid(4))
    assert_false(r.is_valid(5))
    assert_false(r.is_valid(6))
    assert_false(r.is_valid(7))
    assert_true(r.is_valid(8))


def test_or_same_shift_fast_path():
    """OR of two sliced bitmaps with same non-zero sub-byte offset."""
    var fa = _make(16, [3, 5])
    var fb = _make(16, [3, 4])
    var a = fa.slice(3, 5)   # slice indices 0,2 set
    var b = fb.slice(3, 5)   # slice indices 0,1 set
    var r = a | b
    assert_equal(len(r), 5)
    assert_true(r.is_valid(0))
    assert_true(r.is_valid(1))
    assert_true(r.is_valid(2))
    assert_false(r.is_valid(3))
    assert_false(r.is_valid(4))


def test_and_different_offsets():
    """AND of bitmaps with different sub-byte offsets (shift-one path)."""
    # fa bits 3,5,7,9,11 set; sliced at offset 3 → shift_a=3, indices 0,2,4,6,8
    # fb bits 5,7,9,11,13 set; sliced at offset 5 → shift_b=5, indices 0,2,4,6,8
    var fa = _make(16, [3, 5, 7, 9, 11])
    var fb = _make(16, [5, 7, 9, 11, 13])
    var a = fa.slice(3, 9)   # shift_a = 3
    var b = fb.slice(5, 9)   # shift_b = 5
    var r = a & b
    assert_equal(len(r), 9)
    # AND: both have indices 0,2,4,6,8 set → intersection is 0,2,4,6,8
    for i in range(9):
        assert_equal(r.is_valid(i), i % 2 == 0)


def test_and_different_offsets_large_byte_delta():
    """AND where byte-level offsets differ by more than 8 bytes."""
    # Build a large bitmap so we can slice at widely separated positions.
    # Set every even bit in the range we care about.
    var full = _make(600, [
        100, 102, 104, 106, 108, 110, 112, 114,
        500, 502, 504, 506, 508, 510, 512, 514,
    ])
    # a: slice at bit 100 (byte 12, shift 4), 16 bits → indices 0,2,4,6,8,10,12,14 set
    var a = full.slice(100, 16)
    # b: slice at bit 500 (byte 62, shift 4), 16 bits → indices 0,2,4,6,8,10,12,14 set
    var b = full.slice(500, 16)
    # Same sub-byte shift (4), but byte_delta = 50 — well beyond a single SIMD width.
    var r = a & b
    assert_equal(len(r), 16)
    for i in range(16):
        assert_equal(r.is_valid(i), i % 2 == 0)


def test_and_different_offsets_large_byte_delta_different_shift():
    """AND where byte offsets AND sub-byte shifts both differ."""
    # a at bit 100 (byte 12, shift 4); b at bit 503 (byte 62, shift 7).
    # byte_delta = 50, bit shift delta = 3.
    var bits_a = List[Int]()
    var bits_b = List[Int]()
    for i in range(16):
        bits_a.append(100 + i)  # all 16 bits set starting at 100
        bits_b.append(503 + i)  # all 16 bits set starting at 503
    var full_a = _make(600, bits_a)
    var full_b = _make(600, bits_b)
    var a = full_a.slice(100, 16)
    var b = full_b.slice(503, 16)
    var r = a & b
    assert_equal(len(r), 16)
    # Both slices are all-ones, so AND should be all-ones.
    for i in range(16):
        assert_true(r.is_valid(i))


def test_or_different_offsets_large_byte_delta():
    """OR where byte-level offsets differ significantly."""
    var full_a = _make(600, [100, 104, 108])
    var full_b = _make(600, [500, 501, 502])
    var a = full_a.slice(100, 12)  # indices 0,4,8 set
    var b = full_b.slice(500, 12)  # indices 0,1,2 set
    var r = a | b
    assert_equal(len(r), 12)
    assert_true(r.is_valid(0))   # set in both
    assert_true(r.is_valid(1))   # set in b
    assert_true(r.is_valid(2))   # set in b
    assert_false(r.is_valid(3))
    assert_true(r.is_valid(4))   # set in a
    assert_false(r.is_valid(5))
    assert_false(r.is_valid(6))
    assert_false(r.is_valid(7))
    assert_true(r.is_valid(8))   # set in a
    assert_false(r.is_valid(9))
    assert_false(r.is_valid(10))
    assert_false(r.is_valid(11))


def test_xor_different_offsets_large_byte_delta():
    """XOR where byte-level offsets differ by > 64 bytes."""
    var bits_a = List[Int]()
    var bits_b = List[Int]()
    for i in range(16):
        bits_a.append(80 + i)   # all set
        bits_b.append(592 + i)  # all set
    var full_a = _make(700, bits_a)
    var full_b = _make(700, bits_b)
    var a = full_a.slice(80, 16)   # byte 10, shift 0
    var b = full_b.slice(592, 16)  # byte 74, shift 0
    var r = a ^ b
    assert_equal(len(r), 16)
    # Both all-ones → XOR should be all-zeros.
    for i in range(16):
        assert_false(r.is_valid(i))


def test_and_not_different_offsets_large_byte_delta():
    """AND-NOT where byte-level offsets differ significantly."""
    var bits_a = List[Int]()
    for i in range(16):
        bits_a.append(100 + i)  # all set
    var full_a = _make(600, bits_a)
    var full_b = _make(600, [500, 502, 504, 506])  # even indices set
    var a = full_a.slice(100, 16)  # all-ones
    var b = full_b.slice(500, 16)  # indices 0,2,4,6 set
    var r = a.and_not(b)
    assert_equal(len(r), 16)
    # a & ~b: clear bits where b is set → odd indices remain
    for i in range(16):
        if i < 8:
            assert_equal(r.is_valid(i), i % 2 != 0)
        else:
            assert_true(r.is_valid(i))


def test_invert_with_offset():
    var full = _make(16, [4, 5, 6, 7])
    var s = full.slice(4, 8)  # bits 4-11 → [1,1,1,1,0,0,0,0]
    var inv = ~s
    assert_false(inv.is_valid(0))
    assert_false(inv.is_valid(1))
    assert_false(inv.is_valid(2))
    assert_false(inv.is_valid(3))
    assert_true(inv.is_valid(4))
    assert_true(inv.is_valid(5))
    assert_true(inv.is_valid(6))
    assert_true(inv.is_valid(7))


def test_invert_large_byte_offset():
    # __invert__ with byte_offset > 63: exercises the lead_bytes > 0 code path
    # 600-bit source; slice at bit 576 → byte_offset=72, lead_bytes=8, shift=0.
    # Set bits 577 and 578 of the full bitmap (slice indices 1 and 2).
    var full = _make(600, [577, 578])
    var s = full.slice(576, 24)
    assert_false(s.is_valid(0))
    assert_true(s.is_valid(1))
    assert_true(s.is_valid(2))
    var inv = ~s
    assert_equal(len(inv), 24)
    assert_true(inv.is_valid(0))
    assert_false(inv.is_valid(1))
    assert_false(inv.is_valid(2))
    for i in range(3, 24):
        assert_true(inv.is_valid(i))
    assert_equal(inv.count_set_bits(), 22)


def test_invert_large_byte_offset_with_shift():
    # __invert__ with large byte_offset AND non-zero sub-byte shift
    # Slice at bit 577 → byte_offset=72, shift=1, lead_bytes=8.
    # full bits 577, 578, 580 set → slice indices 0, 1, 3 set.
    var full = _make(600, [577, 578, 580])
    var s = full.slice(577, 8)
    assert_true(s.is_valid(0))
    assert_true(s.is_valid(1))
    assert_false(s.is_valid(2))
    assert_true(s.is_valid(3))
    var inv = ~s
    assert_equal(len(inv), 8)
    assert_false(inv.is_valid(0))
    assert_false(inv.is_valid(1))
    assert_true(inv.is_valid(2))
    assert_false(inv.is_valid(3))
    assert_true(inv.is_valid(4))
    assert_true(inv.is_valid(5))
    assert_true(inv.is_valid(6))
    assert_true(inv.is_valid(7))
    assert_equal(inv.count_set_bits(), 5)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()

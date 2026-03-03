from testing import assert_equal, assert_true, assert_false, TestSuite
import marrow.dtypes as dt


def test_bool_type():
    assert_true(materialize[dt.bool_]() == materialize[dt.bool_]())
    assert_false(materialize[dt.bool_]() == materialize[dt.int64]())
    assert_true(materialize[dt.bool_]() is materialize[dt.bool_]())
    assert_false(materialize[dt.bool_]() is materialize[dt.int64]())


def test_list_type():
    assert_true(
        dt.list_(materialize[dt.int64]()) == dt.list_(materialize[dt.int64]())
    )
    assert_false(
        dt.list_(materialize[dt.int64]()) == dt.list_(materialize[dt.int32]())
    )


def test_field():
    var field = dt.Field("a", materialize[dt.int64](), False)
    var expected = "Field(name=a, dtype=int64, nullable=False)"
    assert_equal(String(field), expected)
    assert_equal(field.__repr__(), expected)


def test_struct_type():
    s1 = dt.struct_(
        dt.Field("a", materialize[dt.int64](), False),
        dt.Field("b", materialize[dt.int32](), False),
    )
    s2 = dt.struct_(
        dt.Field("a", materialize[dt.int64](), False),
        dt.Field("b", materialize[dt.int32](), False),
    )
    s3 = dt.struct_(
        dt.Field("a", materialize[dt.int64](), False),
        dt.Field("b", materialize[dt.int32](), False),
        dt.Field("c", materialize[dt.int8](), False),
    )
    assert_true(s1 == s2)
    assert_false(s1 == s3)


def test_is_integer():
    assert_true(materialize[dt.int8]().is_integer())
    assert_true(materialize[dt.int16]().is_integer())
    assert_true(materialize[dt.int32]().is_integer())
    assert_true(materialize[dt.int64]().is_integer())
    assert_true(materialize[dt.uint8]().is_integer())
    assert_true(materialize[dt.uint16]().is_integer())
    assert_true(materialize[dt.uint32]().is_integer())
    assert_true(materialize[dt.uint64]().is_integer())
    assert_false(materialize[dt.bool_]().is_integer())
    assert_false(materialize[dt.float32]().is_integer())
    assert_false(materialize[dt.float64]().is_integer())
    assert_false(dt.list_(materialize[dt.int64]()).is_integer())


def test_is_signed_integer():
    assert_true(materialize[dt.int8]().is_signed_integer())
    assert_true(materialize[dt.int16]().is_signed_integer())
    assert_true(materialize[dt.int32]().is_signed_integer())
    assert_true(materialize[dt.int64]().is_signed_integer())
    assert_false(materialize[dt.uint8]().is_signed_integer())
    assert_false(materialize[dt.uint16]().is_signed_integer())
    assert_false(materialize[dt.uint32]().is_signed_integer())
    assert_false(materialize[dt.uint64]().is_signed_integer())
    assert_false(materialize[dt.bool_]().is_signed_integer())
    assert_false(materialize[dt.float32]().is_signed_integer())
    assert_false(materialize[dt.float64]().is_signed_integer())


def test_is_unsigned_integer():
    assert_false(materialize[dt.int8]().is_unsigned_integer())
    assert_false(materialize[dt.int16]().is_unsigned_integer())
    assert_false(materialize[dt.int32]().is_unsigned_integer())
    assert_false(materialize[dt.int64]().is_unsigned_integer())
    assert_true(materialize[dt.uint8]().is_unsigned_integer())
    assert_true(materialize[dt.uint16]().is_unsigned_integer())
    assert_true(materialize[dt.uint32]().is_unsigned_integer())
    assert_true(materialize[dt.uint64]().is_unsigned_integer())
    assert_false(materialize[dt.bool_]().is_unsigned_integer())
    assert_false(materialize[dt.float32]().is_unsigned_integer())
    assert_false(materialize[dt.float64]().is_unsigned_integer())


def test_is_floating_point():
    assert_false(materialize[dt.int8]().is_floating_point())
    assert_false(materialize[dt.int16]().is_floating_point())
    assert_false(materialize[dt.int32]().is_floating_point())
    assert_false(materialize[dt.int64]().is_floating_point())
    assert_false(materialize[dt.uint8]().is_floating_point())
    assert_false(materialize[dt.uint16]().is_floating_point())
    assert_false(materialize[dt.uint32]().is_floating_point())
    assert_false(materialize[dt.uint64]().is_floating_point())
    assert_false(materialize[dt.bool_]().is_floating_point())
    assert_true(materialize[dt.float32]().is_floating_point())
    assert_true(materialize[dt.float64]().is_floating_point())


def test_bit_width():
    assert_equal(materialize[dt.int8]().bit_width(), 8)
    assert_equal(materialize[dt.int16]().bit_width(), 16)
    assert_equal(materialize[dt.int32]().bit_width(), 32)
    assert_equal(materialize[dt.int64]().bit_width(), 64)
    assert_equal(materialize[dt.uint8]().bit_width(), 8)
    assert_equal(materialize[dt.uint16]().bit_width(), 16)
    assert_equal(materialize[dt.uint32]().bit_width(), 32)
    assert_equal(materialize[dt.uint64]().bit_width(), 64)
    assert_equal(materialize[dt.bool_]().bit_width(), 1)
    assert_equal(materialize[dt.float32]().bit_width(), 32)
    assert_equal(materialize[dt.float64]().bit_width(), 64)
    assert_equal(dt.list_(materialize[dt.int64]()).bit_width(), 0)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()

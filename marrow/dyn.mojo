"""Dynamic dispatch prototype using function-pointer trampolines.

Demonstrates that typed array wrappers (PrimitiveArray, ListArray, StructArray)
can be held in a single type-erased container (AnyArray) that dispatches to the
correct implementation at runtime via function pointers — without Variant
or if/else chains on dtype codes.
"""

from std.memory import ArcPointer
from .dtypes import DataType, int8, int32, int64, float32, float64


# ---------------------------------------------------------------------------
# Trait — the interface every typed array must implement
# ---------------------------------------------------------------------------


trait Array(Movable, ImplicitlyDestructible):
    fn length(self) -> Int: ...
    fn null_count(self) -> Int: ...
    fn is_valid(self, index: Int) -> Bool: ...


# ---------------------------------------------------------------------------
# AnyArray — type-erased array with dynamic dispatch
# ---------------------------------------------------------------------------

# TODO: AnyType?
struct AnyArray(ImplicitlyCopyable, Movable):
    """Type-erased array container.

    Wraps any `Array`-conforming type and dispatches `length()`,
    `null_count()`, and `is_valid()` through function pointers.  The inner
    value lives on the heap behind an `ArcPointer`, so copies are O(1)
    ref-count bumps.
    """

    var _data: ArcPointer[NoneType]
    var _virt_length: fn (ArcPointer[NoneType]) -> Int
    var _virt_null_count: fn (ArcPointer[NoneType]) -> Int
    var _virt_is_valid: fn (ArcPointer[NoneType], Int) -> Bool
    var _virt_drop: fn (var ArcPointer[NoneType])

    # --- trampolines: one per vtable slot, plus cleanup ---

    @staticmethod
    fn _tramp_length[T: Array](ptr: ArcPointer[NoneType]) -> Int:
        return rebind[ArcPointer[T]](ptr)[].length()

    @staticmethod
    fn _tramp_null_count[T: Array](ptr: ArcPointer[NoneType]) -> Int:
        return rebind[ArcPointer[T]](ptr)[].null_count()

    @staticmethod
    fn _tramp_is_valid[T: Array](ptr: ArcPointer[NoneType], index: Int) -> Bool:
        return rebind[ArcPointer[T]](ptr)[].is_valid(index)

    @staticmethod
    fn _tramp_drop[T: Array](var ptr: ArcPointer[NoneType]):
        var typed = rebind[ArcPointer[T]](ptr^)
        _ = typed^

    # --- public API ---

    @implicit
    fn __init__[T: Array](out self, var value: T):
        var ptr = ArcPointer(value^)
        self._data = rebind[ArcPointer[NoneType]](ptr^)
        self._virt_length = Self._tramp_length[T]
        self._virt_null_count = Self._tramp_null_count[T]
        self._virt_is_valid = Self._tramp_is_valid[T]
        self._virt_drop = Self._tramp_drop[T]

    fn __copyinit__(out self, read copy: Self):
        self._data = copy._data.copy()
        self._virt_length = copy._virt_length
        self._virt_null_count = copy._virt_null_count
        self._virt_is_valid = copy._virt_is_valid
        self._virt_drop = copy._virt_drop

    fn length(self) -> Int:
        return self._virt_length(self._data)

    fn null_count(self) -> Int:
        return self._virt_null_count(self._data)

    fn is_valid(self, index: Int) -> Bool:
        return self._virt_is_valid(self._data, index)

    fn downcast[T: Array](self) -> ArcPointer[T]:
        return rebind[ArcPointer[T]](self._data.copy())

    fn __del__(deinit self):
        self._virt_drop(self._data^)


# ---------------------------------------------------------------------------
# PrimitiveArray — leaf array of fixed-size values
# ---------------------------------------------------------------------------


struct PrimitiveArray[T: DataType](Array):
    comptime dtype = Self.T
    comptime scalar = Scalar[Self.T.native]

    var _length: Int
    var _null_count: Int
    var values: List[Self.scalar]

    fn __init__(out self, var values: List[Self.scalar], null_count: Int = 0):
        self._length = len(values)
        self._null_count = null_count
        self.values = values^

    fn length(self) -> Int:
        return self._length

    fn null_count(self) -> Int:
        return self._null_count

    fn is_valid(self, index: Int) -> Bool:
        return True

    fn as_any(var self) -> AnyArray:
        return AnyArray(self^)


# ---------------------------------------------------------------------------
# ListArray — variable-length list array (offsets + type-erased child)
# ---------------------------------------------------------------------------


struct ListArray(Array):
    var _length: Int
    var _null_count: Int
    var offsets: List[Int]
    var child: AnyArray

    fn __init__(out self, var offsets: List[Int], var child: AnyArray):
        self._length = len(offsets) - 1
        self._null_count = 0
        self.offsets = offsets^
        self.child = child^

    fn length(self) -> Int:
        return self._length

    fn null_count(self) -> Int:
        return self._null_count

    fn is_valid(self, index: Int) -> Bool:
        return True

    fn as_any(var self) -> AnyArray:
        return AnyArray(self^)


# ---------------------------------------------------------------------------
# StructArray — struct array (multiple type-erased children)
# ---------------------------------------------------------------------------


struct StructArray(Array):
    var _length: Int
    var _null_count: Int
    var children: List[AnyArray]

    fn __init__(out self, length: Int, var children: List[AnyArray]):
        self._length = length
        self._null_count = 0
        self.children = children^

    fn length(self) -> Int:
        return self._length

    fn null_count(self) -> Int:
        return self._null_count

    fn is_valid(self, index: Int) -> Bool:
        return True

    fn as_any(var self) -> AnyArray:
        return AnyArray(self^)

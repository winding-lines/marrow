from std.python import PythonObject, ConvertibleToPython, ConvertibleFromPython, Python
from std.python.bindings import PythonModuleBuilder
from std.collections import OwnedKwargsDict
from std.python._cpython import CPython, PyObjectPtr, PyTypeObject, PyTypeObjectPtr
from std.memory import ArcPointer, alloc
from marrow.arrays import (
    Array,
    PrimitiveArray,
    BoolArray,
    Int8Array,
    Int16Array,
    Int32Array,
    Int64Array,
    UInt8Array,
    UInt16Array,
    UInt32Array,
    UInt64Array,
    Float32Array,
    Float64Array,
    StringArray,
    ListArray,
    FixedSizeListArray,
    StructArray,
    ChunkedArray,
)
from marrow.builders import AnyBuilder, ListBuilder, StructBuilder, make_builder
import marrow.dtypes as dt


# ---------------------------------------------------------------------------
# pymethod helpers — reduce boilerplate for Python bindings
# ---------------------------------------------------------------------------


fn pymethod[
    T: AnyType,
    R: ConvertibleToPython,
    //,
    method: fn (T) raises -> R,
]() -> fn (UnsafePointer[T, MutAnyOrigin]) raises -> PythonObject:
    """Wrap a zero-arg method returning ConvertibleToPython."""

    fn wrapper(ptr: UnsafePointer[T, MutAnyOrigin]) raises -> PythonObject:
        return method(ptr[]).to_python_object()

    return wrapper


fn pymethod[
    T: AnyType,
    A0: ConvertibleFromPython,
    R: ConvertibleToPython,
    //,
    method: fn (T, A0) raises -> R,
]() -> fn (UnsafePointer[T, MutAnyOrigin], PythonObject) raises -> PythonObject:
    """Wrap a single-arg method returning ConvertibleToPython."""

    fn wrapper(
        ptr: UnsafePointer[T, MutAnyOrigin], arg: PythonObject
    ) raises -> PythonObject:
        return method(ptr[], A0(py=arg)).to_python_object()

    return wrapper


fn pymethod[
    T: AnyType,
    A0: ConvertibleFromPython,
    A1: ConvertibleFromPython,
    R: ConvertibleToPython,
    //,
    method: fn (T, A0, A1) raises -> R,
]() -> fn (
    UnsafePointer[T, MutAnyOrigin], PythonObject, PythonObject
) raises -> PythonObject:
    """Wrap a two-arg method returning ConvertibleToPython."""

    fn wrapper(
        ptr: UnsafePointer[T, MutAnyOrigin],
        arg0: PythonObject,
        arg1: PythonObject,
    ) raises -> PythonObject:
        return method(ptr[], A0(py=arg0), A1(py=arg1)).to_python_object()

    return wrapper


# ---------------------------------------------------------------------------
# Custom wrappers for methods that need special return-type handling
# ---------------------------------------------------------------------------


fn _str_getitem(
    ptr: UnsafePointer[StringArray, MutAnyOrigin],
    index: PythonObject,
) raises -> PythonObject:
    return PythonObject(String(ptr[].__getitem__(Int(py=index))))


# ---------------------------------------------------------------------------
# PyInferrer — mirrors PyArrow's PyInferrer (inference.cc)
# ---------------------------------------------------------------------------


struct PyInferrer(Copyable, Movable):
    """Infers the Arrow DataType of a Python sequence.

    Mirrors PyArrow's PyInferrer (inference.cc): counts occurrences by Python
    type in a single pass, recursing into list/dict elements via _visit_list and
    _visit_dict, then resolves to a DataType without re-iterating the sequence.
    """

    var none_count: Int
    var bool_count: Int
    var int_count: Int
    var float_count: Int
    var unicode_count: Int
    var unicode_bytes: Int
    var bytes_count: Int
    var list_count: Int
    var struct_count: Int

    # Child inferrers — List[PyInferrer] works because PyInferrer declares Copyable
    var _list_child: List[PyInferrer]      # 0 or 1 elements
    var _field_order: List[String]
    var _field_children: List[PyInferrer]  # parallel to _field_order

    # Cached CPython type pointers (obtained once at init)
    var _none_ptr: PyObjectPtr
    var _unicode_type: PyTypeObjectPtr  # PyUnicode_Type via lib.get_symbol
    var _bytes_type: PyTypeObjectPtr    # PyBytes_Type via lib.get_symbol
    var _list_type: PyTypeObjectPtr     # PyList_Type via lib.get_symbol
    var _tuple_type: PyTypeObjectPtr    # PyTuple_Type via lib.get_symbol
    var _dict_type: PyTypeObjectPtr     # cpython.PyDict_Type()

    fn __init__(out self) raises:
        self.none_count = 0
        self.bool_count = 0
        self.int_count = 0
        self.float_count = 0
        self.unicode_count = 0
        self.unicode_bytes = 0
        self.bytes_count = 0
        self.list_count = 0
        self.struct_count = 0
        self._list_child = []
        self._field_order = []
        self._field_children = []
        ref cpy = Python().cpython()
        self._none_ptr = cpy.Py_None()
        self._unicode_type = cpy.lib.get_symbol[PyTypeObject]("PyUnicode_Type")
        self._bytes_type = cpy.lib.get_symbol[PyTypeObject]("PyBytes_Type")
        self._list_type = cpy.lib.get_symbol[PyTypeObject]("PyList_Type")
        self._tuple_type = cpy.lib.get_symbol[PyTypeObject]("PyTuple_Type")
        self._dict_type = cpy.PyDict_Type()

    fn visit(mut self, ptr: PyObjectPtr) raises:
        """Count one element's Python type, following PyArrow's Visit() order."""
        ref cpy = Python().cpython()
        if cpy.Py_Is(ptr, self._none_ptr):
            self.none_count += 1
        elif cpy.PyBool_Check(ptr) != 0:  # exact bool check before PyLong_Check
            self.bool_count += 1
        elif cpy.PyFloat_Check(ptr) != 0:  # float before int
            self.float_count += 1
        elif cpy.PyLong_Check(ptr) != 0:
            self.int_count += 1
        elif cpy.PyObject_TypeCheck(ptr, self._unicode_type) != 0:
            self.unicode_count += 1
            self.unicode_bytes += len(cpy.PyUnicode_AsUTF8AndSize(ptr))
        elif cpy.PyObject_TypeCheck(ptr, self._bytes_type) != 0:
            self.bytes_count += 1
        elif cpy.PyObject_TypeCheck(ptr, self._dict_type) != 0:
            self.struct_count += 1
            self._visit_dict(ptr)
        elif cpy.PyObject_TypeCheck(ptr, self._list_type) != 0:
            self.list_count += 1
            self._visit_list(ptr)
        elif cpy.PyObject_TypeCheck(ptr, self._tuple_type) != 0:
            self.list_count += 1
            self._visit_tuple(ptr)
        else:
            raise Error(
                "cannot include value of type: "
                + String(PythonObject(from_borrowed=ptr).__class__.__name__)
            )

    fn _visit_list(mut self, ptr: PyObjectPtr) raises:
        """Mirrors PyArrow's VisitSequence: recurse into list children."""
        if len(self._list_child) == 0:
            self._list_child.append(PyInferrer())
        ref cpy = Python().cpython()
        var n = Int(cpy.PyObject_Length(ptr))
        for i in range(n):
            self._list_child[0].visit(cpy.PyList_GetItem(ptr, i))

    fn _visit_tuple(mut self, ptr: PyObjectPtr) raises:
        """Mirrors PyArrow's VisitSequence for tuples."""
        if len(self._list_child) == 0:
            self._list_child.append(PyInferrer())
        ref cpy = Python().cpython()
        var n = Int(cpy.PyObject_Length(ptr))
        for i in range(n):
            self._list_child[0].visit(cpy.PyTuple_GetItem(ptr, i))

    fn _visit_dict(mut self, dict_ptr: PyObjectPtr) raises:
        """Mirrors PyArrow's VisitDict: route each value to its field's child inferrer."""
        ref cpy = Python().cpython()
        var n = Int(cpy.PyObject_Length(dict_ptr))
        var pos = Int(0)
        var key_raw = alloc[PyObjectPtr](1)
        var val_raw = alloc[PyObjectPtr](1)
        var pos_ptr = UnsafePointer(to=pos)
        for _ in range(n):
            _ = cpy.PyDict_Next(dict_ptr, pos_ptr, key_raw, val_raw)
            var name = String(cpy.PyUnicode_AsUTF8AndSize(key_raw[]))
            var idx = -1
            for i in range(len(self._field_order)):
                if self._field_order[i] == name:
                    idx = i
                    break
            if idx == -1:
                idx = len(self._field_order)
                self._field_order.append(name)
                self._field_children.append(PyInferrer())
            self._field_children[idx].visit(val_raw[])
        key_raw.free()
        val_raw.free()

    fn _total_count(self) -> Int:
        return (
            self.none_count
            + self.bool_count
            + self.int_count
            + self.float_count
            + self.unicode_count
            + self.bytes_count
            + self.list_count
            + self.struct_count
        )

    fn _get_binary_type(self) raises -> dt.DataType:
        if self.bytes_count + self.none_count != self._total_count():
            raise Error("cannot mix bytes and non-bytes values")
        return dt.binary

    fn _get_list_type(self) raises -> dt.DataType:
        if self.list_count + self.none_count != self._total_count():
            raise Error("cannot mix list and non-list values")
        if len(self._list_child) == 0:
            raise Error("cannot infer type: all-null list")
        return dt.list_(self._list_child[0]._get_type())

    fn _get_struct_type(self) raises -> dt.DataType:
        if self.struct_count + self.none_count != self._total_count():
            raise Error("cannot mix dict and non-dict values")
        var fields: List[dt.Field] = []
        for i in range(len(self._field_order)):
            var child_dtype = self._field_children[i]._get_type()
            fields.append(
                dt.Field(self._field_order[i], child_dtype, nullable=True)
            )
        return dt.struct_(fields)

    fn _get_primitive_type(self) raises -> dt.DataType:
        if self.unicode_count > 0 and (
            self.bool_count + self.int_count + self.float_count
        ) > 0:
            raise Error("cannot mix string and numeric types")
        if self.float_count > 0:
            return dt.float64
        if self.int_count > 0:
            return dt.int64
        if self.bool_count > 0:
            return dt.bool_
        if self.unicode_count > 0:
            return dt.string
        return dt.null  # empty sequence or all-None

    fn _get_type(self) raises -> dt.DataType:
        if self.bytes_count > 0:
            return self._get_binary_type()
        if self.list_count > 0:
            return self._get_list_type()
        if self.struct_count > 0:
            return self._get_struct_type()
        return self._get_primitive_type()

    fn infer(mut self, obj: PythonObject) raises -> dt.DataType:
        """Single pass: visit all elements, then resolve to a DataType."""
        ref cpy = Python().cpython()
        var list_ptr = obj._obj_ptr
        var n = len(obj)
        for i in range(n):
            var item_ptr = cpy.PyList_GetItem(list_ptr, i)
            self.visit(item_ptr)
        return self._get_type()


# ---------------------------------------------------------------------------
# PyConverter trait — interface for typed Python-to-Arrow converters
# ---------------------------------------------------------------------------


trait PyConverter(Movable, ImplicitlyDestructible):
    fn append(mut self, value: PyObjectPtr) raises: ...
    fn extend(mut self, values: PyObjectPtr) raises: ...


# ---------------------------------------------------------------------------
# PyAnyConverter — type-erased converter with dynamic dispatch
# ---------------------------------------------------------------------------


struct PyAnyConverter(ImplicitlyCopyable, Movable):
    """Type-erased converter dispatching through fn-ptr trampolines."""

    var _data: ArcPointer[NoneType]
    var _virt_append: fn (ArcPointer[NoneType], PyObjectPtr) raises
    var _virt_extend: fn (ArcPointer[NoneType], PyObjectPtr) raises
    var _virt_drop: fn (var ArcPointer[NoneType])

    @staticmethod
    fn _tramp_append[T: PyConverter](
        ptr: ArcPointer[NoneType], value: PyObjectPtr
    ) raises:
        rebind[ArcPointer[T]](ptr)[].append(value)

    @staticmethod
    fn _tramp_extend[T: PyConverter](
        ptr: ArcPointer[NoneType], values: PyObjectPtr
    ) raises:
        rebind[ArcPointer[T]](ptr)[].extend(values)

    @staticmethod
    fn _tramp_drop[T: PyConverter](var ptr: ArcPointer[NoneType]):
        var typed = rebind[ArcPointer[T]](ptr^)
        _ = typed^

    @implicit
    fn __init__[T: PyConverter](out self, var value: T):
        var ptr = ArcPointer(value^)
        self._data = rebind[ArcPointer[NoneType]](ptr^)
        self._virt_append = Self._tramp_append[T]
        self._virt_extend = Self._tramp_extend[T]
        self._virt_drop = Self._tramp_drop[T]

    fn __copyinit__(out self, read copy: Self):
        self._data = copy._data.copy()
        self._virt_append = copy._virt_append
        self._virt_extend = copy._virt_extend
        self._virt_drop = copy._virt_drop

    fn append(mut self, value: PyObjectPtr) raises:
        self._virt_append(self._data, value)

    fn extend(mut self, values: PyObjectPtr) raises:
        self._virt_extend(self._data, values)

    fn __del__(deinit self):
        self._virt_drop(self._data^)


# ---------------------------------------------------------------------------
# PyPrimitiveConverter — hot path for numeric/bool types
# ---------------------------------------------------------------------------


struct PyPrimitiveConverter[T: dt.DataType](PyConverter):
    var _builder: AnyBuilder
    var _has_nulls: Bool

    fn __init__(out self, var builder: AnyBuilder, has_nulls: Bool = True):
        self._builder = builder^
        self._has_nulls = has_nulls

    fn extend(mut self, values: PyObjectPtr) raises:
        var b = self._builder.as_primitive[Self.T]()
        ref cpy = Python().cpython()
        var n = Int(cpy.PyObject_Length(values))
        var none_ptr = cpy.Py_None()
        b[].reserve(n)
        if self._has_nulls:
            for i in range(n):
                var item = cpy.PyList_GetItem(values, i)
                if cpy.Py_Is(item, none_ptr):
                    b[].unsafe_append_null()
                else:
                    comptime if Self.T.native.is_floating_point():
                        b[].unsafe_append(
                            Scalar[Self.T.native](cpy.PyFloat_AsDouble(item))
                        )
                    else:
                        b[].unsafe_append(
                            Scalar[Self.T.native](cpy.PyLong_AsSsize_t(item))
                        )
        else:
            for i in range(n):
                var item = cpy.PyList_GetItem(values, i)
                comptime if Self.T.native.is_floating_point():
                    b[].unsafe_append(
                        Scalar[Self.T.native](cpy.PyFloat_AsDouble(item))
                    )
                else:
                    b[].unsafe_append(
                        Scalar[Self.T.native](cpy.PyLong_AsSsize_t(item))
                    )

    fn append(mut self, value: PyObjectPtr) raises:
        var b = self._builder.as_primitive[Self.T]()
        ref pb = b[]
        ref cpy = Python().cpython()
        if cpy.Py_Is(value, cpy.Py_None()):
            pb.append_null()
        else:
            comptime if Self.T == dt.bool_:
                pb.append(Bool(cpy.PyLong_AsSsize_t(value) != 0))
            elif Self.T.native.is_floating_point():
                pb.append(
                    Scalar[Self.T.native](cpy.PyFloat_AsDouble(value))
                )
            else:
                pb.append(
                    Scalar[Self.T.native](cpy.PyLong_AsSsize_t(value))
                )


# ---------------------------------------------------------------------------
# PyStringConverter — hot path for UTF-8 strings
# ---------------------------------------------------------------------------


struct PyStringConverter(PyConverter):
    var _builder: AnyBuilder
    var _has_nulls: Bool
    var _total_bytes: Int

    fn __init__(
        out self,
        var builder: AnyBuilder,
        has_nulls: Bool = True,
        total_bytes: Int = 0,
    ):
        self._builder = builder^
        self._has_nulls = has_nulls
        self._total_bytes = total_bytes

    fn extend(mut self, values: PyObjectPtr) raises:
        var sb = self._builder.as_string()
        ref cpy = Python().cpython()
        var n = Int(cpy.PyObject_Length(values))
        var none_ptr = cpy.Py_None()

        sb[].reserve(n)
        var total_bytes = self._total_bytes
        if total_bytes == 0:
            for i in range(n):
                var item = cpy.PyList_GetItem(values, i)
                if not cpy.Py_Is(item, none_ptr):
                    total_bytes += len(cpy.PyUnicode_AsUTF8AndSize(item))
        sb[].reserve_bytes(total_bytes)

        if self._has_nulls:
            for i in range(n):
                var item = cpy.PyList_GetItem(values, i)
                if cpy.Py_Is(item, none_ptr):
                    sb[].unsafe_append_null()
                else:
                    var s = cpy.PyUnicode_AsUTF8AndSize(item)
                    sb[].unsafe_append(s.unsafe_ptr(), len(s))
        else:
            for i in range(n):
                var item = cpy.PyList_GetItem(values, i)
                var s = cpy.PyUnicode_AsUTF8AndSize(item)
                sb[].unsafe_append(s.unsafe_ptr(), len(s))

    fn append(mut self, value: PyObjectPtr) raises:
        var sb = self._builder.as_string()
        ref b = sb[]
        ref cpy = Python().cpython()
        if cpy.Py_Is(value, cpy.Py_None()):
            b.append_null()
        else:
            var s = cpy.PyUnicode_AsUTF8AndSize(value)
            b.append(s.unsafe_ptr(), len(s))


# ---------------------------------------------------------------------------
# PyListConverter — variable-length list conversion
# ---------------------------------------------------------------------------


struct PyListConverter(PyConverter):
    var _builder: AnyBuilder
    var _child: PyAnyConverter

    fn __init__(out self, var builder: AnyBuilder, var child: PyAnyConverter):
        self._builder = builder^
        self._child = child^

    fn extend(mut self, values: PyObjectPtr) raises:
        ref cpy = Python().cpython()
        var n = Int(cpy.PyObject_Length(values))
        var none_ptr = cpy.Py_None()
        self._builder.reserve(n)
        for i in range(n):
            var item = cpy.PyList_GetItem(values, i)
            if cpy.Py_Is(item, none_ptr):
                self._builder.append_null()
            else:
                var inner_n = Int(cpy.PyObject_Length(item))
                for j in range(inner_n):
                    self._child.append(cpy.PyList_GetItem(item, j))
                self._builder.append_valid()

    fn append(mut self, value: PyObjectPtr) raises:
        ref cpy = Python().cpython()
        if cpy.Py_Is(value, cpy.Py_None()):
            self._builder.append_null()
        else:
            var n = Int(cpy.PyObject_Length(value))
            for i in range(n):
                self._child.append(cpy.PyList_GetItem(value, i))
            self._builder.append_valid()


# ---------------------------------------------------------------------------
# PyStructConverter — struct/dict conversion
# ---------------------------------------------------------------------------


struct PyStructConverter(PyConverter):
    var _builder: AnyBuilder
    var _children: List[PyAnyConverter]
    var _field_keys: List[PythonObject]

    fn __init__(
        out self,
        var builder: AnyBuilder,
        var children: List[PyAnyConverter],
        dtype: dt.DataType,
    ) raises:
        var field_keys = List[PythonObject]()
        for i in range(len(dtype.fields)):
            field_keys.append(PythonObject(dtype.fields[i].name))
        self._builder = builder^
        self._children = children^
        self._field_keys = field_keys^

    fn extend(mut self, values: PyObjectPtr) raises:
        var n_fields = len(self._children)
        ref cpy = Python().cpython()
        var n = Int(cpy.PyObject_Length(values))
        self._builder.reserve(n)
        var none_ptr = cpy.Py_None()
        for row in range(n):
            var item = cpy.PyList_GetItem(values, row)
            if cpy.Py_Is(item, none_ptr):
                for i in range(n_fields):
                    self._children[i].append(none_ptr)
                self._builder.append_null()
            else:
                for i in range(n_fields):
                    var val = cpy.PyDict_GetItemWithError(
                        item, self._field_keys[i]._obj_ptr
                    )
                    if val == PyObjectPtr():
                        cpy.PyErr_Clear()
                        self._children[i].append(none_ptr)
                    else:
                        self._children[i].append(val)
                self._builder.append_valid()

    fn append(mut self, value: PyObjectPtr) raises:
        ref cpy = Python().cpython()
        var none_ptr = cpy.Py_None()
        if cpy.Py_Is(value, none_ptr):
            for i in range(len(self._children)):
                self._children[i].append(none_ptr)
            self._builder.append_null()
        else:
            for i in range(len(self._children)):
                var val = cpy.PyDict_GetItemWithError(
                    value, self._field_keys[i]._obj_ptr
                )
                if val == PyObjectPtr():
                    cpy.PyErr_Clear()
                    self._children[i].append(none_ptr)
                else:
                    self._children[i].append(val)
            self._builder.append_valid()


# ---------------------------------------------------------------------------
# Factory functions for converters
# ---------------------------------------------------------------------------


fn make_converter(
    dtype: dt.DataType,
    var builder: AnyBuilder,
    has_nulls: Bool = True,
    total_bytes: Int = 0,
) raises -> PyAnyConverter:
    """Create a typed converter wrapping the given builder."""
    comptime for T in dt.primitive_dtypes:
        if dtype == T:
            return PyPrimitiveConverter[T](builder^, has_nulls)
    if dtype.is_string():
        return PyStringConverter(builder^, has_nulls, total_bytes)
    elif dtype.is_list():
        var child_builder = builder.as_list()[].values()
        var child = make_converter(dtype.fields[0].dtype, child_builder)
        return PyListConverter(builder^, child^)
    elif dtype.is_struct():
        var children = List[PyAnyConverter]()
        for i in range(len(dtype.fields)):
            var field_builder = builder.as_struct()[].child(i)
            children.append(
                make_converter(dtype.fields[i].dtype, field_builder)
            )
        return PyStructConverter(builder^, children^, dtype)
    else:
        raise Error("unsupported type: " + String(dtype))


# ---------------------------------------------------------------------------
# Public Python functions
# ---------------------------------------------------------------------------


fn infer_type(obj: PythonObject) raises -> PythonObject:
    var inferrer = PyInferrer()
    return inferrer.infer(obj).to_python_object()


fn array(
    obj: PythonObject, kwargs: OwnedKwargsDict[PythonObject]
) raises -> PythonObject:
    var dtype: dt.DataType
    var has_nulls = True
    var total_bytes = 0
    if opt := kwargs.find("type"):
        dtype = opt.value().downcast_value_ptr[dt.DataType]()[]
    else:
        var inferrer = PyInferrer()
        dtype = inferrer.infer(obj)
        has_nulls = inferrer.none_count > 0
        total_bytes = inferrer.unicode_bytes

    if dtype.is_null():
        raise Error(
            "cannot build array: sequence is empty or all-None"
            " (provide type= explicitly)"
        )

    var builder = make_builder(dtype, len(obj))
    var finish_builder = builder
    var converter = make_converter(dtype, builder^, has_nulls, total_bytes)
    converter.extend(obj._obj_ptr)
    return finish_builder.finish().to_python_object()


def add_to_module(mut mb: PythonModuleBuilder) raises -> None:
    """Add array types and constructors to the Python API."""

    # --- BoolArray ---
    _ = (
        mb.add_type[BoolArray]("BoolArray")
        .def_method[pymethod[BoolArray.__len__]()]("__len__")
        .def_method[pymethod[BoolArray.null_count]()]("null_count")
        .def_method[pymethod[BoolArray.__str__]()]("__str__")
        .def_method[pymethod[BoolArray.__str__]()]("__repr__")
        .def_method[pymethod[BoolArray.is_valid]()]("is_valid")
        .def_method[pymethod[BoolArray.__getitem__]()]("__getitem__")
        .def_method[pymethod[BoolArray.slice]()]("slice")
        .def_method[pymethod[BoolArray.true_count]()]("true_count")
        .def_method[pymethod[BoolArray.false_count]()]("false_count")
    )

    # --- Numeric PrimitiveArrays ---
    _ = (
        mb.add_type[Int8Array]("Int8Array")
        .def_method[pymethod[Int8Array.__len__]()]("__len__")
        .def_method[pymethod[Int8Array.null_count]()]("null_count")
        .def_method[pymethod[Int8Array.__str__]()]("__str__")
        .def_method[pymethod[Int8Array.__str__]()]("__repr__")
        .def_method[pymethod[Int8Array.is_valid]()]("is_valid")
        .def_method[pymethod[Int8Array.__getitem__]()]("__getitem__")
        .def_method[pymethod[Int8Array.slice]()]("slice")
    )
    _ = (
        mb.add_type[Int16Array]("Int16Array")
        .def_method[pymethod[Int16Array.__len__]()]("__len__")
        .def_method[pymethod[Int16Array.null_count]()]("null_count")
        .def_method[pymethod[Int16Array.__str__]()]("__str__")
        .def_method[pymethod[Int16Array.__str__]()]("__repr__")
        .def_method[pymethod[Int16Array.is_valid]()]("is_valid")
        .def_method[pymethod[Int16Array.__getitem__]()]("__getitem__")
        .def_method[pymethod[Int16Array.slice]()]("slice")
    )
    _ = (
        mb.add_type[Int32Array]("Int32Array")
        .def_method[pymethod[Int32Array.__len__]()]("__len__")
        .def_method[pymethod[Int32Array.null_count]()]("null_count")
        .def_method[pymethod[Int32Array.__str__]()]("__str__")
        .def_method[pymethod[Int32Array.__str__]()]("__repr__")
        .def_method[pymethod[Int32Array.is_valid]()]("is_valid")
        .def_method[pymethod[Int32Array.__getitem__]()]("__getitem__")
        .def_method[pymethod[Int32Array.slice]()]("slice")
    )
    _ = (
        mb.add_type[Int64Array]("Int64Array")
        .def_method[pymethod[Int64Array.__len__]()]("__len__")
        .def_method[pymethod[Int64Array.null_count]()]("null_count")
        .def_method[pymethod[Int64Array.__str__]()]("__str__")
        .def_method[pymethod[Int64Array.__str__]()]("__repr__")
        .def_method[pymethod[Int64Array.is_valid]()]("is_valid")
        .def_method[pymethod[Int64Array.__getitem__]()]("__getitem__")
        .def_method[pymethod[Int64Array.slice]()]("slice")
    )
    _ = (
        mb.add_type[UInt8Array]("UInt8Array")
        .def_method[pymethod[UInt8Array.__len__]()]("__len__")
        .def_method[pymethod[UInt8Array.null_count]()]("null_count")
        .def_method[pymethod[UInt8Array.__str__]()]("__str__")
        .def_method[pymethod[UInt8Array.__str__]()]("__repr__")
        .def_method[pymethod[UInt8Array.is_valid]()]("is_valid")
        .def_method[pymethod[UInt8Array.__getitem__]()]("__getitem__")
        .def_method[pymethod[UInt8Array.slice]()]("slice")
    )
    _ = (
        mb.add_type[UInt16Array]("UInt16Array")
        .def_method[pymethod[UInt16Array.__len__]()]("__len__")
        .def_method[pymethod[UInt16Array.null_count]()]("null_count")
        .def_method[pymethod[UInt16Array.__str__]()]("__str__")
        .def_method[pymethod[UInt16Array.__str__]()]("__repr__")
        .def_method[pymethod[UInt16Array.is_valid]()]("is_valid")
        .def_method[pymethod[UInt16Array.__getitem__]()]("__getitem__")
        .def_method[pymethod[UInt16Array.slice]()]("slice")
    )
    _ = (
        mb.add_type[UInt32Array]("UInt32Array")
        .def_method[pymethod[UInt32Array.__len__]()]("__len__")
        .def_method[pymethod[UInt32Array.null_count]()]("null_count")
        .def_method[pymethod[UInt32Array.__str__]()]("__str__")
        .def_method[pymethod[UInt32Array.__str__]()]("__repr__")
        .def_method[pymethod[UInt32Array.is_valid]()]("is_valid")
        .def_method[pymethod[UInt32Array.__getitem__]()]("__getitem__")
        .def_method[pymethod[UInt32Array.slice]()]("slice")
    )
    _ = (
        mb.add_type[UInt64Array]("UInt64Array")
        .def_method[pymethod[UInt64Array.__len__]()]("__len__")
        .def_method[pymethod[UInt64Array.null_count]()]("null_count")
        .def_method[pymethod[UInt64Array.__str__]()]("__str__")
        .def_method[pymethod[UInt64Array.__str__]()]("__repr__")
        .def_method[pymethod[UInt64Array.is_valid]()]("is_valid")
        .def_method[pymethod[UInt64Array.__getitem__]()]("__getitem__")
        .def_method[pymethod[UInt64Array.slice]()]("slice")
    )
    _ = (
        mb.add_type[Float32Array]("Float32Array")
        .def_method[pymethod[Float32Array.__len__]()]("__len__")
        .def_method[pymethod[Float32Array.null_count]()]("null_count")
        .def_method[pymethod[Float32Array.__str__]()]("__str__")
        .def_method[pymethod[Float32Array.__str__]()]("__repr__")
        .def_method[pymethod[Float32Array.is_valid]()]("is_valid")
        .def_method[pymethod[Float32Array.__getitem__]()]("__getitem__")
        .def_method[pymethod[Float32Array.slice]()]("slice")
    )
    _ = (
        mb.add_type[Float64Array]("Float64Array")
        .def_method[pymethod[Float64Array.__len__]()]("__len__")
        .def_method[pymethod[Float64Array.null_count]()]("null_count")
        .def_method[pymethod[Float64Array.__str__]()]("__str__")
        .def_method[pymethod[Float64Array.__str__]()]("__repr__")
        .def_method[pymethod[Float64Array.is_valid]()]("is_valid")
        .def_method[pymethod[Float64Array.__getitem__]()]("__getitem__")
        .def_method[pymethod[Float64Array.slice]()]("slice")
    )

    # --- StringArray ---
    _ = (
        mb.add_type[StringArray]("StringArray")
        .def_method[pymethod[StringArray.__len__]()]("__len__")
        .def_method[pymethod[StringArray.null_count]()]("null_count")
        .def_method[pymethod[StringArray.__str__]()]("__str__")
        .def_method[pymethod[StringArray.__str__]()]("__repr__")
        .def_method[pymethod[StringArray.is_valid]()]("is_valid")
        .def_method[_str_getitem]("__getitem__")
        .def_method[pymethod[StringArray.slice]()]("slice")
    )

    # --- ListArray ---
    _ = (
        mb.add_type[ListArray]("ListArray")
        .def_method[pymethod[ListArray.__len__]()]("__len__")
        .def_method[pymethod[ListArray.null_count]()]("null_count")
        .def_method[pymethod[ListArray.__str__]()]("__str__")
        .def_method[pymethod[ListArray.__str__]()]("__repr__")
        .def_method[pymethod[ListArray.is_valid]()]("is_valid")
        .def_method[pymethod[ListArray.__getitem__]()]("__getitem__")
        .def_method[pymethod[ListArray.slice]()]("slice")
        .def_method[pymethod[ListArray.flatten]()]("flatten")
        .def_method[pymethod[ListArray.value_lengths]()]("value_lengths")
    )

    # --- FixedSizeListArray ---
    _ = (
        mb.add_type[FixedSizeListArray]("FixedSizeListArray")
        .def_method[pymethod[FixedSizeListArray.__len__]()]("__len__")
        .def_method[pymethod[FixedSizeListArray.null_count]()]("null_count")
        .def_method[pymethod[FixedSizeListArray.__str__]()]("__str__")
        .def_method[pymethod[FixedSizeListArray.__str__]()]("__repr__")
        .def_method[pymethod[FixedSizeListArray.is_valid]()]("is_valid")
        .def_method[pymethod[FixedSizeListArray.__getitem__]()]("__getitem__")
        .def_method[pymethod[FixedSizeListArray.slice]()]("slice")
        .def_method[pymethod[FixedSizeListArray.flatten]()]("flatten")
    )

    # --- StructArray ---
    _ = (
        mb.add_type[StructArray]("StructArray")
        .def_method[pymethod[StructArray.__len__]()]("__len__")
        .def_method[pymethod[StructArray.null_count]()]("null_count")
        .def_method[pymethod[StructArray.__str__]()]("__str__")
        .def_method[pymethod[StructArray.__str__]()]("__repr__")
        .def_method[pymethod[StructArray.is_valid]()]("is_valid")
        .def_method[pymethod[StructArray.field]()]("field")
    )

    mb.def_function[infer_type](
        "infer_type", docstring="Infer the Arrow type of a Python sequence."
    )
    mb.def_function[array](
        "array", docstring="Create a marrow array from a Python sequence."
    )

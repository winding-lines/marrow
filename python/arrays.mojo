from std.python import PythonObject, ConvertibleToPython, Python
from std.python.bindings import PythonModuleBuilder
from std.collections import OwnedKwargsDict
from std.python._cpython import CPython, PyObjectPtr, PyTypeObject, PyTypeObjectPtr
from std.memory import ArcPointer
from marrow.arrays import Array
from marrow.builders import (
    AnyBuilder,
    PrimitiveBuilder,
    StringBuilder,
    ListBuilder,
    StructBuilder,
    make_builder,
)
import marrow.arrays as arr
import marrow.dtypes as dt


fn pymethod[
    T: AnyType,
    R: ConvertibleToPython,
    //,
    method: fn (T) -> R,
]() -> fn (UnsafePointer[T, MutAnyOrigin]) raises -> PythonObject:
    fn wrapper(ptr: UnsafePointer[T, MutAnyOrigin]) raises -> PythonObject:
        return method(ptr[]).to_python_object()

    return wrapper


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

    fn visit(mut self, element: PythonObject) raises:
        """Count one element's Python type, following PyArrow's Visit() order."""
        self.visit_ptr(element._obj_ptr)

    fn visit_ptr(mut self, ptr: PyObjectPtr) raises:
        """Count one element's Python type from a raw pointer."""
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
            self._visit_dict(PythonObject(from_borrowed=ptr))
        elif cpy.PyObject_TypeCheck(ptr, self._list_type) != 0:
            self.list_count += 1
            self._visit_list(PythonObject(from_borrowed=ptr))
        elif cpy.PyObject_TypeCheck(ptr, self._tuple_type) != 0:
            self.list_count += 1
            self._visit_list(PythonObject(from_borrowed=ptr))
        else:
            raise Error(
                "cannot include value of type: "
                + String(PythonObject(from_borrowed=ptr).__class__.__name__)
            )

    fn _visit_list(mut self, list_obj: PythonObject) raises:
        """Mirrors PyArrow's VisitSequence: recurse into list element's children."""
        if len(self._list_child) == 0:
            self._list_child.append(PyInferrer())
        for element in list_obj:
            self._list_child[0].visit(element)

    fn _visit_dict(mut self, dict_obj: PythonObject) raises:
        """Mirrors PyArrow's VisitDict: route each value to its field's child inferrer."""
        for key_obj in dict_obj.keys():
            var name = String(py=key_obj)
            var idx = -1
            for i in range(len(self._field_order)):
                if self._field_order[i] == name:
                    idx = i
                    break
            if idx == -1:
                idx = len(self._field_order)
                self._field_order.append(name)
                self._field_children.append(PyInferrer())
            self._field_children[idx].visit(dict_obj[key_obj])

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
            self.visit_ptr(item_ptr)
        return self._get_type()


# ---------------------------------------------------------------------------
# PyConverter trait — interface for typed Python-to-Arrow converters
# ---------------------------------------------------------------------------


trait PyConverter(Movable, ImplicitlyDestructible):
    fn append(mut self, value: PythonObject) raises: ...
    fn extend(mut self, values: PythonObject) raises: ...
    fn builder(self) -> AnyBuilder: ...


# ---------------------------------------------------------------------------
# PyAnyConverter — type-erased converter with dynamic dispatch
# ---------------------------------------------------------------------------


struct PyAnyConverter(ImplicitlyCopyable, Movable):
    """Type-erased converter dispatching through fn-ptr trampolines.

    `append` and `extend` are dispatched virtually to the typed converter.
    `finish` is non-virtual — every converter delegates to the same
    `AnyBuilder.finish()`, so the builder is captured at construction time.
    """

    var _data: ArcPointer[NoneType]
    var _builder: AnyBuilder
    var _virt_append: fn (ArcPointer[NoneType], PythonObject) raises
    var _virt_extend: fn (ArcPointer[NoneType], PythonObject) raises
    var _virt_drop: fn (var ArcPointer[NoneType])

    @staticmethod
    fn _tramp_append[T: PyConverter](
        ptr: ArcPointer[NoneType], value: PythonObject
    ) raises:
        rebind[ArcPointer[T]](ptr)[].append(value)

    @staticmethod
    fn _tramp_extend[T: PyConverter](
        ptr: ArcPointer[NoneType], values: PythonObject
    ) raises:
        rebind[ArcPointer[T]](ptr)[].extend(values)

    @staticmethod
    fn _tramp_drop[T: PyConverter](var ptr: ArcPointer[NoneType]):
        var typed = rebind[ArcPointer[T]](ptr^)
        _ = typed^

    @implicit
    fn __init__[T: PyConverter](out self, var value: T):
        self._builder = value.builder()
        var ptr = ArcPointer(value^)
        self._data = rebind[ArcPointer[NoneType]](ptr^)
        self._virt_append = Self._tramp_append[T]
        self._virt_extend = Self._tramp_extend[T]
        self._virt_drop = Self._tramp_drop[T]

    fn __copyinit__(out self, read copy: Self):
        self._data = copy._data.copy()
        self._builder = copy._builder
        self._virt_append = copy._virt_append
        self._virt_extend = copy._virt_extend
        self._virt_drop = copy._virt_drop

    fn append(mut self, value: PythonObject) raises:
        self._virt_append(self._data, value)

    fn extend(mut self, values: PythonObject) raises:
        self._virt_extend(self._data, values)

    fn finish(mut self) raises -> Array:
        return self._builder.finish()

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

    fn builder(self) -> AnyBuilder:
        return self._builder

    fn extend(mut self, values: PythonObject) raises:
        var pb = self._builder.as_primitive[Self.T]()
        ref cpy = Python().cpython()
        var n = len(values)
        var list_ptr = values._obj_ptr
        var none_ptr = cpy.Py_None()
        pb[].reserve(n)
        if self._has_nulls:
            for i in range(n):
                var item = cpy.PyList_GetItem(list_ptr, i)
                if cpy.Py_Is(item, none_ptr):
                    pb[].unsafe_append_null()
                else:
                    comptime if Self.T.native.is_floating_point():
                        pb[].unsafe_append(
                            Scalar[Self.T.native](cpy.PyFloat_AsDouble(item))
                        )
                    else:
                        pb[].unsafe_append(
                            Scalar[Self.T.native](cpy.PyLong_AsSsize_t(item))
                        )
        else:
            for i in range(n):
                var item = cpy.PyList_GetItem(list_ptr, i)
                comptime if Self.T.native.is_floating_point():
                    pb[].unsafe_append(
                        Scalar[Self.T.native](cpy.PyFloat_AsDouble(item))
                    )
                else:
                    pb[].unsafe_append(
                        Scalar[Self.T.native](cpy.PyLong_AsSsize_t(item))
                    )

    fn append(mut self, value: PythonObject) raises:
        var pb = self._builder.as_primitive[Self.T]()
        if value is None:
            pb[].append_null()
        else:
            comptime if Self.T == dt.bool_:
                pb[].append(value.__bool__())
            else:
                pb[].append(Scalar[Self.T.native](py=value))


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

    fn builder(self) -> AnyBuilder:
        return self._builder

    fn extend(mut self, values: PythonObject) raises:
        var sb = self._builder.as_string()
        ref cpy = Python().cpython()
        var n = len(values)
        var list_ptr = values._obj_ptr
        var none_ptr = cpy.Py_None()

        sb[].reserve(n)
        var total_bytes = self._total_bytes
        if total_bytes == 0:
            for i in range(n):
                var item = cpy.PyList_GetItem(list_ptr, i)
                if not cpy.Py_Is(item, none_ptr):
                    total_bytes += len(cpy.PyUnicode_AsUTF8AndSize(item))
        sb[].reserve_bytes(total_bytes)

        if self._has_nulls:
            for i in range(n):
                var item = cpy.PyList_GetItem(list_ptr, i)
                if cpy.Py_Is(item, none_ptr):
                    sb[].unsafe_append_null()
                else:
                    var s = cpy.PyUnicode_AsUTF8AndSize(item)
                    sb[].unsafe_append(s.unsafe_ptr(), len(s))
        else:
            for i in range(n):
                var item = cpy.PyList_GetItem(list_ptr, i)
                var s = cpy.PyUnicode_AsUTF8AndSize(item)
                sb[].unsafe_append(s.unsafe_ptr(), len(s))

    fn append(mut self, value: PythonObject) raises:
        var sb = self._builder.as_string()
        if value is None:
            sb[].append_null()
        else:
            sb[].append(String(py=value))


# ---------------------------------------------------------------------------
# PyListConverter — variable-length list conversion
# ---------------------------------------------------------------------------


struct PyListConverter(PyConverter):
    var _builder: AnyBuilder
    var _child: PyAnyConverter

    fn __init__(out self, var builder: AnyBuilder, var child: PyAnyConverter):
        self._builder = builder^
        self._child = child^

    fn builder(self) -> AnyBuilder:
        return self._builder

    fn extend(mut self, values: PythonObject) raises:
        var lb = self._builder.as_list()
        for element in values:
            if element is None:
                lb[].append_null()
            else:
                self._child.extend(element)
                lb[].append(True)

    fn append(mut self, value: PythonObject) raises:
        var lb = self._builder.as_list()
        if value is None:
            lb[].append_null()
        else:
            self._child.extend(value)
            lb[].append(True)


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

    fn builder(self) -> AnyBuilder:
        return self._builder

    fn extend(mut self, values: PythonObject) raises:
        var sb = self._builder.as_struct()
        var n_fields = len(self._children)
        ref cpy = Python().cpython()
        var list_ptr = values._obj_ptr
        var n = len(values)
        var none_ptr = cpy.Py_None()
        for row in range(n):
            var item = cpy.PyList_GetItem(list_ptr, row)
            if cpy.Py_Is(item, none_ptr):
                for i in range(n_fields):
                    self._children[i].append(PythonObject(None))
                sb[].append(False)
            else:
                var dict_obj = PythonObject(from_borrowed=item)
                for i in range(n_fields):
                    self._children[i].append(dict_obj.get(self._field_keys[i]))
                sb[].append(True)

    fn append(mut self, value: PythonObject) raises:
        var sb = self._builder.as_struct()
        var is_null = value is None
        for i in range(len(self._children)):
            if is_null:
                self._children[i].append(PythonObject(None))
            else:
                self._children[i].append(value.get(self._field_keys[i]))
        sb[].append(not is_null)


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
    var converter = make_converter(dtype, builder^, has_nulls, total_bytes)
    converter.extend(obj)
    return converter.finish().to_python_object()


def add_to_module(mut mb: PythonModuleBuilder) raises -> None:
    """Add array types and constructors to the Python API."""

    _ = (
        mb.add_type[arr.BoolArray]("BoolArray")
        .def_method[pymethod[arr.BoolArray.__len__]()]("__len__")
        .def_method[pymethod[arr.BoolArray.null_count]()]("null_count")
    )
    _ = (
        mb.add_type[arr.Int8Array]("Int8Array")
        .def_method[pymethod[arr.Int8Array.__len__]()]("__len__")
        .def_method[pymethod[arr.Int8Array.null_count]()]("null_count")
    )
    _ = (
        mb.add_type[arr.Int16Array]("Int16Array")
        .def_method[pymethod[arr.Int16Array.__len__]()]("__len__")
        .def_method[pymethod[arr.Int16Array.null_count]()]("null_count")
    )
    _ = (
        mb.add_type[arr.Int32Array]("Int32Array")
        .def_method[pymethod[arr.Int32Array.__len__]()]("__len__")
        .def_method[pymethod[arr.Int32Array.null_count]()]("null_count")
    )
    _ = (
        mb.add_type[arr.Int64Array]("Int64Array")
        .def_method[pymethod[arr.Int64Array.__len__]()]("__len__")
        .def_method[pymethod[arr.Int64Array.null_count]()]("null_count")
    )
    _ = (
        mb.add_type[arr.UInt8Array]("UInt8Array")
        .def_method[pymethod[arr.UInt8Array.__len__]()]("__len__")
        .def_method[pymethod[arr.UInt8Array.null_count]()]("null_count")
    )
    _ = (
        mb.add_type[arr.UInt16Array]("UInt16Array")
        .def_method[pymethod[arr.UInt16Array.__len__]()]("__len__")
        .def_method[pymethod[arr.UInt16Array.null_count]()]("null_count")
    )
    _ = (
        mb.add_type[arr.UInt32Array]("UInt32Array")
        .def_method[pymethod[arr.UInt32Array.__len__]()]("__len__")
        .def_method[pymethod[arr.UInt32Array.null_count]()]("null_count")
    )
    _ = (
        mb.add_type[arr.UInt64Array]("UInt64Array")
        .def_method[pymethod[arr.UInt64Array.__len__]()]("__len__")
        .def_method[pymethod[arr.UInt64Array.null_count]()]("null_count")
    )
    _ = (
        mb.add_type[arr.Float32Array]("Float32Array")
        .def_method[pymethod[arr.Float32Array.__len__]()]("__len__")
        .def_method[pymethod[arr.Float32Array.null_count]()]("null_count")
    )
    _ = (
        mb.add_type[arr.Float64Array]("Float64Array")
        .def_method[pymethod[arr.Float64Array.__len__]()]("__len__")
        .def_method[pymethod[arr.Float64Array.null_count]()]("null_count")
    )
    _ = (
        mb.add_type[arr.StringArray]("StringArray")
        .def_method[pymethod[arr.StringArray.__len__]()]("__len__")
        .def_method[pymethod[arr.StringArray.null_count]()]("null_count")
    )
    _ = (
        mb.add_type[arr.ListArray]("ListArray")
        .def_method[pymethod[arr.ListArray.__len__]()]("__len__")
        .def_method[pymethod[arr.ListArray.null_count]()]("null_count")
    )
    _ = mb.add_type[arr.FixedSizeListArray]("FixedSizeListArray")
    _ = (
        mb.add_type[arr.StructArray]("StructArray")
        .def_method[pymethod[arr.StructArray.__len__]()]("__len__")
        .def_method[pymethod[arr.StructArray.null_count]()]("null_count")
    )

    mb.def_function[infer_type](
        "infer_type", docstring="Infer the Arrow type of a Python sequence."
    )
    mb.def_function[array](
        "array", docstring="Create a marrow array from a Python sequence."
    )

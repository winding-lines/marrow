"""Python interface for primitive array."""

from os import abort
from python.bindings import PythonModuleBuilder, PythonObject
from marrow.dtypes import DataType
from marrow.dtypes import *
from marrow.arrays import Array
from marrow.arrays import PrimitiveArray as _PrimitiveArray
from marrow.builders import PrimitiveBuilder
from python import Python


@fieldwise_init
struct PrimitiveArray(Movable, Representable):
    """Type erased PrimitiveArray so that we can return to python."""

    var data: Array
    var offset: Int
    var capacity: Int

    fn __repr__(self) -> String:
        return "PrimitiveArray"

    @staticmethod
    fn __len__(py_self: PythonObject) raises -> PythonObject:
        """Return the length of the underlying Array."""
        var self_ptr = py_self.downcast_value_ptr[Self]()
        return self_ptr[].data.length

    @staticmethod
    fn __getitem__(
        py_self: PythonObject, index: PythonObject
    ) raises -> PythonObject:
        """Access the element at the given index."""
        var self_ptr = py_self.downcast_value_ptr[Self]()
        return PythonObject(
            _PrimitiveArray[int64](self_ptr[].data.copy()).unsafe_get(
                Int(py=index)
            )
        )


fn array(content: PythonObject) raises -> PythonObject:
    """Create a primitive array, only In64 implemented so far.

    Args:
        content: An iterable of Ints.

    Returns:
        A PrimitiveArray wrapped in a PythonObject.

    """
    var builder = PrimitiveBuilder[int64]()

    for v in content:
        builder.append(Scalar[int64.native](Int(py=v)))

    var actual = builder.finish()
    var result = PrimitiveArray(
        data=Array(actual^),
        offset=0,
        capacity=0,
    )
    return PythonObject(alloc=result^)


def add_to_module(mut builder: PythonModuleBuilder) -> None:
    """Add primitive array support to the python API."""

    _ = (
        builder.add_type[PrimitiveArray]("PrimitiveArray")
        .def_method[PrimitiveArray.__len__]("__len__")
        .def_method[PrimitiveArray.__getitem__]("__getitem__")
    )
    builder.def_function[array](
        "array",
        docstring="Build a primitive array with the given data and datatype",
    )

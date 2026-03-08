
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
import marrow.arrays as arr
import marrow.dtypes as dt


def add_to_module(mut mb: PythonModuleBuilder) raises -> None:
    """Add DataType related data to the Python API."""

    _ = mb.add_type[arr.BoolArray]("BoolArray")
    _ = mb.add_type[arr.Int8Array]("Int8Array")
    _ = mb.add_type[arr.Int16Array]("Int16Array")
    _ = mb.add_type[arr.Int32Array]("Int32Array")
    _ = mb.add_type[arr.Int64Array]("Int64Array")
    _ = mb.add_type[arr.UInt8Array]("UInt8Array")
    _ = mb.add_type[arr.UInt16Array]("UInt16Array")
    _ = mb.add_type[arr.UInt32Array]("UInt32Array")
    _ = mb.add_type[arr.UInt64Array]("UInt64Array")
    _ = mb.add_type[arr.Float16Array]("Float16Array")
    _ = mb.add_type[arr.Float32Array]("Float32Array")
    _ = mb.add_type[arr.Float64Array]("Float64Array")
    _ = mb.add_type[arr.StringArray]("StringArray")
    _ = mb.add_type[arr.ListArray]("ListArray")
    _ = mb.add_type[arr.FixedSizeListArray]("FixedSizeListArray")
    _ = mb.add_type[arr.StructArray]("StructArray")



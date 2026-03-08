# """Python interface for compute kernels."""

# from std.python import PythonObject
# from std.python.bindings import PythonModuleBuilder
# from marrow.arrays import PrimitiveArray, FixedSizeListArray, Array
# from marrow.dtypes import DataType, all_numeric_dtypes, float32, float64
# from marrow.kernels.arithmetic import (
#     add,
#     sub,
#     mul,
#     div,
#     floordiv,
#     mod,
#     min_ as elem_min,
#     max_ as elem_max,
#     neg,
#     abs_,
# )
# from marrow.kernels.aggregate import (
#     sum as agg_sum,
#     product as agg_product,
#     min_ as agg_min,
#     max_ as agg_max,
# )
# from marrow.kernels.similarity import cosine_similarity


# # ---------------------------------------------------------------------------
# # Helpers: PythonObject ↔ Array conversion
# # ---------------------------------------------------------------------------


# fn _to_array(py_obj: PythonObject) raises -> Array:
#     """Extract a runtime-typed Array from a Python-side PrimitiveArray[T]."""
#     var dtype_py = py_obj.dtype()
#     var dtype = dtype_py.downcast_value_ptr[DataType]()[]
#     comptime for T in all_numeric_dtypes:
#         if dtype == T:
#             return Array(py_obj.downcast_value_ptr[PrimitiveArray[T]]()[])
#     raise Error("unsupported dtype: " + String(dtype))


# fn _array_to_py(result: Array) raises -> PythonObject:
#     """Wrap a runtime-typed Array back into the appropriate Python typed array."""
#     comptime for T in all_numeric_dtypes:
#         if result.dtype == T:
#             return PythonObject(alloc=PrimitiveArray[T](result))
#     raise Error("unsupported dtype: " + String(result.dtype))


# # ---------------------------------------------------------------------------
# # Binary element-wise operations
# # ---------------------------------------------------------------------------


# fn add_py(left: PythonObject, right: PythonObject) raises -> PythonObject:
#     return _array_to_py(add(_to_array(left), _to_array(right)))


# fn sub_py(left: PythonObject, right: PythonObject) raises -> PythonObject:
#     return _array_to_py(sub(_to_array(left), _to_array(right)))


# fn mul_py(left: PythonObject, right: PythonObject) raises -> PythonObject:
#     return _array_to_py(mul(_to_array(left), _to_array(right)))


# fn div_py(left: PythonObject, right: PythonObject) raises -> PythonObject:
#     return _array_to_py(div(_to_array(left), _to_array(right)))


# fn floordiv_py(left: PythonObject, right: PythonObject) raises -> PythonObject:
#     return _array_to_py(floordiv(_to_array(left), _to_array(right)))


# fn mod_py(left: PythonObject, right: PythonObject) raises -> PythonObject:
#     return _array_to_py(mod(_to_array(left), _to_array(right)))


# fn elem_min_py(
#     left: PythonObject, right: PythonObject
# ) raises -> PythonObject:
#     return _array_to_py(elem_min(_to_array(left), _to_array(right)))


# fn elem_max_py(
#     left: PythonObject, right: PythonObject
# ) raises -> PythonObject:
#     return _array_to_py(elem_max(_to_array(left), _to_array(right)))


# # ---------------------------------------------------------------------------
# # Unary element-wise operations
# # ---------------------------------------------------------------------------


# fn neg_py(arr: PythonObject) raises -> PythonObject:
#     var data = _to_array(arr)
#     comptime for T in all_numeric_dtypes:
#         if data.dtype == T:
#             return PythonObject(alloc=neg[T](PrimitiveArray[T](data)))
#     raise Error("neg: unsupported dtype: " + String(data.dtype))


# fn abs_py(arr: PythonObject) raises -> PythonObject:
#     var data = _to_array(arr)
#     comptime for T in all_numeric_dtypes:
#         if data.dtype == T:
#             return PythonObject(alloc=abs_[T](PrimitiveArray[T](data)))
#     raise Error("abs_: unsupported dtype: " + String(data.dtype))


# # ---------------------------------------------------------------------------
# # Aggregations (return Python float)
# # ---------------------------------------------------------------------------


# fn sum_py(arr: PythonObject) raises -> PythonObject:
#     return agg_sum(_to_array(arr))


# fn product_py(arr: PythonObject) raises -> PythonObject:
#     return agg_product(_to_array(arr))


# fn min_py(arr: PythonObject) raises -> PythonObject:
#     return agg_min(_to_array(arr))


# fn max_py(arr: PythonObject) raises -> PythonObject:
#     return agg_max(_to_array(arr))


# # ---------------------------------------------------------------------------
# # Similarity
# # ---------------------------------------------------------------------------


# fn cosine_similarity_py(
#     vectors: PythonObject, query: PythonObject
# ) raises -> PythonObject:
#     """Batch cosine similarity: N vectors vs one query → N scores."""
#     var vecs_ptr = vectors.downcast_value_ptr[FixedSizeListArray]()
#     var dtype_py = query.dtype()
#     var dtype = dtype_py.downcast_value_ptr[DataType]()[]
#     if dtype == float32:
#         var q_ptr = query.downcast_value_ptr[PrimitiveArray[float32]]()
#         var result = cosine_similarity[float32](vecs_ptr[], q_ptr[])
#         return PythonObject(alloc=result^)
#     elif dtype == float64:
#         var q_ptr = query.downcast_value_ptr[PrimitiveArray[float64]]()
#         var result = cosine_similarity[float64](vecs_ptr[], q_ptr[])
#         return PythonObject(alloc=result^)
#     else:
#         raise Error(
#             "cosine_similarity: unsupported dtype "
#             + String(dtype)
#             + " (use float32 or float64)"
#         )


# # ---------------------------------------------------------------------------
# # Module registration
# # ---------------------------------------------------------------------------


# def add_to_module(mut builder: PythonModuleBuilder) -> None:
#     """Register compute functions with the Python module."""

#     builder.def_function[add_py]("add", docstring="Element-wise addition.")
#     builder.def_function[sub_py]("sub", docstring="Element-wise subtraction.")
#     builder.def_function[mul_py]("mul", docstring="Element-wise multiplication.")
#     builder.def_function[div_py]("div", docstring="Element-wise true division.")
#     builder.def_function[floordiv_py](
#         "floordiv", docstring="Element-wise floor division."
#     )
#     builder.def_function[mod_py]("mod", docstring="Element-wise modulo.")
#     builder.def_function[elem_min_py](
#         "minimum", docstring="Element-wise minimum of two arrays."
#     )
#     builder.def_function[elem_max_py](
#         "maximum", docstring="Element-wise maximum of two arrays."
#     )
#     builder.def_function[neg_py]("neg", docstring="Element-wise negation.")
#     builder.def_function[abs_py]("abs_", docstring="Element-wise absolute value.")
#     builder.def_function[sum_py]("sum", docstring="Sum of all valid elements.")
#     builder.def_function[product_py](
#         "product", docstring="Product of all valid elements."
#     )
#     builder.def_function[min_py](
#         "min_", docstring="Minimum of all valid elements."
#     )
#     builder.def_function[max_py](
#         "max_", docstring="Maximum of all valid elements."
#     )
#     builder.def_function[cosine_similarity_py](
#         "cosine_similarity",
#         docstring="Batch cosine similarity: N vectors vs one query.",
#     )

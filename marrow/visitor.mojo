from .arrays import *
from .dtypes import *


trait ArrayVisitor:
    """Trait for type-dispatched array operations.

    Implement this trait and call `visitor.visit(array)` to receive a
    concretely-typed array matching the runtime dtype of the Array.

    All typed `visit` overloads have default no-op bodies, so implementors
    only need to override the array kinds they care about. `visit(Array)`
    dispatches to the typed overloads by default. `visit(ChunkedArray)`
    dispatches to each chunk by default.

    Typed arrays passed to visitor methods share the underlying buffer memory
    with the original Array (ArcPointer semantics). Visitor methods are
    `raises` to allow implementations that perform I/O or recursive dispatch
    into nested arrays.
    """

    fn visit[T: DataType](mut self, array: PrimitiveArray[T]) raises:
        pass

    fn visit(mut self, array: StringArray) raises:
        pass

    fn visit(mut self, array: ListArray) raises:
        pass

    fn visit(mut self, array: StructArray) raises:
        pass

    fn visit(mut self, array: ChunkedArray) raises:
        for chunk in array.chunks:
            self.visit(chunk)

    fn visit(mut self, array: Array) raises:
        """Dispatch to the typed overload matching the runtime dtype."""

        @parameter
        for dtype in [
            bool_,
            int8, int16, int32, int64,
            uint8, uint16, uint32, uint64,
            float16, float32, float64,
        ]:
            if array.dtype == materialize[dtype]():
                self.visit[dtype](PrimitiveArray[dtype](array.copy()))
                return

        if array.dtype.is_string():
            self.visit(StringArray(array.copy()))
        elif array.dtype.is_list():
            self.visit(ListArray(array.copy()))
        elif array.dtype.is_struct():
            self.visit(StructArray(data=array.copy()))
        else:
            raise Error("visit: unsupported dtype {}".format(array.dtype))

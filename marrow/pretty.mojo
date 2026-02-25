from .arrays import *
from .visitor import ArrayVisitor


struct ArrayPrinter(ArrayVisitor):
    """Pretty-prints arrays to a String buffer with a configurable element limit.

    Usage:
        var printer = ArrayPrinter(limit=3)
        printer.visit(my_array)
        print(printer.finish())
    """

    var output: String
    var limit: Int

    fn __init__(out self, limit: Int = 3):
        self.output = String()
        self.limit = limit

    fn finish(deinit self) -> String:
        return self.output^

    fn visit[T: DataType](mut self, array: PrimitiveArray[T]) raises:
        self.output.write("PrimitiveArray[")
        self.output.write(String(materialize[T]()))
        self.output.write("]([")
        for i in range(array.length):
            if i > 0:
                self.output.write(", ")
            if i >= self.limit:
                self.output.write("...")
                break
            if array.is_valid(i):
                self.output.write(array.unsafe_get(i))
            else:
                self.output.write("NULL")
        self.output.write("])")

    fn visit(mut self, array: StringArray) raises:
        self.output.write("StringArray([")
        for i in range(array.length):
            if i > 0:
                self.output.write(", ")
            if i >= self.limit:
                self.output.write("...")
                break
            if array.is_valid(i):
                self.output.write(array.unsafe_get(UInt(i)))
            else:
                self.output.write("NULL")
        self.output.write("])")

    fn visit(mut self, array: ListArray) raises:
        self.output.write("ListArray([")
        for i in range(array.length):
            if i > 0:
                self.output.write(", ")
            if i >= self.limit:
                self.output.write("...")
                break
            if array.is_valid(i):
                var start = Int(
                    array.offsets[].unsafe_get[DType.int32](array.offset + i)
                )
                var end = Int(
                    array.offsets[].unsafe_get[DType.int32](
                        array.offset + i + 1
                    )
                )
                ref first_child = array.values[]
                self.visit(
                    Array(
                        dtype=first_child.dtype.copy(),
                        bitmap=first_child.bitmap,
                        buffers=first_child.buffers.copy(),
                        offset=start,
                        length=end - start,
                        children=first_child.children.copy(),
                    )
                )
            else:
                self.output.write("NULL")
        self.output.write("])")

    fn visit(mut self, array: StructArray) raises:
        self.output.write("StructArray({")
        if len(array.children) > 0:
            for i in range(len(array.dtype.fields)):
                if i > 0:
                    self.output.write(", ")
                ref field = array.dtype.fields[i]
                self.output.write("'")
                self.output.write(field.name)
                self.output.write("': ")
                self.visit(array.children[i][])
        self.output.write("})")

    fn visit(mut self, array: ChunkedArray) raises:
        self.output.write("ChunkedArray([")
        for i in range(len(array.chunks)):
            if i > 0:
                self.output.write(", ")
            self.visit(array.chunks[i])
        self.output.write("])")

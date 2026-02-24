from .arrays import *

struct ArrayPrinter:
    """Pretty-prints arrays to a String buffer with a configurable element limit.

    Usage:
        var printer = ArrayPrinter(limit=3)
        printer.visit(my_array)
        print(printer.output)
    """

    var output: String
    var limit: Int

    fn __init__(out self, limit: Int = 3):
        self.output = String()
        self.limit = limit

    fn visit[T: DataType](mut self, array: PrimitiveArray[T]):
        self.output.write("PrimitiveArray[")
        self.output.write(materialize[T]())
        self.output.write("]([")
        for i in range(array.data.length):
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

    fn visit(mut self, array: StringArray):
        self.output.write("StringArray([")
        for i in range(array.data.length):
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

    fn visit(mut self, array: ListArray):
        self.output.write("ListArray([")
        for i in range(array.data.length):
            if i > 0:
                self.output.write(", ")
            if i >= self.limit:
                self.output.write("...")
                break
            if array.is_valid(i):
                var start = Int(array.offsets()[].unsafe_get[DType.int32](array.data.offset + i))
                var end = Int(array.offsets()[].unsafe_get[DType.int32](array.data.offset + i + 1))
                ref first_child = array.data.children[0][]
                self.visit(Array(
                    dtype=first_child.dtype.copy(),
                    bitmap=first_child.bitmap,
                    buffers=first_child.buffers.copy(),
                    offset=start,
                    length=end - start,
                    children=first_child.children.copy(),
                ))
            else:
                self.output.write("NULL")
        self.output.write("])")

    fn visit(mut self, array: StructArray):
        self.output.write("StructArray({")
        if len(array.data.children) > 0:
            for i in range(len(array.fields)):
                if i > 0:
                    self.output.write(", ")
                ref field = array.fields[i]
                self.output.write("'")
                self.output.write(field.name)
                self.output.write("': ")
                self.visit(array.data.children[i][])
        self.output.write("})")

    fn visit(mut self, array: ChunkedArray):
        self.output.write("ChunkedArray([")
        for i in range(len(array.chunks)):
            if i > 0:
                self.output.write(", ")
            self.visit(array.chunks[i])
        self.output.write("])")

    fn visit(mut self, array: Array):
        @parameter
        for dtype in [
            bool_,
            int8, int16, int32, int64,
            uint8, uint16, uint32, uint64,
            float16, float32, float64,
        ]:
            if array.dtype == materialize[dtype]():
                self.output.write("PrimitiveArray[")
                self.output.write(array.dtype)
                self.output.write("]([")
                for i in range(array.length):
                    if i > 0:
                        self.output.write(", ")
                    if i >= self.limit:
                        self.output.write("...")
                        break
                    if array.is_valid(i + array.offset):
                        array._dynamic_write(i + array.offset, self.output)
                    else:
                        self.output.write("NULL")
                self.output.write("])")
                return

        if array.dtype.is_string():
            self.output.write("StringArray([")
            for i in range(array.length):
                if i > 0:
                    self.output.write(", ")
                if i >= self.limit:
                    self.output.write("...")
                    break
                if array.is_valid(i + array.offset):
                    var idx = i + array.offset
                    var start = array.buffers[0][].unsafe_get[DType.uint32](idx)
                    var end = array.buffers[0][].unsafe_get[DType.uint32](idx + 1)
                    var ptr = array.buffers[1][].get_ptr_at(Int(start)).mut_cast[False]()
                    self.output.write(StringSlice(unsafe_from_utf8=Span[Byte](ptr=ptr, length=Int(end) - Int(start))))
                else:
                    self.output.write("NULL")
            self.output.write("])")
        elif array.dtype.is_list():
            self.output.write("ListArray([")
            for i in range(array.length):
                if i > 0:
                    self.output.write(", ")
                if i >= self.limit:
                    self.output.write("...")
                    break
                if array.is_valid(i + array.offset):
                    var start = Int(array.buffers[0][].unsafe_get[DType.int32](array.offset + i))
                    var end = Int(array.buffers[0][].unsafe_get[DType.int32](array.offset + i + 1))
                    ref first_child = array.children[0][]
                    self.visit(Array(
                        dtype=first_child.dtype.copy(),
                        bitmap=first_child.bitmap,
                        buffers=first_child.buffers.copy(),
                        offset=start,
                        length=end - start,
                        children=first_child.children.copy(),
                    ))
                else:
                    self.output.write("NULL")
            self.output.write("])")
        elif array.dtype.is_struct():
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

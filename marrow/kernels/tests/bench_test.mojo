import benchmark
from benchmark import keep
from marrow.arrays import PrimitiveArray
from marrow.builders import PrimitiveBuilder
from marrow.dtypes import int32, DataType
from marrow.kernels import binary_simd
from marrow.kernels.arithmetic import _add

fn _make_array[T: DataType](size: Int) raises -> PrimitiveArray[T]:
    var b = PrimitiveBuilder[T](size)
    for i in range(size):
        b.append(Scalar[T.native](i))
    return b.finish()

@parameter
fn bench_simd_int32_1k() raises:
    var lhs = _make_array[int32](1_000)
    var rhs = _make_array[int32](1_000)
    var result = binary_simd[int32, _add[int32.native]](lhs, rhs)
    keep(result.unsafe_get(0))

@parameter
fn bench_simd_int32_10k() raises:
    var lhs = _make_array[int32](10_000)
    var rhs = _make_array[int32](10_000)
    var result = binary_simd[int32, _add[int32.native]](lhs, rhs)
    keep(result.unsafe_get(0))

def main() raises:
    var r1 = benchmark.run[bench_simd_int32_1k]()
    print("int32 1k:", r1.mean("us"), "us")
    var r2 = benchmark.run[bench_simd_int32_10k]()
    print("int32 10k:", r2.mean("us"), "us")

"""GPU benchmarks for arithmetic kernels.

Compares GPU add vs CPU SIMD add across array sizes.

Run with: pixi run bench_gpu
"""

from sys import has_accelerator
from time import perf_counter_ns

from benchmark import keep
from gpu.host import DeviceContext

from marrow.arrays import PrimitiveArray
from marrow.builders import PrimitiveBuilder
from marrow.dtypes import int32, float32, DataType
from marrow.compute.arithmetic import _add_no_nulls
from marrow.compute.gpu import add as gpu_add


fn _make_array[T: DataType](size: Int) -> PrimitiveArray[T]:
    var b = PrimitiveBuilder[T](size)
    for i in range(size):
        b.unsafe_append(Scalar[T.native](i))
    return b^.freeze()


fn _bench_cpu[T: DataType, size: Int](iters: Int) -> Float64:
    """CPU SIMD add, mean microseconds per iteration."""
    var lhs = _make_array[T](size)
    var rhs = _make_array[T](size)
    for _ in range(3):
        _ = _add_no_nulls[T](lhs, rhs, size)
    var start = perf_counter_ns()
    for _ in range(iters):
        var result = _add_no_nulls[T](lhs, rhs, size)
        keep(result.unsafe_get(0))
    return Float64(perf_counter_ns() - start) / Float64(iters) / 1000.0


fn _bench_gpu[
    T: DataType, size: Int
](iters: Int, ctx: DeviceContext) raises -> Float64:
    """GPU add, mean microseconds per iteration (includes upload, excludes download)."""
    var lhs = _make_array[T](size)
    var rhs = _make_array[T](size)
    # Warmup
    for _ in range(3):
        _ = gpu_add[T](lhs, rhs, ctx)
    ctx.synchronize()
    var start = perf_counter_ns()
    for _ in range(iters):
        var result = gpu_add[T](lhs, rhs, ctx)
        keep(len(result))
    ctx.synchronize()
    return Float64(perf_counter_ns() - start) / Float64(iters) / 1000.0


fn _bench_gpu_preloaded[
    T: DataType, size: Int
](iters: Int, ctx: DeviceContext) raises -> Float64:
    """GPU add with data pre-loaded on device (kernel-only), mean us/iter."""
    var lhs = _make_array[T](size).to_device(ctx)
    var rhs = _make_array[T](size).to_device(ctx)
    # Warmup
    for _ in range(3):
        _ = gpu_add[T](lhs, rhs, ctx)
    ctx.synchronize()
    var start = perf_counter_ns()
    for _ in range(iters):
        var result = gpu_add[T](lhs, rhs, ctx)
        keep(len(result))
    ctx.synchronize()
    return Float64(perf_counter_ns() - start) / Float64(iters) / 1000.0


fn _print_row(kernel: String, dtype: String, size: Int, us: Float64):
    var size_str = String(size)
    var pad1 = " " * (12 - len(kernel))
    var pad2 = " " * (10 - len(dtype))
    var pad3 = " " * (10 - len(size_str))
    print(kernel + pad1 + dtype + pad2 + size_str + pad3 + String(us) + " us")


def main():
    if not has_accelerator():
        print("No GPU accelerator found, skipping GPU benchmarks.")
        return

    var ctx = DeviceContext()
    print("=== CPU vs GPU add benchmarks ===\n")
    print("kernel      dtype     size      mean (us/iter)")
    print("------      -----     ----      --------------")

    comptime sizes = (1_000, 10_000, 100_000, 1_000_000, 10_000_000)
    comptime iters = (1000, 1000, 100, 10, 5)

    # --- int32 ---
    comptime for i in range(5):
        _print_row(
            "cpu", "int32", sizes[i], _bench_cpu[int32, sizes[i]](iters[i])
        )

    comptime for i in range(5):
        _print_row(
            "gpu", "int32", sizes[i], _bench_gpu[int32, sizes[i]](iters[i], ctx)
        )

    comptime for i in range(5):
        _print_row(
            "gpu-preload",
            "int32",
            sizes[i],
            _bench_gpu_preloaded[int32, sizes[i]](iters[i], ctx),
        )

    # --- float32 ---
    comptime for i in range(5):
        _print_row(
            "cpu", "float32", sizes[i], _bench_cpu[float32, sizes[i]](iters[i])
        )

    comptime for i in range(5):
        _print_row(
            "gpu",
            "float32",
            sizes[i],
            _bench_gpu[float32, sizes[i]](iters[i], ctx),
        )

    comptime for i in range(5):
        _print_row(
            "gpu-preload",
            "float32",
            sizes[i],
            _bench_gpu_preloaded[float32, sizes[i]](iters[i], ctx),
        )

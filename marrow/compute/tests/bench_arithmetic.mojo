"""Benchmarks for arithmetic kernel variants.

Compares:
  - simd: SIMD-vectorized add via binary_simd helper
  - dispatch: Public add() with runtime dispatch (no nulls)
  - nulls: Public add() where inputs have 10% nulls

Across array sizes (1k-1M) and dtypes (int32, float64).

Run with: pixi run bench

NOTE: Uses manual timing because the Bench framework crashes with multiple
parametric instantiations of `add` (Mojo codegen bug). The Bench framework
is used for the SIMD-only section which doesn't trigger the bug.
"""

from benchmark import Bench, BenchConfig, Bencher, BenchId, keep
from time import perf_counter_ns

from marrow.arrays import PrimitiveArray
from marrow.builders import PrimitiveBuilder
from marrow.dtypes import int32, float64, DataType
from marrow.compute.kernels.add import _add_simd, add
from marrow.compute.kernels import binary_simd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


fn _make_array[T: DataType](size: Int) -> PrimitiveArray[T]:
    """Build a non-null array of `size` elements [0, 1, 2, ...]."""
    var b = PrimitiveBuilder[T](size)
    for i in range(size):
        b.unsafe_append(Scalar[T.native](i))
    return b.finish()


fn _make_array_with_nulls[T: DataType](size: Int) -> PrimitiveArray[T]:
    """Build an array with 10% nulls (every 10th element)."""
    var b = PrimitiveBuilder[T](size)
    for i in range(size):
        b.unsafe_set(i, Scalar[T.native](i))
        if i % 10 != 0:
            b.bitmap.unsafe_set(i, True)
    b.length = size
    return b.finish()


# ---------------------------------------------------------------------------
# Timed benchmark helpers
# ---------------------------------------------------------------------------


fn _bench_simd[T: DataType, size: Int](iters: Int) -> Float64:
    """Benchmark direct SIMD add, return mean microseconds per iteration."""
    var lhs = _make_array[T](size)
    var rhs = _make_array[T](size)
    for _ in range(3):
        _ = binary_simd[T, _add_simd[T.native]](lhs, rhs)
    var start = perf_counter_ns()
    for _ in range(iters):
        var result = binary_simd[T, _add_simd[T.native]](lhs, rhs)
        keep(result.unsafe_get(0))
    return Float64(perf_counter_ns() - start) / Float64(iters) / 1000.0


fn _bench_dispatch[T: DataType, size: Int](iters: Int) raises -> Float64:
    """Benchmark public add() with non-null inputs, return mean us/iter."""
    var lhs = _make_array[T](size)
    var rhs = _make_array[T](size)
    for _ in range(3):
        _ = add[T](lhs, rhs)
    var start = perf_counter_ns()
    for _ in range(iters):
        var result = add[T](lhs, rhs)
        keep(result.unsafe_get(0))
    return Float64(perf_counter_ns() - start) / Float64(iters) / 1000.0


fn _bench_nulls[T: DataType, size: Int](iters: Int) raises -> Float64:
    """Benchmark add() with 10% nulls, return mean us/iter."""
    var lhs = _make_array_with_nulls[T](size)
    var rhs = _make_array_with_nulls[T](size)
    for _ in range(3):
        _ = add[T](lhs, rhs)
    var start = perf_counter_ns()
    for _ in range(iters):
        var result = add[T](lhs, rhs)
        keep(result.unsafe_get(1))
    return Float64(perf_counter_ns() - start) / Float64(iters) / 1000.0


# ---------------------------------------------------------------------------
# Bench framework (SIMD only — Bench + `add` triggers a Mojo codegen crash
# when multiple parametric sizes coexist in the same binary)
# ---------------------------------------------------------------------------


@parameter
fn bench_add_simd[T: DataType, size: Int](mut b: Bencher) raises:
    """SIMD-vectorized add via binary_simd helper."""
    var lhs = _make_array[T](size)
    var rhs = _make_array[T](size)

    @always_inline
    @parameter
    fn call_fn() raises:
        var result = binary_simd[T, _add_simd[T.native]](lhs, rhs)
        keep(result.unsafe_get(0))

    b.iter[call_fn]()


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


fn _print_row(kernel: String, dtype: String, size: Int, us: Float64):
    var size_str = String(size)
    var pad1 = " " * (16 - len(kernel))
    var pad2 = " " * (10 - len(dtype))
    var pad3 = " " * (10 - len(size_str))
    print(kernel + pad1 + dtype + pad2 + size_str + pad3 + String(us) + " us")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=== Arithmetic add benchmarks ===\n")
    print("kernel          dtype     size      mean (us/iter)")
    print("------          -----     ----      --------------")

    # --- int32 ---
    _print_row("simd", "int32", 1_000, _bench_simd[int32, 1_000](1000))
    _print_row("simd", "int32", 10_000, _bench_simd[int32, 10_000](1000))
    _print_row("simd", "int32", 100_000, _bench_simd[int32, 100_000](100))
    _print_row("simd", "int32", 1_000_000, _bench_simd[int32, 1_000_000](10))

    _print_row("dispatch", "int32", 1_000, _bench_dispatch[int32, 1_000](1000))
    _print_row(
        "dispatch", "int32", 10_000, _bench_dispatch[int32, 10_000](1000)
    )
    _print_row(
        "dispatch", "int32", 100_000, _bench_dispatch[int32, 100_000](100)
    )
    _print_row(
        "dispatch", "int32", 1_000_000, _bench_dispatch[int32, 1_000_000](10)
    )

    _print_row("nulls", "int32", 1_000, _bench_nulls[int32, 1_000](1000))
    _print_row("nulls", "int32", 10_000, _bench_nulls[int32, 10_000](1000))
    _print_row("nulls", "int32", 100_000, _bench_nulls[int32, 100_000](100))
    _print_row("nulls", "int32", 1_000_000, _bench_nulls[int32, 1_000_000](10))

    # --- float64 ---
    _print_row("simd", "float64", 1_000, _bench_simd[float64, 1_000](1000))
    _print_row("simd", "float64", 10_000, _bench_simd[float64, 10_000](1000))
    _print_row("simd", "float64", 100_000, _bench_simd[float64, 100_000](100))
    _print_row(
        "simd", "float64", 1_000_000, _bench_simd[float64, 1_000_000](10)
    )

    _print_row(
        "dispatch", "float64", 1_000, _bench_dispatch[float64, 1_000](1000)
    )
    _print_row(
        "dispatch", "float64", 10_000, _bench_dispatch[float64, 10_000](1000)
    )
    _print_row(
        "dispatch", "float64", 100_000, _bench_dispatch[float64, 100_000](100)
    )
    _print_row(
        "dispatch",
        "float64",
        1_000_000,
        _bench_dispatch[float64, 1_000_000](10),
    )

    _print_row("nulls", "float64", 1_000, _bench_nulls[float64, 1_000](1000))
    _print_row("nulls", "float64", 10_000, _bench_nulls[float64, 10_000](1000))
    _print_row("nulls", "float64", 100_000, _bench_nulls[float64, 100_000](100))
    _print_row(
        "nulls", "float64", 1_000_000, _bench_nulls[float64, 1_000_000](10)
    )

    # --- Bench framework table (SIMD only, proper statistical measurement) ---
    print("\n=== Bench framework (statistical, SIMD only) ===\n")
    var m = Bench(BenchConfig(num_repetitions=3))
    m.bench_function[bench_add_simd[int32, 1_000]](
        BenchId("simd[int32]", "1000")
    )
    m.bench_function[bench_add_simd[int32, 10_000]](
        BenchId("simd[int32]", "10000")
    )
    m.bench_function[bench_add_simd[int32, 100_000]](
        BenchId("simd[int32]", "100000")
    )
    m.bench_function[bench_add_simd[int32, 1_000_000]](
        BenchId("simd[int32]", "1000000")
    )
    m.bench_function[bench_add_simd[float64, 1_000]](
        BenchId("simd[float64]", "1000")
    )
    m.bench_function[bench_add_simd[float64, 10_000]](
        BenchId("simd[float64]", "10000")
    )
    m.bench_function[bench_add_simd[float64, 100_000]](
        BenchId("simd[float64]", "100000")
    )
    m.bench_function[bench_add_simd[float64, 1_000_000]](
        BenchId("simd[float64]", "1000000")
    )
    m.dump_report()

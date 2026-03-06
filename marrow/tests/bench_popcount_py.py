"""Benchmark popcount (sum) and AND throughput for PyArrow and Polars.

Uses pairs of sizes to cancel fixed Python call overhead:
    t(N) = overhead + N / throughput
    throughput = (N2 - N1) / (t(N2) - t(N1))

Prints both raw per-call times and overhead-corrected throughput.
"""

import time
import pyarrow as pa
import pyarrow.compute as pc
import polars as pl

SIZES = [10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]
ITERS = 200


def measure(fn, iters: int) -> float:
    for _ in range(10):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def measure_polars_sum(arr: pa.Array, iters: int) -> float:
    # Polars caches the sum on a series, so construct a fresh series each call
    # from the pre-built PyArrow array (near zero-copy) and subtract alloc cost.
    with_sum = measure(lambda: pl.from_arrow(arr).sum(), iters)
    alloc    = measure(lambda: pl.from_arrow(arr), iters)
    return with_sum - alloc


def print_results(label: str, times: dict):
    # Estimate overhead and corrected throughput from consecutive size pairs.
    pairs = []
    for s1, s2 in zip(SIZES, SIZES[1:]):
        t1, t2 = times[s1], times[s2]
        throughput = (s2 - s1) / (t2 - t1)
        overhead = t1 - s1 / throughput
        pairs.append((s1, s2, throughput, overhead))
    avg_overhead = sum(p[3] for p in pairs) / len(pairs)

    print(f"\n{label}")
    print(f"\n{'size':>12}  {'mean (µs)':>12}  {'overhead (µs)':>14}  {'raw GElems/s':>14}  {'corrected GElems/s':>20}")
    print("-" * 80)
    for size in SIZES:
        t = times[size]
        corrected = max(t - avg_overhead, 1e-12)
        print(f"{size:>12,}  {t*1e6:>12.2f}  {avg_overhead*1e6:>14.2f}  {(size/t)/1e9:>14.1f}  {(size/corrected)/1e9:>20.1f}")

    print(f"\n{'pair':>24}  {'throughput (GElems/s)':>22}  {'overhead (µs)':>14}")
    print("-" * 66)
    for s1, s2, throughput, overhead in pairs:
        print(f"{s1:>10,} → {s2:>10,}  {throughput/1e9:>22.1f}  {overhead*1e6:>14.2f}")


def main():
    data  = {size: [i % 2 == 0 for i in range(size)] for size in SIZES}
    data2 = {size: [i % 3 == 0 for i in range(size)] for size in SIZES}

    pa_a  = {size: pa.array(data[size],  type=pa.bool_()) for size in SIZES}
    pa_b  = {size: pa.array(data2[size], type=pa.bool_()) for size in SIZES}
    pl_a  = {size: pl.from_arrow(pa_a[size]) for size in SIZES}
    pl_b  = {size: pl.from_arrow(pa_b[size]) for size in SIZES}

    # --- popcount (sum) ---
    pa_sum_times = {size: measure(lambda a=pa_a[size]: pc.sum(a), ITERS) for size in SIZES}
    pl_sum_times = {size: measure_polars_sum(pa_a[size], ITERS) for size in SIZES}

    print("\n=== POPCOUNT (sum of boolean array) ===")
    print_results("PyArrow  pc.sum(bool_array)", pa_sum_times)
    print_results("Polars   Series.sum()", pl_sum_times)

    # --- AND ---
    pa_and_times = {size: measure(lambda a=pa_a[size], b=pa_b[size]: pc.and_(a, b), ITERS) for size in SIZES}
    pl_and_times = {size: measure(lambda a=pl_a[size], b=pl_b[size]: a & b,          ITERS) for size in SIZES}

    print("\n=== BITWISE AND ===")
    print_results("PyArrow  pc.and_(a, b)", pa_and_times)
    print_results("Polars   a & b", pl_and_times)

    # --- INVERT ---
    pa_inv_times = {size: measure(lambda a=pa_a[size]: pc.invert(a), ITERS) for size in SIZES}
    pl_inv_times = {size: measure(lambda a=pl_a[size]: ~a,           ITERS) for size in SIZES}

    print("\n=== BITWISE INVERT ===")
    print_results("PyArrow  pc.invert(a)", pa_inv_times)
    print_results("Polars   ~a", pl_inv_times)


if __name__ == "__main__":
    main()

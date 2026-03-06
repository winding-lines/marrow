"""Compare BitSet (stdlib, stack-allocated, compile-time size, Int64 words)
vs Bitmap (marrow, heap-allocated, runtime size, uint8 SIMD accumulators).

BitSet uses `comptime for` over all words — fully unrolled at compile time.
This is ideal for small, fixed sizes but generates O(words) instructions
inline, which thrashes the instruction cache for large sizes.

Bitmap uses a runtime SIMD loop with interleaved uint8 accumulators,
which scales to arbitrarily large sizes with constant code size.

Sizes tested: 1k, 10k, 100k bits.

Run with: pixi run bench_bitset
"""

from benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
    keep,
)
from collections import BitSet

from marrow.bitmap import Bitmap, BitmapBuilder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


fn _make_bitmap(size: Int) -> Bitmap:
    var b = BitmapBuilder.alloc(size)
    var i = 0
    while i < size:
        b.set_bit(i, True)
        i += 2
    return b.finish(size)


# ---------------------------------------------------------------------------
# BitSet benchmarks (one function per size to avoid parameterized-fn bug)
# ---------------------------------------------------------------------------


@parameter
fn bench_bitset_count_1k(mut b: Bencher) raises:
    var bs = BitSet[1_000]()
    for i in range(0, 1_000, 2):
        bs.set(i)

    @always_inline
    @parameter
    fn call_fn():
        keep(len(bs))

    b.iter[call_fn]()


@parameter
fn bench_bitset_count_10k(mut b: Bencher) raises:
    var bs = BitSet[10_000]()
    for i in range(0, 10_000, 2):
        bs.set(i)

    @always_inline
    @parameter
    fn call_fn():
        keep(len(bs))

    b.iter[call_fn]()


@parameter
fn bench_bitset_count_100k(mut b: Bencher) raises:
    var bs = BitSet[100_000]()
    for i in range(0, 100_000, 2):
        bs.set(i)

    @always_inline
    @parameter
    fn call_fn():
        keep(len(bs))

    b.iter[call_fn]()


# ---------------------------------------------------------------------------
# Bitmap benchmarks (same sizes)
# ---------------------------------------------------------------------------


@parameter
fn bench_bitmap_count(mut b: Bencher, size: Int) raises:
    var bm = _make_bitmap(size)

    @always_inline
    @parameter
    fn call_fn():
        keep(bm.count_set_bits())

    b.iter[call_fn]()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() raises:
    var m = Bench(BenchConfig(num_repetitions=3))

    m.bench_function[bench_bitset_count_1k](
        BenchId("bitset_count", "1k"),
        [ThroughputMeasure(BenchMetric.elements, 1_000)],
    )
    m.bench_function[bench_bitset_count_10k](
        BenchId("bitset_count", "10k"),
        [ThroughputMeasure(BenchMetric.elements, 10_000)],
    )
    m.bench_function[bench_bitset_count_100k](
        BenchId("bitset_count", "100k"),
        [ThroughputMeasure(BenchMetric.elements, 100_000)],
    )

    comptime sizes = (1_000, 10_000, 100_000)
    comptime labels = ("1k", "10k", "100k")
    comptime for si in range(3):
        m.bench_with_input[Int, bench_bitmap_count](
            BenchId("bitmap_count", labels[si]),
            sizes[si],
            [ThroughputMeasure(BenchMetric.elements, sizes[si])],
        )

    m.dump_report()

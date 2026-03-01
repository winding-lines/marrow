"""Batch cosine similarity kernel — CPU SIMD + optional GPU."""

import math
from sys import size_of
from sys.info import simd_byte_width

from gpu.host import DeviceContext

from marrow.arrays import PrimitiveArray, FixedSizeListArray
from marrow.buffers import MemorySpace
from marrow.builders import PrimitiveBuilder
from marrow.dtypes import DataType


fn _cosine_similarity_no_nulls[
    T: DataType
](
    vectors: FixedSizeListArray,
    query: PrimitiveArray[T],
    n_vectors: Int,
    dim: Int,
) -> PrimitiveArray[T]:
    """SIMD-vectorized batch cosine similarity.

    For each of n_vectors vectors of dimension dim, computes:
        score[i] = dot(v_i, q) / (||v_i|| * ||q||)
    """
    comptime native = T.native
    comptime width = simd_byte_width() // size_of[native]()

    var result = PrimitiveBuilder[T](n_vectors)
    result.bitmap.unsafe_range_set(0, n_vectors, True)
    var op = result.buffer.ptr.bitcast[Scalar[native]]()

    # Flat values pointer from the child array
    ref child = vectors.values
    var vp = (
        child.buffers[0].ptr.bitcast[Scalar[native]]()
        + child.offset
        + child.buffers[0].offset
    )
    var qp = (
        query.buffer.ptr.bitcast[Scalar[native]]()
        + query.offset
        + query.buffer.offset
    )

    # Pre-compute query norm
    var norm_q = Scalar[native](0)
    var j = 0
    while j + width <= dim:
        var q = (qp + j).load[width=width]()
        norm_q += (q * q).reduce_add()
        j += width
    while j < dim:
        norm_q += qp[j] * qp[j]
        j += 1
    var query_norm = math.sqrt(norm_q)

    # Compute similarity for each vector
    for i in range(n_vectors):
        var offset = (vectors.offset + i) * dim
        var dot = Scalar[native](0)
        var norm_v = Scalar[native](0)

        j = 0
        while j + width <= dim:
            var v = (vp + offset + j).load[width=width]()
            var q = (qp + j).load[width=width]()
            dot += (v * q).reduce_add()
            norm_v += (v * v).reduce_add()
            j += width
        while j < dim:
            dot += vp[offset + j] * qp[j]
            norm_v += vp[offset + j] * vp[offset + j]
            j += 1

        var denom = math.sqrt(norm_v) * query_norm
        if denom > 0:
            op[i] = dot / denom
        else:
            op[i] = Scalar[native](0)

    result.length = n_vectors
    return result^.freeze()


fn cosine_similarity[
    T: DataType
](
    vectors: FixedSizeListArray,
    query: PrimitiveArray[T],
) raises -> PrimitiveArray[T]:
    """Batch cosine similarity on CPU: N vectors vs one query → N scores.

    Args:
        vectors: FixedSizeListArray of N vectors, each of dimension D.
        query: PrimitiveArray[T] of D elements (the query vector).

    Returns:
        PrimitiveArray[T] of N cosine similarity scores in [-1, 1].
    """
    var dim = vectors.dtype.size
    var n_vectors = len(vectors)

    if len(query) != dim:
        raise Error(
            "cosine_similarity: query length {} != vector dim {}".format(
                len(query), dim
            )
        )

    return _cosine_similarity_no_nulls[T](vectors, query, n_vectors, dim)


fn cosine_similarity[
    T: DataType
](
    vectors: FixedSizeListArray[MemorySpace.DEVICE],
    query: PrimitiveArray[T, MemorySpace.DEVICE],
    ctx: DeviceContext,
) raises -> PrimitiveArray[T, MemorySpace.DEVICE]:
    """GPU-accelerated batch cosine similarity on device-resident data.

    Args:
        vectors: Device-resident FixedSizeListArray of N vectors.
        query: Device-resident PrimitiveArray[T] of D elements.
        ctx: GPU device context.

    Returns:
        Device-resident PrimitiveArray[T] of N cosine similarity scores.
        Call `.to_host(ctx)` to download to CPU memory.
    """
    var dim = vectors.dtype.size
    var n_vectors = len(vectors)

    if len(query) != dim:
        raise Error(
            "cosine_similarity: query length {} != vector dim {}".format(
                len(query), dim
            )
        )

    from .gpu import _cosine_similarity_gpu

    return _cosine_similarity_gpu[T](
        vectors, query, n_vectors, dim, ctx
    )

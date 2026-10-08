/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/block/block_merge_sort.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cuda/iterator>
#include <cuda/std/tuple>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace cudf::detail {

// Experimental radix key: null rank is most significant, followed by the prefix.
// Row IDs are values, never part of the radix key.
struct string_radix_prefix_key {
  uint64_t hi;
  uint32_t lo;
  uint32_t null_rank;
  __host__ __device__ bool operator==(string_radix_prefix_key const& rhs) const
  {
    return hi == rhs.hi && lo == rhs.lo && null_rank == rhs.null_rank;
  }
};

template <int Bytes>
struct string_radix_decomposer {
  __host__ __device__ auto operator()(string_radix_prefix_key& key) const
  {
    if constexpr (Bytes <= 8) {
      return cuda::std::tuple<uint32_t&, uint64_t&>{key.null_rank, key.hi};
    } else {
      return cuda::std::tuple<uint32_t&, uint64_t&, uint32_t&>{key.null_rank, key.hi, key.lo};
    }
  }
};

struct radix_segment {
  size_type begin;
  size_type end;
};

// Locate the segment owning a tile by upper-bounding the scanned tile offsets.
__device__ inline size_type radix_tile_segment(size_type tile,
                                               size_type const* offsets,
                                               size_type count)
{
  size_type lo = 0, hi = count;
  while (lo < hi) {
    auto const mid = lo + (hi - lo) / 2;
    if (offsets[mid + 1] <= tile)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}

template <int TileSize>
__global__ void radix_prepare_tiles(radix_segment const* segments,
                                    size_type count,
                                    size_type* tile_counts,
                                    size_type* max_length)
{
  auto const i = static_cast<size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < count) {
    auto const length = segments[i].end - segments[i].begin;
    tile_counts[i]    = 1 + (length - 1) / TileSize;
    atomicMax(max_length, length);
  }
  if (i == count) tile_counts[i] = 0;
}

template <int Threads, int Items, bool Stable, typename Comparator>
__global__ void radix_segment_block_sort(size_type const* input,
                                         size_type* output,
                                         radix_segment const* segments,
                                         size_type const* tile_offsets,
                                         size_type segment_count,
                                         Comparator comp)
{
  constexpr int tile_size = Threads * Items;
  auto const tile         = static_cast<size_type>(blockIdx.x);
  auto const segment      = radix_tile_segment(tile, tile_offsets, segment_count);
  auto const begin        = segments[segment].begin + (tile - tile_offsets[segment]) * tile_size;
  auto const valid        = min(tile_size, segments[segment].end - begin);
  using sorter            = cub::BlockMergeSort<size_type, Threads, Items>;
  __shared__ typename sorter::TempStorage storage;
  size_type rows[Items];
  for (int j = 0; j < Items; ++j) {
    auto const offset = static_cast<int>(threadIdx.x) * Items + j;
    rows[j]           = offset < valid ? input[begin + offset] : -1;
  }
  if constexpr (Stable)
    sorter(storage).StableSort(rows, comp, valid, -1);
  else
    sorter(storage).Sort(rows, comp, valid, -1);
  for (int j = 0; j < Items; ++j) {
    auto const offset = static_cast<int>(threadIdx.x) * Items + j;
    if (offset < valid) output[begin + offset] = rows[j];
  }
}

// Each thread finds one position on the stable merge path and emits one row.
// Left-run equality precedes right-run equality, preserving stable radix order.
template <int Threads, int Items, typename Comparator>
__global__ void radix_segment_merge(size_type const* input,
                                    size_type* output,
                                    radix_segment const* segments,
                                    size_type const* tile_offsets,
                                    size_type segment_count,
                                    int64_t run_length,
                                    Comparator comp)
{
  constexpr int tile_size = Threads * Items;
  auto const tile         = static_cast<size_type>(blockIdx.x);
  auto const segment      = radix_tile_segment(tile, tile_offsets, segment_count);
  auto const seg          = segments[segment];
  auto const local_tile   = tile - tile_offsets[segment];
  auto const tile_begin   = static_cast<int64_t>(local_tile) * tile_size;
  auto const pair_begin   = tile_begin / (2 * run_length) * (2 * run_length);
  auto const length       = static_cast<int64_t>(seg.end - seg.begin);
  auto const left_count   = min(run_length, length - pair_begin);
  auto const right_count  = max(int64_t{0}, min(run_length, length - pair_begin - run_length));
  auto const* left        = input + seg.begin + pair_begin;
  auto const* right       = left + left_count;
  if (right_count == 0) {
    for (int j = 0; j < Items; ++j) {
      auto const position = tile_begin + threadIdx.x + j * Threads;
      if (position < length) output[seg.begin + position] = input[seg.begin + position];
    }
    return;
  }
  // Partition each output tile once; thread searches then stay within this tile's inputs.
  __shared__ int64_t partition_a[2];
  if (threadIdx.x < 2) {
    auto const diagonal = min(tile_begin + threadIdx.x * tile_size, length) - pair_begin;
    int64_t lo          = max(int64_t{0}, diagonal - right_count);
    int64_t hi          = min(diagonal, left_count);
    while (lo < hi) {
      auto const a = lo + (hi - lo) / 2;
      auto const b = diagonal - a;
      if (b > 0 && a < left_count && !comp(right[b - 1], left[a]))
        lo = a + 1;
      else
        hi = a;
    }
    partition_a[threadIdx.x] = lo;
  }
  __syncthreads();
  auto const first_b = tile_begin - pair_begin - partition_a[0];
  auto const last_b  = min(tile_begin + tile_size, length) - pair_begin - partition_a[1];
  for (int j = 0; j < Items; ++j) {
    auto const position = tile_begin + threadIdx.x + j * Threads;
    if (position >= length) continue;
    auto const diagonal = position - pair_begin;
    int64_t lo          = max(partition_a[0], diagonal - last_b);
    int64_t hi          = min(partition_a[1], diagonal - first_b);
    while (lo < hi) {
      auto const a = lo + (hi - lo) / 2;
      auto const b = diagonal - a;
      if (b > 0 && a < left_count && !comp(right[b - 1], left[a]))
        lo = a + 1;
      else
        hi = a;
    }
    auto const a = lo, b = diagonal - a;
    output[seg.begin + position] =
      a < left_count && (b >= right_count || !comp(right[b], left[a])) ? left[a] : right[b];
  }
}

template <int Bytes, int Threads, int Items, bool Stable, typename Extractor, typename Comparator>
void radix_prefix_refine(size_type size,
                         size_type* output,
                         Extractor extractor,
                         Comparator comparator,
                         uint32_t null_rank,
                         bool use_rle,
                         cuda::stream_ref stream)
{
  // Evaluation path: boundary selection requires three host synchronizations; RLE requires two.
  CUDF_EXPECTS(size < std::numeric_limits<size_type>::max(), "Radix boundary count overflow");
  auto const mr     = cudf::get_current_device_resource_ref();
  auto const policy = rmm::exec_policy_nosync(stream, mr);
  auto rows         = cuda::counting_iterator<size_type>{0};
  rmm::device_uvector<string_radix_prefix_key> keys_in(size, stream, mr);
  rmm::device_uvector<string_radix_prefix_key> keys_out(size, stream, mr);
  rmm::device_uvector<size_type> scratch(size, stream, mr);
  thrust::transform(policy, rows, rows + size, keys_in.begin(), extractor);
  thrust::sequence(policy, scratch.begin(), scratch.end(), size_type{0});
  std::size_t bytes = 0;
  auto radix_sort   = [&](void* storage) {
    return cub::DeviceRadixSort::SortPairs(storage,
                                           bytes,
                                           keys_in.data(),
                                           keys_out.data(),
                                           scratch.data(),
                                           output,
                                           size,
                                           string_radix_decomposer<Bytes>{},
                                           0,
                                           Bytes <= 8 ? 65 : 97,
                                           stream.get());
  };
  CUDF_CUDA_TRY(radix_sort(nullptr));
  {
    rmm::device_buffer temp(bytes, stream, mr);
    CUDF_CUDA_TRY(radix_sort(temp.data()));
  }
  keys_in.resize(0, stream);

  rmm::device_uvector<size_type> count(1, stream, mr);
  rmm::device_uvector<radix_segment> segments(0, stream, mr);
  auto const* sorted_keys = keys_out.data();
  if (use_rle) {
    auto const max_runs = size / 2;
    rmm::device_uvector<size_type> offsets(max_runs, stream, mr);
    rmm::device_uvector<size_type> lengths(max_runs, stream, mr);
    rmm::device_uvector<size_type> run_count(1, stream, mr);
    auto encode = [&](void* storage) {
      return cub::DeviceRunLengthEncode::NonTrivialRuns(storage,
                                                        bytes,
                                                        sorted_keys,
                                                        offsets.data(),
                                                        lengths.data(),
                                                        run_count.data(),
                                                        size,
                                                        stream.get());
    };
    bytes = 0;
    CUDF_CUDA_TRY(encode(nullptr));
    {
      rmm::device_buffer temp(bytes, stream, mr);
      CUDF_CUDA_TRY(encode(temp.data()));
    }
    segments.resize(max_runs, stream);
    auto const* starts = offsets.data();
    auto const* sizes  = lengths.data();
    auto const* runs   = run_count.data();
    auto make_segment  = [starts, sizes, runs] __device__(size_type i) -> radix_segment {
      if (i >= *runs) return {0, 0};
      return {starts[i], starts[i] + sizes[i]};
    };
    auto segment_input = cuda::make_transform_iterator(rows, make_segment);
    auto non_null      = [sorted_keys, null_rank] __device__(radix_segment seg) -> bool {
      return seg.end > seg.begin && sorted_keys[seg.begin].null_rank != null_rank;
    };
    auto filter = [&](void* storage) {
      return cub::DeviceSelect::If(storage,
                                   bytes,
                                   segment_input,
                                   segments.data(),
                                   count.data(),
                                   max_runs,
                                   non_null,
                                   stream.get());
    };
    bytes = 0;
    CUDF_CUDA_TRY(filter(nullptr));
    {
      rmm::device_buffer temp(bytes, stream, mr);
      CUDF_CUDA_TRY(filter(temp.data()));
    }
  } else {
    rmm::device_uvector<size_type> boundaries(static_cast<std::size_t>(size) + 1, stream, mr);
    auto is_boundary = [sorted_keys, size] __device__(size_type i) -> bool {
      return i == 0 || i == size || !(sorted_keys[i] == sorted_keys[i - 1]);
    };
    auto select_boundaries = [&](void* storage) {
      return cub::DeviceSelect::If(
        storage, bytes, rows, boundaries.data(), count.data(), size + 1, is_boundary, stream.get());
    };
    bytes = 0;
    CUDF_CUDA_TRY(select_boundaries(nullptr));
    {
      rmm::device_buffer temp(bytes, stream, mr);
      CUDF_CUDA_TRY(select_boundaries(temp.data()));
    }
    size_type boundary_count;
    CUDF_CUDA_TRY(
      cudf::detail::memcpy_async(&boundary_count, count.data(), sizeof(size_type), stream));
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
    segments.resize(boundary_count - 1, stream);
    auto const* starts = boundaries.data();
    auto make_segment  = [starts] __device__(size_type i) -> radix_segment {
      return radix_segment{starts[i], starts[i + 1]};
    };
    auto segment_input    = cuda::make_transform_iterator(rows, make_segment);
    auto needs_refinement = [sorted_keys, null_rank] __device__(radix_segment segment) -> bool {
      return segment.end - segment.begin > 1 && sorted_keys[segment.begin].null_rank != null_rank;
    };
    auto select_segments = [&](void* storage) {
      return cub::DeviceSelect::If(storage,
                                   bytes,
                                   segment_input,
                                   segments.data(),
                                   count.data(),
                                   boundary_count - 1,
                                   needs_refinement,
                                   stream.get());
    };
    bytes = 0;
    CUDF_CUDA_TRY(select_segments(nullptr));
    {
      rmm::device_buffer temp(bytes, stream, mr);
      CUDF_CUDA_TRY(select_segments(temp.data()));
    }
  }
  size_type segment_count;
  CUDF_CUDA_TRY(
    cudf::detail::memcpy_async(&segment_count, count.data(), sizeof(size_type), stream));
  CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
  if (segment_count == 0) return;

  constexpr int tile_size = Threads * Items;
  rmm::device_uvector<size_type> tile_counts(segment_count + 1, stream, mr);
  rmm::device_uvector<size_type> tile_offsets(segment_count + 1, stream, mr);
  CUDF_CUDA_TRY(cudaMemsetAsync(count.data(), 0, sizeof(size_type), stream.get()));
  radix_prepare_tiles<tile_size><<<(segment_count + 256) / 256, 256, 0, stream.get()>>>(
    segments.data(), segment_count, tile_counts.data(), count.data());
  CUDF_CUDA_TRY(cudaGetLastError());
  bytes = 0;
  CUDF_CUDA_TRY(cub::DeviceScan::ExclusiveSum(
    nullptr, bytes, tile_counts.data(), tile_offsets.data(), segment_count + 1, stream.get()));
  {
    rmm::device_buffer temp(bytes, stream, mr);
    CUDF_CUDA_TRY(cub::DeviceScan::ExclusiveSum(temp.data(),
                                                bytes,
                                                tile_counts.data(),
                                                tile_offsets.data(),
                                                segment_count + 1,
                                                stream.get()));
  }
  size_type tile_count, max_length;
  CUDF_CUDA_TRY(cudf::detail::memcpy_async(
    &tile_count, tile_offsets.data() + segment_count, sizeof(size_type), stream));
  CUDF_CUDA_TRY(cudf::detail::memcpy_async(&max_length, count.data(), sizeof(size_type), stream));
  CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
  CUDF_CUDA_TRY(
    cudf::detail::memcpy_async(scratch.data(), output, sizeof(size_type) * size, stream));
  radix_segment_block_sort<Threads, Items, Stable><<<tile_count, Threads, 0, stream.get()>>>(
    output, scratch.data(), segments.data(), tile_offsets.data(), segment_count, comparator);
  CUDF_CUDA_TRY(cudaGetLastError());
  auto* src = scratch.data();
  auto* dst = output;
  for (int64_t run = tile_size; run < max_length; run *= 2) {
    radix_segment_merge<Threads, Items><<<tile_count, Threads, 0, stream.get()>>>(
      src, dst, segments.data(), tile_offsets.data(), segment_count, run, comparator);
    CUDF_CUDA_TRY(cudaGetLastError());
    std::swap(src, dst);
  }
  if (src != output) {
    CUDF_CUDA_TRY(cudf::detail::memcpy_async(output, src, sizeof(size_type) * size, stream));
  }
}
}  // namespace cudf::detail

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

#include <cooperative_groups.h>
#include <cub/block/block_merge_sort.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/tuple>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

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

// Nonnullable prefixes carry no null rank or padding through radix sorting.
struct string_radix_prefix12 {
  uint32_t hi;
  uint32_t mid;
  uint32_t lo;
  __host__ __device__ bool operator==(string_radix_prefix12 const& rhs) const
  {
    return hi == rhs.hi && mid == rhs.mid && lo == rhs.lo;
  }
};
static_assert(sizeof(string_radix_prefix12) == 12);

template <typename Key>
__device__ bool radix_key_is_null(Key const& key, uint32_t null_rank)
{
  if constexpr (std::is_same_v<Key, string_radix_prefix_key>)
    return key.null_rank == null_rank;
  else
    return false;
}

template <int Bytes>
struct string_radix_decomposer {
  template <typename Key>
  __host__ __device__ auto operator()(Key& key) const
  {
    if constexpr (std::is_same_v<Key, string_radix_prefix12>) {
      return cuda::std::tuple<uint32_t&, uint32_t&, uint32_t&>{key.hi, key.mid, key.lo};
    } else if constexpr (Bytes <= 8) {
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
                                    size_type* max_length,
                                    size_type const* device_count)
{
  auto const i            = static_cast<size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  auto const actual_count = device_count == nullptr ? count : *device_count;
  if (i < actual_count) {
    auto const length = segments[i].end - segments[i].begin;
    tile_counts[i]    = 1 + (length - 1) / TileSize;
    atomicMax(max_length, length);
  }
  if (i >= actual_count && i <= count) tile_counts[i] = 0;
}

template <int Threads, int Items, bool Stable, typename Comparator>
__device__ void radix_block_sort_tile(size_type const* input,
                                      size_type* output,
                                      radix_segment const* segments,
                                      size_type const* tile_offsets,
                                      size_type segment_count,
                                      size_type tile,
                                      Comparator comp)
{
  constexpr int tile_size = Threads * Items;
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

template <int Threads, int Items, bool Stable, typename Comparator>
__global__ void radix_segment_block_sort(size_type const* input,
                                         size_type* output,
                                         radix_segment const* segments,
                                         size_type const* tile_offsets,
                                         size_type segment_count,
                                         Comparator comp)
{
  radix_block_sort_tile<Threads, Items, Stable>(
    input, output, segments, tile_offsets, segment_count, static_cast<size_type>(blockIdx.x), comp);
}

// Each thread finds one position on the stable merge path and emits one row.
// Left-run equality precedes right-run equality, preserving stable radix order.
template <int Threads, int Items, typename Comparator>
__device__ void radix_merge_tile(size_type const* input,
                                 size_type* output,
                                 radix_segment const* segments,
                                 size_type const* tile_offsets,
                                 size_type segment_count,
                                 size_type tile,
                                 int64_t run_length,
                                 Comparator comp)
{
  constexpr int tile_size = Threads * Items;
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

template <int Threads, int Items, typename Comparator>
__global__ void radix_segment_merge(size_type const* input,
                                    size_type* output,
                                    radix_segment const* segments,
                                    size_type const* tile_offsets,
                                    size_type segment_count,
                                    int64_t run_length,
                                    Comparator comp)
{
  radix_merge_tile<Threads, Items>(input,
                                   output,
                                   segments,
                                   tile_offsets,
                                   segment_count,
                                   static_cast<size_type>(blockIdx.x),
                                   run_length,
                                   comp);
}

// A block reserves segment IDs and tile offsets together in a packed prefix sum.
// Reservations may reorder whole segments; row order inside each segment is unchanged.
template <int TileSize, typename Key>
__global__ void radix_compact_runs(size_type const* starts,
                                   size_type const* lengths,
                                   size_type const* run_count,
                                   Key const* sorted_keys,
                                   uint32_t null_rank,
                                   radix_segment* segments,
                                   size_type* tile_offsets,
                                   unsigned long long* metadata)
{
  using Scan   = cub::BlockScan<unsigned long long, 256>;
  using Reduce = cub::BlockReduce<size_type, 256>;
  __shared__ union {
    typename Scan::TempStorage scan;
    typename Reduce::TempStorage reduce;
  } temp;
  __shared__ unsigned long long reservation;
  auto const count = *run_count;
  for (size_type base = blockIdx.x * 256; base < count; base += gridDim.x * 256) {
    auto const i      = base + threadIdx.x;
    auto const begin  = i < count ? starts[i] : 0;
    auto const length = i < count ? lengths[i] : 0;
    bool const valid  = length > 1 && !radix_key_is_null(sorted_keys[begin], null_rank);
    auto const tiles  = valid ? 1 + (length - 1) / TileSize : 0;
    auto const packed = (static_cast<unsigned long long>(tiles) << 32) | (valid ? 1ULL : 0ULL);
    unsigned long long prefix, total;
    Scan(temp.scan).ExclusiveSum(packed, prefix, total);
    __syncthreads();
    auto const max_length = Reduce(temp.reduce).Reduce(valid ? length : 0, cuda::maximum<>{});
    if (threadIdx.x == 0) {
      reservation = total != 0 ? atomicAdd(metadata, total) : 0;
      atomicMax(reinterpret_cast<size_type*>(metadata + 1), max_length);
    }
    __syncthreads();
    if (valid) {
      auto const location   = reservation + prefix;
      auto const segment    = static_cast<uint32_t>(location);
      segments[segment]     = {begin, begin + length};
      tile_offsets[segment] = static_cast<size_type>(location >> 32);
    }
    __syncthreads();
  }
}

__global__ void radix_finish_offsets(size_type* offsets, unsigned long long const* metadata)
{
  auto const packed                      = *metadata;
  offsets[static_cast<uint32_t>(packed)] = static_cast<size_type>(packed >> 32);
}

// Schedule block sorts freely before the lighter cooperative merge kernel.
__global__ void radix_device_copy(size_type size,
                                  size_type const* input,
                                  size_type* output,
                                  unsigned long long const* metadata)
{
  if (static_cast<uint32_t>(metadata[0]) == 0) return;
  auto const stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (auto i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < size; i += stride)
    output[i] = input[i];
}

template <int Threads, int Items, bool Stable, typename Comparator>
__global__ void radix_device_block_sort(size_type const* input,
                                        size_type* output,
                                        radix_segment const* segments,
                                        size_type const* offsets,
                                        unsigned long long const* metadata,
                                        Comparator comp)
{
  auto const packed = metadata[0];
  auto const count  = static_cast<size_type>(static_cast<uint32_t>(packed));
  auto const tiles  = static_cast<size_type>(packed >> 32);
  for (size_type tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    radix_block_sort_tile<Threads, Items, Stable>(
      input, output, segments, offsets, count, tile, comp);
    __syncthreads();
  }
}

// Guarded launches retain the ordinary CUDA scheduler without host metadata.
template <int Threads, int Items, typename Comparator>
__global__ void radix_device_merge(size_type const* input,
                                   size_type* output,
                                   radix_segment const* segments,
                                   size_type const* offsets,
                                   unsigned long long const* metadata,
                                   int64_t run,
                                   Comparator comp)
{
  auto const packed = metadata[0];
  auto const count  = static_cast<size_type>(static_cast<uint32_t>(packed));
  if (count == 0 || run >= static_cast<int64_t>(metadata[1])) return;
  auto const tiles = static_cast<size_type>(packed >> 32);
  for (size_type tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    radix_merge_tile<Threads, Items>(input, output, segments, offsets, count, tile, run, comp);
    __syncthreads();
  }
}

template <int TileSize>
__global__ void radix_device_finish(size_type size,
                                    size_type const* scratch,
                                    size_type* output,
                                    unsigned long long const* metadata)
{
  if (static_cast<uint32_t>(metadata[0]) == 0) return;
  int levels = 0;
  for (int64_t run = TileSize; run < static_cast<int64_t>(metadata[1]); run *= 2)
    ++levels;
  if (levels % 2 != 0) return;  // Odd merge counts already leave the final result in output.
  auto const stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (auto i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < size; i += stride)
    output[i] = scratch[i];
}

template <int Threads, int Items, bool Stable, typename Comparator>
void radix_launch_device_schedule(size_type size,
                                  size_type* output,
                                  size_type* scratch,
                                  radix_segment const* segments,
                                  size_type const* offsets,
                                  unsigned long long const* metadata,
                                  Comparator comp,
                                  cuda::stream_ref stream)
{
  int device, multiprocessors;
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  CUDF_CUDA_TRY(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device));
  auto const blocks = std::min(multiprocessors * 32, std::max(1, size / 2));
  radix_device_copy<<<blocks, 256, 0, stream.get()>>>(size, output, scratch, metadata);
  CUDF_CUDA_TRY(cudaGetLastError());
  radix_device_block_sort<Threads, Items, Stable>
    <<<blocks, Threads, 0, stream.get()>>>(output, scratch, segments, offsets, metadata, comp);
  CUDF_CUDA_TRY(cudaGetLastError());
  auto* src = scratch;
  auto* dst = output;
  for (int64_t run = Threads * Items; run < size; run *= 2) {
    radix_device_merge<Threads, Items>
      <<<blocks, Threads, 0, stream.get()>>>(src, dst, segments, offsets, metadata, run, comp);
    CUDF_CUDA_TRY(cudaGetLastError());
    std::swap(src, dst);
  }
  radix_device_finish<Threads * Items>
    <<<blocks, 256, 0, stream.get()>>>(size, scratch, output, metadata);
  CUDF_CUDA_TRY(cudaGetLastError());
}

// All blocks remain resident; grid barriers replace host scheduling/readbacks.
template <int Threads, int Items, bool Stable, typename Comparator>
__global__ void radix_cooperative_refine(size_type size,
                                         size_type* output,
                                         size_type* scratch,
                                         radix_segment const* segments,
                                         size_type const* tile_offsets,
                                         unsigned long long const* metadata,
                                         Comparator comp)
{
  auto const packed        = metadata[0];
  auto const segment_count = static_cast<size_type>(static_cast<uint32_t>(packed));
  if (segment_count == 0) return;
  auto const tile_count = static_cast<size_type>(packed >> 32);
  auto const max_length = static_cast<size_type>(metadata[1]);
  auto const grid       = cooperative_groups::this_grid();
  auto const rank       = static_cast<int64_t>(blockIdx.x) * Threads + threadIdx.x;
  auto const stride     = static_cast<int64_t>(gridDim.x) * Threads;
  auto* src             = scratch;
  auto* dst             = output;
  for (int64_t run = Threads * Items; run < max_length; run *= 2) {
    for (size_type tile = blockIdx.x; tile < tile_count; tile += gridDim.x) {
      radix_merge_tile<Threads, Items>(
        src, dst, segments, tile_offsets, segment_count, tile, run, comp);
      __syncthreads();
    }
    grid.sync();
    auto* tmp = src;
    src       = dst;
    dst       = tmp;
  }
  if (src != output) {
    for (auto i = rank; i < size; i += stride)
      output[i] = src[i];
  }
}

template <int Threads, int Items, bool Stable, typename Comparator>
void radix_launch_cooperative(size_type size,
                              size_type* output,
                              size_type* scratch,
                              radix_segment const* segments,
                              size_type const* tile_offsets,
                              unsigned long long const* metadata,
                              Comparator comparator,
                              cuda::stream_ref stream)
{
  int device, supported, multiprocessors, blocks_per_sm;
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  CUDF_CUDA_TRY(cudaDeviceGetAttribute(&supported, cudaDevAttrCooperativeLaunch, device));
  CUDF_EXPECTS(supported, "Cooperative radix refinement requires cooperative launch support");
  CUDF_CUDA_TRY(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device));
  auto const work_blocks = std::min(multiprocessors * 16, std::max(1, size / 2));
  radix_device_copy<<<work_blocks, 256, 0, stream.get()>>>(size, output, scratch, metadata);
  CUDF_CUDA_TRY(cudaGetLastError());
  radix_device_block_sort<Threads, Items, Stable><<<work_blocks, Threads, 0, stream.get()>>>(
    output, scratch, segments, tile_offsets, metadata, comparator);
  CUDF_CUDA_TRY(cudaGetLastError());
  auto const kernel = radix_cooperative_refine<Threads, Items, Stable, Comparator>;
  CUDF_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, Threads, 0));
  auto const blocks = std::min(multiprocessors * blocks_per_sm, std::max(1, size / 2));
  CUDF_EXPECTS(blocks > 0, "No resident blocks available for cooperative refinement");
  void* args[] = {&size, &output, &scratch, &segments, &tile_offsets, &metadata, &comparator};
  CUDF_CUDA_TRY(
    cudaLaunchCooperativeKernel(kernel, dim3(blocks), dim3(Threads), args, 0, stream.get()));
}

template <int Threads, int Items, bool Stable, typename Comparator>
void radix_refine_segments(size_type size,
                           size_type* output,
                           size_type* scratch,
                           radix_segment const* segments,
                           size_type const* tile_offsets,
                           size_type segment_count,
                           size_type tile_count,
                           size_type max_length,
                           Comparator comparator,
                           cuda::stream_ref stream)
{
  if (segment_count == 0) return;
  constexpr int tile_size = Threads * Items;
  CUDF_CUDA_TRY(cudf::detail::memcpy_async(scratch, output, sizeof(size_type) * size, stream));
  radix_segment_block_sort<Threads, Items, Stable><<<tile_count, Threads, 0, stream.get()>>>(
    output, scratch, segments, tile_offsets, segment_count, comparator);
  CUDF_CUDA_TRY(cudaGetLastError());
  auto* src = scratch;
  auto* dst = output;
  for (int64_t run = tile_size; run < max_length; run *= 2) {
    radix_segment_merge<Threads, Items><<<tile_count, Threads, 0, stream.get()>>>(
      src, dst, segments, tile_offsets, segment_count, run, comparator);
    CUDF_CUDA_TRY(cudaGetLastError());
    std::swap(src, dst);
  }
  if (src != output) {
    CUDF_CUDA_TRY(cudf::detail::memcpy_async(output, src, sizeof(size_type) * size, stream));
  }
}

template <int Bytes, int Threads, int Items, bool Stable, typename Extractor, typename Comparator>
void radix_prefix_refine(size_type size,
                         size_type* output,
                         Extractor extractor,
                         Comparator comparator,
                         uint32_t null_rank,
                         bool use_rle,
                         bool device_metadata,
                         bool compact_runs,
                         int schedule,
                         cuda::stream_ref stream)
{
  // Device metadata mode prepares all scheduling information before one host readback.
  device_metadata = device_metadata && use_rle;
  CUDF_EXPECTS(size < std::numeric_limits<size_type>::max(), "Radix boundary count overflow");
  auto const mr     = cudf::get_current_device_resource_ref();
  auto const policy = rmm::exec_policy_nosync(stream, mr);
  auto rows         = cuda::counting_iterator<size_type>{0};
  using key_type    = decltype(extractor(size_type{0}));
  rmm::device_uvector<key_type> keys_in(size, stream, mr);
  rmm::device_uvector<key_type> keys_out(size, stream, mr);
  rmm::device_uvector<size_type> scratch(size, stream, mr);
  thrust::transform(policy, rows, rows + size, keys_in.begin(), extractor);
  thrust::sequence(policy, scratch.begin(), scratch.end(), size_type{0});
  std::size_t bytes = 0;
  auto radix_sort   = [&](void* storage) {
    if constexpr (std::is_same_v<key_type, uint64_t>) {
      return cub::DeviceRadixSort::SortPairs(storage,
                                             bytes,
                                             keys_in.data(),
                                             keys_out.data(),
                                             scratch.data(),
                                             output,
                                             size,
                                             0,
                                             64,
                                             stream.get());
    } else {
      return cub::DeviceRadixSort::SortPairs(
        storage,
        bytes,
        keys_in.data(),
        keys_out.data(),
        scratch.data(),
        output,
        size,
        string_radix_decomposer<Bytes>{},
        0,
        std::is_same_v<key_type, string_radix_prefix12> ? 96 : (Bytes <= 8 ? 65 : 97),
        stream.get());
    }
  };
  CUDF_CUDA_TRY(radix_sort(nullptr));
  {
    rmm::device_buffer temp(bytes, stream, mr);
    CUDF_CUDA_TRY(radix_sort(temp.data()));
  }
  keys_in.resize(0, stream);

  rmm::device_uvector<size_type> count(3, stream, mr);
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
    if (compact_runs) {
      rmm::device_uvector<size_type> tile_offsets(max_runs + 1, stream, mr);
      rmm::device_uvector<unsigned long long> metadata(2, stream, mr);
      CUDF_CUDA_TRY(cudaMemsetAsync(
        metadata.data(), 0, metadata.size() * sizeof(unsigned long long), stream.get()));
      radix_compact_runs<Threads * Items>
        <<<std::min(256, (max_runs + 255) / 256), 256, 0, stream.get()>>>(offsets.data(),
                                                                          lengths.data(),
                                                                          run_count.data(),
                                                                          sorted_keys,
                                                                          null_rank,
                                                                          segments.data(),
                                                                          tile_offsets.data(),
                                                                          metadata.data());
      CUDF_CUDA_TRY(cudaGetLastError());
      radix_finish_offsets<<<1, 1, 0, stream.get()>>>(tile_offsets.data(), metadata.data());
      CUDF_CUDA_TRY(cudaGetLastError());
      if (schedule == 2) {
        radix_launch_device_schedule<Threads, Items, Stable>(size,
                                                             output,
                                                             scratch.data(),
                                                             segments.data(),
                                                             tile_offsets.data(),
                                                             metadata.data(),
                                                             comparator,
                                                             stream);
        return;
      }
      if (schedule == 1) {
        radix_launch_cooperative<Threads, Items, Stable>(size,
                                                         output,
                                                         scratch.data(),
                                                         segments.data(),
                                                         tile_offsets.data(),
                                                         metadata.data(),
                                                         comparator,
                                                         stream);
        return;
      }
      unsigned long long host_metadata[2];
      CUDF_CUDA_TRY(
        cudf::detail::memcpy_async(host_metadata, metadata.data(), sizeof(host_metadata), stream));
      CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
      auto const segment_count = static_cast<size_type>(static_cast<uint32_t>(host_metadata[0]));
      auto const tile_count    = static_cast<size_type>(host_metadata[0] >> 32);
      auto const max_length    = static_cast<size_type>(host_metadata[1]);
      radix_refine_segments<Threads, Items, Stable>(size,
                                                    output,
                                                    scratch.data(),
                                                    segments.data(),
                                                    tile_offsets.data(),
                                                    segment_count,
                                                    tile_count,
                                                    max_length,
                                                    comparator,
                                                    stream);
      return;
    }
    auto const* starts = offsets.data();
    auto const* sizes  = lengths.data();
    auto const* runs   = run_count.data();
    auto make_segment  = [starts, sizes, runs] __device__(size_type i) -> radix_segment {
      if (i >= *runs) return {0, 0};
      return {starts[i], starts[i] + sizes[i]};
    };
    auto segment_input = cuda::make_transform_iterator(rows, make_segment);
    auto non_null      = [sorted_keys, null_rank] __device__(radix_segment seg) -> bool {
      return seg.end > seg.begin && !radix_key_is_null(sorted_keys[seg.begin], null_rank);
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
      return segment.end - segment.begin > 1 &&
             !radix_key_is_null(sorted_keys[segment.begin], null_rank);
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
  size_type segment_count = 0;
  if (!device_metadata) {
    CUDF_CUDA_TRY(
      cudf::detail::memcpy_async(&segment_count, count.data(), sizeof(size_type), stream));
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
    if (segment_count == 0) return;
  }

  constexpr int tile_size = Threads * Items;
  auto const capacity     = device_metadata ? size / 2 : segment_count;
  rmm::device_uvector<size_type> tile_counts(capacity + 1, stream, mr);
  rmm::device_uvector<size_type> tile_offsets(capacity + 1, stream, mr);
  CUDF_CUDA_TRY(cudaMemsetAsync(count.data() + 2, 0, sizeof(size_type), stream.get()));
  radix_prepare_tiles<tile_size>
    <<<(capacity + 256) / 256, 256, 0, stream.get()>>>(segments.data(),
                                                       capacity,
                                                       tile_counts.data(),
                                                       count.data() + 2,
                                                       device_metadata ? count.data() : nullptr);
  CUDF_CUDA_TRY(cudaGetLastError());
  bytes = 0;
  CUDF_CUDA_TRY(cub::DeviceScan::ExclusiveSum(
    nullptr, bytes, tile_counts.data(), tile_offsets.data(), capacity + 1, stream.get()));
  {
    rmm::device_buffer temp(bytes, stream, mr);
    CUDF_CUDA_TRY(cub::DeviceScan::ExclusiveSum(
      temp.data(), bytes, tile_counts.data(), tile_offsets.data(), capacity + 1, stream.get()));
  }
  CUDF_CUDA_TRY(cudf::detail::memcpy_async(
    count.data() + 1, tile_offsets.data() + capacity, sizeof(size_type), stream));
  size_type metadata[3];
  CUDF_CUDA_TRY(cudf::detail::memcpy_async(metadata, count.data(), sizeof(metadata), stream));
  CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
  segment_count         = metadata[0];
  auto const tile_count = metadata[1];
  auto const max_length = metadata[2];
  if (segment_count == 0) return;
  radix_refine_segments<Threads, Items, Stable>(size,
                                                output,
                                                scratch.data(),
                                                segments.data(),
                                                tile_offsets.data(),
                                                segment_count,
                                                tile_count,
                                                max_length,
                                                comparator,
                                                stream);
}
}  // namespace cudf::detail

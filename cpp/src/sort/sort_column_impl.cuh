/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "sort.hpp"
#include "sort_radix.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/row_operator/common_utils.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_merge_sort.cuh>
#include <cuda/iterator>
#include <cuda/std/execution>
#include <cuda/stream>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <type_traits>

namespace cudf {
namespace detail {

/**
 * @brief Extracts the first bytes of a string as an unsigned big-endian integer.
 *
 * Zero padding is order preserving. It can create a prefix tie between a short string and a
 * longer string containing zero bytes; string lengths resolve that case.
 */
template <typename PrefixKey, bool has_nulls>
struct string_prefix_extractor {
  static_assert(std::is_unsigned_v<PrefixKey>);
  static constexpr auto prefix_bytes  = static_cast<size_type>(sizeof(PrefixKey));
  static constexpr auto bits_per_byte = std::numeric_limits<uint8_t>::digits;

  __device__ PrefixKey operator()(size_type row) const
  {
    if constexpr (has_nulls) {
      if (d_column.is_null(row)) { return 0; }
    }

    auto const string = d_column.element<string_view>(row);
    PrefixKey prefix  = 0;
    for (size_type byte = 0; byte < prefix_bytes; ++byte) {
      prefix <<= bits_per_byte;
      if (byte < string.size_bytes()) {
        prefix |= static_cast<PrefixKey>(static_cast<uint8_t>(string.data()[byte]));
      }
    }
    return prefix;
  }

  column_device_view const d_column;
};

/**
 * @brief String comparator accelerated by a contiguous array of cached prefix keys.
 *
 * This optimization was inspired by Eiger (https://arxiv.org/abs/2607.04489), which optionally
 * caches four-byte prefixes based on runtime prefix-distribution statistics. This implementation
 * instead uses a fixed-width prefix (currently eight bytes) for every nontrivial single-column
 * string sort and does not perform Eiger's runtime profiling or algorithm selection.
 *
 * This implementation also right-pads short strings with zero bytes and resolves the resulting
 * prefix collisions using string lengths. Comparisons tied after a complete prefix resume at the
 * first uncached byte, and nullable and non-nullable inputs use separate comparator
 * specializations.
 */
template <typename PrefixKey, bool has_nulls>
struct string_prefix_comparator {
  static_assert(std::is_unsigned_v<PrefixKey>);
  static constexpr auto prefix_bytes = static_cast<size_type>(sizeof(PrefixKey));

  __device__ bool operator()(size_type lhs, size_type rhs)
  {
    if constexpr (has_nulls) {
      bool const lhs_null{d_column.is_null(lhs)};
      bool const rhs_null{d_column.is_null(rhs)};
      if (lhs_null || rhs_null) {
        return null_compare(lhs_null, rhs_null, null_precedence) ==
               (ascending ? weak_ordering::LESS : weak_ordering::GREATER);
      }
    }

    auto const lhs_prefix = prefixes[lhs];
    auto const rhs_prefix = prefixes[rhs];
    if (lhs_prefix != rhs_prefix) {
      return ascending ? lhs_prefix < rhs_prefix : lhs_prefix > rhs_prefix;
    }

    auto const left_element  = d_column.element<string_view>(lhs);
    auto const right_element = d_column.element<string_view>(rhs);
    auto const left_size     = left_element.size_bytes();
    auto const right_size    = right_element.size_bytes();
    if (left_size <= prefix_bytes or right_size <= prefix_bytes) {
      // Equal zero-padded prefixes prove that all bytes in the shorter value match and that any
      // represented bytes beyond it are zero. The shorter value is therefore lexicographically
      // smaller, while equal lengths prove equality without rereading either string.
      return ascending ? left_size < right_size : right_size < left_size;
    }

    // Both values contain a complete cached prefix, so resume comparison at the first byte not
    // represented by the key instead of rescanning known-equal bytes.
    auto const left_suffix =
      string_view{left_element.data() + prefix_bytes, left_element.size_bytes() - prefix_bytes};
    auto const right_suffix =
      string_view{right_element.data() + prefix_bytes, right_element.size_bytes() - prefix_bytes};
    return ascending ? left_suffix < right_suffix : right_suffix < left_suffix;
  }

  column_device_view const d_column;
  PrefixKey const* prefixes;
  bool ascending;
  null_order null_precedence{};
};

// Temporary evaluation selector (CUDF_STRING_SORT_VARIANT env var, not for merge):
//   0 = upstream comparator (no prefix cache)
//   1 = P: uint64_t prefixes[] gathered by row id (PR 24267 head)
//   2 = A: {uint64 prefix; int32 row; pad} sorted as the key
//   3 = B: {uint64 hi; uint32 lo; int32 row}, 12 cached bytes
//   4 = C: {uint64 prefix; int32 row; uint32 is_null}
//   5 = D: uint64 packing a 4-byte prefix, a null bit, and a 31-bit row id
inline int string_sort_variant()
{
  static int const variant = [] {
    auto const value = std::getenv("CUDF_STRING_SORT_VARIANT");
    return value == nullptr ? 1 : std::atoi(value);
  }();
  return variant;
}

/**
 * @brief Reads CUDF_STRING_SORT_IPT (merge-sort items per thread for 16 B keys; 0 = CUB default).
 */
inline int string_sort_items_per_thread()
{
  static int const items = [] {
    auto const value = std::getenv("CUDF_STRING_SORT_IPT");
    return value == nullptr ? 0 : std::atoi(value);
  }();
  return items;
}

template <int ItemsPerThread>
struct prefix_row_merge_sort_policy {
  __host__ __device__ constexpr auto operator()(cuda::compute_capability) const
    -> cub::MergeSortPolicy
  {
    return {256,
            ItemsPerThread,
            cub::BLOCK_LOAD_WARP_TRANSPOSE,
            cub::LOAD_DEFAULT,
            cub::BLOCK_STORE_WARP_TRANSPOSE};
  }
};

/**
 * @brief Loads `sizeof(Integer)` bytes of `str` starting at `offset` as a zero-padded big-endian
 * integer, using the same byte loop as `string_prefix_extractor`.
 */
template <typename Integer>
__device__ Integer load_big_endian(string_view const& str, size_type offset)
{
  constexpr auto width = static_cast<size_type>(sizeof(Integer));
  Integer value        = 0;
  for (size_type byte = 0; byte < width; ++byte) {
    value <<= std::numeric_limits<uint8_t>::digits;
    if (offset + byte < str.size_bytes()) {
      value |= static_cast<Integer>(static_cast<uint8_t>(str.data()[offset + byte]));
    }
  }
  return value;
}

/**
 * @brief Orders two non-null rows whose first `CachedBytes` zero-padded bytes are equal.
 */
template <size_type CachedBytes>
__device__ bool cached_prefix_tie_break(column_device_view const& d_column,
                                        size_type lhs,
                                        size_type rhs,
                                        bool ascending)
{
  auto const left_element  = d_column.element<string_view>(lhs);
  auto const right_element = d_column.element<string_view>(rhs);
  auto const left_size     = left_element.size_bytes();
  auto const right_size    = right_element.size_bytes();
  if (left_size <= CachedBytes or right_size <= CachedBytes) {
    return ascending ? left_size < right_size : right_size < left_size;
  }
  auto const left_suffix = string_view{left_element.data() + CachedBytes, left_size - CachedBytes};
  auto const right_suffix =
    string_view{right_element.data() + CachedBytes, right_size - CachedBytes};
  return ascending ? left_suffix < right_suffix : right_suffix < left_suffix;
}

__device__ inline bool null_less(bool lhs_null, bool rhs_null, bool ascending, null_order order)
{
  return null_compare(lhs_null, rhs_null, order) ==
         (ascending ? weak_ordering::LESS : weak_ordering::GREATER);
}

// Variant A
struct prefix_row_a {
  uint64_t prefix;
  size_type row;
  uint32_t padding;
};
static_assert(sizeof(prefix_row_a) == 16 and alignof(prefix_row_a) == 8);

template <bool has_nulls>
struct prefix_row_a_ops {
  static constexpr size_type cached_bytes = 8;
  using key_type                          = prefix_row_a;

  __device__ key_type operator()(size_type row) const
  {
    if constexpr (has_nulls) {
      if (d_column.is_null(row)) { return {0, row, 0}; }
    }
    return {load_big_endian<uint64_t>(d_column.element<string_view>(row), 0), row, 0};
  }

  __device__ bool operator()(key_type const& lhs, key_type const& rhs) const
  {
    if constexpr (has_nulls) {
      bool const lhs_null{d_column.is_null(lhs.row)};
      bool const rhs_null{d_column.is_null(rhs.row)};
      if (lhs_null || rhs_null) {
        return null_less(lhs_null, rhs_null, ascending, null_precedence);
      }
    }
    if (lhs.prefix != rhs.prefix) {
      return ascending ? lhs.prefix < rhs.prefix : lhs.prefix > rhs.prefix;
    }
    return cached_prefix_tie_break<cached_bytes>(d_column, lhs.row, rhs.row, ascending);
  }

  column_device_view const d_column;
  bool ascending;
  null_order null_precedence{};
};

// Variant B
struct prefix_row_b {
  uint64_t hi;
  uint32_t lo;
  size_type row;
};
static_assert(sizeof(prefix_row_b) == 16 and alignof(prefix_row_b) == 8);

template <bool has_nulls>
struct prefix_row_b_ops {
  static constexpr size_type cached_bytes = 12;
  using key_type                          = prefix_row_b;

  __device__ key_type operator()(size_type row) const
  {
    if constexpr (has_nulls) {
      if (d_column.is_null(row)) { return {0, 0, row}; }
    }
    auto const str = d_column.element<string_view>(row);
    return {load_big_endian<uint64_t>(str, 0), load_big_endian<uint32_t>(str, 8), row};
  }

  __device__ bool operator()(key_type const& lhs, key_type const& rhs) const
  {
    if constexpr (has_nulls) {
      bool const lhs_null{d_column.is_null(lhs.row)};
      bool const rhs_null{d_column.is_null(rhs.row)};
      if (lhs_null || rhs_null) {
        return null_less(lhs_null, rhs_null, ascending, null_precedence);
      }
    }
    if (lhs.hi != rhs.hi) { return ascending ? lhs.hi < rhs.hi : lhs.hi > rhs.hi; }
    if (lhs.lo != rhs.lo) { return ascending ? lhs.lo < rhs.lo : lhs.lo > rhs.lo; }
    return cached_prefix_tie_break<cached_bytes>(d_column, lhs.row, rhs.row, ascending);
  }

  column_device_view const d_column;
  bool ascending;
  null_order null_precedence{};
};

// Variant C
struct prefix_row_c {
  uint64_t prefix;
  size_type row;
  uint32_t is_null;
};
static_assert(sizeof(prefix_row_c) == 16 and alignof(prefix_row_c) == 8);

template <bool has_nulls>
struct prefix_row_c_ops {
  static constexpr size_type cached_bytes = 8;
  using key_type                          = prefix_row_c;

  __device__ key_type operator()(size_type row) const
  {
    if constexpr (has_nulls) {
      if (d_column.is_null(row)) { return {0, row, 1}; }
    }
    return {load_big_endian<uint64_t>(d_column.element<string_view>(row), 0), row, 0};
  }

  __device__ bool operator()(key_type const& lhs, key_type const& rhs) const
  {
    if constexpr (has_nulls) {
      if (lhs.is_null || rhs.is_null) {
        return null_less(lhs.is_null, rhs.is_null, ascending, null_precedence);
      }
    }
    if (lhs.prefix != rhs.prefix) {
      return ascending ? lhs.prefix < rhs.prefix : lhs.prefix > rhs.prefix;
    }
    return cached_prefix_tie_break<cached_bytes>(d_column, lhs.row, rhs.row, ascending);
  }

  column_device_view const d_column;
  bool ascending;
  null_order null_precedence{};
};

template <typename Key>
struct project_row {
  __device__ size_type operator()(Key const& key) const { return key.row; }
};

// Variant D
/**
 * @brief 8-byte sort key: bits 63-32 hold a 4-byte big-endian prefix, bit 31 is the null flag,
 * and bits 30-0 hold the row index.
 *
 * Row indices fit in 31 bits because `size_type` is a signed 32-bit integer.
 */
struct packed_prefix_row {
  static constexpr uint64_t null_bit = uint64_t{1} << 31;
  static constexpr uint64_t row_mask = 0x7FFF'FFFF;

  uint64_t bits;

  __device__ uint32_t prefix() const { return static_cast<uint32_t>(bits >> 32); }
  __device__ bool is_null() const { return (bits & null_bit) != 0; }
  __device__ size_type row() const { return static_cast<size_type>(bits & row_mask); }
};
static_assert(sizeof(packed_prefix_row) == 8 and alignof(packed_prefix_row) == 8);
static_assert(packed_prefix_row::row_mask ==
              static_cast<uint64_t>(std::numeric_limits<size_type>::max()));

template <bool has_nulls>
struct packed_prefix_row_ops {
  static constexpr size_type cached_bytes = 4;
  using key_type                          = packed_prefix_row;

  __device__ key_type operator()(size_type row) const
  {
    auto const row_bits = static_cast<uint64_t>(row);
    if constexpr (has_nulls) {
      if (d_column.is_null(row)) { return {key_type::null_bit | row_bits}; }
    }
    auto const prefix = load_big_endian<uint32_t>(d_column.element<string_view>(row), 0);
    return {(static_cast<uint64_t>(prefix) << 32) | row_bits};
  }

  __device__ bool operator()(key_type const& lhs, key_type const& rhs) const
  {
    if constexpr (has_nulls) {
      bool const lhs_null = lhs.is_null();
      bool const rhs_null = rhs.is_null();
      if (lhs_null || rhs_null) {
        return null_less(lhs_null, rhs_null, ascending, null_precedence);
      }
    }
    auto const lhs_prefix = lhs.prefix();
    auto const rhs_prefix = rhs.prefix();
    if (lhs_prefix != rhs_prefix) {
      return ascending ? lhs_prefix < rhs_prefix : lhs_prefix > rhs_prefix;
    }
    return cached_prefix_tie_break<cached_bytes>(d_column, lhs.row(), rhs.row(), ascending);
  }

  column_device_view const d_column;
  bool ascending;
  null_order null_precedence{};
};

template <>
struct project_row<packed_prefix_row> {
  __device__ size_type operator()(packed_prefix_row const& key) const { return key.row(); }
};

/**
 * @brief Comparator functor needed for single column sort.
 *
 * @tparam Column element type.
 */
template <typename T>
struct simple_comparator {
  __device__ bool operator()(size_type lhs, size_type rhs)
  {
    if (has_nulls) {
      bool const lhs_null{d_column.is_null(lhs)};
      bool const rhs_null{d_column.is_null(rhs)};
      if (lhs_null || rhs_null) {
        return null_compare(lhs_null, rhs_null, null_precedence) ==
               (ascending ? weak_ordering::LESS : weak_ordering::GREATER);
      }
    }

    auto const left_element  = d_column.element<T>(lhs);
    auto const right_element = d_column.element<T>(rhs);
    return relational_compare(left_element, right_element) ==
           (ascending ? weak_ordering::LESS : weak_ordering::GREATER);
  }
  column_device_view const d_column;
  bool has_nulls;
  bool ascending;
  null_order null_precedence{};
};

template <sort_method method>
struct column_sorted_order_fn {
 private:
  template <typename Comparator>
  void merge_sort(mutable_column_view& indices, Comparator comp, cuda::stream_ref stream)
  {
    auto in_keys  = cuda::counting_iterator<cudf::size_type>{0};
    auto out_keys = indices.begin<size_type>();
    auto env      = cuda::std::execution::env{
      cuda::std::execution::prop{cuda::get_stream_t{}, stream},
      cuda::std::execution::prop{cuda::mr::get_memory_resource_t{},
                                 cudf::get_current_device_resource_ref()}};
    if constexpr (method == sort_method::STABLE) {
      CUDF_CUDA_TRY(
        cub::DeviceMergeSort::StableSortKeysCopy(in_keys, out_keys, indices.size(), comp, env));
    } else {
      CUDF_CUDA_TRY(
        cub::DeviceMergeSort::SortKeysCopy(in_keys, out_keys, indices.size(), comp, env));
    }
  }

  template <typename PrefixKey, bool has_nulls>
  void prefix_sorted_order_impl(column_view const& input,
                                column_device_view const& keys,
                                mutable_column_view& indices,
                                bool ascending,
                                null_order null_precedence,
                                cuda::stream_ref stream)
  {
    auto prefixes =
      rmm::device_uvector<PrefixKey>(input.size(), stream, cudf::get_current_device_resource_ref());
    auto rows = cuda::counting_iterator<cudf::size_type>{0};
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      rows,
                      rows + input.size(),
                      prefixes.begin(),
                      string_prefix_extractor<PrefixKey, has_nulls>{keys});

    auto comp = string_prefix_comparator<PrefixKey, has_nulls>{
      keys, prefixes.data(), ascending, null_precedence};
    merge_sort(indices, comp, stream);
  }

  template <typename Key, typename Ops, typename Tuning>
  void in_place_merge_sort(rmm::device_uvector<Key>& keys,
                           Ops const& ops,
                           Tuning const& tuning,
                           cuda::stream_ref stream)
  {
    auto const env =
      cuda::std::execution::env{cuda::std::execution::prop{cuda::get_stream_t{}, stream}, tuning};
    auto tmp_bytes    = std::size_t{0};
    auto sort_keys_fn = [&](void* tmp_stg) {
      if constexpr (method == sort_method::STABLE) {
        return cub::DeviceMergeSort::StableSortKeys(
          tmp_stg, tmp_bytes, keys.begin(), keys.size(), ops, env);
      } else {
        return cub::DeviceMergeSort::SortKeys(
          tmp_stg, tmp_bytes, keys.begin(), keys.size(), ops, env);
      }
    };
    CUDF_CUDA_TRY(sort_keys_fn(nullptr));
    auto tmp_stg = rmm::device_buffer(tmp_bytes, stream, cudf::get_current_device_resource_ref());
    CUDF_CUDA_TRY(sort_keys_fn(tmp_stg.data()));
  }

  template <template <bool> class Ops, bool has_nulls>
  void prefix_row_sorted_order_impl(column_device_view const& keys,
                                    mutable_column_view& indices,
                                    bool ascending,
                                    null_order null_precedence,
                                    cuda::stream_ref stream)
  {
    using ops_type  = Ops<has_nulls>;
    using key_type  = typename ops_type::key_type;
    auto const ops  = ops_type{keys, ascending, null_precedence};
    auto const size = indices.size();
    auto const mr   = cudf::get_current_device_resource_ref();
    auto sort_keys  = rmm::device_uvector<key_type>(size, stream, mr);
    auto rows       = cuda::counting_iterator<cudf::size_type>{0};
    thrust::transform(
      rmm::exec_policy_nosync(stream, mr), rows, rows + size, sort_keys.begin(), ops);

    switch (string_sort_items_per_thread()) {
      case 0: in_place_merge_sort(sort_keys, ops, cuda::std::execution::env{}, stream); break;
      case 8:
        in_place_merge_sort(
          sort_keys, ops, cuda::execution::tune(prefix_row_merge_sort_policy<8>{}), stream);
        break;
      case 11:
        in_place_merge_sort(
          sort_keys, ops, cuda::execution::tune(prefix_row_merge_sort_policy<11>{}), stream);
        break;
      default: CUDF_FAIL("Unsupported CUDF_STRING_SORT_IPT");
    }

    thrust::transform(rmm::exec_policy_nosync(stream, mr),
                      sort_keys.begin(),
                      sort_keys.end(),
                      indices.begin<size_type>(),
                      project_row<key_type>{});
  }

  template <bool has_nulls>
  void dispatch_variant(column_view const& input,
                        column_device_view const& keys,
                        mutable_column_view& indices,
                        bool ascending,
                        null_order null_precedence,
                        cuda::stream_ref stream)
  {
    switch (string_sort_variant()) {
      case 1:
        prefix_sorted_order_impl<uint64_t, has_nulls>(
          input, keys, indices, ascending, null_precedence, stream);
        break;
      case 2:
        prefix_row_sorted_order_impl<prefix_row_a_ops, has_nulls>(
          keys, indices, ascending, null_precedence, stream);
        break;
      case 3:
        prefix_row_sorted_order_impl<prefix_row_b_ops, has_nulls>(
          keys, indices, ascending, null_precedence, stream);
        break;
      case 4:
        prefix_row_sorted_order_impl<prefix_row_c_ops, has_nulls>(
          keys, indices, ascending, null_precedence, stream);
        break;
      case 5:
        prefix_row_sorted_order_impl<packed_prefix_row_ops, has_nulls>(
          keys, indices, ascending, null_precedence, stream);
        break;
      default: CUDF_FAIL("Unknown CUDF_STRING_SORT_VARIANT");
    }
  }

  template <typename PrefixKey>
  void prefix_sorted_order(column_view const& input,
                           mutable_column_view& indices,
                           bool ascending,
                           null_order null_precedence,
                           cuda::stream_ref stream)
  {
    // A non-null strings column with no chars buffer contains only empty strings. Checking the
    // buffer pointer avoids the host synchronization required to read the terminal offset.
    auto const all_values_equal =
      not input.has_nulls() and strings_column_view{input}.chars_begin(stream) == nullptr;
    if (input.size() < 2 or input.null_count() == input.size() or all_values_equal) {
      thrust::sequence(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                       indices.begin<size_type>(),
                       indices.end<size_type>(),
                       size_type{0});
      return;
    }

    auto keys = column_device_view::create(input, stream);
    if (input.has_nulls()) {
      dispatch_variant<true>(input, *keys, indices, ascending, null_precedence, stream);
    } else {
      dispatch_variant<false>(input, *keys, indices, ascending, null_precedence, stream);
    }
  }

 public:
  /**
   * @brief Sorts a single column with a relationally comparable type.
   *
   * This is used when a comparator is required.
   *
   * @param input Column to sort
   * @param indices Output sorted indices
   * @param ascending True if sort order is ascending
   * @param null_precedence How null rows are to be ordered
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  template <typename T>
  void sorted_order(column_view const& input,
                    mutable_column_view& indices,
                    bool ascending,
                    null_order null_precedence,
                    cuda::stream_ref stream)
  {
    if constexpr (std::is_same_v<T, string_view>) {
      if (string_sort_variant() != 0) {
        prefix_sorted_order<uint64_t>(input, indices, ascending, null_precedence, stream);
        return;
      }
    }
    {
      auto keys = column_device_view::create(input, stream);
      auto comp = simple_comparator<T>{*keys, input.has_nulls(), ascending, null_precedence};
      merge_sort(indices, comp, stream);
    }
  }

  template <typename T>
    requires(cudf::is_relationally_comparable<T, T>() and not cudf::is_dictionary<T>())
  void operator()(column_view const& input,
                  mutable_column_view& indices,
                  bool ascending,
                  null_order null_precedence,
                  cuda::stream_ref stream)
  {
    sorted_order<T>(input, indices, ascending, null_precedence, stream);
  }

  template <typename T>
    requires(not cudf::is_relationally_comparable<T, T>())
  void operator()(column_view const&, mutable_column_view&, bool, null_order, cuda::stream_ref)
  {
    CUDF_FAIL("Column type must be relationally comparable");
  }

  template <typename T>
    requires(is_dictionary<T>())
  void operator()(column_view const& input,
                  mutable_column_view& indices,
                  bool ascending,
                  null_order null_precedence,
                  cuda::stream_ref stream)
  {
    auto const keys = dictionary_column_view(input).keys();
    // For the keys we do an arg-sort of arg-sort to get the rank and use that as a map
    // to sort the indices in rank order.
    // First, get sorted-order of just the keys (slow but expect keys.size <<< indices.size)
    auto temp_mr = cudf::get_current_device_resource_ref();
    auto ordered_indices =
      cudf::detail::sorted_order<method>(keys, order::ASCENDING, null_precedence, stream, temp_mr);
    // Now, sort the ordered indices to get their ordered positions (very fast integer sort)
    ordered_indices = cudf::detail::sorted_order<method>(
      ordered_indices->view(), order::ASCENDING, null_precedence, stream, temp_mr);
    // And use the result as a map over the dictionary indices
    auto map = ordered_indices->view().template data<size_type>();
    auto itr = cudf::detail::indexalator_factory::make_input_iterator(
      dictionary_column_view(input).indices());
    auto mapped_indices = rmm::device_uvector<size_type>(input.size(), stream);
    thrust::gather(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   itr,
                   itr + input.size(),
                   map,
                   mapped_indices.begin());

    // Finally, sort-order the dictionary indices using mapped values
    auto mapped_view = column_view(data_type{type_to_id<size_type>()},
                                   input.size(),
                                   mapped_indices.data(),
                                   input.null_mask(),
                                   input.null_count());
    // these should be very fast since they are sorting integers
    if (input.has_nulls()) {
      sorted_order<size_type>(mapped_view, indices, ascending, null_precedence, stream);
    } else {
      sorted_order_radix(mapped_view, indices, ascending, stream);
    }
  }
};

}  // namespace detail
}  // namespace cudf

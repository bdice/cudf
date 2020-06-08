/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cudf/detail/utilities/hash_functions.cuh>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <hash/concurrent_unordered_multimap.cuh>

#include <algorithm>
#include <limits>
#include <memory>
#include <numeric>

namespace cudf {
namespace detail {
constexpr size_type MAX_JOIN_SIZE{std::numeric_limits<size_type>::max()};

constexpr int DEFAULT_JOIN_BLOCK_SIZE = 128;
constexpr int DEFAULT_JOIN_CACHE_SIZE = 128;
constexpr size_type JoinNoneValue     = -1;

using VectorPair = std::pair<rmm::device_vector<size_type>, rmm::device_vector<size_type>>;

using multimap_type =
  concurrent_unordered_multimap<hash_value_type,
                                size_type,
                                size_t,
                                std::numeric_limits<hash_value_type>::max(),
                                std::numeric_limits<size_type>::max(),
                                default_hash<hash_value_type>,
                                equal_to<hash_value_type>,
                                default_allocator<thrust::pair<hash_value_type, size_type>>>;

using row_hash = cudf::row_hasher<default_hash>;

using row_equality = cudf::row_equality_comparator<true>;

enum class join_kind { INNER_JOIN, LEFT_JOIN, FULL_JOIN, LEFT_SEMI_JOIN, LEFT_ANTI_JOIN };

inline bool is_trivial_join(table_view const& left,
                            table_view const& right,
                            std::vector<size_type> const& left_on,
                            std::vector<size_type> const& right_on,
                            join_kind join_type)
{
  // If there is nothing to join, then send empty table with all columns
  if (left_on.empty() || right_on.empty()) { return true; }

  // If left join and the left table is empty, return immediately
  if ((join_kind::LEFT_JOIN == join_type) && (0 == left.num_rows())) { return true; }

  // If Inner Join and either table is empty, return immediately
  if ((join_kind::INNER_JOIN == join_type) && ((0 == left.num_rows()) || (0 == right.num_rows()))) {
    return true;
  }

  // If left semi join (contains) and right table is empty,
  // return immediately
  if ((join_kind::LEFT_SEMI_JOIN == join_type) && (0 == right.num_rows())) { return true; }

  return false;
}

/* --------------------------------------------------------------------------*/
/**
 * @brief  Computes the trivial left join operation for the case when the
 * right table is empty. In this case all the valid indices of the left table
 * are returned with their corresponding right indices being set to
 * JoinNoneValue, i.e. -1.
 *
 * @param left  Table of left columns to join
 * @param stream CUDA stream used for device memory operations and kernel launches.
 *
 * @returns Join output indices vector pair
 */
/* ----------------------------------------------------------------------------*/
inline std::pair<rmm::device_vector<size_type>, rmm::device_vector<size_type>>
get_trivial_left_join_indices(table_view const& left, cudaStream_t stream)
{
  rmm::device_vector<size_type> left_indices(left.num_rows());
  thrust::sequence(
    rmm::exec_policy(stream)->on(stream), left_indices.begin(), left_indices.end(), 0);
  rmm::device_vector<size_type> right_indices(left.num_rows());
  thrust::fill(rmm::exec_policy(stream)->on(stream),
               right_indices.begin(),
               right_indices.end(),
               JoinNoneValue);
  return std::make_pair(std::move(left_indices), std::move(right_indices));
}

}  // namespace detail

}  // namespace cudf

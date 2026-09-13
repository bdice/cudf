/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/stream>

#include <memory>
#include <optional>
#include <utility>

namespace cudf::detail {

/**
 * @brief Materializes passing FULL join pairs and one unmatched entry for each row with no match.
 *
 * Shared by the AST and JIT predicate evaluators. Sentinel pairs are not matches; unmatched rows
 * are reconstructed after all candidate pairs have been considered.
 */
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
filter_full_join_indices(size_type left_num_rows,
                         size_type right_num_rows,
                         device_span<size_type const> left_indices,
                         device_span<size_type const> right_indices,
                         device_span<bool const> predicate_results,
                         std::optional<std::size_t> output_size,
                         cuda::stream_ref stream,
                         rmm::device_async_resource_ref mr);

}  // namespace cudf::detail

/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cudf/ast/linearizer.cuh>
#include <cudf/ast/operators.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

namespace cudf {

namespace ast {

template <typename Element>
__device__ Element
resolve_input_data_reference(const detail::device_data_reference device_data_reference,
                             const table_device_view table,
                             const cudf::detail::fixed_width_scalar_device_view_base* literals,
                             const std::int64_t* thread_intermediate_storage,
                             cudf::size_type row_index);

template <typename Element>
__device__ Element* resolve_output_data_reference(
  const detail::device_data_reference device_data_reference,
  const table_device_view table,
  mutable_column_device_view output_column,
  std::int64_t* thread_intermediate_storage,
  cudf::size_type row_index);

struct typed_operator_dispatch_functor {
  template <typename OperatorFunctor,
            typename LHS,
            typename RHS,
            typename Out = simt::std::invoke_result_t<OperatorFunctor, LHS, RHS>,
            std::enable_if_t<cudf::ast::is_valid_binary_op<OperatorFunctor, LHS, RHS>>* = nullptr>
  CUDA_HOST_DEVICE_CALLABLE decltype(auto) operator()(
    const table_device_view table,
    mutable_column_device_view output_column,
    const cudf::detail::fixed_width_scalar_device_view_base* literals,
    std::int64_t* thread_intermediate_storage,
    cudf::size_type row_index,
    const detail::device_data_reference lhs,
    const detail::device_data_reference rhs,
    const detail::device_data_reference output);

  template <typename OperatorFunctor,
            typename LHS,
            typename RHS,
            typename Out                                                                 = void,
            std::enable_if_t<!cudf::ast::is_valid_binary_op<OperatorFunctor, LHS, RHS>>* = nullptr>
  CUDA_HOST_DEVICE_CALLABLE decltype(auto) operator()(
    const table_device_view table,
    mutable_column_device_view output_column,
    const cudf::detail::fixed_width_scalar_device_view_base* literals,
    std::int64_t* thread_intermediate_storage,
    cudf::size_type row_index,
    const detail::device_data_reference lhs,
    const detail::device_data_reference rhs,
    const detail::device_data_reference output);
};

__device__ void operate(ast_operator op,
                        const table_device_view table,
                        mutable_column_device_view output_column,
                        std::int64_t* thread_intermediate_storage,
                        cudf::size_type row_index,
                        const detail::device_data_reference lhs,
                        const detail::device_data_reference rhs,
                        const detail::device_data_reference output);

__device__ void evaluate_row_expression(
  const table_device_view table,
  const detail::device_data_reference* data_references,
  const cudf::detail::fixed_width_scalar_device_view_base* literals,
  const ast_operator* operators,
  const cudf::size_type* operator_source_indices,
  cudf::size_type num_operators,
  cudf::size_type row_index,
  std::int64_t* thread_intermediate_storage,
  mutable_column_device_view output_column);

template <size_type block_size>
__launch_bounds__(block_size) __global__
  void compute_column_kernel(const table_device_view table,
                             const detail::device_data_reference* data_references,
                             const cudf::detail::fixed_width_scalar_device_view_base* literals,
                             const ast_operator* operators,
                             const cudf::size_type* operator_source_indices,
                             cudf::size_type num_operators,
                             cudf::size_type num_intermediates,
                             mutable_column_device_view output_column);

std::unique_ptr<column> compute_column(
  table_view const table,
  std::reference_wrapper<const expression> expr,
  cudaStream_t stream                 = 0,  // TODO use detail API
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

}  // namespace ast

}  // namespace cudf

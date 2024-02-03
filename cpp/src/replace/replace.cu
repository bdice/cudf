/*
 * Copyright 2018 BlazingDB, Inc.

 *     Copyright 2018 Cristhian Alberto Gonzales Castillo <cristhian@blazingdb.com>
 *     Copyright 2018 Alexander Ocsa <alexander@blazingdb.com>
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
/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/concatenate.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/replace.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/device_atomics.cuh>
#include <cudf/dictionary/detail/update_keys.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/replace.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>

#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/find.h>
#include <thrust/pair.h>
#include <thrust/tuple.h>

namespace {  // anonymous

static constexpr int BLOCK_SIZE = 256;

// return the new_value for output column at index `idx`
template <class T, bool replacement_has_nulls>
__device__ auto get_new_value(cudf::size_type idx,
                              T const* __restrict__ input_data,
                              T const* __restrict__ values_to_replace_begin,
                              T const* __restrict__ values_to_replace_end,
                              T const* __restrict__ d_replacement_values,
                              cudf::bitmask_type const* __restrict__ replacement_valid)
{
  auto found_ptr =
    thrust::find(thrust::seq, values_to_replace_begin, values_to_replace_end, input_data[idx]);
  T new_value{};
  bool output_is_valid{true};

  if (found_ptr != values_to_replace_end) {
    auto d    = thrust::distance(values_to_replace_begin, found_ptr);
    new_value = d_replacement_values[d];
    if (replacement_has_nulls) { output_is_valid = cudf::bit_is_set(replacement_valid, d); }
  } else {
    new_value = input_data[idx];
  }
  return thrust::make_pair(new_value, output_is_valid);
}

__device__ int get_new_string_value(cudf::size_type idx,
                                    cudf::column_device_view& input,
                                    cudf::column_device_view& values_to_replace,
                                    cudf::column_device_view&)
{
  cudf::string_view input_string = input.element<cudf::string_view>(idx);
  int match                      = -1;
  for (int i = 0; i < values_to_replace.size(); i++) {
    cudf::string_view value_string = values_to_replace.element<cudf::string_view>(i);
    if (input_string == value_string) {
      match = i;
      break;
    }
  }
  return match;
}

/**
 * @brief Kernel which does the first pass of strings replace.
 *
 * It computes the output null_mask, null_count, and the offsets.
 *
 * @param input The input column to replace strings in.
 * @param values_to_replace The string values to replace.
 * @param replacement The replacement values.
 * @param offsets The column which will contain the offsets of the new string column
 * @param indices Temporary column used to store the replacement indices
 * @param output_valid The output null_mask
 * @param output_valid_count The output valid count
 */
template <bool input_has_nulls, bool replacement_has_nulls>
CUDF_KERNEL void replace_strings_first_pass(cudf::column_device_view input,
                                            cudf::column_device_view values_to_replace,
                                            cudf::column_device_view replacement,
                                            cudf::mutable_column_device_view offsets,
                                            cudf::mutable_column_device_view indices,
                                            cudf::bitmask_type* output_valid,
                                            cudf::size_type* __restrict__ output_valid_count)
{
  cudf::size_type nrows = input.size();
  auto tid              = cudf::detail::grid_1d::global_thread_id();
  auto const stride     = cudf::detail::grid_1d::grid_stride();
  uint32_t active_mask  = 0xffff'ffffu;
  active_mask           = __ballot_sync(active_mask, tid < nrows);
  auto const lane_id{threadIdx.x % cudf::detail::warp_size};
  uint32_t valid_sum{0};

  while (tid < nrows) {
    auto const idx      = static_cast<cudf::size_type>(tid);
    bool input_is_valid = true;

    if (input_has_nulls) input_is_valid = input.is_valid_nocheck(idx);
    bool output_is_valid = input_is_valid;

    if (input_is_valid) {
      int result               = get_new_string_value(idx, input, values_to_replace, replacement);
      cudf::string_view output = (result == -1) ? input.element<cudf::string_view>(idx)
                                                : replacement.element<cudf::string_view>(result);
      offsets.data<cudf::size_type>()[idx] = output.size_bytes();
      indices.data<cudf::size_type>()[idx] = result;
      if (replacement_has_nulls && result != -1) {
        output_is_valid = replacement.is_valid_nocheck(result);
      }
    } else {
      offsets.data<cudf::size_type>()[idx] = 0;
      indices.data<cudf::size_type>()[idx] = -1;
    }

    uint32_t bitmask = __ballot_sync(active_mask, output_is_valid);
    if (0 == lane_id) {
      output_valid[cudf::word_index(idx)] = bitmask;
      valid_sum += __popc(bitmask);
    }

    tid += stride;
    active_mask = __ballot_sync(active_mask, tid < nrows);
  }

  // Compute total valid count for this block and add it to global count
  uint32_t block_valid_count = cudf::detail::single_lane_block_sum_reduce<BLOCK_SIZE, 0>(valid_sum);
  // one thread computes and adds to output_valid_count
  if (threadIdx.x == 0) {
    cudf::detail::atomic_add(output_valid_count, static_cast<cudf::size_type>(block_valid_count));
  }
}

/**
 * @brief Kernel which does the second pass of strings replace.
 *
 * It copies the string data needed from input and replacement into the new strings column chars
 * column.
 *
 * @param input The input column
 * @param replacement The replacement values
 * @param offsets The offsets column of the new strings column
 * @param strings The chars column of the new strings column
 * @param indices Temporary column used to store the replacement indices.
 */
template <bool input_has_nulls, bool replacement_has_nulls>
CUDF_KERNEL void replace_strings_second_pass(cudf::column_device_view input,
                                             cudf::column_device_view replacement,
                                             cudf::mutable_column_device_view offsets,
                                             char* strings,
                                             cudf::mutable_column_device_view indices)
{
  cudf::size_type nrows = input.size();
  auto tid              = cudf::detail::grid_1d::global_thread_id();
  auto const stride     = cudf::detail::grid_1d::grid_stride();

  while (tid < nrows) {
    auto const idx         = static_cast<cudf::size_type>(tid);
    auto const replace_idx = indices.element<cudf::size_type>(idx);
    bool output_is_valid   = true;
    bool input_is_valid    = true;

    if (input_has_nulls) {
      input_is_valid  = input.is_valid_nocheck(idx);
      output_is_valid = input_is_valid;
    }
    if (replacement_has_nulls && replace_idx != -1) {
      output_is_valid = replacement.is_valid_nocheck(replace_idx);
    }
    if (output_is_valid) {
      cudf::string_view output = (replace_idx == -1)
                                   ? input.element<cudf::string_view>(idx)
                                   : replacement.element<cudf::string_view>(replace_idx);
      std::memcpy(
        strings + offsets.data<cudf::size_type>()[idx], output.data(), output.size_bytes());
    }

    tid += stride;
  }
}

/**
 * @brief Kernel that replaces elements from `output_data` given the following
 *        rule: replace all `values_to_replace[i]` in [values_to_replace_begin`,
 *        `values_to_replace_end`) present in `output_data` with `d_replacement_values[i]`.
 *
 * @tparam input_has_nulls `true` if output column has valid mask, `false` otherwise
 * @tparam replacement_has_nulls `true` if replacement_values column has valid mask, `false`
 * otherwise The input_has_nulls and replacement_has_nulls template parameters allows us to
 * specialize this kernel for the different scenario for performance without writing different
 * kernel.
 *
 * @param[in] input_data Device array with the data to be modified
 * @param[in] input_valid Valid mask associated with input_data
 * @param[out] output_data Device array to store the data from input_data
 * @param[out] output_valid Valid mask associated with output_data
 * @param[out] output_valid_count #valid in output column
 * @param[in] nrows # rows in `output_data`
 * @param[in] values_to_replace_begin Device pointer to the beginning of the sequence
 * of old values to be replaced
 * @param[in] values_to_replace_end  Device pointer to the end of the sequence
 * of old values to be replaced
 * @param[in] d_replacement_values Device array with the new values
 * @param[in] replacement_valid Valid mask associated with d_replacement_values
 */
template <class T, bool input_has_nulls, bool replacement_has_nulls>
CUDF_KERNEL void replace_kernel(cudf::column_device_view input,
                                cudf::mutable_column_device_view output,
                                cudf::size_type* __restrict__ output_valid_count,
                                cudf::size_type nrows,
                                cudf::column_device_view values_to_replace,
                                cudf::column_device_view replacement)
{
  T* __restrict__ output_data = output.data<T>();

  auto tid          = cudf::detail::grid_1d::global_thread_id();
  auto const stride = cudf::detail::grid_1d::grid_stride();

  uint32_t active_mask = 0xffff'ffffu;
  active_mask          = __ballot_sync(active_mask, tid < nrows);
  auto const lane_id{threadIdx.x % cudf::detail::warp_size};
  uint32_t valid_sum{0};

  while (tid < nrows) {
    auto const idx = static_cast<cudf::size_type>(tid);
    bool output_is_valid{true};
    bool input_is_valid{true};
    if (input_has_nulls) {
      input_is_valid  = input.is_valid_nocheck(idx);
      output_is_valid = input_is_valid;
    }
    if (input_is_valid)
      thrust::tie(output_data[idx], output_is_valid) = get_new_value<T, replacement_has_nulls>(
        idx,
        input.data<T>(),
        values_to_replace.data<T>(),
        values_to_replace.data<T>() + values_to_replace.size(),
        replacement.data<T>(),
        replacement.null_mask());

    /* output valid counts calculations*/
    if (input_has_nulls or replacement_has_nulls) {
      uint32_t bitmask = __ballot_sync(active_mask, output_is_valid);
      if (0 == lane_id) {
        output.set_mask_word(cudf::word_index(idx), bitmask);
        valid_sum += __popc(bitmask);
      }
    }

    tid += stride;
    active_mask = __ballot_sync(active_mask, tid < nrows);
  }
  if (input_has_nulls or replacement_has_nulls) {
    // Compute total valid count for this block and add it to global count
    uint32_t block_valid_count =
      cudf::detail::single_lane_block_sum_reduce<BLOCK_SIZE, 0>(valid_sum);
    // one thread computes and adds to output_valid_count
    if (threadIdx.x == 0) {
      cudf::detail::atomic_add(output_valid_count, static_cast<cudf::size_type>(block_valid_count));
    }
  }
}

/**
 * @brief Functor called by the `type_dispatcher` in order to invoke and instantiate
 *        `replace_kernel` with the appropriate data types.
 */
struct replace_kernel_forwarder {
  template <typename col_type, std::enable_if_t<cudf::is_fixed_width<col_type>()>* = nullptr>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& input_col,
                                           cudf::column_view const& values_to_replace,
                                           cudf::column_view const& replacement_values,
                                           rmm::cuda_stream_view stream,
                                           rmm::mr::device_memory_resource* mr)
  {
    rmm::device_scalar<cudf::size_type> valid_counter(0, stream);
    cudf::size_type* valid_count = valid_counter.data();

    auto replace = [&] {
      if (input_col.has_nulls())
        return replacement_values.has_nulls() ? replace_kernel<col_type, true, true>
                                              : replace_kernel<col_type, true, false>;
      else
        return replacement_values.has_nulls() ? replace_kernel<col_type, false, true>
                                              : replace_kernel<col_type, false, false>;
    }();

    auto output = [&] {
      auto const mask_allocation_policy = input_col.has_nulls() || replacement_values.has_nulls()
                                            ? cudf::mask_allocation_policy::ALWAYS
                                            : cudf::mask_allocation_policy::NEVER;
      return cudf::detail::allocate_like(
        input_col, input_col.size(), mask_allocation_policy, stream, mr);
    }();

    auto output_view = output->mutable_view();
    auto grid        = cudf::detail::grid_1d{output_view.size(), BLOCK_SIZE, 1};

    auto device_in                 = cudf::column_device_view::create(input_col, stream);
    auto device_out                = cudf::mutable_column_device_view::create(output_view, stream);
    auto device_values_to_replace  = cudf::column_device_view::create(values_to_replace, stream);
    auto device_replacement_values = cudf::column_device_view::create(replacement_values, stream);

    replace<<<grid.num_blocks, BLOCK_SIZE, 0, stream.value()>>>(*device_in,
                                                                *device_out,
                                                                valid_count,
                                                                output_view.size(),
                                                                *device_values_to_replace,
                                                                *device_replacement_values);

    if (output_view.nullable()) {
      output->set_null_count(output->size() - valid_counter.value(stream));
    }
    return output;
  }

  template <typename col_type, std::enable_if_t<not cudf::is_fixed_width<col_type>()>* = nullptr>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const&,
                                           cudf::column_view const&,
                                           cudf::column_view const&,
                                           rmm::cuda_stream_view,
                                           rmm::mr::device_memory_resource*)
  {
    CUDF_FAIL("No specialization exists for this type");
  }
};

template <>
std::unique_ptr<cudf::column> replace_kernel_forwarder::operator()<cudf::string_view>(
  cudf::column_view const& input_col,
  cudf::column_view const& values_to_replace,
  cudf::column_view const& replacement_values,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  rmm::device_scalar<cudf::size_type> valid_counter(0, stream);
  cudf::size_type* valid_count = valid_counter.data();

  auto replace_first  = replace_strings_first_pass<true, false>;
  auto replace_second = replace_strings_second_pass<true, false>;
  if (input_col.has_nulls()) {
    if (replacement_values.has_nulls()) {
      replace_first  = replace_strings_first_pass<true, true>;
      replace_second = replace_strings_second_pass<true, true>;
    }
  } else {
    if (replacement_values.has_nulls()) {
      replace_first  = replace_strings_first_pass<false, true>;
      replace_second = replace_strings_second_pass<false, true>;
    } else {
      replace_first  = replace_strings_first_pass<false, false>;
      replace_second = replace_strings_second_pass<false, false>;
    }
  }

  // Create new offsets column to use in kernel
  std::unique_ptr<cudf::column> sizes =
    cudf::make_numeric_column(cudf::data_type{cudf::type_to_id<cudf::size_type>()},
                              input_col.size(),
                              cudf::mask_state::UNALLOCATED,
                              stream);
  std::unique_ptr<cudf::column> indices =
    cudf::make_numeric_column(cudf::data_type{cudf::type_to_id<cudf::size_type>()},
                              input_col.size(),
                              cudf::mask_state::UNALLOCATED,
                              stream);

  auto sizes_view   = sizes->mutable_view();
  auto indices_view = indices->mutable_view();

  auto device_in                = cudf::column_device_view::create(input_col, stream);
  auto device_values_to_replace = cudf::column_device_view::create(values_to_replace, stream);
  auto device_replacement       = cudf::column_device_view::create(replacement_values, stream);
  auto device_sizes             = cudf::mutable_column_device_view::create(sizes_view, stream);
  auto device_indices           = cudf::mutable_column_device_view::create(indices_view, stream);

  rmm::device_buffer valid_bits =
    cudf::detail::create_null_mask(input_col.size(), cudf::mask_state::UNINITIALIZED, stream, mr);

  // Call first pass kernel to get sizes in offsets
  cudf::detail::grid_1d grid{input_col.size(), BLOCK_SIZE, 1};
  replace_first<<<grid.num_blocks, BLOCK_SIZE, 0, stream.value()>>>(
    *device_in,
    *device_values_to_replace,
    *device_replacement,
    *device_sizes,
    *device_indices,
    reinterpret_cast<cudf::bitmask_type*>(valid_bits.data()),
    valid_count);

  auto [offsets, bytes] = cudf::detail::make_offsets_child_column(
    sizes_view.begin<cudf::size_type>(), sizes_view.end<cudf::size_type>(), stream, mr);
  auto offsets_view   = offsets->mutable_view();
  auto device_offsets = cudf::mutable_column_device_view::create(offsets_view, stream);

  // Allocate chars array and output null mask
  cudf::size_type null_count = input_col.size() - valid_counter.value(stream);
  rmm::device_uvector<char> output_chars(bytes, stream, mr);
  auto d_chars = output_chars.data();

  replace_second<<<grid.num_blocks, BLOCK_SIZE, 0, stream.value()>>>(
    *device_in, *device_replacement, *device_offsets, d_chars, *device_indices);

  return cudf::make_strings_column(input_col.size(),
                                   std::move(offsets),
                                   output_chars.release(),
                                   null_count,
                                   std::move(valid_bits));
}

template <>
std::unique_ptr<cudf::column> replace_kernel_forwarder::operator()<cudf::dictionary32>(
  cudf::column_view const& input_col,
  cudf::column_view const& values_to_replace,
  cudf::column_view const& replacement_values,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  auto input        = cudf::dictionary_column_view(input_col);
  auto values       = cudf::dictionary_column_view(values_to_replace);
  auto replacements = cudf::dictionary_column_view(replacement_values);

  auto matched_input = [&] {
    auto new_keys = cudf::detail::concatenate(
      std::vector<cudf::column_view>({values.keys(), replacements.keys()}),
      stream,
      rmm::mr::get_current_device_resource());
    return cudf::dictionary::detail::add_keys(input, new_keys->view(), stream, mr);
  }();
  auto matched_view   = cudf::dictionary_column_view(matched_input->view());
  auto matched_values = cudf::dictionary::detail::set_keys(
    values, matched_view.keys(), stream, rmm::mr::get_current_device_resource());
  auto matched_replacements = cudf::dictionary::detail::set_keys(
    replacements, matched_view.keys(), stream, rmm::mr::get_current_device_resource());

  auto indices_type = matched_view.indices().type();
  auto new_indices  = cudf::type_dispatcher<cudf::dispatch_storage_type>(
    indices_type,
    replace_kernel_forwarder{},
    matched_view.get_indices_annotated(),
    cudf::dictionary_column_view(matched_values->view()).indices(),
    cudf::dictionary_column_view(matched_replacements->view()).get_indices_annotated(),
    stream,
    mr);
  auto null_count     = new_indices->null_count();
  auto contents       = new_indices->release();
  auto indices_column = std::make_unique<cudf::column>(
    indices_type, input.size(), std::move(*(contents.data.release())), rmm::device_buffer{}, 0);
  std::unique_ptr<cudf::column> keys_column(std::move(matched_input->release().children.back()));
  return cudf::make_dictionary_column(std::move(keys_column),
                                      std::move(indices_column),
                                      std::move(*(contents.null_mask.release())),
                                      null_count);
}

}  // end anonymous namespace

namespace cudf {
namespace detail {
std::unique_ptr<cudf::column> find_and_replace_all(cudf::column_view const& input_col,
                                                   cudf::column_view const& values_to_replace,
                                                   cudf::column_view const& replacement_values,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(values_to_replace.size() == replacement_values.size(),
               "values_to_replace and replacement_values size mismatch.");

  CUDF_EXPECTS(
    input_col.type() == values_to_replace.type() && input_col.type() == replacement_values.type(),
    "Columns type mismatch");
  CUDF_EXPECTS(not values_to_replace.has_nulls(), "values_to_replace must not have nulls");

  if (input_col.is_empty() or values_to_replace.is_empty() or replacement_values.is_empty()) {
    return std::make_unique<cudf::column>(input_col, stream, mr);
  }

  return cudf::type_dispatcher<dispatch_storage_type>(input_col.type(),
                                                      replace_kernel_forwarder{},
                                                      input_col,
                                                      values_to_replace,
                                                      replacement_values,
                                                      stream,
                                                      mr);
}

}  // namespace detail

/**
 * @brief Replace elements from `input_col` according to the mapping `values_to_replace` to
 *        `replacement_values`, that is, replace all `values_to_replace[i]` present in `input_col`
 *        with `replacement_values[i]`.
 *
 * @param[in] input_col column_view of the data to be modified
 * @param[in] values_to_replace column_view of the old values to be replaced
 * @param[in] replacement_values column_view of the new values
 *
 * @returns output cudf::column with the modified data
 */
std::unique_ptr<cudf::column> find_and_replace_all(cudf::column_view const& input_col,
                                                   cudf::column_view const& values_to_replace,
                                                   cudf::column_view const& replacement_values,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::mr::device_memory_resource* mr)
{
  return detail::find_and_replace_all(input_col, values_to_replace, replacement_values, stream, mr);
}
}  // namespace cudf

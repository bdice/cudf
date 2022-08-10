/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

/**
 * @file reader_impl.hpp
 * @brief cuDF-IO Parquet reader class implementation header
 */

#pragma once

#include "parquet.hpp"
#include "parquet_gpu.hpp"

#include <io/utilities/column_buffer.hpp>
#include <io/utilities/hostdevice_vector.hpp>

#include <cudf/io/datasource.hpp>
#include <cudf/io/detail/parquet.hpp>
#include <cudf/io/parquet.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace cudf {
namespace io {
namespace detail {
namespace parquet {
using namespace cudf::io::parquet;
using namespace cudf::io;

// Forward declarations
class aggregate_reader_metadata;

/**
 * @brief Implementation for Parquet reader
 */
class reader::impl {
 public:
  /**
   * @brief Constructor from an array of dataset sources with reader options.
   *
   * @param sources Dataset sources
   * @param options Settings for controlling reading behavior
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit impl(std::vector<std::unique_ptr<datasource>>&& sources,
                parquet_reader_options const& options,
                rmm::cuda_stream_view stream,
                rmm::mr::device_memory_resource* mr);

  /**
   * @brief Read an entire set or a subset of data and returns a set of columns
   *
   * @param skip_rows Number of rows to skip from the start
   * @param num_rows Number of rows to read
   * @param uses_custom_row_bounds Whether or not num_rows and min_rows represents user-specific
   * bounds
   * @param row_group_indices Lists of row groups to read, one per source
   *
   * @return The set of columns along with metadata
   */
  table_with_metadata read(size_type skip_rows,
                           size_type num_rows,
                           bool uses_custom_row_bounds,
                           std::vector<std::vector<size_type>> const& row_group_indices);

 private:
  /**
   * @brief Reads compressed page data to device memory
   *
   * @param page_data Buffers to hold compressed page data for each chunk
   * @param chunks List of column chunk descriptors
   * @param begin_chunk Index of first column chunk to read
   * @param end_chunk Index after the last column chunk to read
   * @param column_chunk_offsets File offset for all chunks
   *
   */
  std::future<void> read_column_chunks(std::vector<std::unique_ptr<datasource::buffer>>& page_data,
                                       hostdevice_vector<gpu::ColumnChunkDesc>& chunks,
                                       size_t begin_chunk,
                                       size_t end_chunk,
                                       const std::vector<size_t>& column_chunk_offsets,
                                       std::vector<size_type> const& chunk_source_map);

  /**
   * @brief Returns the number of total pages from the given column chunks
   *
   * @param chunks List of column chunk descriptors
   *
   * @return The total number of pages
   */
  size_t count_page_headers(hostdevice_vector<gpu::ColumnChunkDesc>& chunks);

  /**
   * @brief Returns the page information from the given column chunks.
   *
   * @param chunks List of column chunk descriptors
   * @param pages List of page information
   */
  void decode_page_headers(hostdevice_vector<gpu::ColumnChunkDesc>& chunks,
                           hostdevice_vector<gpu::PageInfo>& pages);

  /**
   * @brief Decompresses the page data, at page granularity.
   *
   * @param chunks List of column chunk descriptors
   * @param pages List of page information
   *
   * @return Device buffer to decompressed page data
   */
  rmm::device_buffer decompress_page_data(hostdevice_vector<gpu::ColumnChunkDesc>& chunks,
                                          hostdevice_vector<gpu::PageInfo>& pages);

  /**
   * @brief Allocate nesting information storage for all pages and set pointers
   *        to it.
   *
   * One large contiguous buffer of PageNestingInfo structs is allocated and
   * distributed among the PageInfo structs.
   *
   * Note that this gets called even in the flat schema case so that we have a
   * consistent place to store common information such as value counts, etc.
   *
   * @param chunks List of column chunk descriptors
   * @param pages List of page information
   * @param page_nesting_info The allocated nesting info structs.
   */
  void allocate_nesting_info(hostdevice_vector<gpu::ColumnChunkDesc> const& chunks,
                             hostdevice_vector<gpu::PageInfo>& pages,
                             hostdevice_vector<gpu::PageNestingInfo>& page_nesting_info);

  /**
   * @brief Preprocess column information for nested schemas.
   *
   * There are several pieces of information we can't compute directly from row counts in
   * the parquet headers when dealing with nested schemas.
   * - The total sizes of all output columns at all nesting levels
   * - The starting output buffer offset for each page, for each nesting level
   *
   * For flat schemas, these values are computed during header decoding (see gpuDecodePageHeaders)
   *
   * @param chunks All chunks to be decoded
   * @param pages All pages to be decoded
   * @param min_rows crop all rows below min_row
   * @param total_rows Maximum number of rows to read
   * @param uses_custom_row_bounds Whether or not num_rows and min_rows represents user-specific
   * bounds
   * @param has_lists Whether or not this data contains lists and requires
   * a preprocess.
   */
  void preprocess_columns(hostdevice_vector<gpu::ColumnChunkDesc>& chunks,
                          hostdevice_vector<gpu::PageInfo>& pages,
                          size_t min_row,
                          size_t total_rows,
                          bool uses_custom_row_bounds,
                          bool has_lists);

  /**
   * @brief Converts the page data and outputs to columns.
   *
   * @param chunks List of column chunk descriptors
   * @param pages List of page information
   * @param page_nesting Page nesting array
   * @param min_row Minimum number of rows from start
   * @param total_rows Number of rows to output
   */
  void decode_page_data(hostdevice_vector<gpu::ColumnChunkDesc>& chunks,
                        hostdevice_vector<gpu::PageInfo>& pages,
                        hostdevice_vector<gpu::PageNestingInfo>& page_nesting,
                        size_t min_row,
                        size_t total_rows);

  /**
   * @brief Indicates if a column should be written as a byte array
   *
   * @param col column to check
   * @return true if the column should be written as a byte array
   * @return false if the column should be written as normal for that type
   */
  bool should_write_byte_array(int col)
  {
    return _output_columns[col].type.id() == type_id::STRING &&
           _force_binary_columns_as_strings.has_value() &&
           !_force_binary_columns_as_strings.value()[col];
  }

 private:
  rmm::cuda_stream_view _stream;
  rmm::mr::device_memory_resource* _mr = nullptr;

  std::vector<std::unique_ptr<datasource>> _sources;
  std::unique_ptr<aggregate_reader_metadata> _metadata;

  // input columns to be processed
  std::vector<input_column_info> _input_columns;
  // output columns to be generated
  std::vector<column_buffer> _output_columns;
  // _output_columns associated schema indices
  std::vector<int> _output_column_schemas;

  bool _strings_to_categorical = false;
  std::optional<std::vector<bool>> _force_binary_columns_as_strings;
  data_type _timestamp_type{type_id::EMPTY};
};

}  // namespace parquet
}  // namespace detail
}  // namespace io
}  // namespace cudf

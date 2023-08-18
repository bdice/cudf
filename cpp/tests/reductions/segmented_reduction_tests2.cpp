/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/types.hpp>

#include <limits>
#include <utility>
#include <vector>

// String min/max test grid
// Segment: Length 0, length 1, length 2
// Element nulls: No nulls, all nulls, some nulls
// String: Empty string,
// Position of the min/max: start of segment, end of segment
// Include null, exclude null

#define XXX ""  // null placeholder

struct SegmentedReductionStringTest : public cudf::test::BaseFixture {
  std::pair<cudf::test::strings_column_wrapper,
            cudf::test::fixed_width_column_wrapper<cudf::size_type>>
  input()
  {
    return std::pair(
      cudf::test::strings_column_wrapper{
        {"world", "cudf", XXX, "", "rapids", "i am", "ai", "apples", "zebras", XXX, XXX, XXX},
        {true, true, false, true, true, true, true, true, true, false, false, false}},
      cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 1, 4, 7, 9, 9, 10, 12});
  }
};

/*
TEST_F(SegmentedReductionStringTest, MaxIncludeNulls)
{
  // data: ['world'], ['cudf', NULL, ''], ['rapids', 'i am', 'ai'], ['apples', 'zebras'],
  //       [], [NULL], [NULL, NULL]
  // values:  {"world", "cudf", XXX, "", "rapids", "i am", "ai", "apples", "zebras", XXX, XXX, XXX}
  // nullmask:{1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0}
  // offsets: {0, 1, 4, 7, 9, 9, 10, 12}
  // output_dtype: string dtype
  // outputs: {"world", XXX, "rapids", "zebras", XXX, XXX, XXX}
  // output nullmask: {1, 0, 1, 1, 0, 0, 0}

  auto const [input, offsets] = this->input();
  cudf::data_type output_dtype{cudf::type_id::STRING};

  cudf::test::strings_column_wrapper expect{{"world", XXX, "rapids", "zebras", XXX, XXX, XXX},
                                            {true, false, true, true, false, false, false}};

  auto res =
    cudf::segmented_reduce(input,
                           cudf::column_view(offsets),
                           *cudf::make_max_aggregation<cudf::segmented_reduce_aggregation>(),
                           output_dtype,
                           cudf::null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}

TEST_F(SegmentedReductionStringTest, MaxExcludeNulls)
{
  // data: ['world'], ['cudf', NULL, ''], ['rapids', 'i am', 'ai'], ['apples', 'zebras'],
  //       [], [NULL], [NULL, NULL]
  // values:  {"world", "cudf", XXX, "", "rapids", "i am", "ai", "apples", "zebras", XXX, XXX, XXX}
  // nullmask:{1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0}
  // offsets: {0, 1, 4, 7, 9, 9, 10, 12}
  // output_dtype: string dtype
  // outputs: {"world", "cudf", "rapids", "zebras", XXX, XXX, XXX}
  // output nullmask: {1, 1, 1, 1, 0, 0, 0}

  auto const [input, offsets] = this->input();
  cudf::data_type output_dtype{cudf::type_id::STRING};

  cudf::test::strings_column_wrapper expect{{"world", "cudf", "rapids", "zebras", XXX, XXX, XXX},
                                            {true, true, true, true, false, false, false}};

  auto res =
    cudf::segmented_reduce(input,
                           cudf::column_view(offsets),
                           *cudf::make_max_aggregation<cudf::segmented_reduce_aggregation>(),
                           output_dtype,
                           cudf::null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}

TEST_F(SegmentedReductionStringTest, MinIncludeNulls)
{
  // data: ['world'], ['cudf', NULL, ''], ['rapids', 'i am', 'ai'], ['apples', 'zebras'],
  //       [], [NULL], [NULL, NULL]
  // values:  {"world", "cudf", XXX, "", "rapids", "i am", "ai", "apples", "zebras", XXX, XXX, XXX}
  // nullmask:{1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0}
  // offsets: {0, 1, 4, 7, 9, 9, 10, 12}
  // output_dtype: string dtype
  // outputs: {"world", XXX, "ai", "apples", XXX, XXX, XXX}
  // output nullmask: {1, 0, 1, 1, 0, 0, 0}

  auto const [input, offsets] = this->input();
  cudf::data_type output_dtype{cudf::type_id::STRING};

  cudf::test::strings_column_wrapper expect{{"world", XXX, "ai", "apples", XXX, XXX, XXX},
                                            {true, false, true, true, false, false, false}};

  auto res =
    cudf::segmented_reduce(input,
                           cudf::column_view(offsets),
                           *cudf::make_min_aggregation<cudf::segmented_reduce_aggregation>(),
                           output_dtype,
                           cudf::null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}
*/

TEST_F(SegmentedReductionStringTest, MinExcludeNulls)
{
  // data: ['world'], ['cudf', NULL, ''], ['rapids', 'i am', 'ai'], ['apples', 'zebras'],
  //       [], [NULL], [NULL, NULL]
  // values:  {"world", "cudf", XXX, "", "rapids", "i am", "ai", "apples", "zebras", XXX, XXX, XXX}
  // nullmask:{1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0}
  // offsets: {0, 1, 4, 7, 9, 9, 10, 12}
  // output_dtype: string dtype
  // outputs: {"world", "", "ai", "apples", XXX, XXX, XXX}
  // output nullmask: {1, 1, 1, 1, 0, 0, 0}

  auto const [input, offsets] = this->input();
  cudf::data_type output_dtype{cudf::type_id::STRING};

  cudf::test::strings_column_wrapper expect{{"world", "", "ai", "apples", XXX, XXX, XXX},
                                            {true, true, true, true, false, false, false}};

  auto res =
    cudf::segmented_reduce(input,
                           cudf::column_view(offsets),
                           *cudf::make_min_aggregation<cudf::segmented_reduce_aggregation>(),
                           output_dtype,
                           cudf::null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}

TEST_F(SegmentedReductionStringTest, EmptyInputWithOffsets)
{
  auto const input     = cudf::test::strings_column_wrapper{};
  auto const offsets   = std::vector<cudf::size_type>{0, 0, 0, 0};
  auto const d_offsets = cudf::detail::make_device_uvector_async(
    offsets, cudf::get_default_stream(), rmm::mr::get_current_device_resource());
  auto const expect = cudf::test::strings_column_wrapper({XXX, XXX, XXX}, {0, 0, 0});

  auto result =
    cudf::segmented_reduce(input,
                           d_offsets,
                           *cudf::make_max_aggregation<cudf::segmented_reduce_aggregation>(),
                           cudf::data_type{cudf::type_id::STRING},
                           cudf::null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
  result = cudf::segmented_reduce(input,
                                  d_offsets,
                                  *cudf::make_min_aggregation<cudf::segmented_reduce_aggregation>(),
                                  cudf::data_type{cudf::type_id::STRING},
                                  cudf::null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

#undef XXX

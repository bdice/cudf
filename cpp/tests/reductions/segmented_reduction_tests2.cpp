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
#include <cudf/types.hpp>

#include <utility>

struct SegmentedReductionStringTest : public cudf::test::BaseFixture {};

TEST_F(SegmentedReductionStringTest, MinExcludeNulls)
{
  auto const input   = cudf::test::strings_column_wrapper{{"", ""}, {true, true}};
  auto const offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 2};
  cudf::data_type output_dtype{cudf::type_id::STRING};

  std::cout << "Starting segmented reduce..." << std::endl;
  auto res =
    cudf::segmented_reduce(input,
                           cudf::column_view(offsets),
                           *cudf::make_min_aggregation<cudf::segmented_reduce_aggregation>(),
                           output_dtype,
                           cudf::null_policy::EXCLUDE);
  std::cout << "Ending segmented reduce..." << std::endl;

  cudf::test::strings_column_wrapper expect{{""}, {true}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}

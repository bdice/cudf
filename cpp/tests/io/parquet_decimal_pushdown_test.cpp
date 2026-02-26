/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "parquet_common.hpp"

#include <cudf_test/base_fixture.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

namespace {

constexpr int32_t kScale     = -2;
constexpr __int128_t kBig128 = (static_cast<__int128_t>(1) << 70) + 1234;

// ---------------------------------------------------------------------------
// Column / table helpers
// ---------------------------------------------------------------------------

template <typename Rep>
std::unique_ptr<cudf::column> make_decimal_column(cudf::type_id type_id,
                                                  int32_t scale,
                                                  std::vector<Rep> const& values,
                                                  std::vector<bool> const* valid = nullptr)
{
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();
  auto col    = cudf::make_fixed_width_column(cudf::data_type{type_id, scale},
                                           static_cast<cudf::size_type>(values.size()),
                                           cudf::mask_state::UNALLOCATED,
                                           stream);
  if (!values.empty()) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(col->mutable_view().data<Rep>(),
                                  values.data(),
                                  values.size() * sizeof(Rep),
                                  cudaMemcpyHostToDevice,
                                  stream.value()));
    stream.synchronize();
  }
  if (valid != nullptr) {
    bool any_null = std::any_of(valid->begin(), valid->end(), [](bool v) { return !v; });
    if (any_null) {
      auto mask = cudf::create_null_mask(
        static_cast<cudf::size_type>(values.size()), cudf::mask_state::ALL_VALID, stream, mr);
      auto* mask_ptr             = static_cast<cudf::bitmask_type*>(mask.data());
      cudf::size_type null_count = 0;
      for (cudf::size_type i = 0; i < static_cast<cudf::size_type>(valid->size()); ++i) {
        if (!(*valid)[i]) {
          cudf::set_null_mask(mask_ptr, i, i + 1, false, stream);
          ++null_count;
        }
      }
      stream.synchronize();
      col->set_null_mask(std::move(mask), null_count);
    }
  }
  return col;
}

std::unique_ptr<cudf::table> make_decimal_table(std::vector<int64_t> const& c0,
                                                std::vector<__int128_t> const& c1,
                                                int32_t scale)
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.emplace_back(make_decimal_column<int64_t>(cudf::type_id::DECIMAL64, scale, c0));
  cols.emplace_back(make_decimal_column<__int128_t>(cudf::type_id::DECIMAL128, scale, c1));
  return std::make_unique<cudf::table>(std::move(cols));
}

// ---------------------------------------------------------------------------
// AST expression helpers
// ---------------------------------------------------------------------------

template <typename DecimalT, typename Rep>
cudf::ast::expression const& add_decimal_literal(
  cudf::ast::tree& tree,
  std::vector<std::unique_ptr<cudf::scalar>>& scalars,
  Rep unscaled,
  int32_t scale,
  rmm::cuda_stream_view stream)
{
  using scalar_t = cudf::fixed_point_scalar<DecimalT>;
  scalars.emplace_back(
    std::make_unique<scalar_t>(unscaled, ::numeric::scale_type{scale}, true, stream));
  return tree.push(cudf::ast::literal{*static_cast<scalar_t*>(scalars.back().get())});
}

cudf::ast::expression const& make_range_expr(cudf::ast::tree& tree,
                                             cudf::ast::expression const& col_ref,
                                             cudf::ast::expression const& lower,
                                             cudf::ast::expression const& upper)
{
  using Op = cudf::ast::ast_operator;
  auto& ge = tree.push(cudf::ast::operation{Op::GREATER_EQUAL, col_ref, lower});
  auto& le = tree.push(cudf::ast::operation{Op::LESS_EQUAL, col_ref, upper});
  return tree.push(cudf::ast::operation{Op::NULL_LOGICAL_AND, ge, le});
}

cudf::ast::expression const& combine_with_or(
  cudf::ast::tree& tree,
  std::vector<std::reference_wrapper<cudf::ast::expression const>> const& exprs)
{
  cudf::ast::expression const* cur = &exprs.front().get();
  for (size_t i = 1; i < exprs.size(); ++i) {
    cur = &tree.push(
      cudf::ast::operation{cudf::ast::ast_operator::NULL_LOGICAL_OR, *cur, exprs[i].get()});
  }
  return *cur;
}

template <typename DecimalT, typename Rep>
cudf::ast::expression const& make_in_list_expr(cudf::ast::tree& tree,
                                               cudf::ast::expression const& col_ref,
                                               std::vector<std::unique_ptr<cudf::scalar>>& scalars,
                                               std::vector<Rep> const& values,
                                               int32_t scale,
                                               rmm::cuda_stream_view stream)
{
  std::vector<std::reference_wrapper<cudf::ast::expression const>> exprs;
  exprs.reserve(values.size());
  for (auto v : values) {
    auto& lit = add_decimal_literal<DecimalT, Rep>(tree, scalars, v, scale, stream);
    auto& eq  = tree.push(cudf::ast::operation{cudf::ast::ast_operator::EQUAL, col_ref, lit});
    exprs.emplace_back(eq);
  }
  return combine_with_or(tree, exprs);
}

// ---------------------------------------------------------------------------
// Host result helpers
// ---------------------------------------------------------------------------

struct BoolResult {
  std::vector<uint8_t> values;
  std::vector<bool> is_null;
};

BoolResult copy_bool_column(cudf::column_view const& view, rmm::cuda_stream_view stream)
{
  BoolResult out;
  out.values.resize(view.size());
  out.is_null.assign(view.size(), false);
  if (view.size() == 0) { return out; }
  CUDF_CUDA_TRY(cudaMemcpyAsync(out.values.data(),
                                view.data<bool>(),
                                view.size() * sizeof(bool),
                                cudaMemcpyDeviceToHost,
                                stream.value()));
  if (view.nullable()) {
    auto words = cudf::num_bitmask_words(view.size());
    std::vector<cudf::bitmask_type> mask(words);
    CUDF_CUDA_TRY(cudaMemcpyAsync(mask.data(),
                                  view.null_mask(),
                                  words * sizeof(cudf::bitmask_type),
                                  cudaMemcpyDeviceToHost,
                                  stream.value()));
    stream.synchronize();
    for (cudf::size_type i = 0; i < view.size(); ++i) {
      out.is_null[i] = !cudf::bit_is_set(mask.data(), i + view.offset());
    }
  } else {
    stream.synchronize();
  }
  return out;
}

}  // namespace

// =============================================================================
// Test fixture
// =============================================================================

struct ParquetDecimalPushdownTest : public cudf::test::BaseFixture {};

// =============================================================================
// Stats-filter (row-group pruning) tests
// =============================================================================

// Helper: write a multi-row-group parquet with two decimal columns (c0=DECIMAL64,
// c1=DECIMAL128), then read it back with a combined filter and verify:
//  - num_input_row_groups equals the number of groups written
//  - num_row_groups_after_stats_filter equals the expected surviving groups
//  - the output row count equals the expected matching rows
//
// The `use_jit` parameter selects the JIT vs AST filter path.
static void run_stats_filter_test(std::vector<std::vector<int64_t>> const& c0_groups,
                                  std::vector<std::vector<__int128_t>> const& c1_groups,
                                  // c0 filter: range [c0_lo, c0_hi]
                                  int64_t c0_lo,
                                  int64_t c0_hi,
                                  // c1 filter: range [c1_lo, c1_hi]
                                  __int128_t c1_lo,
                                  __int128_t c1_hi,
                                  bool use_jit)
{
  ASSERT_EQ(c0_groups.size(), c1_groups.size());

  auto const filepath = temp_env->get_temp_filepath("decimal_stats_filter.parquet");
  auto const sink     = cudf::io::sink_info(filepath);

  // Write each group separately via chunked writer so each becomes its own row group.
  auto first_table = make_decimal_table(c0_groups[0], c1_groups[0], kScale);
  auto metadata    = cudf::io::table_input_metadata(first_table->view());
  metadata.column_metadata[0].set_name("c0");
  metadata.column_metadata[1].set_name("c1");

  auto opts = cudf::io::chunked_parquet_writer_options::builder(sink).metadata(metadata).build();
  cudf::io::chunked_parquet_writer writer(opts);
  for (size_t g = 0; g < c0_groups.size(); ++g) {
    auto tbl = make_decimal_table(c0_groups[g], c1_groups[g], kScale);
    writer.write(tbl->view());
  }
  writer.close();

  // Build filter: (c0 >= c0_lo AND c0 <= c0_hi) AND (c1 >= c1_lo AND c1 <= c1_hi)
  auto stream = cudf::get_default_stream();
  cudf::ast::tree tree;
  std::vector<std::unique_ptr<cudf::scalar>> scalars;

  cudf::ast::column_reference c0_ref(0);
  cudf::ast::column_reference c1_ref(1);

  auto& lo0   = add_decimal_literal<::numeric::decimal64>(tree, scalars, c0_lo, kScale, stream);
  auto& hi0   = add_decimal_literal<::numeric::decimal64>(tree, scalars, c0_hi, kScale, stream);
  auto& expr0 = make_range_expr(tree, c0_ref, lo0, hi0);

  auto& lo1   = add_decimal_literal<::numeric::decimal128>(tree, scalars, c1_lo, kScale, stream);
  auto& hi1   = add_decimal_literal<::numeric::decimal128>(tree, scalars, c1_hi, kScale, stream);
  auto& expr1 = make_range_expr(tree, c1_ref, lo1, hi1);

  auto& combined =
    tree.push(cudf::ast::operation{cudf::ast::ast_operator::NULL_LOGICAL_AND, expr0, expr1});

  auto read_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info(filepath))
                     .use_jit_filter(use_jit)
                     .build();
  read_opts.set_filter(combined);
  auto result = cudf::io::read_parquet(read_opts);

  // Compute expected survivors
  cudf::size_type expected_row_groups = 0;
  cudf::size_type expected_rows       = 0;
  for (size_t g = 0; g < c0_groups.size(); ++g) {
    auto const& c0v = c0_groups[g];
    auto const& c1v = c1_groups[g];
    if (c0v.empty()) { continue; }
    int64_t c0min      = *std::min_element(c0v.begin(), c0v.end());
    int64_t c0max      = *std::max_element(c0v.begin(), c0v.end());
    __int128_t c1min   = *std::min_element(c1v.begin(), c1v.end());
    __int128_t c1max   = *std::max_element(c1v.begin(), c1v.end());
    bool group_matches = !(c0max < c0_lo || c0min > c0_hi) && !(c1max < c1_lo || c1min > c1_hi);
    if (group_matches) { ++expected_row_groups; }
    for (size_t i = 0; i < c0v.size(); ++i) {
      if (c0v[i] >= c0_lo && c0v[i] <= c0_hi && c1v[i] >= c1_lo && c1v[i] <= c1_hi) {
        ++expected_rows;
      }
    }
  }

  EXPECT_EQ(result.metadata.num_input_row_groups, static_cast<int64_t>(c0_groups.size()));
  ASSERT_TRUE(result.metadata.num_row_groups_after_stats_filter.has_value());
  EXPECT_EQ(result.metadata.num_row_groups_after_stats_filter.value(), expected_row_groups);
  EXPECT_EQ(result.tbl->num_rows(), expected_rows);
}

// ---------------------------------------------------------------------------
// positive range: 3 row groups, filter selects exactly the middle one
// ---------------------------------------------------------------------------

TEST_F(ParquetDecimalPushdownTest, StatsFilterPositiveRangeJit)
{
  run_stats_filter_test({{100, 200}, {300, 400}, {500, 600}},
                        {{__int128_t{1000}, __int128_t{2000}},
                         {__int128_t{3000}, __int128_t{4000}},
                         {__int128_t{5000}, __int128_t{6000}}},
                        300,
                        400,
                        __int128_t{3000},
                        __int128_t{4000},
                        /*use_jit=*/true);
}

TEST_F(ParquetDecimalPushdownTest, StatsFilterPositiveRangeAst)
{
  run_stats_filter_test({{100, 200}, {300, 400}, {500, 600}},
                        {{__int128_t{1000}, __int128_t{2000}},
                         {__int128_t{3000}, __int128_t{4000}},
                         {__int128_t{5000}, __int128_t{6000}}},
                        300,
                        400,
                        __int128_t{3000},
                        __int128_t{4000},
                        /*use_jit=*/false);
}

// ---------------------------------------------------------------------------
// negative range: filter selects the middle group (negative values)
// ---------------------------------------------------------------------------

TEST_F(ParquetDecimalPushdownTest, StatsFilterNegativeRangeJit)
{
  run_stats_filter_test({{-600, -500}, {-400, -300}, {-200, -100}},
                        {{__int128_t{-6000}, __int128_t{-5000}},
                         {__int128_t{-4000}, __int128_t{-3000}},
                         {__int128_t{-2000}, __int128_t{-1000}}},
                        -450,
                        -350,
                        __int128_t{-4000},
                        __int128_t{-3000},
                        /*use_jit=*/true);
}

TEST_F(ParquetDecimalPushdownTest, StatsFilterNegativeRangeAst)
{
  run_stats_filter_test({{-600, -500}, {-400, -300}, {-200, -100}},
                        {{__int128_t{-6000}, __int128_t{-5000}},
                         {__int128_t{-4000}, __int128_t{-3000}},
                         {__int128_t{-2000}, __int128_t{-1000}}},
                        -450,
                        -350,
                        __int128_t{-4000},
                        __int128_t{-3000},
                        /*use_jit=*/false);
}

// ---------------------------------------------------------------------------
// no pruning: filter overlaps all row groups; row-level eval prunes rows
// ---------------------------------------------------------------------------

TEST_F(ParquetDecimalPushdownTest, StatsFilterNoPruneRowEvalJit)
{
  run_stats_filter_test({{100, 200}, {300, 400}, {500, 600}},
                        {{__int128_t{1000}, __int128_t{2000}},
                         {__int128_t{3000}, __int128_t{4000}},
                         {__int128_t{5000}, __int128_t{6000}}},
                        150,
                        550,
                        __int128_t{1500},
                        __int128_t{5500},
                        /*use_jit=*/true);
}

TEST_F(ParquetDecimalPushdownTest, StatsFilterNoPruneRowEvalAst)
{
  run_stats_filter_test({{100, 200}, {300, 400}, {500, 600}},
                        {{__int128_t{1000}, __int128_t{2000}},
                         {__int128_t{3000}, __int128_t{4000}},
                         {__int128_t{5000}, __int128_t{6000}}},
                        150,
                        550,
                        __int128_t{1500},
                        __int128_t{5500},
                        /*use_jit=*/false);
}

// ---------------------------------------------------------------------------
// Parquet physical type check: DECIMAL64 -> INT64, DECIMAL128 -> FIXED_LEN_BYTE_ARRAY
// ---------------------------------------------------------------------------

TEST_F(ParquetDecimalPushdownTest, PhysicalTypes)
{
  auto const filepath = temp_env->get_temp_filepath("decimal_physical_types.parquet");

  auto tbl      = make_decimal_table({100, 200}, {__int128_t{1000}, __int128_t{2000}}, kScale);
  auto metadata = cudf::io::table_input_metadata(tbl->view());
  metadata.column_metadata[0].set_name("c0");
  metadata.column_metadata[1].set_name("c1");

  cudf::io::parquet_writer_options opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, tbl->view())
      .metadata(std::move(metadata))
      .build();
  cudf::io::write_parquet(opts);

  auto file_meta   = cudf::io::read_parquet_metadata(cudf::io::source_info(filepath));
  auto const& root = file_meta.schema().root();

  auto find_col = [&](std::string const& name) -> cudf::io::parquet_column_schema const* {
    for (auto const& child : root.children()) {
      if (child.name() == name) { return &child; }
    }
    return nullptr;
  };

  auto const* c0_schema = find_col("c0");
  auto const* c1_schema = find_col("c1");
  ASSERT_NE(c0_schema, nullptr);
  ASSERT_NE(c1_schema, nullptr);
  EXPECT_EQ(c0_schema->type(), cudf::io::parquet::Type::INT64);
  EXPECT_EQ(c1_schema->type(), cudf::io::parquet::Type::FIXED_LEN_BYTE_ARRAY);
}

// =============================================================================
// Row-level evaluation tests (compute_column / compute_column_jit)
// =============================================================================

// Helper: build a single-column decimal table, evaluate a range filter via
// compute_column or compute_column_jit, and verify each output row.
template <typename DecimalT, typename Rep>
static void run_row_range_test(cudf::type_id type_id,
                               std::vector<Rep> const& values,
                               std::vector<bool> const& valid,
                               Rep lo,
                               Rep hi,
                               bool use_jit)
{
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.emplace_back(make_decimal_column<Rep>(type_id, kScale, values, &valid));
  auto tbl = std::make_unique<cudf::table>(std::move(cols));

  cudf::ast::tree tree;
  std::vector<std::unique_ptr<cudf::scalar>> scalars;
  cudf::ast::column_reference col_ref(0);
  auto& lower = add_decimal_literal<DecimalT>(tree, scalars, lo, kScale, stream);
  auto& upper = add_decimal_literal<DecimalT>(tree, scalars, hi, kScale, stream);
  auto& expr  = make_range_expr(tree, col_ref, lower, upper);

  std::unique_ptr<cudf::column> result;
  if (use_jit) {
    result = cudf::compute_column_jit(tbl->view(), expr, stream, mr);
  } else {
    result = cudf::compute_column(tbl->view(), expr, stream, mr);
  }

  ASSERT_EQ(result->type().id(), cudf::type_id::BOOL8);
  auto host = copy_bool_column(result->view(), stream);
  ASSERT_EQ(static_cast<size_t>(result->size()), values.size());

  for (size_t i = 0; i < values.size(); ++i) {
    bool expected_null = !valid[i];
    EXPECT_EQ(host.is_null[i], expected_null) << "null mismatch at index " << i;
    if (!expected_null) {
      bool expected_val = (values[i] >= lo && values[i] <= hi);
      EXPECT_EQ(host.values[i], static_cast<uint8_t>(expected_val))
        << "value mismatch at index " << i;
    }
  }
}

// Helper: same as above but for an IN-list filter.
template <typename DecimalT, typename Rep>
static void run_row_in_list_test(cudf::type_id type_id,
                                 std::vector<Rep> const& values,
                                 std::vector<bool> const& valid,
                                 std::vector<Rep> const& in_list,
                                 bool use_jit)
{
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.emplace_back(make_decimal_column<Rep>(type_id, kScale, values, &valid));
  auto tbl = std::make_unique<cudf::table>(std::move(cols));

  cudf::ast::tree tree;
  std::vector<std::unique_ptr<cudf::scalar>> scalars;
  cudf::ast::column_reference col_ref(0);
  auto& expr = make_in_list_expr<DecimalT, Rep>(tree, col_ref, scalars, in_list, kScale, stream);

  std::unique_ptr<cudf::column> result;
  if (use_jit) {
    result = cudf::compute_column_jit(tbl->view(), expr, stream, mr);
  } else {
    result = cudf::compute_column(tbl->view(), expr, stream, mr);
  }

  ASSERT_EQ(result->type().id(), cudf::type_id::BOOL8);
  auto host = copy_bool_column(result->view(), stream);
  ASSERT_EQ(static_cast<size_t>(result->size()), values.size());

  for (size_t i = 0; i < values.size(); ++i) {
    bool expected_null = !valid[i];
    EXPECT_EQ(host.is_null[i], expected_null) << "null mismatch at index " << i;
    if (!expected_null) {
      bool expected_val = std::find(in_list.begin(), in_list.end(), values[i]) != in_list.end();
      EXPECT_EQ(host.values[i], static_cast<uint8_t>(expected_val))
        << "value mismatch at index " << i;
    }
  }
}

// ---------------------------------------------------------------------------
// DECIMAL64 range with nulls
// ---------------------------------------------------------------------------

TEST_F(ParquetDecimalPushdownTest, RowRangeDecimal64NullsJit)
{
  run_row_range_test<::numeric::decimal64>(cudf::type_id::DECIMAL64,
                                           {100, 200, -150, 300, 450},
                                           {true, true, true, false, true},
                                           int64_t{-200},
                                           int64_t{250},
                                           /*use_jit=*/true);
}

TEST_F(ParquetDecimalPushdownTest, RowRangeDecimal64NullsAst)
{
  run_row_range_test<::numeric::decimal64>(cudf::type_id::DECIMAL64,
                                           {100, 200, -150, 300, 450},
                                           {true, true, true, false, true},
                                           int64_t{-200},
                                           int64_t{250},
                                           /*use_jit=*/false);
}

// ---------------------------------------------------------------------------
// DECIMAL64 IN-list with nulls
// ---------------------------------------------------------------------------

TEST_F(ParquetDecimalPushdownTest, RowInListDecimal64NullsJit)
{
  run_row_in_list_test<::numeric::decimal64, int64_t>(cudf::type_id::DECIMAL64,
                                                      {100, 200, -150, 300, 450},
                                                      {true, false, true, true, true},
                                                      {int64_t{100}, int64_t{300}, int64_t{-150}},
                                                      /*use_jit=*/true);
}

TEST_F(ParquetDecimalPushdownTest, RowInListDecimal64NullsAst)
{
  run_row_in_list_test<::numeric::decimal64, int64_t>(cudf::type_id::DECIMAL64,
                                                      {100, 200, -150, 300, 450},
                                                      {true, false, true, true, true},
                                                      {int64_t{100}, int64_t{300}, int64_t{-150}},
                                                      /*use_jit=*/false);
}

// ---------------------------------------------------------------------------
// DECIMAL128 range with nulls
// ---------------------------------------------------------------------------

TEST_F(ParquetDecimalPushdownTest, RowRangeDecimal128NullsJit)
{
  run_row_range_test<::numeric::decimal128>(
    cudf::type_id::DECIMAL128,
    {__int128_t{1000}, __int128_t{2000}, __int128_t{-1500}, __int128_t{3000}, __int128_t{4500}},
    {true, true, true, false, true},
    __int128_t{-2000},
    __int128_t{3500},
    /*use_jit=*/true);
}

TEST_F(ParquetDecimalPushdownTest, RowRangeDecimal128NullsAst)
{
  run_row_range_test<::numeric::decimal128>(
    cudf::type_id::DECIMAL128,
    {__int128_t{1000}, __int128_t{2000}, __int128_t{-1500}, __int128_t{3000}, __int128_t{4500}},
    {true, true, true, false, true},
    __int128_t{-2000},
    __int128_t{3500},
    /*use_jit=*/false);
}

// ---------------------------------------------------------------------------
// DECIMAL128 IN-list with large value
// ---------------------------------------------------------------------------

TEST_F(ParquetDecimalPushdownTest, RowInListDecimal128BigValueJit)
{
  run_row_in_list_test<::numeric::decimal128, __int128_t>(
    cudf::type_id::DECIMAL128,
    {__int128_t{1000}, __int128_t{2000}, __int128_t{-1500}, kBig128, __int128_t{4500}},
    {true, true, false, true, true},
    {__int128_t{1000}, kBig128, __int128_t{4500}},
    /*use_jit=*/true);
}

TEST_F(ParquetDecimalPushdownTest, RowInListDecimal128BigValueAst)
{
  run_row_in_list_test<::numeric::decimal128, __int128_t>(
    cudf::type_id::DECIMAL128,
    {__int128_t{1000}, __int128_t{2000}, __int128_t{-1500}, kBig128, __int128_t{4500}},
    {true, true, false, true, true},
    {__int128_t{1000}, kBig128, __int128_t{4500}},
    /*use_jit=*/false);
}

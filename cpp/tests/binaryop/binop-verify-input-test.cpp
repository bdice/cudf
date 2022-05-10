/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
 *
 * Copyright 2018-2019 BlazingDB, Inc.
 *     Copyright 2018 Christian Noboa Mardini <christian@blazingdb.com>
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

#include <cudf/binaryop.hpp>

#include <binaryop/compiled/binary_ops.hpp>

#include <tests/binaryop/binop-fixture.hpp>

#include <iostream>

namespace cudf {
namespace test {
namespace binop {
struct BinopVerifyInputTest : public BinaryOperationTest {
};

TEST_F(BinopVerifyInputTest, Vector_Scalar_ErrorOutputVectorType)
{
  using TypeLhs = int64_t;
  using TypeRhs = int64_t;

  auto lhs = make_random_wrapped_scalar<TypeLhs>();
  auto rhs = make_random_wrapped_column<TypeRhs>(10);

  EXPECT_THROW(
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::ADD, data_type(type_id::NUM_TYPE_IDS)),
    cudf::logic_error);
}

TEST_F(BinopVerifyInputTest, Vector_Vector_ErrorSecondOperandVectorZeroSize)
{
  using TypeOut = int64_t;
  using TypeLhs = int64_t;
  using TypeRhs = int64_t;

  auto lhs = make_random_wrapped_column<TypeLhs>(1);
  auto rhs = make_random_wrapped_column<TypeRhs>(10);

  EXPECT_THROW(
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::ADD, data_type(type_to_id<TypeOut>())),
    cudf::logic_error);
}

struct BinopTypeTest : public BinaryOperationTest {
};
TEST_F(BinopTypeTest, GetCommonTypeTest)
{
  auto const type_ids = std::vector<cudf::type_id>{
    // cudf::type_id::EMPTY,  ///< Always null with no underlying data
    cudf::type_id::INT8,               ///< 1 byte signed integer
    cudf::type_id::INT16,              ///< 2 byte signed integer
    cudf::type_id::INT32,              ///< 4 byte signed integer
    cudf::type_id::INT64,              ///< 8 byte signed integer
    cudf::type_id::UINT8,              ///< 1 byte unsigned integer
    cudf::type_id::UINT16,             ///< 2 byte unsigned integer
    cudf::type_id::UINT32,             ///< 4 byte unsigned integer
    cudf::type_id::UINT64,             ///< 8 byte unsigned integer
    cudf::type_id::FLOAT32,            ///< 4 byte floating point
    cudf::type_id::FLOAT64,            ///< 8 byte floating point
    cudf::type_id::BOOL8,              ///< Boolean using one byte per value, 0 == false, else true
    cudf::type_id::TIMESTAMP_DAYS,     ///< point in time in days since Unix Epoch in int32
    cudf::type_id::TIMESTAMP_SECONDS,  ///< point in time in seconds since Unix Epoch in int64
    cudf::type_id::TIMESTAMP_MILLISECONDS,  ///< point in time in milliseconds since Unix Epoch in
                                            ///< int64
    cudf::type_id::TIMESTAMP_MICROSECONDS,  ///< point in time in microseconds since Unix Epoch in
                                            ///< int64
    cudf::type_id::TIMESTAMP_NANOSECONDS,   ///< point in time in nanoseconds since Unix Epoch in
                                            ///< int64
    cudf::type_id::DURATION_DAYS,           ///< time interval of days in int32
    cudf::type_id::DURATION_SECONDS,        ///< time interval of seconds in int64
    cudf::type_id::DURATION_MILLISECONDS,   ///< time interval of milliseconds in int64
    cudf::type_id::DURATION_MICROSECONDS,   ///< time interval of microseconds in int64
    cudf::type_id::DURATION_NANOSECONDS,    ///< time interval of nanoseconds in int64
    cudf::type_id::DICTIONARY32,            ///< Dictionary type using int32 indices
    cudf::type_id::STRING,                  ///< String elements
    cudf::type_id::LIST,                    ///< List elements
    cudf::type_id::DECIMAL32,               ///< Fixed-point type with int32_t
    cudf::type_id::DECIMAL64,               ///< Fixed-point type with int64_t
    cudf::type_id::DECIMAL128,              ///< Fixed-point type with __int128_t
    cudf::type_id::STRUCT                   ///< Struct elements
  };
  for (auto const& t1 : type_ids) {
    for (auto const& t2 : type_ids) {
      for (auto const& t3 : type_ids) {
        auto const d1  = cudf::data_type(t1);
        auto const d2  = cudf::data_type(t2);
        auto const d3  = cudf::data_type(t3);
        auto const old = cudf::binops::compiled::get_common_type_old(d1, d2, d3);
        (void)old;
        auto const new_ = cudf::binops::compiled::get_common_type(d1, d2, d3);
        (void)new_;
        std::cerr << static_cast<int32_t>(t1) << ", " << static_cast<int32_t>(t2) << ", "
                  << static_cast<int32_t>(t3) << std::endl;
        EXPECT_EQ(old, new_);
      }
    }
  }
}

TEST_F(BinopTypeTest, IsSupportedOperationTest)
{
  auto const binops = std::vector<cudf::binary_operator>{
    cudf::binary_operator::ADD,        ///< operator +
    cudf::binary_operator::SUB,        ///< operator -
    cudf::binary_operator::MUL,        ///< operator *
    cudf::binary_operator::DIV,        ///< operator / using common type of lhs and rhs
    cudf::binary_operator::TRUE_DIV,   ///< operator / after promoting type to floating point
    cudf::binary_operator::FLOOR_DIV,  ///< operator / after promoting to 64 bit floating point and
                                       ///< then
    cudf::binary_operator::MOD,        ///< operator %
    cudf::binary_operator::PMOD,       ///< positive modulo operator
    cudf::binary_operator::PYMOD,  ///< operator % but following Python's sign rules for negatives
    cudf::binary_operator::POW,    ///< lhs ^ rhs
    cudf::binary_operator::LOG_BASE,              ///< logarithm to the base
    cudf::binary_operator::ATAN2,                 ///< 2-argument arctangent
    cudf::binary_operator::SHIFT_LEFT,            ///< operator <<
    cudf::binary_operator::SHIFT_RIGHT,           ///< operator >>
    cudf::binary_operator::SHIFT_RIGHT_UNSIGNED,  ///< operator >>> (from Java)
    cudf::binary_operator::BITWISE_AND,           ///< operator &
    cudf::binary_operator::BITWISE_OR,            ///< operator |
    cudf::binary_operator::BITWISE_XOR,           ///< operator ^
    cudf::binary_operator::LOGICAL_AND,           ///< operator &&
    cudf::binary_operator::LOGICAL_OR,            ///< operator ||
    cudf::binary_operator::EQUAL,                 ///< operator ==
    cudf::binary_operator::NOT_EQUAL,             ///< operator !=
    cudf::binary_operator::LESS,                  ///< operator <
    cudf::binary_operator::GREATER,               ///< operator >
    cudf::binary_operator::LESS_EQUAL,            ///< operator <=
    cudf::binary_operator::GREATER_EQUAL,         ///< operator >=
    cudf::binary_operator::NULL_EQUALS,  ///< Returns true when both operands are null; false when
                                         ///< one is null; the
    cudf::binary_operator::NULL_MAX,  ///< Returns max of operands when both are non-null; returns
                                      ///< the non-null
    cudf::binary_operator::NULL_MIN,  ///< Returns min of operands when both are non-null; returns
                                      ///< the non-null
    cudf::binary_operator::GENERIC_BINARY,  ///< generic binary operator to be generated with input
    cudf::binary_operator::NULL_LOGICAL_AND,  ///< operator && with Spark rules: (null, null) is
                                              ///< null, (null, true) is
    cudf::binary_operator::NULL_LOGICAL_OR,   ///< operator || with Spark rules: (null, null) is
                                              ///< null, (null, true) is true,
    cudf::binary_operator::INVALID_BINARY     ///< invalid operation
  };

  auto const type_ids = std::vector<cudf::type_id>{
    // cudf::type_id::EMPTY,  ///< Always null with no underlying data
    cudf::type_id::INT8,               ///< 1 byte signed integer
    cudf::type_id::INT16,              ///< 2 byte signed integer
    cudf::type_id::INT32,              ///< 4 byte signed integer
    cudf::type_id::INT64,              ///< 8 byte signed integer
    cudf::type_id::UINT8,              ///< 1 byte unsigned integer
    cudf::type_id::UINT16,             ///< 2 byte unsigned integer
    cudf::type_id::UINT32,             ///< 4 byte unsigned integer
    cudf::type_id::UINT64,             ///< 8 byte unsigned integer
    cudf::type_id::FLOAT32,            ///< 4 byte floating point
    cudf::type_id::FLOAT64,            ///< 8 byte floating point
    cudf::type_id::BOOL8,              ///< Boolean using one byte per value, 0 == false, else true
    cudf::type_id::TIMESTAMP_DAYS,     ///< point in time in days since Unix Epoch in int32
    cudf::type_id::TIMESTAMP_SECONDS,  ///< point in time in seconds since Unix Epoch in int64
    cudf::type_id::TIMESTAMP_MILLISECONDS,  ///< point in time in milliseconds since Unix Epoch in
                                            ///< int64
    cudf::type_id::TIMESTAMP_MICROSECONDS,  ///< point in time in microseconds since Unix Epoch in
                                            ///< int64
    cudf::type_id::TIMESTAMP_NANOSECONDS,   ///< point in time in nanoseconds since Unix Epoch in
                                            ///< int64
    cudf::type_id::DURATION_DAYS,           ///< time interval of days in int32
    cudf::type_id::DURATION_SECONDS,        ///< time interval of seconds in int64
    cudf::type_id::DURATION_MILLISECONDS,   ///< time interval of milliseconds in int64
    cudf::type_id::DURATION_MICROSECONDS,   ///< time interval of microseconds in int64
    cudf::type_id::DURATION_NANOSECONDS,    ///< time interval of nanoseconds in int64
    cudf::type_id::DICTIONARY32,            ///< Dictionary type using int32 indices
    cudf::type_id::STRING,                  ///< String elements
    cudf::type_id::LIST,                    ///< List elements
    cudf::type_id::DECIMAL32,               ///< Fixed-point type with int32_t
    cudf::type_id::DECIMAL64,               ///< Fixed-point type with int64_t
    cudf::type_id::DECIMAL128,              ///< Fixed-point type with __int128_t
    cudf::type_id::STRUCT                   ///< Struct elements
  };
  for (auto const& op : binops) {
    for (auto const& t1 : type_ids) {
      for (auto const& t2 : type_ids) {
        for (auto const& t3 : type_ids) {
          auto const d1   = cudf::data_type(t1);
          auto const d2   = cudf::data_type(t2);
          auto const d3   = cudf::data_type(t3);
          auto const old  = cudf::binops::compiled::is_supported_operation_old(d1, d2, d3, op);
          auto const new_ = cudf::binops::compiled::is_supported_operation(d1, d2, d3, op);
          std::cerr << static_cast<int32_t>(t1) << ", " << static_cast<int32_t>(t2) << ", "
                    << static_cast<int32_t>(t3) << ", " << static_cast<int32_t>(op) << std::endl;
          EXPECT_EQ(old, new_);
        }
      }
    }
  }
}

}  // namespace binop
}  // namespace test
}  // namespace cudf

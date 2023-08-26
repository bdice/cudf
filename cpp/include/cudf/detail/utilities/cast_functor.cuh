/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

/**
 * @brief A casting functor for use with CUB.
 * @file
 */

#include <cudf/types.hpp>

#include <cuda/functional>

#include <functional>
#include <type_traits>

namespace cudf {
namespace detail {

/**
 * @brief Functor that casts another functor's result to a specified type.
 *
 * CUB 2.0.0 reductions require that the binary operator returns the same type
 * as the initial value type, so we wrap binary operators with this when used
 * by CUB.
 */
template <typename ResultType, typename F>
struct cast_functor_fn {
  F f;

  template <typename... Ts>
  CUDF_HOST_DEVICE inline ResultType operator()(Ts&&... args)
  {
    return static_cast<ResultType>(f(std::forward<Ts>(args)...));
  }
};

/**
 * @brief Function creating a casting functor.
 */
template <typename ResultType, typename F>
cast_functor_fn<ResultType, std::decay_t<F>> cast_functor(F&& f)
{
  return cast_functor_fn<ResultType, std::decay_t<F>>{std::forward<F>(f)};
}

template <typename ResultType, typename F>
struct cast_device_functor_fn {
  F f;

  template <typename... Ts>
  __device__ inline ResultType operator()(Ts&&... args)
  {
    return static_cast<ResultType>(f(std::forward<Ts>(args)...));
  }
};

/**
 * @brief Function creating a casting functor.
 */
template <typename ResultType, typename F>
auto cast_device_functor(F&& f)
{
  auto cast_f = cast_device_functor_fn<ResultType, F>{std::forward<F>(f)};
  return cuda::proclaim_return_type<ResultType>(cast_f);
}

}  // namespace detail

}  // namespace cudf

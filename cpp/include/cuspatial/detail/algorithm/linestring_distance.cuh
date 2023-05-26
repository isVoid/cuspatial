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

#include <cuspatial/detail/utility/device_atomics.cuh>
#include <cuspatial/detail/utility/linestring.cuh>

#include <rmm/device_uvector.hpp>

#include <thrust/binary_search.h>
#include <thrust/optional.h>

#include <cub/cub.cuh>

#include <cooperative_groups.h>

#include <limits>

namespace cg = cooperative_groups;

namespace cuspatial {
namespace detail {

/**
 * @internal
 * @brief The kernel to compute linestring to linestring distance
 *
 * Each thread of the kernel computes the distance between a segment in a linestring in pair 1 to a
 * linestring in pair 2. For a segment in pair 1, the linestring index is looked up from the offset
 * array and mapped to the linestring in the pair 2. The segment is then computed with all segments
 * in the corresponding linestring in pair 2. This forms a local minima of the shortest distance,
 * which is then combined with other segment results via an atomic operation to form the global
 * minimum distance between the linestrings.
 *
 * `intersects` is an optional pointer to a boolean range where the `i`th element indicates the
 * `i`th output should be set to 0 and bypass distance computation. This argument is optional, if
 * set to nullopt, no distance computation will be bypassed.
 */
template <class MultiLinestringRange1, class MultiLinestringRange2, class OutputIt>
__global__ void linestring_distance(MultiLinestringRange1 multilinestrings1,
                                    MultiLinestringRange2 multilinestrings2,
                                    thrust::optional<uint8_t*> intersects,
                                    OutputIt distances_first)
{
  using T = typename MultiLinestringRange1::element_t;

  for (auto idx = threadIdx.x + blockIdx.x * blockDim.x; idx < multilinestrings1.num_points();
       idx += gridDim.x * blockDim.x) {
    auto const part_idx = multilinestrings1.part_idx_from_point_idx(idx);
    if (!multilinestrings1.is_valid_segment_id(idx, part_idx)) continue;
    auto const geometry_idx = multilinestrings1.geometry_idx_from_part_idx(part_idx);

    if (intersects.has_value() && intersects.value()[geometry_idx]) {
      distances_first[geometry_idx] = 0;
      continue;
    }

    auto [a, b]            = multilinestrings1.segment(idx);
    T min_distance_squared = std::numeric_limits<T>::max();

    for (auto const& linestring2 : multilinestrings2[geometry_idx]) {
      for (auto [c, d] : linestring2) {
        min_distance_squared = min(min_distance_squared, squared_segment_distance(a, b, c, d));
      }
    }
    atomicMin(&distances_first[geometry_idx], static_cast<T>(sqrt(min_distance_squared)));
  }
}

template <typename T, typename MultiLineString>
auto __device__ segment_multilinestring_distance(segment<T> s, MultiLineString multilinestring)
{
  T min_distance_squared = std::numeric_limits<T>::max();

  auto [a, b] = s;
  for (auto const& linestring : multilinestring)
    for (auto [c, d] : linestring)
    {
      // printf("blockid: %d from: (%f, %f) -> (%f, %f) to: (%f, %f) -> (%f, %f)\n", static_cast<int>(cg::this_grid().block_rank()), a.x, a.y, b.x, b.y, c.x, c.y, d.x, d.y);
      min_distance_squared = min(min_distance_squared, squared_segment_distance(a, b, c, d));
    }

  return min_distance_squared;
}

template <typename MultiLineString1, typename MultiLineString2>
auto __device__ linestring_distance_thread(
                                           MultiLineString1 lhs,
                                           MultiLineString2 rhs)
{
  using T = typename MultiLineString1::element_t;

  auto rank = cg::this_thread_block().thread_rank();
  auto block_size            = cg::this_thread_block().size();
  std::size_t points_per_thread = (lhs.num_points() + block_size - 1) / block_size;

  T min_distance_squared = std::numeric_limits<T>::max();

  // printf("thread_rank: %d points_per_thread: %d\n", static_cast<int>(rank), static_cast<int>(points_per_thread));

  for (auto i = rank * points_per_thread;
       i < (rank + 1) * points_per_thread && i < lhs.num_points();
       ++i) {
    auto it = thrust::upper_bound(thrust::seq, lhs.local_part_begin(), lhs.local_part_end(), i);
    auto local_part_idx = thrust::distance(lhs.local_part_begin(), thrust::prev(it));
    if (!lhs.is_valid_segment_id(i, local_part_idx)) continue;

    vec_2d<T> a = lhs.point_begin()[i];
    vec_2d<T> b = lhs.point_begin()[i + 1];

    min_distance_squared =
      min(min_distance_squared, segment_multilinestring_distance(segment<T>{a, b}, rhs));

    // printf("thread_rank: %d local_point_index: %d local_part_idx: %d \n",
    //        static_cast<int>(rank),
    //        static_cast<int>(i),
    //        static_cast<int>(local_part_idx));

  }

  return min_distance_squared;
}

template <std::size_t BlockSize, typename MultiLinestringsIter1, typename MultiLinestringsIter2, typename OutputIt>
void __global__ linestring_distance_block(MultiLinestringsIter1 lhs,
                                          MultiLinestringsIter2 rhs,
                                          OutputIt dist)
{
  using T = typename MultiLinestringsIter1::element_t;
  using BlockReduce = cub::BlockReduce<T, BlockSize>;

  auto id  = cg::this_grid().block_rank();
  // printf("BlockID: %d\n", static_cast<int>(id));
  T partial = std::sqrt(linestring_distance_thread(lhs[id], rhs[id]));

  __shared__ typename BlockReduce::TempStorage temp_storage;

  T result = BlockReduce(temp_storage).Reduce(partial, cub::Min());

  if (cg::this_thread_block().thread_rank() == 0) dist[id] = result;
}

}  // namespace detail
}  // namespace cuspatial

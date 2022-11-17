/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cuspatial_test/test_util.cuh>

#include <cuspatial/cuda_utils.hpp>
#include <cuspatial/detail/utility/device_atomics.cuh>
#include <cuspatial/detail/utility/linestring.cuh>
#include <cuspatial/error.hpp>
#include <cuspatial/experimental/detail/linestring_intersection_count.cuh>
#include <cuspatial/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/tuple.h>
#include <thrust/uninitialized_fill.h>

#include <cuda/atomic>

#include <cstdint>

namespace cuspatial {

template <typename T, typename OffsetType>
struct intersection_result;

namespace detail {

/**
 * @brief Compute union column's offset buffer
 *
 * This is performing a group-by cummulative sum (pandas semantic) operation
 * to an "all 1s vector", using `types_buffer` as the key column.
 */
template <typename index_t>
rmm::device_uvector<index_t> compute_offset_buffer(rmm::device_uvector<uint8_t> const& types_buffer,
                                                   rmm::mr::device_memory_resource* mr,
                                                   rmm::cuda_stream_view stream)
{
  auto N            = types_buffer.size();
  auto keys_copy    = rmm::device_uvector(types_buffer, stream);
  auto indices_temp = rmm::device_uvector<index_t>(N, stream);
  thrust::sequence(rmm::exec_policy(stream), indices_temp.begin(), indices_temp.end());
  thrust::stable_sort_by_key(
    rmm::exec_policy(stream), keys_copy.begin(), keys_copy.end(), indices_temp.begin());
  auto offset_buffer = rmm::device_uvector<index_t>(N, stream, mr);
  thrust::uninitialized_fill_n(rmm::exec_policy(stream), offset_buffer.begin(), N, 1);
  thrust::exclusive_scan_by_key(rmm::exec_policy(stream),
                                keys_copy.begin(),
                                keys_copy.end(),
                                offset_buffer.begin(),
                                offset_buffer.begin());
  thrust::scatter(rmm::exec_policy(stream),
                  offset_buffer.begin(),
                  offset_buffer.end(),
                  indices_temp.begin(),
                  offset_buffer.begin());
  return offset_buffer;
}

/**
 * @brief Kernel to compute the linestring intersections and writes the result to the output buffer
 *
 * Use naive algorithm of O(N^2).
 *
 * @tparam MultiLinestringRange1
 * @tparam MultiLinestringRange2
 * @tparam TempIt1
 * @tparam TempIt2
 * @tparam Offsets1
 * @tparam Offsets2
 * @tparam OutputIt1
 * @tparam OutputIt2
 * @param multilinestrings1
 * @param multilinestrings2
 * @param n_points_stored
 * @param n_segments_stored
 * @param num_points_offsets_first
 * @param num_segments_offsets_first
 * @param points_first
 * @param segments_first
 */
template <typename MultiLinestringRange1,
          typename MultiLinestringRange2,
          typename TempIt1,
          typename TempIt2,
          typename Offsets1,
          typename Offsets2,
          typename Offsets3,
          typename Offsets4,
          typename Types1,
          typename Offsets5,
          typename OutputIt1,
          typename OutputIt2>
void __global__ pairwise_linestring_intersection_simple(MultiLinestringRange1 multilinestrings1,
                                                        MultiLinestringRange2 multilinestrings2,
                                                        TempIt1 n_points_stored,
                                                        TempIt2 n_segments_stored,
                                                        Offsets1 num_points_offsets_first,
                                                        Offsets2 num_segments_offsets_first,
                                                        Offsets3 geometry_collection_offset_first,
                                                        Offsets4 num_points_per_pair_first,
                                                        Types1 types_code_first,
                                                        Offsets5 lhs_linestring_id_first,
                                                        Offsets5 lhs_segment_id_first,
                                                        Offsets5 rhs_linestring_id_first,
                                                        Offsets5 rhs_segment_id_first,
                                                        OutputIt1 points_first,
                                                        OutputIt2 segments_first)
{
  using T       = typename MultiLinestringRange1::element_t;
  using types_t = uint8_t;
  using count_t = iterator_value_type<Offsets1>;
  for (auto idx = threadIdx.x + blockIdx.x * blockDim.x; idx < multilinestrings1.num_points();
       idx += gridDim.x * blockDim.x) {
    auto const part_idx = multilinestrings1.part_idx_from_point_idx(idx);
    if (!multilinestrings1.is_valid_segment_id(idx, part_idx)) continue;
    auto const lhs_linestring_idx = multilinestrings1.intra_part_idx(part_idx);
    auto const lhs_segment_idx    = multilinestrings1.intra_point_idx(idx);
    auto [a, b]                   = multilinestrings1.segment(idx);
    auto const geometry_idx       = multilinestrings1.geometry_idx_from_part_idx(part_idx);
    auto const multilinestring2   = multilinestrings2[geometry_idx];

    for (auto rhs_linestring_idx = 0; rhs_linestring_idx < multilinestring2.size();
         ++rhs_linestring_idx) {
      auto const linestring2 = multilinestring2[rhs_linestring_idx];
      for (auto rhs_segment_idx = 0; rhs_segment_idx < linestring2.num_segments();
           ++rhs_segment_idx) {
        auto [c, d]                   = linestring2.segment(rhs_segment_idx);
        auto [point_opt, segment_opt] = segment_intersection(segment<T>{a, b}, segment<T>{c, d});

        if (point_opt.has_value()) {
          auto r              = cuda::atomic_ref<count_t>{*(n_points_stored + geometry_idx)};
          auto next_point_idx = r.fetch_add(1);
          points_first[num_points_offsets_first[geometry_idx] + next_point_idx] = point_opt.value();
          auto union_column_idx = geometry_collection_offset_first[geometry_idx] + next_point_idx;
          types_code_first[union_column_idx]        = IntersectionTypeCode::POINT;
          lhs_linestring_id_first[union_column_idx] = lhs_linestring_idx;
          lhs_segment_id_first[union_column_idx]    = lhs_segment_idx;
          rhs_linestring_id_first[union_column_idx] = rhs_linestring_idx;
          rhs_segment_id_first[union_column_idx]    = rhs_segment_idx;
        } else if (segment_opt.has_value()) {
          auto r                = cuda::atomic_ref<count_t>{*(n_segments_stored + geometry_idx)};
          auto next_segment_idx = r.fetch_add(1);
          segments_first[num_segments_offsets_first[geometry_idx] + next_segment_idx] =
            segment_opt.value();
          auto union_column_idx = geometry_collection_offset_first[geometry_idx] +
                                  num_points_per_pair_first[geometry_idx] + next_segment_idx;
          types_code_first[union_column_idx]        = IntersectionTypeCode::LINESTRING;
          lhs_linestring_id_first[union_column_idx] = lhs_linestring_idx;
          lhs_segment_id_first[union_column_idx]    = lhs_segment_idx;
          rhs_linestring_id_first[union_column_idx] = rhs_linestring_idx;
          rhs_segment_id_first[union_column_idx]    = rhs_segment_idx;
        }
      }
    }
  }
}

}  // namespace detail

/**
 * @brief Compute intersections between multilnestrings.
 */
template <typename MultiLinestringRange1,
          typename MultiLinestringRange2,
          typename index_t,
          typename T>
intersection_result<T, index_t> pairwise_linestring_intersection_with_duplicate(
  MultiLinestringRange1 multilinestrings1,
  MultiLinestringRange2 multilinestrings2,
  rmm::mr::device_memory_resource* mr,
  rmm::cuda_stream_view stream)
{
  using types_t = typename intersection_result<T, index_t>::types_t;

  static_assert(is_same_floating_point<T, typename MultiLinestringRange2::element_t>(),
                "Inputs and output must have the same floating point value type.");

  static_assert(is_same<vec_2d<T>,
                        typename MultiLinestringRange1::point_t,
                        typename MultiLinestringRange2::point_t>(),
                "All input types must be cuspatial::vec_2d with the same value type");

  CUSPATIAL_EXPECTS(multilinestrings1.size() == multilinestrings2.size(),
                    "The size input multilinestrings mismatch.");

  auto const num_pairs = multilinestrings1.size();

  // Compute the upper bound of spaces required to store intersection results.
  rmm::device_uvector<index_t> num_points_per_pair(num_pairs, stream);
  rmm::device_uvector<index_t> num_segments_per_pair(num_pairs, stream);

  thrust::uninitialized_fill_n(rmm::exec_policy(stream), num_points_per_pair.begin(), num_pairs, 0);
  thrust::uninitialized_fill_n(
    rmm::exec_policy(stream), num_segments_per_pair.begin(), num_pairs, 0);

  detail::pairwise_linestring_intersection_upper_bound_count(multilinestrings1,
                                                             multilinestrings2,
                                                             num_points_per_pair.begin(),
                                                             num_segments_per_pair.begin(),
                                                             stream);

  // Allocate the space needed to store the result point and segments.
  auto num_points = thrust::reduce(
    rmm::exec_policy(stream), num_points_per_pair.begin(), num_points_per_pair.end());
  auto num_segments = thrust::reduce(
    rmm::exec_policy(stream), num_segments_per_pair.begin(), num_segments_per_pair.end());

  rmm::device_uvector<vec_2d<T>> points(num_points, stream, mr);
  rmm::device_uvector<segment<T>> segments(num_segments, stream, mr);

  // Compute the offset from which the thread should start writing results
  rmm::device_uvector<index_t> num_points_offsets(num_points_per_pair, stream);
  rmm::device_uvector<index_t> num_segments_offsets(num_segments_per_pair, stream);

  thrust::exclusive_scan(rmm::exec_policy(stream),
                         num_points_offsets.begin(),
                         num_points_offsets.end(),
                         num_points_offsets.begin());
  thrust::exclusive_scan(rmm::exec_policy(stream),
                         num_segments_offsets.begin(),
                         num_segments_offsets.end(),
                         num_segments_offsets.begin());

  // Allocate a temporary vector so that each thread can keep track of how many results
  // of the current multilinestring pair has written.
  rmm::device_uvector<index_t> num_points_stored_temp(num_pairs, stream);
  rmm::device_uvector<index_t> num_segments_stored_temp(num_pairs, stream);

  thrust::uninitialized_fill_n(
    rmm::exec_policy(stream), num_points_stored_temp.begin(), num_pairs, 0);
  thrust::uninitialized_fill_n(
    rmm::exec_policy(stream), num_segments_stored_temp.begin(), num_pairs, 0);

  // Compute GeometryCollectionOffset
  rmm::device_uvector<index_t> geometry_collection_offset(num_pairs + 1, stream, mr);
  thrust::uninitialized_fill_n(
    rmm::exec_policy(stream), geometry_collection_offset.begin(), num_pairs + 1, 0);
  auto num_points_segment_per_pair_it =
    thrust::make_zip_iterator(num_segments_per_pair.begin(), num_points_per_pair.begin());

  auto geometry_collection_output_it = thrust::next(geometry_collection_offset.begin());
  thrust::transform(rmm::exec_policy(stream),
                    num_points_segment_per_pair_it,
                    num_points_segment_per_pair_it + num_pairs,
                    geometry_collection_output_it,
                    [] __device__(auto p) {
                      index_t num_points, num_segments;
                      thrust::tie(num_points, num_segments) = p;
                      return num_points + num_segments;
                    });

  thrust::inclusive_scan(rmm::exec_policy(stream),
                         geometry_collection_offset.begin(),
                         geometry_collection_offset.end(),
                         geometry_collection_offset.begin());

  // Allocate types buffer
  auto num_union_column_rows = num_points + num_segments;
  rmm::device_uvector<uint8_t> types_buffer(num_union_column_rows, stream, mr);

  // Compute the intersections
  auto [threads_per_block, num_blocks] = grid_1d(multilinestrings1.num_points());

  // Allocate buffer for the look-back indices
  rmm::device_uvector<index_t> lhs_linestring_id(num_union_column_rows, stream, mr);
  rmm::device_uvector<index_t> lhs_segment_id(num_union_column_rows, stream, mr);
  rmm::device_uvector<index_t> rhs_linestring_id(num_union_column_rows, stream, mr);
  rmm::device_uvector<index_t> rhs_segment_id(num_union_column_rows, stream, mr);

  detail::
    pairwise_linestring_intersection_simple<<<num_blocks, threads_per_block, 0, stream.value()>>>(
      multilinestrings1,
      multilinestrings2,
      num_points_stored_temp.begin(),
      num_segments_stored_temp.begin(),
      num_points_offsets.begin(),
      num_segments_offsets.begin(),
      geometry_collection_offset.begin(),
      num_points_per_pair.begin(),
      types_buffer.begin(),
      lhs_linestring_id.begin(),
      lhs_segment_id.begin(),
      rhs_linestring_id.begin(),
      rhs_segment_id.begin(),
      points.begin(),
      segments.begin());

  // Types buffer is computed, computed offsets buffer.
  auto offsets_buffer = detail::compute_offset_buffer<index_t>(types_buffer, mr, stream);

  auto dummy = rmm::device_uvector<index_t>(0, stream, mr);
  return intersection_result<T, index_t>{std::move(geometry_collection_offset),
                                         std::move(types_buffer),
                                         std::move(offsets_buffer),
                                         std::move(points),
                                         std::move(segments),
                                         std::move(lhs_linestring_id),
                                         std::move(lhs_segment_id),
                                         std::move(rhs_linestring_id),
                                         std::move(rhs_segment_id)};
}

}  // namespace cuspatial

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

template <typename T>
struct intersection_result;

namespace detail {

enum IntersectionTypeCode : uint8_t { MULTIPOINT = 0, MULTILINESTRING = 1 };

template <typename Int>
void __device__ print(Int i)
{
  printf("%d ", static_cast<int>(i));
}

rmm::device_uvector<std::size_t> compute_offset_buffer(
  rmm::device_uvector<uint8_t> const& types_buffer,
  rmm::mr::device_memory_resource* mr,
  rmm::cuda_stream_view stream)
{
  auto N            = types_buffer.size();
  auto keys_copy    = rmm::device_uvector(types_buffer, stream);
  auto indices_temp = rmm::device_uvector<std::size_t>(N, stream);
  thrust::sequence(rmm::exec_policy(stream), indices_temp.begin(), indices_temp.end());
  thrust::stable_sort_by_key(
    rmm::exec_policy(stream), keys_copy.begin(), keys_copy.end(), indices_temp.begin());
  auto offset_buffer = rmm::device_uvector<std::size_t>(N, stream, mr);
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

template <typename OffsetIterator, typename CountIteratorA, typename CountIteratorB>
struct types_buffer_functor {
  OffsetIterator _geometry_offset_begin;
  OffsetIterator _geometry_offset_end;

  CountIteratorA _point_count_begin;
  CountIteratorB _segment_count_begin;

  types_buffer_functor(OffsetIterator geometry_offset_begin,
                       OffsetIterator geometry_offset_end,
                       CountIteratorA point_count_begin,
                       CountIteratorB segment_count_begin)
    : _geometry_offset_begin(geometry_offset_begin),
      _geometry_offset_end(geometry_offset_end),
      _point_count_begin(point_count_begin),
      _segment_count_begin(segment_count_begin)
  {
  }

  uint8_t __device__ operator()(std::size_t idx)
  {
    auto geometry_iter = thrust::prev(
      thrust::upper_bound(thrust::seq, _geometry_offset_begin, _geometry_offset_end, idx));
    auto geometry_idx = thrust::distance(_geometry_offset_begin, geometry_iter);

    if (idx == 3) {
      print(idx);
      print(geometry_idx);
      print(_point_count_begin[geometry_idx]);
      print(_segment_count_begin[geometry_idx]);
      printf("\n");
    }

    if (_point_count_begin[geometry_idx] == 0 && _segment_count_begin[geometry_idx] == 0)
      return;
    else if (_point_count_begin[geometry_idx] == 0)
      return IntersectionTypeCode::MULTILINESTRING;
    else if (_segment_count_begin[geometry_idx] == 0)
      return IntersectionTypeCode::MULTIPOINT;
    else {
      // This group contains both type of geometries
      // In each group, we (arbitrarily) enforce that multipoint should precede multilinestring.
      if (idx == _geometry_offset_begin[geometry_idx]) { return IntersectionTypeCode::MULTIPOINT; }
      return IntersectionTypeCode::MULTILINESTRING;
    }
  }
};

template <typename MultiLinestringRange1,
          typename MultiLinestringRange2,
          typename TempIt1,
          typename TempIt2,
          typename Offsets1,
          typename Offsets2,
          typename OutputIt1,
          typename OutputIt2>
void __global__ pairwise_linestring_intersection_simple(MultiLinestringRange1 multilinestrings1,
                                                        MultiLinestringRange2 multilinestrings2,
                                                        TempIt1 n_points_stored,
                                                        TempIt2 n_segments_stored,
                                                        Offsets1 num_points_offsets_first,
                                                        Offsets2 num_segments_offsets_first,
                                                        OutputIt1 points_first,
                                                        OutputIt2 segments_first)
{
  using T          = typename MultiLinestringRange1::element_t;
  using types_t    = uint8_t;
  using count_type = unsigned int;  // TODO: dynamically infer
  for (auto idx = threadIdx.x + blockIdx.x * blockDim.x; idx < multilinestrings1.num_points();
       idx += gridDim.x * blockDim.x) {
    auto const part_idx = multilinestrings1.part_idx_from_point_idx(idx);
    if (!multilinestrings1.is_valid_segment_id(idx, part_idx)) continue;
    int32_t const geometry_idx = multilinestrings1.geometry_idx_from_part_idx(part_idx);
    auto [a, b]                = multilinestrings1.segment(idx);
    for (auto const& linestring2 : multilinestrings2[geometry_idx]) {
      for (auto [c, d] : linestring2) {
        auto [point_opt, segment_opt] = segment_intersection(segment<T>{a, b}, segment<T>{c, d});
        if (point_opt.has_value()) {
          auto r              = cuda::atomic_ref<std::size_t>{*(n_points_stored + geometry_idx)};
          auto next_point_idx = r.fetch_add(1);
          points_first[num_points_offsets_first[geometry_idx] + next_point_idx] = point_opt.value();
        } else if (segment_opt.has_value()) {
          auto r = cuda::atomic_ref<std::size_t>{*(n_segments_stored + geometry_idx)};
          auto next_segment_idx = r.fetch_add(1);
          segments_first[num_segments_offsets_first[geometry_idx] + next_segment_idx] =
            segment_opt.value();
        }
      }
    }
  }
}

/**
 * @brief Compute the geometry offset from the number of intersections/overlaps per pair, applicable
 * to both intersecting points and overlapping segments.
 */
rmm::device_uvector<std::size_t> compute_geometry_offsets(
  rmm::device_uvector<std::size_t> const& num_intersections_per_pair,
  rmm::mr::device_memory_resource* mr,
  rmm::cuda_stream_view stream)
{
  rmm::device_uvector<std::size_t> offsets_temp(num_intersections_per_pair, stream);
  auto offset_end  = thrust::remove_if(rmm::exec_policy(stream),
                                      offsets_temp.begin(),
                                      offsets_temp.end(),
                                      [] __device__(std::size_t const& i) { return i == 0; });
  std::size_t size = thrust::distance(offsets_temp.begin(), offset_end);
  rmm::device_uvector<std::size_t> offsets(size + 1, stream, mr);
  thrust::uninitialized_fill_n(rmm::exec_policy(stream), offsets.begin(), size + 1, 0);
  thrust::inclusive_scan(
    rmm::exec_policy(stream), offsets_temp.begin(), offset_end, thrust::next(offsets.begin()));
  return offsets;
}

}  // namespace detail

/**
 * @brief Compute the number of intersections between multilnestrings.
 */
template <typename MultiLinestringRange1, typename MultiLinestringRange2, typename T>
intersection_result<T> pairwise_linestring_intersection_with_duplicate(
  MultiLinestringRange1 multilinestrings1,
  MultiLinestringRange2 multilinestrings2,
  rmm::mr::device_memory_resource* mr,
  rmm::cuda_stream_view stream)
{
  // TODO type checks..
  using types_t = uint8_t;

  CUSPATIAL_EXPECTS(multilinestrings1.size() == multilinestrings2.size(),
                    "The size input multilinestrings mismatch.");
  auto const num_pairs = multilinestrings1.size();

  // Step 1: Compute the upper bound of spaces required to store intersection results.
  rmm::device_uvector<std::size_t> num_points_per_pair(num_pairs, stream);
  rmm::device_uvector<std::size_t> num_segments_per_pair(num_pairs, stream);

  thrust::uninitialized_fill_n(rmm::exec_policy(stream), num_points_per_pair.begin(), num_pairs, 0);
  thrust::uninitialized_fill_n(
    rmm::exec_policy(stream), num_segments_per_pair.begin(), num_pairs, 0);

  detail::pairwise_linestring_intersection_upper_bound_count(multilinestrings1,
                                                             multilinestrings2,
                                                             num_points_per_pair.begin(),
                                                             num_segments_per_pair.begin(),
                                                             stream);

  cuspatial::test::print_device_vector(num_points_per_pair);
  cuspatial::test::print_device_vector(num_segments_per_pair);

  std::cout << "Step2a: compute geometry offsets" << std::endl;

  // Compute geometry offsets for the points and the segments
  auto points_geometry_offsets = detail::compute_geometry_offsets(num_points_per_pair, mr, stream);
  auto segments_geometry_offsets =
    detail::compute_geometry_offsets(num_segments_per_pair, mr, stream);

  std::cout << "Step2c: allocate points and segments" << std::endl;

  // Allocate the space needed to store the result point and segments.
  auto num_points = thrust::reduce(
    rmm::exec_policy(stream), num_points_per_pair.begin(), num_points_per_pair.end());
  auto num_segments = thrust::reduce(
    rmm::exec_policy(stream), num_segments_per_pair.begin(), num_segments_per_pair.end());

  rmm::device_uvector<vec_2d<T>> points(num_points, stream, mr);
  rmm::device_uvector<segment<T>> segments(num_segments, stream, mr);

  std::cout << "Step2d: compute offsets from which the thread should start writing results."
            << std::endl;

  // Compute the offset from which the thread should start writing results
  rmm::device_uvector<std::size_t> num_points_offsets(num_points_per_pair, stream);
  rmm::device_uvector<std::size_t> num_segments_offsets(num_segments_per_pair, stream);

  thrust::exclusive_scan(rmm::exec_policy(stream),
                         num_points_offsets.begin(),
                         num_points_offsets.end(),
                         num_points_offsets.begin());
  thrust::exclusive_scan(rmm::exec_policy(stream),
                         num_segments_offsets.begin(),
                         num_segments_offsets.end(),
                         num_segments_offsets.begin());

  std::cout << "Step2e: allocate temporary buffer where thread should start writing results to."
            << std::endl;

  // Allocate a temporary vector such that each thead can keep track of how many results
  // of the current multilinestring pair has written.
  rmm::device_uvector<std::size_t> num_geometries_stored_temp(num_pairs, stream);
  rmm::device_uvector<std::size_t> num_points_stored_temp(num_pairs, stream);
  rmm::device_uvector<std::size_t> num_segments_stored_temp(num_pairs, stream);

  thrust::uninitialized_fill_n(
    rmm::exec_policy(stream), num_geometries_stored_temp.begin(), num_pairs, 0);
  thrust::uninitialized_fill_n(
    rmm::exec_policy(stream), num_points_stored_temp.begin(), num_pairs, 0);
  thrust::uninitialized_fill_n(
    rmm::exec_policy(stream), num_segments_stored_temp.begin(), num_pairs, 0);

  std::cout << "Step2f: geometry collection offsets." << std::endl;

  // Compute GeometryCollectionOffset
  rmm::device_uvector<std::size_t> geometry_collection_offset(num_pairs + 1, stream, mr);
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
                      std::size_t num_points, num_segments;
                      thrust::tie(num_points, num_segments) = p;
                      return int(num_points > 0) + int(num_segments > 0);
                    });

  thrust::inclusive_scan(rmm::exec_policy(stream),
                         geometry_collection_offset.begin(),
                         geometry_collection_offset.end(),
                         geometry_collection_offset.begin());

  // Compute types_buffer and offsets_buffer
  std::cout << "Step2b: compute types and offsets buffer" << std::endl;

  auto num_union_column_rows =
    points_geometry_offsets.size() + segments_geometry_offsets.size() - 2;

  rmm::device_uvector<uint8_t> types_buffer(num_union_column_rows, stream, mr);

  thrust::tabulate(rmm::exec_policy(stream),
                   types_buffer.begin(),
                   types_buffer.end(),
                   detail::types_buffer_functor{geometry_collection_offset.begin(),
                                                geometry_collection_offset.end(),
                                                num_points_per_pair.begin(),
                                                num_segments_per_pair.begin()});

  auto offsets_buffer = detail::compute_offset_buffer(types_buffer, mr, stream);

  std::cout << "Step2g: invoke geometry computation kernel." << std::endl;

  // Step 2: Compute the intersections
  auto [threads_per_block, num_blocks] = grid_1d(multilinestrings1.num_points());

  cuspatial::test::print_device_vector(num_points_offsets);
  cuspatial::test::print_device_vector(num_segments_offsets);

  detail::
    pairwise_linestring_intersection_simple<<<num_blocks, threads_per_block, 0, stream.value()>>>(
      multilinestrings1,
      multilinestrings2,
      num_points_stored_temp.begin(),
      num_segments_stored_temp.begin(),
      num_points_offsets.begin(),
      num_segments_offsets.begin(),
      points.begin(),
      segments.begin());

  auto dummy = rmm::device_uvector<std::size_t>(0, stream, mr);
  return intersection_result<T>{std::move(geometry_collection_offset),
                                std::move(types_buffer),
                                std::move(offsets_buffer),
                                std::move(points_geometry_offsets),
                                std::move(points),
                                std::move(segments_geometry_offsets),
                                std::move(segments),
                                std::move(dummy),
                                std::move(dummy),
                                std::move(dummy),
                                std::move(dummy)};
}

}  // namespace cuspatial

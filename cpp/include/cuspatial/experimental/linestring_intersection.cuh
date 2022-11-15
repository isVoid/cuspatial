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

#include <cuspatial/experimental/geometry/segment.cuh>
#include <cuspatial/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <thrust/pair.h>

namespace cuspatial {

enum IntersectionTypeCode : uint8_t { MULTIPOINT = 0, MULTILINESTRING = 1 };

template <typename T, typename OffsetType>
struct intersection_result {
  using point_t   = vec_2d<T>;
  using segment_t = segment<T>;
  using types_t   = uint8_t;
  using index_t   = OffsetType;

  rmm::device_uvector<index_t> geometry_collection_offset;

  rmm::device_uvector<types_t> types_buffer;
  rmm::device_uvector<index_t> offset_buffer;

  // Point Results
  rmm::device_uvector<index_t> points_geometry_offsets;
  rmm::device_uvector<point_t> points_coords;

  // Segment Results
  rmm::device_uvector<index_t> segments_geometry_offset;
  rmm::device_uvector<segment_t> segments_coords;

  // look-back indices
  rmm::device_uvector<index_t> lhs_linestring_id;
  rmm::device_uvector<index_t> lhs_segment_id;
  rmm::device_uvector<index_t> rhs_linestring_id;
  rmm::device_uvector<index_t> rhs_segment_id;
};

/**
 * @brief Compute the intersections between multilinestrings and ids to the intersecting
 * linestrings.
 */
template <typename MultiLinestringRange1,
          typename MultiLinestringRange2,
          typename index_t = std::size_t,
          typename T       = typename MultiLinestringRange1::element_t>
intersection_result<T, index_t> pairwise_linestring_intersection_with_duplicate(
  MultiLinestringRange1 multilinestrings1,
  MultiLinestringRange2 multilinestrings2,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default);

}  // namespace cuspatial

#include <cuspatial/experimental/detail/linestring_intersection.cuh>

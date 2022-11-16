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

#include <cuspatial_test/vector_equality.hpp>
#include <cuspatial_test/vector_factories.cuh>

#include <cuspatial/error.hpp>
#include <cuspatial/experimental/iterator_factory.cuh>
#include <cuspatial/experimental/linestring_intersection.cuh>
#include <cuspatial/traits.hpp>
#include <cuspatial/vec_2d.hpp>

#include <rmm/device_vector.hpp>

#include <rmm/mr/device/pool_memory_resource.hpp>
#include <thrust/iterator/zip_iterator.h>

#include <initializer_list>
#include <type_traits>

using namespace cuspatial;
using namespace cuspatial::test;

template <typename SegmentVector, typename T = typename SegmentVector::value_type::value_type>
std::pair<rmm::device_vector<vec_2d<T>>, rmm::device_vector<vec_2d<T>>> unpack_segment_vector(
  SegmentVector const& segments)
{
  rmm::device_vector<vec_2d<T>> first(segments.size()), second(segments.size());
  auto zipped_output = thrust::make_zip_iterator(first.begin(), second.begin());
  thrust::transform(
    segments.begin(), segments.end(), zipped_output, [] __device__(segment<T> const& segment) {
      return thrust::make_tuple(segment.first, segment.second);
    });
  return {std::move(first), std::move(second)};
}

template <typename SegmentVector1, typename SegmentVector2>
void expect_segment_equivalent(SegmentVector1 expected, SegmentVector2 got)
{
  auto [expected_first, expected_second] = unpack_segment_vector(expected);
  auto [got_first, got_second]           = unpack_segment_vector(got);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_first, got_first);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_second, got_second);
}

template <typename T>
struct LinestringIntersectionTest : public ::testing::Test {};

// float and double are logically the same but would require seperate tests due to precision.
using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(LinestringIntersectionTest, TestTypes);

TYPED_TEST(LinestringIntersectionTest, Example)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  using index_t = typename intersection_result<T, std::size_t>::index_t;
  using types_t = typename intersection_result<T, std::size_t>::types_t;

  auto multilinestrings1 = make_multilinestring_array({0, 1, 2, 3, 4, 5, 6, 7},
                                                      {0, 2, 4, 6, 8, 10, 12, 14},
                                                      {P{0, 0},
                                                       P{1, 1},
                                                       P{0, 0},
                                                       P{1, 1},
                                                       P{0, 0},
                                                       P{1, 1},
                                                       P{0, 0},
                                                       P{1, 1},
                                                       P{0, 0},
                                                       P{1, 1},
                                                       P{0, 0},
                                                       P{1, 1},
                                                       P{0, 0},
                                                       P{1, 1}});

  auto multilinestrings2 = make_multilinestring_array(
    {0, 1, 2, 3, 4, 5, 6, 7},
    {0, 2, 5, 7, 12, 16, 18, 20},
    {P{1, 0},       P{0, 1},     P{0.5, 0},    P{0, 0.5},     P{1, 0.5},
     P{0.5, 0.5},   P{1.5, 1.5}, P{-1, -1},    P{0.25, 0.25}, P{0.25, 0.0},
     P{0.75, 0.75}, P{1.5, 1.5}, P{0.25, 0.0}, P{0.25, 0.5},  P{0.75, 0.75},
     P{1.5, 1.5},   P{2, 2},     P{3, 3},      P{1, 0},       P{2, 0}});

  auto got = pairwise_linestring_intersection_with_duplicate(multilinestrings1.range(),
                                                             multilinestrings2.range());

  auto expected_geometry_collection_offset =
    make_device_vector<index_t>({0, 1, 3, 4, 8, 11, 11, 11});
  auto expected_types_buffer  = make_device_vector<types_t>({0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1});
  auto expected_offset_buffer = make_device_vector<index_t>({0, 1, 2, 0, 3, 4, 1, 2, 5, 6, 3});
  auto expected_points_coords = make_device_vector<P>({P{0.5, 0.5},
                                                       P{0.25, 0.25},
                                                       P{0.5, 0.5},
                                                       P{0.25, 0.25},
                                                       P{0.75, 0.75},
                                                       P{0.25, 0.25},
                                                       P{0.75, 0.75}});
  auto expected_segments_coords =
    make_device_vector<segment<T>>({segment<T>{P{0.5, 0.5}, P{1, 1}},
                                    segment<T>{P{0, 0}, P{0.25, 0.25}},
                                    segment<T>{P{0.75, 0.75}, P{1, 1}},
                                    segment<T>{P{0.75, 0.75}, P{1, 1}}});

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_geometry_collection_offset,
                                      std::move(got.geometry_collection_offset));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_types_buffer, std::move(got.types_buffer));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_offset_buffer, std::move(got.offset_buffer));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_points_coords, std::move(got.points_coords));
  expect_segment_equivalent(expected_segments_coords, std::move(got.segments_coords));
}

// Same Test Case as above, reversing the order of multilinestrings1 and multilinestrings2
TYPED_TEST(LinestringIntersectionTest, ExampleReversed)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  using index_t = typename intersection_result<T, std::size_t>::index_t;
  using types_t = typename intersection_result<T, std::size_t>::types_t;

  auto multilinestrings1 = make_multilinestring_array({0, 1, 2, 3, 4, 5, 6, 7},
                                                      {0, 2, 4, 6, 8, 10, 12, 14},
                                                      {P{0, 0},
                                                       P{1, 1},
                                                       P{0, 0},
                                                       P{1, 1},
                                                       P{0, 0},
                                                       P{1, 1},
                                                       P{0, 0},
                                                       P{1, 1},
                                                       P{0, 0},
                                                       P{1, 1},
                                                       P{0, 0},
                                                       P{1, 1},
                                                       P{0, 0},
                                                       P{1, 1}});

  auto multilinestrings2 = make_multilinestring_array(
    {0, 1, 2, 3, 4, 5, 6, 7},
    {0, 2, 5, 7, 12, 16, 18, 20},
    {P{1, 0},       P{0, 1},     P{0.5, 0},    P{0, 0.5},     P{1, 0.5},
     P{0.5, 0.5},   P{1.5, 1.5}, P{-1, -1},    P{0.25, 0.25}, P{0.25, 0.0},
     P{0.75, 0.75}, P{1.5, 1.5}, P{0.25, 0.0}, P{0.25, 0.5},  P{0.75, 0.75},
     P{1.5, 1.5},   P{2, 2},     P{3, 3},      P{1, 0},       P{2, 0}});

  auto got = pairwise_linestring_intersection_with_duplicate(multilinestrings2.range(),
                                                             multilinestrings1.range());

  auto expected_geometry_collection_offset =
    make_device_vector<index_t>({0, 1, 3, 4, 8, 11, 11, 11});
  auto expected_types_buffer  = make_device_vector<types_t>({0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1});
  auto expected_offset_buffer = make_device_vector<index_t>({0, 1, 2, 0, 3, 4, 1, 2, 5, 6, 3});
  auto expected_points_coords = make_device_vector<P>({P{0.5, 0.5},
                                                       P{0.25, 0.25},
                                                       P{0.5, 0.5},
                                                       P{0.25, 0.25},
                                                       P{0.75, 0.75},
                                                       P{0.25, 0.25},
                                                       P{0.75, 0.75}});
  auto expected_segments_coords =
    make_device_vector<segment<T>>({segment<T>{P{0.5, 0.5}, P{1, 1}},
                                    segment<T>{P{0, 0}, P{0.25, 0.25}},
                                    segment<T>{P{0.75, 0.75}, P{1, 1}},
                                    segment<T>{P{0.75, 0.75}, P{1, 1}}});

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_geometry_collection_offset,
                                      std::move(got.geometry_collection_offset));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_types_buffer, std::move(got.types_buffer));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_offset_buffer, std::move(got.offset_buffer));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_points_coords, std::move(got.points_coords));
  expect_segment_equivalent(expected_segments_coords, std::move(got.segments_coords));
}

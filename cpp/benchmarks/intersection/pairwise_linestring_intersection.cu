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

#include <cuspatial_test/geometry_generator.cuh>

#include <cuspatial/intersection.cuh>

#include <benchmarks/fixture/rmm_pool_raii.hpp>
#include <nvbench/nvbench.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_vector.hpp>

using namespace cuspatial;

template <typename T>
void pairwise_linestring_intersection_benchmark(nvbench::state& state, nvbench::type_list<T>)
{
  // TODO: to be replaced by nvbench fixture once it's ready
  cuspatial::rmm_pool_raii rmm_pool;
  auto stream = rmm::cuda_stream_default;

  auto num_pairs = static_cast<std::size_t>(state.get_int64("num_pairs"));
  auto num_linestring_per_multilinestring =
    static_cast<std::size_t>(state.get_int64("num_linestring_per_multilinestring"));
  auto num_segments_per_linestring =
    static_cast<std::size_t>(state.get_int64("num_segments_per_linestring"));

  auto param = test::multilinestring_generator_parameter<T>{
    num_pairs, num_linestring_per_multilinestring, num_segments_per_linestring, 10, {0, 0}};

  auto lhs = test::generate_multilinestring_array(param, stream);
  auto rhs = test::generate_multilinestring_array(param, stream);

  state.add_element_count(num_pairs, "Num Pairs");
  state.add_element_count(num_linestring_per_multilinestring, "Num Linestring per Multilinestring");
  state.add_element_count(num_segments_per_linestring, "Num Segments per Linestring");

  state.exec(nvbench::exec_tag::sync,
             [lrange = lhs.range(), rrange = rhs.range()](nvbench::launch& launch) {
               pairwise_linestring_intersection<T, std::size_t>(lrange, rrange);
             });
}

using floating_point_types = nvbench::type_list<float, double>;
NVBENCH_BENCH_TYPES(pairwise_linestring_intersection_benchmark,
                    NVBENCH_TYPE_AXES(floating_point_types))
  .set_type_axes_names({"FP type"})
  .add_int64_axis("num_pairs", {1000, 10000, 100000, 1000000})
  .add_int64_axis("num_linestring_per_multilinestring", {1})
  .add_int64_axis("num_segments_per_linestring", {100});

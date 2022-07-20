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

#include <cuspatial/cuda_utils.hpp>

#include <utility>
namespace cuspatial {

/**
 * @addtogroup types
 * @{
 */

/**
 * @brief A generic 2D vector type.
 *
 * This is the base type used in cuspatial for both Longitude/Latitude (LonLat) coordinate pairs and
 * Cartesian (X/Y) coordinate pairs. For LonLat pairs, the `x` member represents Longitude, and `y`
 * represents Latitude.
 *
 * @tparam T the base type for the coordinates
 */
template <typename T>
struct alignas(2 * sizeof(T)) vec_2d {
  using value_type = T;
  value_type x;
  value_type y;
};

/**
 * @brief A geographical Longitude/Latitude (LonLat) coordinate pair
 *
 * `x` is the longitude coordinate, `y` is the latitude coordinate.
 *
 * @tparam T the base type for the coordinates
 */
template <typename T>
struct alignas(2 * sizeof(T)) lonlat_2d : vec_2d<T> {
  CUSPATIAL_HOST_DEVICE lonlat_2d<T>() = default;
  CUSPATIAL_HOST_DEVICE lonlat_2d<T>(T x, T y) : vec_2d<T>{x, y} {}
  CUSPATIAL_HOST_DEVICE lonlat_2d<T>(const vec_2d<T>& v) : vec_2d<T>(v) {}
  CUSPATIAL_HOST_DEVICE lonlat_2d<T>(vec_2d<T>&& v) noexcept : vec_2d<T>(std::move(v)) {}
  lonlat_2d& CUSPATIAL_HOST_DEVICE operator=(vec_2d<T> const& other)
  {
    return *this = lonlat_2d<T>(other);
  }
  lonlat_2d& CUSPATIAL_HOST_DEVICE operator=(vec_2d<T>&& other) noexcept
  {
    return *this = lonlat_2d<T>(std::move(other));
  }
  CUSPATIAL_HOST_DEVICE ~lonlat_2d() = default;
};

/**
 * @brief A Cartesian (x/y) coordinate pair.
 *
 * @tparam T the base type for the coordinates.
 */
template <typename T>
struct alignas(2 * sizeof(T)) cartesian_2d : vec_2d<T> {
  CUSPATIAL_HOST_DEVICE cartesian_2d() = default;
  CUSPATIAL_HOST_DEVICE cartesian_2d(T x, T y) : vec_2d<T>{x, y} {}
  CUSPATIAL_HOST_DEVICE cartesian_2d(const vec_2d<T>& v) : vec_2d<T>(v) {}
  CUSPATIAL_HOST_DEVICE cartesian_2d(vec_2d<T>&& v) noexcept : vec_2d<T>(std::move(v)) {}
  cartesian_2d& CUSPATIAL_HOST_DEVICE operator=(const vec_2d<T>& other)
  {
    return *this = cartesian_2d<T>(other);
  }
  cartesian_2d& CUSPATIAL_HOST_DEVICE operator=(vec_2d<T>&& other) noexcept
  {
    return *this = cartesian_2d<T>(std::move(other));
  }
  CUSPATIAL_HOST_DEVICE ~cartesian_2d() = default;
};

/**
 * @brief Compare two 2D vectors for equality.
 */
template <typename T>
bool operator==(vec_2d<T> const& lhs, vec_2d<T> const& rhs)
{
  return (lhs.x == rhs.x) && (lhs.y == rhs.y);
}

/**
 * @brief Element-wise addition of two 2D vectors.
 */
template <typename T>
vec_2d<T> CUSPATIAL_HOST_DEVICE operator+(vec_2d<T> const& a, vec_2d<T> const& b)
{
  return vec_2d<T>{a.x + b.x, a.y + b.y};
}

/**
 * @brief Element-wise subtraction of two 2D vectors.
 */
template <typename T>
vec_2d<T> CUSPATIAL_HOST_DEVICE operator-(vec_2d<T> const& a, vec_2d<T> const& b)
{
  return vec_2d<T>{a.x - b.x, a.y - b.y};
}

/**
 * @brief Scale a 2D vector by a factor @p r.
 */
template <typename T>
vec_2d<T> CUSPATIAL_HOST_DEVICE operator*(vec_2d<T> vec, T const& r)
{
  return vec_2d<T>{vec.x * r, vec.y * r};
}

/**
 * @brief Scale a 2d vector by ratio @p r.
 */
template <typename T>
vec_2d<T> CUSPATIAL_HOST_DEVICE operator*(T const& r, vec_2d<T> vec)
{
  return vec * r;
}

/**
 * @brief Compute dot product of two 2D vectors.
 */
template <typename T>
T CUSPATIAL_HOST_DEVICE dot(vec_2d<T> const& a, vec_2d<T> const& b)
{
  return a.x * b.x + a.y * b.y;
}

/**
 * @brief Compute 2D determinant of a 2x2 matrix with column vectors @p a and @p b.
 */
template <typename T>
T CUSPATIAL_HOST_DEVICE det(vec_2d<T> const& a, vec_2d<T> const& b)
{
  return a.x * b.y - a.y * b.x;
}

/**
 * @} // end of doxygen group
 */

}  // namespace cuspatial

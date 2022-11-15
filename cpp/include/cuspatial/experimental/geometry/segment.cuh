#pragma once

#include <cuspatial/vec_2d.hpp>

namespace cuspatial {

template <typename T>
struct segment {
  using value_type = T;
  vec_2d<T> first;
  vec_2d<T> second;
};

}  // namespace cuspatial

# distutils: language = c++

# Copyright (c) 2022, NVIDIA CORPORATION.

cdef extern from "cuspatial/experimental/point_distance.cuh" \
        namespace "cuspatial" nogil:
    cdef OutputIt pairwise_point_distance[Cart2dItA, Cart2dItB, OutputIt](
        Cart2dItA points1_first,
        Cart2dItA points1_last,
        Cart2dItB points2_first,
        OutputIt distances_first
    ) except +

# Copyright (c) 2022, NVIDIA CORPORATION.

import cupy as cp

from libc.stdint cimport uintptr_t

from cuspatial._lib.cpp.experimental.point_distance cimport pairwise_point_distance as cpp_pairwise_point_distance
from cuspatial._lib.cpp.experimental.type_utils cimport make_cartesian_2d_iterator, ret_type_float

def pairwise_point_distance(
    points1_x,
    points1_y,
    points2_x,
    points2_y
):
    # Only prototyping for float32 type, double can be implemented with if-else
    output = cp.zeros(shape=(len(points1_x), ), dtype=cp.float32)

    cdef float* points1_x_begin = <float*> <uintptr_t> points1_x.__cuda_array_interface__.data[0]
    cdef float* points1_y_begin = <float*> <uintptr_t> points1_y.__cuda_array_interface__.data[0]
    cdef float* points2_x_begin = <float*> <uintptr_t> points2_x.__cuda_array_interface__.data[0]
    cdef float* points2_y_begin = <float*> <uintptr_t> points2_y.__cuda_array_interface__.data[0]
    cdef float* output_begin = <float*> <uintptr_t> output.__cuda_array_interface__.data[0]
    
    cdef int size = len(points1_x)

    cdef ret_type_float points1_begin = make_cartesian_2d_iterator(points1_x_begin, points1_y_begin)
    cdef ret_type_float points2_begin = make_cartesian_2d_iterator(points2_x_begin, points2_y_begin)

    with nogil:
        cpp_pairwise_point_distance(
            points1_begin,
            points1_begin + size,
            points2_begin,
            output_begin
        )
    return output

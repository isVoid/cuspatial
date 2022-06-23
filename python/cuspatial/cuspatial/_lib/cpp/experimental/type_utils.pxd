# distutils: language = c++

# Copyright (c) 2022, NVIDIA CORPORATION.

cdef extern from *:
    """
    #include <cuspatial/experimental/type_utils.hpp>
    #include <type_traits>

    using ret_type_float = std::invoke_result<decltype(cuspatial::make_cartesian_2d_iterator<float*, float*>), float*, float*>::type;
    """

    ctypedef int ret_type_float "ret_type_float"

cdef extern from "cuspatial/experimental/type_utils.hpp" \
        namespace "cuspatial" nogil:
    cdef ret_type_float make_cartesian_2d_iterator[FirstIter, SecondIter](
        FirstIter points1_first,
        SecondIter points1_last,
    ) except +

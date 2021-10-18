

set(MYSRC test.cpp)

add_executable(kokk_yakl_array ${MYSRC})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/kokkos)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/kokkos/core/src)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/kokkos)

target_link_libraries(kokk_yakl_array Kokkos::kokkos)
target_compile_features(kokk_yakl_array PUBLIC cxx_std_14)
include(../../yakl_utils.cmake)
yakl_process_target(kokk_yakl_array)

separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND ${MPI_COMMAND})

add_test(NAME kokk_yakl_array_test COMMAND ${REAL_MPI_COMMAND} ./kokk_yakl_array)


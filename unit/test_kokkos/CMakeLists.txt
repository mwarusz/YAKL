

set(MYSRC test.cpp)

add_executable(test_kokk ${MYSRC})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/kokkos)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/kokkos/core/src)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/kokkos)

target_link_libraries(test_kokk Kokkos::kokkos)
target_compile_features(test_kokk PUBLIC cxx_std_14)
include(../../yakl_utils.cmake)
yakl_process_target(test_kokk)

separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND ${MPI_COMMAND})

add_test(NAME test_kokk_test COMMAND ${REAL_MPI_COMMAND} ./test_kokk)


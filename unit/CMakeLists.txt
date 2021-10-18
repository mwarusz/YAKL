cmake_minimum_required(VERSION 3.0)
project(yakl_unit)
enable_testing()

enable_language(Fortran)
enable_language(CXX)
enable_language(C)
if ("${YAKL_ARCH}" STREQUAL "CUDA")
  enable_language(CUDA)
  if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.18.0")
    set(CMAKE_CUDA_ARCHITECTURES OFF)
  endif()

  #Kokkos flags need to be set in cmakescript or else they won't be globally set  
  #set(Kokkos_ENABLE_CUDA ON)
  #set(Kokkos_ENABLE_CUDA_LAMBDA ON)
  #set(Kokkos_ENABLE_CUDA_CONSTEXPR ON)

  #if("${YAKL_CUDA_ARCH}" STREQUAL "SM35")
  #  set(Kokkos_ARCH_KEPLER35 ON)
  #endif()
endif()

# YAKL library
add_subdirectory(.. ./yakl)

# Kokkos Library
add_subdirectory(../kokkos ./kokkos )

# Unit test directories
add_subdirectory(CArray)
add_subdirectory(FArray)
add_subdirectory(Gator)
add_subdirectory(Random)
add_subdirectory(FFT)
add_subdirectory(Reductions)
add_subdirectory(Atomics)
add_subdirectory(Pentadiagonal)
add_subdirectory(Tridiagonal)
add_subdirectory(Lambda)
add_subdirectory(Fortran_Link)
add_subdirectory(Fortran_Gator)
add_subdirectory(OpenMP_Regions)
add_subdirectory(Intrinsics)
add_subdirectory(ParForC)
add_subdirectory(ParForFortran)

add_subdirectory(test_kokkos)
add_subdirectory(test_kokkos_yakl)
add_subdirectory(kokk_yakl_array)

if (DEFINED NETCDF_LINK_FLAGS)
  add_subdirectory(NetCDF)
endif()


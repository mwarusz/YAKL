#!/bin/bash

../../cmakeclean.sh

unset GATOR_DISABLE

export CC=gcc
export CXX=g++
export FC=gfortran
unset CXXFLAGS
unset FFLAGS

cmake -DYAKL_ARCH="CUDA"                                                                                                          \
      -DYAKL_CUDA_ARCH="SM35"                                                                                                     \
      -DYAKL_CUDA_FLAGS="-O3 -arch sm_35 --use_fast_math -ccbin g++ -DTHRUST_IGNORE_CUB_VERSION_CHECK -I/opt/netcdf_gnu/include"  \
      -DYAKL_C_FLAGS="-O3"                                                                                                        \
      -DYAKL_F90_FLAGS="-O3"                                                                                                      \
      -DNETCDF_LINK_FLAGS="`/opt/netcdf_gnu/bin/nc-config --libs`"                                                                \
      -DKokkos_ENABLE_CUDA=ON                                                                                                     \
      -DKokkos_ENABLE_CUDA_LAMBDA=ON                                                                                              \
      -DKokkos_ENABLE_CUDA_CONSTEXPR=ON                                                                                           \
      -DKokkos_ARCH_KEPLER35=ON                                                                                                   \
      ../../..


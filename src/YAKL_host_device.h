
#pragma once

namespace yakl {

  bool constexpr YAKL_INLINE currently_on_host() {
    #ifdef YAKL_ARCH_CUDA
      #if defined(__CUDA_ARCH__)
        return false;
      #else
        return true;
      #endif
    #elif defined(YAKL_ARCH_HIP)
      #if defined(__HIP_DEVICE_COMPILE__)
        return false;
      #else
        return true;
      #endif
    #elif defined(YAKL_ARCH_SYCL)
      #if defined(__SYCL_DEVICE_ONLY__)
        return false;
      #else
        return true;
      #endif
    #elif defined(YAKL_ARCH_OPENMP)
      return true;
    #else
      return true;
    #endif
  }


  bool constexpr YAKL_INLINE currently_on_device() {
    #ifdef YAKL_ARCH_CUDA
      #if defined(__CUDA_ARCH__)
        return true;
      #else
        return false;
      #endif
    #elif defined(YAKL_ARCH_HIP)
      #if defined(__HIP_DEVICE_COMPILE__)
        return true;
      #else
        return false;
      #endif
    #elif defined(YAKL_ARCH_SYCL)
      #if defined(__SYCL_DEVICE_ONLY__)
        return true;
      #else
        return false;
      #endif
    #elif defined(YAKL_ARCH_OPENMP)
      return true;
    #else
      return true;
    #endif
  }


  bool constexpr YAKL_INLINE separate_device_address_space() {
    #ifdef YAKL_MANAGED_MEMORY
      return false;
    #else
      #if   defined(YAKL_ARCH_CUDA)
        return true;
      #elif defined(YAKL_ARCH_HIP)
        return true;
      #elif defined(YAKL_ARCH_SYCL)
        return true;
      #elif defined(YAKL_ARCH_OPENMP)
        return false;
      #else
        return false;
      #endif
    #endif
  }


  bool constexpr YAKL_INLINE ignore_address_space() {
    #if defined(YAKL_IGNORE_ADDRESS_SPACE)
      return true;
    #else
      return false;
    #endif
  }


  bool constexpr yakl_debug() {
    #ifdef YAKL_DEBUG
      return true;
    #else
      return false;
    #endif
  }

}



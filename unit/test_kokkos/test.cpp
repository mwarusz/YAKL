#include <Kokkos_Core.hpp>

typedef float real;
typedef Kokkos::HostSpace memHost;
#ifdef __CUDACC__
  typedef Kokkos::CudaSpace memDevice;
#else
  typedef Kokkos::HostSpace memDevice;
#endif


int main(int argc, char* argv[]) {
  Kokkos::initialize(argc,argv);
  {
  }
  Kokkos::finalize();

  return 0;

}

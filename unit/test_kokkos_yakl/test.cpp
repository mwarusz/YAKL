//#include <Kokkos_Core.hpp>
#include "YAKL.h"
//#include "KokkosWrap.h"

typedef float real;
typedef Kokkos::HostSpace memHost;
#ifdef __CUDACC__
  typedef Kokkos::CudaSpace memDevice;
#else
  typedef Kokkos::HostSpace memDevice;
#endif

using KokkosWrap::c::CView;
using KokkosWrap::c::LoopBounds;
using KokkosWrap::c::create_mirror;

typedef CView<real * ,memHost  > realHost1d;
typedef CView<real **,memHost  > realHost2d;
typedef CView<real * ,memDevice> real1d;
typedef CView<real **,memDevice> real2d;
typedef CView<real * ,memDevice,Kokkos::MemoryUnmanaged> realUmg1d;
typedef CView<real **,memDevice,Kokkos::MemoryUnmanaged> realUmg2d;


int main(int argc, char* argv[]) {
  Kokkos::initialize(argc,argv);
  yakl::init();
  {
    int constexpr n1 = 100;
    int constexpr n2 = 10;
    real1d tmp1d_orig("tmp1d",n1);
    real2d tmp2d_orig("tmp2d",n1,{2,n2+1});

    realUmg1d tmp1d(tmp1d_orig.data(),n1);
    realUmg2d tmp2d(tmp2d_orig.data(),n1,{2,n2+1});

    // do i = 1,n1
    Kokkos::parallel_for( LoopBounds<1>(n1) , KOKKOS_LAMBDA (int i) {
      tmp1d(i) = i;
    });
    // do j = 2,n2+1
    //   do i = 1,n1
    Kokkos::parallel_for( LoopBounds<2>({2,n2+1},n1) , KOKKOS_LAMBDA (int j, int i) {
      tmp2d(i,j) = (j-2)*n1+i;
    });

    Kokkos::fence();

    realHost1d tmp1d_host = create_mirror(tmp1d);
    realHost2d tmp2d_host = create_mirror(tmp2d);

    Kokkos::deep_copy(tmp1d_host , tmp1d);
    Kokkos::deep_copy(tmp2d_host , tmp2d);

    for (int i=0; i < 100; i++) {
      std::cout << tmp1d_host(i) << " , ";
    }
    std::cout << "\n\n";
    for (int j=2; j < 12; j++) {
      for (int i=0; i < 100; i++) {
        std::cout << tmp2d_host(i,j) << " , ";
      }
    }
    std::cout << "\n\n";
  }
  Kokkos::finalize();
  yakl::finalize();

  return 0;

}

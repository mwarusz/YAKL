#include <Kokkos_Core.hpp>
#include "YAKL.h"
#include "KokkosWrap.h"

using yakl::Array;
using yakl::styleFortran;
//using yakl::c::Bound
//using yakl::memHost;
//using yakl::memDevice;

typedef float real;
typedef Kokkos::HostSpace memHost;
#ifdef __CUDACC__
  typedef Kokkos::CudaSpace memDevice;
#else
  typedef Kokkos::HostSpace memDevice;
#endif

//typedef Array<real,1,yakl::memDevice,styleC> real1d;
//typedef Array<real,2,yakl::memDevice,styleC> real2d;
//typedef Array<real,3,yakl::memDevice,styleC> real3d;
//typedef Array<real,4,yakl::memDevice,styleC> real4d;
//typedef Array<real,5,yakl::memDevice,styleC> real5d;
//typedef Array<real,6,yakl::memDevice,styleC> real6d;
//typedef Array<real,7,yakl::memDevice,styleC> real7d;
//typedef Array<real,8,yakl::memDevice,styleC> real8d;
typedef Array<real,1,yakl::memDevice,styleFortran> real1d;
typedef Array<real,2,yakl::memDevice,styleFortran> real2d;
typedef Array<real,3,yakl::memDevice,styleFortran> real3d;
typedef Array<real,4,yakl::memDevice,styleFortran> real4d;
typedef Array<real,5,yakl::memDevice,styleFortran> real5d;
typedef Array<real,6,yakl::memDevice,styleFortran> real6d;
typedef Array<real,7,yakl::memDevice,styleFortran> real7d;
typedef Array<real,8,yakl::memDevice,styleFortran> real8d;

//typedef Array<real,1,yakl::memHost,styleC> realHost1d;
//typedef Array<real,2,yakl::memHost,styleC> realHost2d;
//typedef Array<real,3,yakl::memHost,styleC> realHost3d;
//typedef Array<real,4,yakl::memHost,styleC> realHost4d;
//typedef Array<real,5,yakl::memHost,styleC> realHost5d;
//typedef Array<real,6,yakl::memHost,styleC> realHost6d;
//typedef Array<real,7,yakl::memHost,styleC> realHost7d;
//typedef Array<real,8,yakl::memHost,styleC> realHost8d;
typedef Array<real,1,yakl::memHost,styleFortran> realHost1d;
typedef Array<real,2,yakl::memHost,styleFortran> realHost2d;
typedef Array<real,3,yakl::memHost,styleFortran> realHost3d;
typedef Array<real,4,yakl::memHost,styleFortran> realHost4d;
typedef Array<real,5,yakl::memHost,styleFortran> realHost5d;
typedef Array<real,6,yakl::memHost,styleFortran> realHost6d;
typedef Array<real,7,yakl::memHost,styleFortran> realHost7d;
typedef Array<real,8,yakl::memHost,styleFortran> realHost8d;

//using KokkosWrap::c::CView;
using KokkosWrap::fortran::LoopBounds;
//using KokkosWrap::c::create_mirror;



//typedef CView<real * ,memHost  > realHost1d;
//typedef CView<real **,memHost  > realHost2d;
//typedef CView<real * ,memDevice> real1d;
//typedef CView<real **,memDevice> real2d;
//typedef CView<real * ,memDevice,Kokkos::MemoryUnmanaged> realUmg1d;
//typedef CView<real **,memDevice,Kokkos::MemoryUnmanaged> realUmg2d;


int main(int argc, char* argv[]) {
  Kokkos::initialize(argc,argv);
  yakl::init();
  {
    int constexpr n1 = 100;
    int constexpr n2 = 10;
    real1d tmp1d_orig("tmp1d",n1);
    real2d tmp2d_orig("tmp2d",n1,{2,n2+1});

    real1d tmp1d("tmp1d_ptr",tmp1d_orig.data(),n1);
    real2d tmp2d("tmp2d_ptr",tmp2d_orig.data(),n1,{2,n2+1});

    // do i = 1,n1
    Kokkos::parallel_for( LoopBounds<1>(n1) , KOKKOS_LAMBDA (int i) {
      //tmp1d_orig(i) = i;
      tmp1d(i) = i;
    });
    // do j = 2,n2+1
    //   do i = 1,n1
    Kokkos::parallel_for( LoopBounds<2>({2,n2+1},n1) , KOKKOS_LAMBDA (int j, int i) {
      //tmp2d_orig(i,j) = (j-2)*n1+i;
      tmp2d(i,j) = (j-2)*n1+i;
    });

    Kokkos::fence();

    //realHost1d tmp1d_host = create_mirror(tmp1d);
    //realHost2d tmp2d_host = create_mirror(tmp2d);

    //realHost1d tmp1d_host = tmp1d_orig.createHostCopy();
    realHost1d tmp1d_host = tmp1d.createHostCopy();
    //realHost2d tmp2d_host = tmp2d_orig.createHostCopy();
    realHost2d tmp2d_host = tmp2d.createHostCopy();

    //Kokkos::deep_copy(tmp1d_host , tmp1d);
    //Kokkos::deep_copy(tmp2d_host , tmp2d);

    //unmanaged doesn't handle 
    tmp1d.deep_copy_to(tmp1d_host);
    tmp2d.deep_copy_to(tmp2d_host);
    //tmp1d_orig.deep_copy_to(tmp1d_host);
    //tmp2d_orig.deep_copy_to(tmp2d_host);
    

    for (int i=1; i <= 100; i++) {
      std::cout << tmp1d_host(i) << " , ";
    }
    std::cout << "\n\n";
    for (int j=2; j < 12; j++) {
      for (int i=1; i <= 100; i++) {
        std::cout << tmp2d_host(i,j) << " , ";
      }
    }
    std::cout << "\n\n";
  }
  Kokkos::finalize();
  yakl::finalize();

  return 0;

}

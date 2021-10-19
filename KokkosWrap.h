#pragma once

#include <Kokkos_Core.hpp>


//extern "C" void c_kokkos_initialize_without_args() {
//  Kokkos::initialize();
//}
//
//extern "C" void c_kokkos_finalize() {
//  Kokkos::finalize();
//}



namespace KokkosWrap {

namespace fortran {


  typedef Kokkos::HostSpace memHost;
  // If we're using CUDA, then set CUDA as the device memory space
  // Otherwise, set host space as the device memory space
  #ifdef __CUDACC__
    typedef Kokkos::CudaSpace memDevice;
  #else
    typedef Kokkos::HostSpace memDevice;
  #endif


  class Bnd {
  public:
    int l, u, s;
    KOKKOS_INLINE_FUNCTION Bnd(                  ) { l = 1   ; u = 1   ; s = 0              ; }
    KOKKOS_INLINE_FUNCTION Bnd(          int u_in) { l = 1   ; u = u_in; s = u_in           ; }
    KOKKOS_INLINE_FUNCTION Bnd(int l_in, int u_in) { l = l_in; u = u_in; s = u_in - l_in + 1; }
  };



  template <class DataType, class MemorySpace, class MemoryTraits = Kokkos::MemoryManaged>
  class FortranView;



  template <class DataType, class MemorySpace>
  class FortranView<DataType,MemorySpace,Kokkos::MemoryManaged> : public Kokkos::View<DataType,Kokkos::LayoutLeft,MemorySpace,Kokkos::MemoryManaged> {
  public:
    typedef Kokkos::View<DataType,Kokkos::LayoutLeft,MemorySpace,Kokkos::MemoryManaged> ViewLoc;
    typedef typename ViewLoc::reference_type T;
    int static constexpr rank = ViewLoc::rank;
    int lbounds[rank];


    FortranView(std::string name , Bnd b0) : ViewLoc(name,b0.s) {
      lbounds[0] = b0.l;
    }
    FortranView(std::string name , Bnd b0, Bnd b1) : ViewLoc(name,b0.s,b1.s) {
      lbounds[0] = b0.l;
      lbounds[1] = b1.l;
    }
    FortranView(std::string name , Bnd b0, Bnd b1, Bnd b2) : ViewLoc(name,b0.s,b1.s,b2.s) {
      lbounds[0] = b0.l;
      lbounds[1] = b1.l;
      lbounds[2] = b2.l;
    }
    FortranView(std::string name , Bnd b0, Bnd b1, Bnd b2, Bnd b3) : ViewLoc(name,b0.s,b1.s,b2.s,b3.s) {
      lbounds[0] = b0.l;
      lbounds[1] = b1.l;
      lbounds[2] = b2.l;
      lbounds[3] = b3.l;
    }
    FortranView(std::string name , Bnd b0, Bnd b1, Bnd b2, Bnd b3, Bnd b4) : ViewLoc(name,b0.s,b1.s,b2.s,b3.s,b4.s) {
      lbounds[0] = b0.l;
      lbounds[1] = b1.l;
      lbounds[2] = b2.l;
      lbounds[3] = b3.l;
      lbounds[4] = b4.l;
    }
    FortranView(std::string name , Bnd b0, Bnd b1, Bnd b2, Bnd b3, Bnd b4, Bnd b5) : ViewLoc(name,b0.s,b1.s,b2.s,b3.s,b4.s,b5.s) {
      lbounds[0] = b0.l;
      lbounds[1] = b1.l;
      lbounds[2] = b2.l;
      lbounds[3] = b3.l;
      lbounds[4] = b4.l;
      lbounds[5] = b5.l;
    }
    FortranView(std::string name , Bnd b0, Bnd b1, Bnd b2, Bnd b3, Bnd b4, Bnd b5, Bnd b6) : ViewLoc(name,b0.s,b1.s,b2.s,b3.s,b4.s,b5.s,b6.s) {
      lbounds[0] = b0.l;
      lbounds[1] = b1.l;
      lbounds[2] = b2.l;
      lbounds[3] = b3.l;
      lbounds[4] = b4.l;
      lbounds[5] = b5.l;
      lbounds[6] = b6.l;
    }
    FortranView(std::string name , Bnd b0, Bnd b1, Bnd b2, Bnd b3, Bnd b4, Bnd b5, Bnd b6, Bnd b7) : ViewLoc(name,b0.s,b1.s,b2.s,b3.s,b4.s,b5.s,b6.s,b7.s) {
      lbounds[0] = b0.l;
      lbounds[1] = b1.l;
      lbounds[2] = b2.l;
      lbounds[3] = b3.l;
      lbounds[4] = b4.l;
      lbounds[5] = b5.l;
      lbounds[6] = b6.l;
      lbounds[7] = b7.l;
    }
    FortranView(ViewLoc const &rhs , int const *lbounds_in) : ViewLoc(rhs) {
      for (int i=0; i < rank; i++) { lbounds[i] = lbounds_in[i]; }
    }


    // COPY CONSTRUCTORS / FUNCTIONS
    KOKKOS_INLINE_FUNCTION FortranView            (FortranView const &rhs) {
      static_cast<ViewLoc &>(*this) = static_cast<ViewLoc const &>(rhs);
      for (int i=0; i<rank; i++) { this->lbounds[i] = rhs.lbounds[i]; }
    }
    KOKKOS_INLINE_FUNCTION FortranView & operator=(FortranView const &rhs) {
      if (this == &rhs) { return *this; }
      static_cast<ViewLoc &>(*this) = static_cast<ViewLoc const &>(rhs);
      for (int i=0; i<rank; i++) { this->lbounds[i] = rhs.lbounds[i]; }
      return *this;
    }
    // MOVE CONSTRUCTORS
    KOKKOS_INLINE_FUNCTION FortranView            (FortranView &&rhs) {
      static_cast<ViewLoc &>(*this) = std::move(static_cast<ViewLoc const &>(rhs));
      for (int i=0; i<rank; i++) { this->lbounds[i] = rhs.lbounds[i]; }
    }
    KOKKOS_INLINE_FUNCTION FortranView & operator=(FortranView &&rhs) {
      if (this == &rhs) { return *this; }
      static_cast<ViewLoc &>(*this) = std::move(static_cast<ViewLoc const &>(rhs));
      for (int i=0; i<rank; i++) { this->lbounds[i] = rhs.lbounds[i]; }
      return *this;
    }

    KOKKOS_INLINE_FUNCTION T &operator() (int i0) const {
      return ViewLoc::operator()(i0 - lbounds[0]);
    }
    KOKKOS_INLINE_FUNCTION T &operator() (int i0, int i1) const {
      return ViewLoc::operator()(i0 - lbounds[0],
                                 i1 - lbounds[1]);
    }
    KOKKOS_INLINE_FUNCTION T &operator() (int i0, int i1, int i2) const {
      return ViewLoc::operator()(i0 - lbounds[0],
                                 i1 - lbounds[1],
                                 i2 - lbounds[2]);
    }
    KOKKOS_INLINE_FUNCTION T &operator() (int i0, int i1, int i2, int i3) const {
      return ViewLoc::operator()(i0 - lbounds[0],
                                 i1 - lbounds[1],
                                 i2 - lbounds[2],
                                 i3 - lbounds[3]);
    }
    KOKKOS_INLINE_FUNCTION T &operator() (int i0, int i1, int i2, int i3, int i4) const {
      return ViewLoc::operator()(i0 - lbounds[0],
                                 i1 - lbounds[1],
                                 i2 - lbounds[2],
                                 i3 - lbounds[3],
                                 i4 - lbounds[4]);
    }
    KOKKOS_INLINE_FUNCTION T &operator() (int i0, int i1, int i2, int i3, int i4, int i5) const {
      return ViewLoc::operator()(i0 - lbounds[0],
                                 i1 - lbounds[1],
                                 i2 - lbounds[2],
                                 i3 - lbounds[3],
                                 i4 - lbounds[4],
                                 i5 - lbounds[5]);
    }
    KOKKOS_INLINE_FUNCTION T &operator() (int i0, int i1, int i2, int i3, int i4, int i5, int i6) const {
      return ViewLoc::operator()(i0 - lbounds[0],
                                 i1 - lbounds[1],
                                 i2 - lbounds[2],
                                 i3 - lbounds[3],
                                 i4 - lbounds[4],
                                 i5 - lbounds[5],
                                 i6 - lbounds[6]);
    }
    KOKKOS_INLINE_FUNCTION T &operator() (int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7) const {
      return ViewLoc::operator()(i0 - lbounds[0],
                                 i1 - lbounds[1],
                                 i2 - lbounds[2],
                                 i3 - lbounds[3],
                                 i4 - lbounds[4],
                                 i5 - lbounds[5],
                                 i6 - lbounds[6],
                                 i7 - lbounds[7]);
    }
    KOKKOS_INLINE_FUNCTION T &operator() (int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8) const {
      return ViewLoc::operator()(i0 - lbounds[0],
                                 i1 - lbounds[1],
                                 i2 - lbounds[2],
                                 i3 - lbounds[3],
                                 i4 - lbounds[4],
                                 i5 - lbounds[5],
                                 i6 - lbounds[6],
                                 i7 - lbounds[7],
                                 i8 - lbounds[8]);
    }
  };



  template <class DataType, class MemorySpace>
  class FortranView<DataType,MemorySpace,Kokkos::MemoryUnmanaged> : public Kokkos::View<DataType,Kokkos::LayoutLeft,MemorySpace,Kokkos::MemoryUnmanaged> {
  public:
    typedef Kokkos::View<DataType,Kokkos::LayoutLeft,MemorySpace,Kokkos::MemoryUnmanaged> ViewLoc;
    typedef typename ViewLoc::reference_type TR;
    typedef typename ViewLoc::pointer_type TP;
    int static constexpr rank = ViewLoc::rank;
    int lbounds[rank];


    FortranView(TP const &data, Bnd b0) : ViewLoc(data,b0.s) {
      lbounds[0] = b0.l;
    }
    FortranView(TP const &data, Bnd b0, Bnd b1) : ViewLoc(data,b0.s,b1.s) {
      lbounds[0] = b0.l;
      lbounds[1] = b1.l;
    }
    FortranView(TP const &data, Bnd b0, Bnd b1, Bnd b2) : ViewLoc(data,b0.s,b1.s,b2.s) {
      lbounds[0] = b0.l;
      lbounds[1] = b1.l;
      lbounds[2] = b2.l;
    }
    FortranView(TP const &data, Bnd b0, Bnd b1, Bnd b2, Bnd b3) : ViewLoc(data,b0.s,b1.s,b2.s,b3.s) {
      lbounds[0] = b0.l;
      lbounds[1] = b1.l;
      lbounds[2] = b2.l;
      lbounds[3] = b3.l;
    }
    FortranView(TP const &data, Bnd b0, Bnd b1, Bnd b2, Bnd b3, Bnd b4) : ViewLoc(data,b0.s,b1.s,b2.s,b3.s,b4.s) {
      lbounds[0] = b0.l;
      lbounds[1] = b1.l;
      lbounds[2] = b2.l;
      lbounds[3] = b3.l;
      lbounds[4] = b4.l;
    }
    FortranView(TP const &data, Bnd b0, Bnd b1, Bnd b2, Bnd b3, Bnd b4, Bnd b5) : ViewLoc(data,b0.s,b1.s,b2.s,b3.s,b4.s,b5.s) {
      lbounds[0] = b0.l;
      lbounds[1] = b1.l;
      lbounds[2] = b2.l;
      lbounds[3] = b3.l;
      lbounds[4] = b4.l;
      lbounds[5] = b5.l;
    }
    FortranView(TP const &data, Bnd b0, Bnd b1, Bnd b2, Bnd b3, Bnd b4, Bnd b5, Bnd b6) : ViewLoc(data,b0.s,b1.s,b2.s,b3.s,b4.s,b5.s,b6.s) {
      lbounds[0] = b0.l;
      lbounds[1] = b1.l;
      lbounds[2] = b2.l;
      lbounds[3] = b3.l;
      lbounds[4] = b4.l;
      lbounds[5] = b5.l;
      lbounds[6] = b6.l;
    }
    FortranView(TP const &data, Bnd b0, Bnd b1, Bnd b2, Bnd b3, Bnd b4, Bnd b5, Bnd b6, Bnd b7) : ViewLoc(data,b0.s,b1.s,b2.s,b3.s,b4.s,b5.s,b6.s,b7.s) {
      lbounds[0] = b0.l;
      lbounds[1] = b1.l;
      lbounds[2] = b2.l;
      lbounds[3] = b3.l;
      lbounds[4] = b4.l;
      lbounds[5] = b5.l;
      lbounds[6] = b6.l;
      lbounds[7] = b7.l;
    }
    FortranView(ViewLoc const &rhs , int const *lbounds_in) : ViewLoc(rhs) {
      for (int i=0; i < rank; i++) { lbounds[i] = lbounds_in[i]; }
    }


    // COPY CONSTRUCTORS / FUNCTIONS
    KOKKOS_INLINE_FUNCTION FortranView            (FortranView const &rhs) {
      static_cast<ViewLoc &>(*this) = static_cast<ViewLoc const &>(rhs);
      for (int i=0; i<rank; i++) { this->lbounds[i] = rhs.lbounds[i]; }
    }
    KOKKOS_INLINE_FUNCTION FortranView & operator=(FortranView const &rhs) {
      if (this == &rhs) { return *this; }
      static_cast<ViewLoc &>(*this) = static_cast<ViewLoc const &>(rhs);
      for (int i=0; i<rank; i++) { this->lbounds[i] = rhs.lbounds[i]; }
      return *this;
    }
    // MOVE CONSTRUCTORS
    KOKKOS_INLINE_FUNCTION FortranView            (FortranView &&rhs) {
      static_cast<ViewLoc &>(*this) = std::move(static_cast<ViewLoc const &>(rhs));
      for (int i=0; i<rank; i++) { this->lbounds[i] = rhs.lbounds[i]; }
    }
    KOKKOS_INLINE_FUNCTION FortranView & operator=(FortranView &&rhs) {
      if (this == &rhs) { return *this; }
      static_cast<ViewLoc &>(*this) = std::move(static_cast<ViewLoc const &>(rhs));
      for (int i=0; i<rank; i++) { this->lbounds[i] = rhs.lbounds[i]; }
      return *this;
    }

    KOKKOS_INLINE_FUNCTION TR &operator() (int i0) const {
      return ViewLoc::operator()(i0 - lbounds[0]);
    }
    KOKKOS_INLINE_FUNCTION TR &operator() (int i0, int i1) const {
      return ViewLoc::operator()(i0 - lbounds[0],
                                 i1 - lbounds[1]);
    }
    KOKKOS_INLINE_FUNCTION TR &operator() (int i0, int i1, int i2) const {
      return ViewLoc::operator()(i0 - lbounds[0],
                                 i1 - lbounds[1],
                                 i2 - lbounds[2]);
    }
    KOKKOS_INLINE_FUNCTION TR &operator() (int i0, int i1, int i2, int i3) const {
      return ViewLoc::operator()(i0 - lbounds[0],
                                 i1 - lbounds[1],
                                 i2 - lbounds[2],
                                 i3 - lbounds[3]);
    }
    KOKKOS_INLINE_FUNCTION TR &operator() (int i0, int i1, int i2, int i3, int i4) const {
      return ViewLoc::operator()(i0 - lbounds[0],
                                 i1 - lbounds[1],
                                 i2 - lbounds[2],
                                 i3 - lbounds[3],
                                 i4 - lbounds[4]);
    }
    KOKKOS_INLINE_FUNCTION TR &operator() (int i0, int i1, int i2, int i3, int i4, int i5) const {
      return ViewLoc::operator()(i0 - lbounds[0],
                                 i1 - lbounds[1],
                                 i2 - lbounds[2],
                                 i3 - lbounds[3],
                                 i4 - lbounds[4],
                                 i5 - lbounds[5]);
    }
    KOKKOS_INLINE_FUNCTION TR &operator() (int i0, int i1, int i2, int i3, int i4, int i5, int i6) const {
      return ViewLoc::operator()(i0 - lbounds[0],
                                 i1 - lbounds[1],
                                 i2 - lbounds[2],
                                 i3 - lbounds[3],
                                 i4 - lbounds[4],
                                 i5 - lbounds[5],
                                 i6 - lbounds[6]);
    }
    KOKKOS_INLINE_FUNCTION TR &operator() (int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7) const {
      return ViewLoc::operator()(i0 - lbounds[0],
                                 i1 - lbounds[1],
                                 i2 - lbounds[2],
                                 i3 - lbounds[3],
                                 i4 - lbounds[4],
                                 i5 - lbounds[5],
                                 i6 - lbounds[6],
                                 i7 - lbounds[7]);
    }
    KOKKOS_INLINE_FUNCTION TR &operator() (int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8) const {
      return ViewLoc::operator()(i0 - lbounds[0],
                                 i1 - lbounds[1],
                                 i2 - lbounds[2],
                                 i3 - lbounds[3],
                                 i4 - lbounds[4],
                                 i5 - lbounds[5],
                                 i6 - lbounds[6],
                                 i7 - lbounds[7],
                                 i8 - lbounds[8]);
    }
  };



  template <class DataType, class MemorySpace, class MemoryTraits>
  inline FortranView<DataType,Kokkos::HostSpace,Kokkos::MemoryManaged> create_mirror(FortranView<DataType,MemorySpace,MemoryTraits> const &src) {
    typedef Kokkos::View<DataType,Kokkos::LayoutLeft,Kokkos::HostSpace,Kokkos::MemoryManaged> ViewLoc;
    ViewLoc dest_view = Kokkos::create_mirror( static_cast<typename FortranView<DataType,MemorySpace,MemoryTraits>::ViewLoc const &>(src) );
    FortranView<DataType,Kokkos::HostSpace,Kokkos::MemoryManaged> dest(dest_view, src.lbounds);
    return dest;
  }



  template <int N> class LoopBounds;



  template<> class LoopBounds<1> : public Kokkos::RangePolicy<Kokkos::IndexType<int>> {
    public:
    LoopBounds(Bnd b) : Kokkos::RangePolicy<Kokkos::IndexType<int>>(b.l,b.u+1) { };
    Kokkos::RangePolicy<Kokkos::IndexType<int>> to_range() const { return static_cast<Kokkos::RangePolicy<Kokkos::IndexType<int>>>(*this); }
    
  };
  template<> class LoopBounds<2> : public Kokkos::MDRangePolicy<Kokkos::Rank<2>,Kokkos::IndexType<int>> {
    public:
    LoopBounds(Bnd b0, Bnd b1) : Kokkos::MDRangePolicy<Kokkos::Rank<2>,Kokkos::IndexType<int>>({b0.l  ,b1.l  },
                                                                                               {b0.u+1,b1.u+1}) { };
    Kokkos::MDRangePolicy<Kokkos::Rank<2>,Kokkos::IndexType<int>> to_range() const { return static_cast<Kokkos::MDRangePolicy<Kokkos::Rank<2>,Kokkos::IndexType<int>>>(*this); }
  };
  template<> class LoopBounds<3> : public Kokkos::MDRangePolicy<Kokkos::Rank<3>,Kokkos::IndexType<int>> {
    public:
    LoopBounds(Bnd b0, Bnd b1, Bnd b2) : Kokkos::MDRangePolicy<Kokkos::Rank<3>,Kokkos::IndexType<int>>({b0.l  ,b1.l  ,b2.l  },
                                                                                                       {b0.u+1,b1.u+1,b2.u+1}) { };
    Kokkos::MDRangePolicy<Kokkos::Rank<3>,Kokkos::IndexType<int>> to_range() const { return static_cast<Kokkos::MDRangePolicy<Kokkos::Rank<3>,Kokkos::IndexType<int>>>(*this); }
  };
  template<> class LoopBounds<4> : public Kokkos::MDRangePolicy<Kokkos::Rank<4>,Kokkos::IndexType<int>> {
    public:
    LoopBounds(Bnd b0, Bnd b1, Bnd b2, Bnd b3) : Kokkos::MDRangePolicy<Kokkos::Rank<4>,Kokkos::IndexType<int>>({b0.l  ,b1.l  ,b2.l  ,b3.l  },
                                                                                                               {b0.u+1,b1.u+1,b2.u+1,b3.u+1}) { };
    Kokkos::MDRangePolicy<Kokkos::Rank<4>,Kokkos::IndexType<int>> to_range() const { return static_cast<Kokkos::MDRangePolicy<Kokkos::Rank<4>,Kokkos::IndexType<int>>>(*this); }
  };
  template<> class LoopBounds<5> : public Kokkos::MDRangePolicy<Kokkos::Rank<5>,Kokkos::IndexType<int>> {
    public:
    LoopBounds(Bnd b0, Bnd b1, Bnd b2, Bnd b3, Bnd b4) : Kokkos::MDRangePolicy<Kokkos::Rank<5>,Kokkos::IndexType<int>>({b0.l  ,b1.l  ,b2.l  ,b3.l  ,b4.l  },
                                                                                                                       {b0.u+1,b1.u+1,b2.u+1,b3.u+1,b4.u+1}) { };
    Kokkos::MDRangePolicy<Kokkos::Rank<5>,Kokkos::IndexType<int>> to_range() const { return static_cast<Kokkos::MDRangePolicy<Kokkos::Rank<5>,Kokkos::IndexType<int>>>(*this); }
  };
  template<> class LoopBounds<6> : public Kokkos::MDRangePolicy<Kokkos::Rank<6>,Kokkos::IndexType<int>> {
    public:
    LoopBounds(Bnd b0, Bnd b1, Bnd b2, Bnd b3, Bnd b4, Bnd b5) : Kokkos::MDRangePolicy<Kokkos::Rank<6>,Kokkos::IndexType<int>>({b0.l  ,b1.l  ,b2.l  ,b3.l  ,b4.l  ,b5.l  },
                                                                                                                               {b0.u+1,b1.u+1,b2.u+1,b3.u+1,b4.u+1,b5.u+1}) { };
    Kokkos::MDRangePolicy<Kokkos::Rank<6>,Kokkos::IndexType<int>> to_range() const { return static_cast<Kokkos::MDRangePolicy<Kokkos::Rank<6>,Kokkos::IndexType<int>>>(*this); }
  };



  #ifdef __CUDACC__
  template <class DataType, class MemoryTraits>
  inline typename Kokkos::View<DataType>::non_const_value_type sum(FortranView<DataType,memDevice,MemoryTraits> const &arr_in) {
    typedef typename Kokkos::View<DataType>::non_const_value_type TYPE;
    TYPE s = 0;
    Kokkos::View<TYPE *,Kokkos::LayoutLeft,memDevice,Kokkos::MemoryUnmanaged> arr(arr_in.data(),arr_in.size());
    Kokkos::parallel_reduce(arr.size() , KOKKOS_LAMBDA (int const i, TYPE &update) {
      update += arr(i);
    } , s );
    return s;
  }
  #endif

  template <class DataType, class MemoryTraits>
  inline typename Kokkos::View<DataType>::non_const_value_type sum(FortranView<DataType,memHost,MemoryTraits> const &arr_in) {
    typedef typename Kokkos::View<DataType>::non_const_value_type TYPE;
    TYPE s = 0;
    Kokkos::View<TYPE *,Kokkos::LayoutLeft,memHost,Kokkos::MemoryUnmanaged> arr(arr_in.data(),arr_in.size());
    for (int i=0; i < arr.size(); i++) {
      s += arr(i);
    }
    return s;
  }



  #ifdef __CUDACC__
  template <class DataType, class MemoryTraits>
  inline typename Kokkos::View<DataType>::non_const_value_type sum(Kokkos::View<DataType,Kokkos::LayoutLeft,memDevice,MemoryTraits> const &arr_in) {
    typedef typename Kokkos::View<DataType>::non_const_value_type TYPE;
    TYPE s = 0;
    Kokkos::View<TYPE *,Kokkos::LayoutLeft,memDevice,Kokkos::MemoryUnmanaged> arr(arr_in.data(),arr_in.size());
    Kokkos::parallel_reduce(arr.size() , KOKKOS_LAMBDA (int const i, TYPE &update) {
      update += arr(i);
    } , s );
    return s;
  }
  #endif

  template <class DataType, class MemoryTraits>
  inline typename Kokkos::View<DataType>::non_const_value_type sum(Kokkos::View<DataType,Kokkos::LayoutLeft,memHost,MemoryTraits> const &arr_in) {
    typedef typename Kokkos::View<DataType>::non_const_value_type TYPE;
    TYPE s = 0;
    Kokkos::View<TYPE *,Kokkos::LayoutLeft,memHost,Kokkos::MemoryUnmanaged> arr(arr_in.data(),arr_in.size());
    for (int i=0; i < arr.size(); i++) {
      s += arr(i);
    }
    return s;
  }


} //fortran

namespace c {


  typedef Kokkos::HostSpace memHost;
  // If we're using CUDA, then set CUDA as the device memory space
  // Otherwise, set host space as the device memory space
  #ifdef __CUDACC__
    typedef Kokkos::CudaSpace memDevice;
  #else
    typedef Kokkos::HostSpace memDevice;
  #endif


  class Bnd {
  public:
    int l, u, s;
    KOKKOS_INLINE_FUNCTION Bnd(                  ) { l = 0   ; u = 0   ; s = 0              ; }
    KOKKOS_INLINE_FUNCTION Bnd(          int u_in) { l = 0   ; u = u_in-1; s = u_in       ; }
    KOKKOS_INLINE_FUNCTION Bnd(int l_in, int u_in) { l = l_in; u = u_in; s = u_in - l_in + 1; }
  };



  template <class DataType, class MemorySpace, class MemoryTraits = Kokkos::MemoryManaged>
  class CView;



  template <class DataType, class MemorySpace>
  class CView<DataType,MemorySpace,Kokkos::MemoryManaged> : public Kokkos::View<DataType,Kokkos::LayoutRight,MemorySpace,Kokkos::MemoryManaged> {
  public:
    typedef Kokkos::View<DataType,Kokkos::LayoutRight,MemorySpace,Kokkos::MemoryManaged> ViewLoc;
    typedef typename ViewLoc::reference_type T;
    int static constexpr rank = ViewLoc::rank;
    int lbounds[rank];


    CView(std::string name , Bnd b0) : ViewLoc(name,b0.s) {
      lbounds[0] = b0.l;
    }
    CView(std::string name , Bnd b0, Bnd b1) : ViewLoc(name,b0.s,b1.s) {
      lbounds[0] = b0.l;
      lbounds[1] = b1.l;
    }
    CView(std::string name , Bnd b0, Bnd b1, Bnd b2) : ViewLoc(name,b0.s,b1.s,b2.s) {
      lbounds[0] = b0.l;
      lbounds[1] = b1.l;
      lbounds[2] = b2.l;
    }
    CView(std::string name , Bnd b0, Bnd b1, Bnd b2, Bnd b3) : ViewLoc(name,b0.s,b1.s,b2.s,b3.s) {
      lbounds[0] = b0.l;
      lbounds[1] = b1.l;
      lbounds[2] = b2.l;
      lbounds[3] = b3.l;
    }
    CView(std::string name , Bnd b0, Bnd b1, Bnd b2, Bnd b3, Bnd b4) : ViewLoc(name,b0.s,b1.s,b2.s,b3.s,b4.s) {
      lbounds[0] = b0.l;
      lbounds[1] = b1.l;
      lbounds[2] = b2.l;
      lbounds[3] = b3.l;
      lbounds[4] = b4.l;
    }
    CView(std::string name , Bnd b0, Bnd b1, Bnd b2, Bnd b3, Bnd b4, Bnd b5) : ViewLoc(name,b0.s,b1.s,b2.s,b3.s,b4.s,b5.s) {
      lbounds[0] = b0.l;
      lbounds[1] = b1.l;
      lbounds[2] = b2.l;
      lbounds[3] = b3.l;
      lbounds[4] = b4.l;
      lbounds[5] = b5.l;
    }
    CView(std::string name , Bnd b0, Bnd b1, Bnd b2, Bnd b3, Bnd b4, Bnd b5, Bnd b6) : ViewLoc(name,b0.s,b1.s,b2.s,b3.s,b4.s,b5.s,b6.s) {
      lbounds[0] = b0.l;
      lbounds[1] = b1.l;
      lbounds[2] = b2.l;
      lbounds[3] = b3.l;
      lbounds[4] = b4.l;
      lbounds[5] = b5.l;
      lbounds[6] = b6.l;
    }
    CView(std::string name , Bnd b0, Bnd b1, Bnd b2, Bnd b3, Bnd b4, Bnd b5, Bnd b6, Bnd b7) : ViewLoc(name,b0.s,b1.s,b2.s,b3.s,b4.s,b5.s,b6.s,b7.s) {
      lbounds[0] = b0.l;
      lbounds[1] = b1.l;
      lbounds[2] = b2.l;
      lbounds[3] = b3.l;
      lbounds[4] = b4.l;
      lbounds[5] = b5.l;
      lbounds[6] = b6.l;
      lbounds[7] = b7.l;
    }
    CView(ViewLoc const &rhs , int const *lbounds_in) : ViewLoc(rhs) {
      for (int i=0; i < rank; i++) { lbounds[i] = lbounds_in[i]; }
    }


    // COPY CONSTRUCTORS / FUNCTIONS
    KOKKOS_INLINE_FUNCTION CView            (CView const &rhs) {
      static_cast<ViewLoc &>(*this) = static_cast<ViewLoc const &>(rhs);
      for (int i=0; i<rank; i++) { this->lbounds[i] = rhs.lbounds[i]; }
    }
    KOKKOS_INLINE_FUNCTION CView & operator=(CView const &rhs) {
      if (this == &rhs) { return *this; }
      static_cast<ViewLoc &>(*this) = static_cast<ViewLoc const &>(rhs);
      for (int i=0; i<rank; i++) { this->lbounds[i] = rhs.lbounds[i]; }
      return *this;
    }
    // MOVE CONSTRUCTORS
    KOKKOS_INLINE_FUNCTION CView            (CView &&rhs) {
      static_cast<ViewLoc &>(*this) = std::move(static_cast<ViewLoc const &>(rhs));
      for (int i=0; i<rank; i++) { this->lbounds[i] = rhs.lbounds[i]; }
    }
    KOKKOS_INLINE_FUNCTION CView & operator=(CView &&rhs) {
      if (this == &rhs) { return *this; }
      static_cast<ViewLoc &>(*this) = std::move(static_cast<ViewLoc const &>(rhs));
      for (int i=0; i<rank; i++) { this->lbounds[i] = rhs.lbounds[i]; }
      return *this;
    }

    KOKKOS_INLINE_FUNCTION T &operator() (int i0) const {
      return ViewLoc::operator()(i0 - lbounds[0]);
    }
    KOKKOS_INLINE_FUNCTION T &operator() (int i0, int i1) const {
      return ViewLoc::operator()(i0 - lbounds[0],
                                 i1 - lbounds[1]);
    }
    KOKKOS_INLINE_FUNCTION T &operator() (int i0, int i1, int i2) const {
      return ViewLoc::operator()(i0 - lbounds[0],
                                 i1 - lbounds[1],
                                 i2 - lbounds[2]);
    }
    KOKKOS_INLINE_FUNCTION T &operator() (int i0, int i1, int i2, int i3) const {
      return ViewLoc::operator()(i0 - lbounds[0],
                                 i1 - lbounds[1],
                                 i2 - lbounds[2],
                                 i3 - lbounds[3]);
    }
    KOKKOS_INLINE_FUNCTION T &operator() (int i0, int i1, int i2, int i3, int i4) const {
      return ViewLoc::operator()(i0 - lbounds[0],
                                 i1 - lbounds[1],
                                 i2 - lbounds[2],
                                 i3 - lbounds[3],
                                 i4 - lbounds[4]);
    }
    KOKKOS_INLINE_FUNCTION T &operator() (int i0, int i1, int i2, int i3, int i4, int i5) const {
      return ViewLoc::operator()(i0 - lbounds[0],
                                 i1 - lbounds[1],
                                 i2 - lbounds[2],
                                 i3 - lbounds[3],
                                 i4 - lbounds[4],
                                 i5 - lbounds[5]);
    }
    KOKKOS_INLINE_FUNCTION T &operator() (int i0, int i1, int i2, int i3, int i4, int i5, int i6) const {
      return ViewLoc::operator()(i0 - lbounds[0],
                                 i1 - lbounds[1],
                                 i2 - lbounds[2],
                                 i3 - lbounds[3],
                                 i4 - lbounds[4],
                                 i5 - lbounds[5],
                                 i6 - lbounds[6]);
    }
    KOKKOS_INLINE_FUNCTION T &operator() (int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7) const {
      return ViewLoc::operator()(i0 - lbounds[0],
                                 i1 - lbounds[1],
                                 i2 - lbounds[2],
                                 i3 - lbounds[3],
                                 i4 - lbounds[4],
                                 i5 - lbounds[5],
                                 i6 - lbounds[6],
                                 i7 - lbounds[7]);
    }
    KOKKOS_INLINE_FUNCTION T &operator() (int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8) const {
      return ViewLoc::operator()(i0 - lbounds[0],
                                 i1 - lbounds[1],
                                 i2 - lbounds[2],
                                 i3 - lbounds[3],
                                 i4 - lbounds[4],
                                 i5 - lbounds[5],
                                 i6 - lbounds[6],
                                 i7 - lbounds[7],
                                 i8 - lbounds[8]);
    }
  };



  template <class DataType, class MemorySpace>
  class CView<DataType,MemorySpace,Kokkos::MemoryUnmanaged> : public Kokkos::View<DataType,Kokkos::LayoutRight,MemorySpace,Kokkos::MemoryUnmanaged> {
  public:
    typedef Kokkos::View<DataType,Kokkos::LayoutRight,MemorySpace,Kokkos::MemoryUnmanaged> ViewLoc;
    typedef typename ViewLoc::reference_type TR;
    typedef typename ViewLoc::pointer_type TP;
    int static constexpr rank = ViewLoc::rank;
    int lbounds[rank];


    CView(TP const &data, Bnd b0) : ViewLoc(data,b0.s) {
      lbounds[0] = b0.l;
    }
    CView(TP const &data, Bnd b0, Bnd b1) : ViewLoc(data,b0.s,b1.s) {
      lbounds[0] = b0.l;
      lbounds[1] = b1.l;
    }
    CView(TP const &data, Bnd b0, Bnd b1, Bnd b2) : ViewLoc(data,b0.s,b1.s,b2.s) {
      lbounds[0] = b0.l;
      lbounds[1] = b1.l;
      lbounds[2] = b2.l;
    }
    CView(TP const &data, Bnd b0, Bnd b1, Bnd b2, Bnd b3) : ViewLoc(data,b0.s,b1.s,b2.s,b3.s) {
      lbounds[0] = b0.l;
      lbounds[1] = b1.l;
      lbounds[2] = b2.l;
      lbounds[3] = b3.l;
    }
    CView(TP const &data, Bnd b0, Bnd b1, Bnd b2, Bnd b3, Bnd b4) : ViewLoc(data,b0.s,b1.s,b2.s,b3.s,b4.s) {
      lbounds[0] = b0.l;
      lbounds[1] = b1.l;
      lbounds[2] = b2.l;
      lbounds[3] = b3.l;
      lbounds[4] = b4.l;
    }
    CView(TP const &data, Bnd b0, Bnd b1, Bnd b2, Bnd b3, Bnd b4, Bnd b5) : ViewLoc(data,b0.s,b1.s,b2.s,b3.s,b4.s,b5.s) {
      lbounds[0] = b0.l;
      lbounds[1] = b1.l;
      lbounds[2] = b2.l;
      lbounds[3] = b3.l;
      lbounds[4] = b4.l;
      lbounds[5] = b5.l;
    }
    CView(TP const &data, Bnd b0, Bnd b1, Bnd b2, Bnd b3, Bnd b4, Bnd b5, Bnd b6) : ViewLoc(data,b0.s,b1.s,b2.s,b3.s,b4.s,b5.s,b6.s) {
      lbounds[0] = b0.l;
      lbounds[1] = b1.l;
      lbounds[2] = b2.l;
      lbounds[3] = b3.l;
      lbounds[4] = b4.l;
      lbounds[5] = b5.l;
      lbounds[6] = b6.l;
    }
    CView(TP const &data, Bnd b0, Bnd b1, Bnd b2, Bnd b3, Bnd b4, Bnd b5, Bnd b6, Bnd b7) : ViewLoc(data,b0.s,b1.s,b2.s,b3.s,b4.s,b5.s,b6.s,b7.s) {
      lbounds[0] = b0.l;
      lbounds[1] = b1.l;
      lbounds[2] = b2.l;
      lbounds[3] = b3.l;
      lbounds[4] = b4.l;
      lbounds[5] = b5.l;
      lbounds[6] = b6.l;
      lbounds[7] = b7.l;
    }
    CView(ViewLoc const &rhs , int const *lbounds_in) : ViewLoc(rhs) {
      for (int i=0; i < rank; i++) { lbounds[i] = lbounds_in[i]; }
    }


    // COPY CONSTRUCTORS / FUNCTIONS
    KOKKOS_INLINE_FUNCTION CView            (CView const &rhs) {
      static_cast<ViewLoc &>(*this) = static_cast<ViewLoc const &>(rhs);
      for (int i=0; i<rank; i++) { this->lbounds[i] = rhs.lbounds[i]; }
    }
    KOKKOS_INLINE_FUNCTION CView & operator=(CView const &rhs) {
      if (this == &rhs) { return *this; }
      static_cast<ViewLoc &>(*this) = static_cast<ViewLoc const &>(rhs);
      for (int i=0; i<rank; i++) { this->lbounds[i] = rhs.lbounds[i]; }
      return *this;
    }
    // MOVE CONSTRUCTORS
    KOKKOS_INLINE_FUNCTION CView            (CView &&rhs) {
      static_cast<ViewLoc &>(*this) = std::move(static_cast<ViewLoc const &>(rhs));
      for (int i=0; i<rank; i++) { this->lbounds[i] = rhs.lbounds[i]; }
    }
    KOKKOS_INLINE_FUNCTION CView & operator=(CView &&rhs) {
      if (this == &rhs) { return *this; }
      static_cast<ViewLoc &>(*this) = std::move(static_cast<ViewLoc const &>(rhs));
      for (int i=0; i<rank; i++) { this->lbounds[i] = rhs.lbounds[i]; }
      return *this;
    }

    KOKKOS_INLINE_FUNCTION TR &operator() (int i0) const {
      return ViewLoc::operator()(i0 - lbounds[0]);
    }
    KOKKOS_INLINE_FUNCTION TR &operator() (int i0, int i1) const {
      return ViewLoc::operator()(i0 - lbounds[0],
                                 i1 - lbounds[1]);
    }
    KOKKOS_INLINE_FUNCTION TR &operator() (int i0, int i1, int i2) const {
      return ViewLoc::operator()(i0 - lbounds[0],
                                 i1 - lbounds[1],
                                 i2 - lbounds[2]);
    }
    KOKKOS_INLINE_FUNCTION TR &operator() (int i0, int i1, int i2, int i3) const {
      return ViewLoc::operator()(i0 - lbounds[0],
                                 i1 - lbounds[1],
                                 i2 - lbounds[2],
                                 i3 - lbounds[3]);
    }
    KOKKOS_INLINE_FUNCTION TR &operator() (int i0, int i1, int i2, int i3, int i4) const {
      return ViewLoc::operator()(i0 - lbounds[0],
                                 i1 - lbounds[1],
                                 i2 - lbounds[2],
                                 i3 - lbounds[3],
                                 i4 - lbounds[4]);
    }
    KOKKOS_INLINE_FUNCTION TR &operator() (int i0, int i1, int i2, int i3, int i4, int i5) const {
      return ViewLoc::operator()(i0 - lbounds[0],
                                 i1 - lbounds[1],
                                 i2 - lbounds[2],
                                 i3 - lbounds[3],
                                 i4 - lbounds[4],
                                 i5 - lbounds[5]);
    }
    KOKKOS_INLINE_FUNCTION TR &operator() (int i0, int i1, int i2, int i3, int i4, int i5, int i6) const {
      return ViewLoc::operator()(i0 - lbounds[0],
                                 i1 - lbounds[1],
                                 i2 - lbounds[2],
                                 i3 - lbounds[3],
                                 i4 - lbounds[4],
                                 i5 - lbounds[5],
                                 i6 - lbounds[6]);
    }
    KOKKOS_INLINE_FUNCTION TR &operator() (int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7) const {
      return ViewLoc::operator()(i0 - lbounds[0],
                                 i1 - lbounds[1],
                                 i2 - lbounds[2],
                                 i3 - lbounds[3],
                                 i4 - lbounds[4],
                                 i5 - lbounds[5],
                                 i6 - lbounds[6],
                                 i7 - lbounds[7]);
    }
    KOKKOS_INLINE_FUNCTION TR &operator() (int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8) const {
      return ViewLoc::operator()(i0 - lbounds[0],
                                 i1 - lbounds[1],
                                 i2 - lbounds[2],
                                 i3 - lbounds[3],
                                 i4 - lbounds[4],
                                 i5 - lbounds[5],
                                 i6 - lbounds[6],
                                 i7 - lbounds[7],
                                 i8 - lbounds[8]);
    }
  };



  template <class DataType, class MemorySpace, class MemoryTraits>
  inline CView<DataType,Kokkos::HostSpace,Kokkos::MemoryManaged> create_mirror(CView<DataType,MemorySpace,MemoryTraits> const &src) {
    typedef Kokkos::View<DataType,Kokkos::LayoutRight,Kokkos::HostSpace,Kokkos::MemoryManaged> ViewLoc;
    ViewLoc dest_view = Kokkos::create_mirror( static_cast<typename CView<DataType,MemorySpace,MemoryTraits>::ViewLoc const &>(src) );
    CView<DataType,Kokkos::HostSpace,Kokkos::MemoryManaged> dest(dest_view, src.lbounds);
    return dest;
  }


  template <int N> class LoopBounds;



  template<> class LoopBounds<1> : public Kokkos::RangePolicy<Kokkos::IndexType<int>> {
    public:
    LoopBounds(Bnd b) : Kokkos::RangePolicy<Kokkos::IndexType<int>>(b.l,b.u+1) { };
    Kokkos::RangePolicy<Kokkos::IndexType<int>> to_range() const { return static_cast<Kokkos::RangePolicy<Kokkos::IndexType<int>>>(*this); }
    
  };
  template<> class LoopBounds<2> : public Kokkos::MDRangePolicy<Kokkos::Rank<2>,Kokkos::IndexType<int>> {
    public:
    LoopBounds(Bnd b0, Bnd b1) : Kokkos::MDRangePolicy<Kokkos::Rank<2>,Kokkos::IndexType<int>>({b0.l  ,b1.l  },
                                                                                               {b0.u+1,b1.u+1}) { };
    Kokkos::MDRangePolicy<Kokkos::Rank<2>,Kokkos::IndexType<int>> to_range() const { return static_cast<Kokkos::MDRangePolicy<Kokkos::Rank<2>,Kokkos::IndexType<int>>>(*this); }
  };
  template<> class LoopBounds<3> : public Kokkos::MDRangePolicy<Kokkos::Rank<3>,Kokkos::IndexType<int>> {
    public:
    LoopBounds(Bnd b0, Bnd b1, Bnd b2) : Kokkos::MDRangePolicy<Kokkos::Rank<3>,Kokkos::IndexType<int>>({b0.l  ,b1.l  ,b2.l  },
                                                                                                       {b0.u+1,b1.u+1,b2.u+1}) { };
    Kokkos::MDRangePolicy<Kokkos::Rank<3>,Kokkos::IndexType<int>> to_range() const { return static_cast<Kokkos::MDRangePolicy<Kokkos::Rank<3>,Kokkos::IndexType<int>>>(*this); }
  };
  template<> class LoopBounds<4> : public Kokkos::MDRangePolicy<Kokkos::Rank<4>,Kokkos::IndexType<int>> {
    public:
    LoopBounds(Bnd b0, Bnd b1, Bnd b2, Bnd b3) : Kokkos::MDRangePolicy<Kokkos::Rank<4>,Kokkos::IndexType<int>>({b0.l  ,b1.l  ,b2.l  ,b3.l  },
                                                                                                               {b0.u+1,b1.u+1,b2.u+1,b3.u+1}) { };
    Kokkos::MDRangePolicy<Kokkos::Rank<4>,Kokkos::IndexType<int>> to_range() const { return static_cast<Kokkos::MDRangePolicy<Kokkos::Rank<4>,Kokkos::IndexType<int>>>(*this); }
  };
  template<> class LoopBounds<5> : public Kokkos::MDRangePolicy<Kokkos::Rank<5>,Kokkos::IndexType<int>> {
    public:
    LoopBounds(Bnd b0, Bnd b1, Bnd b2, Bnd b3, Bnd b4) : Kokkos::MDRangePolicy<Kokkos::Rank<5>,Kokkos::IndexType<int>>({b0.l  ,b1.l  ,b2.l  ,b3.l  ,b4.l  },
                                                                                                                       {b0.u+1,b1.u+1,b2.u+1,b3.u+1,b4.u+1}) { };
    Kokkos::MDRangePolicy<Kokkos::Rank<5>,Kokkos::IndexType<int>> to_range() const { return static_cast<Kokkos::MDRangePolicy<Kokkos::Rank<5>,Kokkos::IndexType<int>>>(*this); }
  };
  template<> class LoopBounds<6> : public Kokkos::MDRangePolicy<Kokkos::Rank<6>,Kokkos::IndexType<int>> {
    public:
    LoopBounds(Bnd b0, Bnd b1, Bnd b2, Bnd b3, Bnd b4, Bnd b5) : Kokkos::MDRangePolicy<Kokkos::Rank<6>,Kokkos::IndexType<int>>({b0.l  ,b1.l  ,b2.l  ,b3.l  ,b4.l  ,b5.l  },
                                                                                                                               {b0.u+1,b1.u+1,b2.u+1,b3.u+1,b4.u+1,b5.u+1}) { };
    Kokkos::MDRangePolicy<Kokkos::Rank<6>,Kokkos::IndexType<int>> to_range() const { return static_cast<Kokkos::MDRangePolicy<Kokkos::Rank<6>,Kokkos::IndexType<int>>>(*this); }
  };


  #ifdef __CUDACC__
  template <class DataType, class MemoryTraits>
  inline typename Kokkos::View<DataType>::non_const_value_type sum(CView<DataType,memDevice,MemoryTraits> const &arr_in) {
    typedef typename Kokkos::View<DataType>::non_const_value_type TYPE;
    TYPE s = 0;
    Kokkos::View<TYPE *,Kokkos::LayoutRight,memDevice,Kokkos::MemoryUnmanaged> arr(arr_in.data(),arr_in.size());
    Kokkos::parallel_reduce(arr.size() , KOKKOS_LAMBDA (int const i, TYPE &update) {
      update += arr(i);
    } , s );
    return s;
  }
  #endif

  template <class DataType, class MemoryTraits>
  inline typename Kokkos::View<DataType>::non_const_value_type sum(CView<DataType,memHost,MemoryTraits> const &arr_in) {
    typedef typename Kokkos::View<DataType>::non_const_value_type TYPE;
    TYPE s = 0;
    Kokkos::View<TYPE *,Kokkos::LayoutRight,memHost,Kokkos::MemoryUnmanaged> arr(arr_in.data(),arr_in.size());
    for (int i=0; i < arr.size(); i++) {
      s += arr(i);
    }
    return s;
  }



  #ifdef __CUDACC__
  template <class DataType, class MemoryTraits>
  inline typename Kokkos::View<DataType>::non_const_value_type sum(Kokkos::View<DataType,Kokkos::LayoutRight,memDevice,MemoryTraits> const &arr_in) {
    typedef typename Kokkos::View<DataType>::non_const_value_type TYPE;
    TYPE s = 0;
    Kokkos::View<TYPE *,Kokkos::LayoutRight,memDevice,Kokkos::MemoryUnmanaged> arr(arr_in.data(),arr_in.size());
    Kokkos::parallel_reduce(arr.size() , KOKKOS_LAMBDA (int const i, TYPE &update) {
      update += arr(i);
    } , s );
    return s;
  }
  #endif

  template <class DataType, class MemoryTraits>
  inline typename Kokkos::View<DataType>::non_const_value_type sum(Kokkos::View<DataType,Kokkos::LayoutRight,memHost,MemoryTraits> const &arr_in) {
    typedef typename Kokkos::View<DataType>::non_const_value_type TYPE;
    TYPE s = 0;
    Kokkos::View<TYPE *,Kokkos::LayoutRight,memHost,Kokkos::MemoryUnmanaged> arr(arr_in.data(),arr_in.size());
    for (int i=0; i < arr.size(); i++) {
      s += arr(i);
    }
    return s;
  }


} //c
} //KokkosWrap



namespace Kokkos {
  template <int N, class F>
  void parallel_for( KokkosWrap::fortran::LoopBounds<N> const &LB , F const &f ) {
    Kokkos::parallel_for( LB.to_range() , f );
  }
  template <int N, class F>
  void parallel_for( std::string str , KokkosWrap::fortran::LoopBounds<N> const &LB , F const &f ) {
    Kokkos::parallel_for( str , LB.to_range() , f );
  }

  template <int N, class F>
  void parallel_for( KokkosWrap::c::LoopBounds<N> const &LB , F const &f ) {
    Kokkos::parallel_for( LB.to_range() , f );
  }
  template <int N, class F>
  void parallel_for( std::string str , KokkosWrap::c::LoopBounds<N> const &LB , F const &f ) {
    Kokkos::parallel_for( str , LB.to_range() , f );
  }
}

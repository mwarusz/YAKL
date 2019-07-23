
#ifndef _ARRAY_H_
#define _ARRAY_H_

#include <iostream>
#include <iomanip>
#include <time.h>
#include <math.h>

#ifdef ARRAY_DEBUG
#include <stdexcept>
#include <sstream>
#include <string>
#endif

#ifdef __NVCC__
#define _HOSTDEV __host__ __device__
#else
#define _HOSTDEV 
#endif


/* Array<T>
Multi-dimensional array with functor indexing up to eight dimensions.
*/


template <class T> class Array {

  public :

  typedef unsigned long ulong;


  int   ndims;
  ulong dimSizes[8];
  long  offsets [8];
  ulong totElems;
  T     * __restrict data;
  Array<T> *orig;


  inline void nullify() {
    data = NULL;
    ndims = 0;
    totElems = 0;
    for (int i=0; i<8; i++) {
      dimSizes[i] = 0;
      offsets [i] = 0;
    }
  }

  /* CONSTRUCTORS
  You can declare the array empty or with many dimensions
  Always nullify before beginning so that data == NULL upon init. This allows the
  setup() functions to keep from deallocating data upon initialization, since
  you don't know what "data" will be when the object is created.
  */
  Array() {
    nullify();
    orig = this;
  }
  //Define the dimension ranges using an array of upper bounds, assuming lower bounds to be zero
  Array(ulong const d1) {
    nullify();
    orig = this;
    setup(d1);
  }
  Array(ulong const d1, ulong const d2) {
    nullify();
    orig = this;
    setup(d1,d2);
  }
  Array(ulong const d1, ulong const d2, ulong const d3) {
    nullify();
    orig = this;
    setup(d1,d2,d3);
  }
  Array(ulong const d1, ulong const d2, ulong const d3, ulong const d4) {
    nullify();
    orig = this;
    setup(d1,d2,d3,d4);
  }
  Array(ulong const d1, ulong const d2, ulong const d3, ulong const d4, ulong const d5) {
    nullify();
    orig = this;
    setup(d1,d2,d3,d4,d5);
  }
  Array(ulong const d1, ulong const d2, ulong const d3, ulong const d4, ulong const d5, ulong const d6) {
    nullify();
    orig = this;
    setup(d1,d2,d3,d4,d5,d6);
  }
  Array(ulong const d1, ulong const d2, ulong const d3, ulong const d4, ulong const d5, ulong const d6, ulong const d7) {
    nullify();
    orig = this;
    setup(d1,d2,d3,d4,d5,d6,d7);
  }
  Array(ulong const d1, ulong const d2, ulong const d3, ulong const d4, ulong const d5, ulong const d6, ulong const d7, ulong const d8) {
    nullify();
    orig = this;
    setup(d1,d2,d3,d4,d5,d6,d7,d8);
  }


  /* DESTRUCTOR
  Make sure the internal arrays are allocated before freeing them
  */
  ~Array() {
    // Only deallocate data if it's not NULL and if this is the original object (not a copy)
    if (data != NULL && orig == this) {
      deallocate();
    }
  }

  /* SETUP FUNCTIONS
  Initialize the array with the given dimensions
  */
  inline void setup(ulong const d1) {
    ulong tmp[1];
    tmp[0] = d1;
    setup_arr((ulong) 1,tmp);
  }
  inline void setup(ulong const d1, ulong const d2) {
    ulong tmp[2];
    tmp[0] = d1;
    tmp[1] = d2;
    setup_arr((ulong) 2,tmp);
  }
  inline void setup(ulong const d1, ulong const d2, ulong const d3) {
    ulong tmp[3];
    tmp[0] = d1;
    tmp[1] = d2;
    tmp[2] = d3;
    setup_arr((ulong) 3,tmp);
  }
  inline void setup(ulong const d1, ulong const d2, ulong const d3, ulong const d4) {
    ulong tmp[4];
    tmp[0] = d1;
    tmp[1] = d2;
    tmp[2] = d3;
    tmp[3] = d4;
    setup_arr((ulong) 4,tmp);
  }
  inline void setup(ulong const d1, ulong const d2, ulong const d3, ulong const d4, ulong const d5) {
    ulong tmp[5];
    tmp[0] = d1;
    tmp[1] = d2;
    tmp[2] = d3;
    tmp[3] = d4;
    tmp[4] = d5;
    setup_arr((ulong) 5,tmp);
  }
  inline void setup(ulong const d1, ulong const d2, ulong const d3, ulong const d4, ulong const d5, ulong const d6) {
    ulong tmp[6];
    tmp[0] = d1;
    tmp[1] = d2;
    tmp[2] = d3;
    tmp[3] = d4;
    tmp[4] = d5;
    tmp[5] = d6;
    setup_arr((ulong) 6,tmp);
  }
  inline void setup(ulong const d1, ulong const d2, ulong const d3, ulong const d4, ulong const d5, ulong const d6, ulong const d7) {
    ulong tmp[7];
    tmp[0] = d1;
    tmp[1] = d2;
    tmp[2] = d3;
    tmp[3] = d4;
    tmp[4] = d5;
    tmp[5] = d6;
    tmp[6] = d7;
    setup_arr((ulong) 7,tmp);
  }
  inline void setup(ulong const d1, ulong const d2, ulong const d3, ulong const d4, ulong const d5, ulong const d6, ulong const d7, ulong const d8) {
    ulong tmp[8];
    tmp[0] = d1;
    tmp[1] = d2;
    tmp[2] = d3;
    tmp[3] = d4;
    tmp[4] = d5;
    tmp[5] = d6;
    tmp[6] = d7;
    tmp[7] = d8;
    setup_arr((ulong) 8,tmp);
  }
  inline void setup_arr(ulong const ndims, ulong const dimSizes[]) {
    // If a buffer exists, destroy it and start over
    if ( data != NULL ) {
      deallocate();
    }

    // Setup this Array with the given number of dimensions and dimension sizes
    this->ndims = ndims;
    totElems = 1;
    for (ulong i=0; i<ndims; i++) {
      this->dimSizes[i] = dimSizes[i];
      totElems *= this->dimSizes[i];
    }
    offsets[ndims-1] = 1;
    for (int i=ndims-2; i>=0; i--) {
      offsets[i] = offsets[i+1] * dimSizes[i+1];
    }
    #ifdef __NVCC__
      cudaMallocManaged(&data,totElems*sizeof(T));
    #else
      data = new T[totElems];
    #endif
  }


  inline void deallocate() {
    #ifdef __NVCC__
      cudaFree(data);
    #else
      delete[] data; nullify();
    #endif
  }


  /* ARRAY INDEXERS (FORTRAN index ordering)
  Return the element at the given index (either read-only or read-write)
  */
  inline _HOSTDEV T &operator()(ulong const i0) const {
    #ifdef ARRAY_DEBUG
    this->check_dims(1,ndims,__FILE__,__LINE__);
    this->check_index(0,i0,0,dimSizes[0]-1,__FILE__,__LINE__);
    #endif
    ulong ind = i0;
    return data[ind];
  }
  inline _HOSTDEV T &operator()(ulong const i0, ulong const i1) const {
    #ifdef ARRAY_DEBUG
    this->check_dims(2,ndims,__FILE__,__LINE__);
    this->check_index(0,i0,0,dimSizes[0]-1,__FILE__,__LINE__);
    this->check_index(1,i1,0,dimSizes[1]-1,__FILE__,__LINE__);
    #endif
    ulong ind = i0*offsets[0] + i1;
    return data[ind];
  }
  inline _HOSTDEV T &operator()(ulong const i0, ulong const i1, ulong const i2) const {
    #ifdef ARRAY_DEBUG
    this->check_dims(3,ndims,__FILE__,__LINE__);
    this->check_index(0,i0,0,dimSizes[0]-1,__FILE__,__LINE__);
    this->check_index(1,i1,0,dimSizes[1]-1,__FILE__,__LINE__);
    this->check_index(2,i2,0,dimSizes[2]-1,__FILE__,__LINE__);
    #endif
    ulong ind = i0*offsets[0] + i1*offsets[1] + i2;
    return data[ind];
  }
  inline _HOSTDEV T &operator()(ulong const i0, ulong const i1, ulong const i2, ulong const i3) const {
    #ifdef ARRAY_DEBUG
    this->check_dims(4,ndims,__FILE__,__LINE__);
    this->check_index(0,i0,0,dimSizes[0]-1,__FILE__,__LINE__);
    this->check_index(1,i1,0,dimSizes[1]-1,__FILE__,__LINE__);
    this->check_index(2,i2,0,dimSizes[2]-1,__FILE__,__LINE__);
    this->check_index(3,i3,0,dimSizes[3]-1,__FILE__,__LINE__);
    #endif
    ulong ind = i0*offsets[0] + i1*offsets[1] + i2*offsets[2] + i3;
    return data[ind];
  }
  inline _HOSTDEV T &operator()(ulong const i0, ulong const i1, ulong const i2, ulong const i3, ulong const i4) const {
    #ifdef ARRAY_DEBUG
    this->check_dims(5,ndims,__FILE__,__LINE__);
    this->check_index(0,i0,0,dimSizes[0]-1,__FILE__,__LINE__);
    this->check_index(1,i1,0,dimSizes[1]-1,__FILE__,__LINE__);
    this->check_index(2,i2,0,dimSizes[2]-1,__FILE__,__LINE__);
    this->check_index(3,i3,0,dimSizes[3]-1,__FILE__,__LINE__);
    this->check_index(4,i4,0,dimSizes[4]-1,__FILE__,__LINE__);
    #endif
    ulong ind = i0*offsets[0] + i1*offsets[1] + i2*offsets[2] + i3*offsets[3] + i4;
    return data[ind];
  }
  inline _HOSTDEV T &operator()(ulong const i0, ulong const i1, ulong const i2, ulong const i3, ulong const i4, ulong const i5) const {
    #ifdef ARRAY_DEBUG
    this->check_dims(6,ndims,__FILE__,__LINE__);
    this->check_index(0,i0,0,dimSizes[0]-1,__FILE__,__LINE__);
    this->check_index(1,i1,0,dimSizes[1]-1,__FILE__,__LINE__);
    this->check_index(2,i2,0,dimSizes[2]-1,__FILE__,__LINE__);
    this->check_index(3,i3,0,dimSizes[3]-1,__FILE__,__LINE__);
    this->check_index(4,i4,0,dimSizes[4]-1,__FILE__,__LINE__);
    this->check_index(5,i5,0,dimSizes[5]-1,__FILE__,__LINE__);
    #endif
    ulong ind = i0*offsets[0] + i1*offsets[1] + i2*offsets[2] + i3*offsets[3] + i4*offsets[4] + i5;
    return data[ind];
  }
  inline _HOSTDEV T &operator()(ulong const i0, ulong const i1, ulong const i2, ulong const i3, ulong const i4, ulong const i5, ulong const i6) const {
    #ifdef ARRAY_DEBUG
    this->check_dims(7,ndims,__FILE__,__LINE__);
    this->check_index(0,i0,0,dimSizes[0]-1,__FILE__,__LINE__);
    this->check_index(1,i1,0,dimSizes[1]-1,__FILE__,__LINE__);
    this->check_index(2,i2,0,dimSizes[2]-1,__FILE__,__LINE__);
    this->check_index(3,i3,0,dimSizes[3]-1,__FILE__,__LINE__);
    this->check_index(4,i4,0,dimSizes[4]-1,__FILE__,__LINE__);
    this->check_index(5,i5,0,dimSizes[5]-1,__FILE__,__LINE__);
    this->check_index(6,i6,0,dimSizes[6]-1,__FILE__,__LINE__);
    #endif
    ulong ind = i0*offsets[0] + i1*offsets[1] + i2*offsets[2] + i3*offsets[3] + i4*offsets[4] + i5*offsets[5] + i6;
    return data[ind];
  }
  inline _HOSTDEV T &operator()(ulong const i0, ulong const i1, ulong const i2, ulong const i3, ulong const i4, ulong const i5, ulong const i6, ulong const i7) const {
    #ifdef ARRAY_DEBUG
    this->check_dims(8,ndims,__FILE__,__LINE__);
    this->check_index(0,i0,0,dimSizes[0]-1,__FILE__,__LINE__);
    this->check_index(1,i1,0,dimSizes[1]-1,__FILE__,__LINE__);
    this->check_index(2,i2,0,dimSizes[2]-1,__FILE__,__LINE__);
    this->check_index(3,i3,0,dimSizes[3]-1,__FILE__,__LINE__);
    this->check_index(4,i4,0,dimSizes[4]-1,__FILE__,__LINE__);
    this->check_index(5,i5,0,dimSizes[5]-1,__FILE__,__LINE__);
    this->check_index(6,i6,0,dimSizes[6]-1,__FILE__,__LINE__);
    this->check_index(7,i7,0,dimSizes[7]-1,__FILE__,__LINE__);
    #endif
    ulong ind = i0*offsets[0] + i1*offsets[1] + i2*offsets[2] + i3*offsets[3] + i4*offsets[4] + i5*offsets[5] + i6*offsets[6] + i7;
    return data[ind];
  }

  inline _HOSTDEV void check_dims(int const ndims_called, int const ndims_actual, char const *file, int const line) const {
    #ifdef ARRAY_DEBUG
    if (ndims_called != ndims_actual) {
      std::stringstream ss;
      ss << "Using " << ndims_called << " dimensions to index an Array with " << ndims_actual << " dimensions\n";
      ss << "File, Line: " << file << ", " << line << "\n";
      throw std::out_of_range(ss.str());
    }
    #endif
  }
  inline _HOSTDEV void check_index(int const dim, long const ind, long const lb, long const ub, char const *file, int const line) const {
    #ifdef ARRAY_DEBUG
    if (ind < lb || ind > ub) {
      std::stringstream ss;
      ss << "Index " << dim << " of " << this->ndims << " out of bounds\n";
      ss << "File, Line: " << file << ", " << line << "\n";
      ss << "Index: " << ind << ". Bounds: (" << lb << "," << ub << ")\n";
      throw std::out_of_range(ss.str());
    }
    #endif
  }

  /* OPERATOR=
  Allow the user to set the entire Array to a single value */
  template <class I> inline _HOSTDEV void operator=(I const rhs) {
    for (ulong i=0; i < totElems; i++) {
      data[i] = rhs;
    }
  }
  /* Copy another Array's data to this one */
  inline _HOSTDEV void operator=(Array const &rhs) {
    #ifdef ARRAY_DEBUG
    if (this->totElems != rhs.totElems) {
      std::stringstream ss;
      ss << "Attempted value-copy via operator= between Arrays with incompatible lengths\n";
      ss << "File, Line: " << __FILE__ << ", " << __LINE__ << "\n";
      ss << "This size, rhs size:" << this->totElems << ", " << rhs.totElems << "\n";
      throw std::out_of_range(ss.str());
    }
    #endif
    for (ulong i=0; i < rhs.totElems; i++) {
      data[i] = rhs.data[i];
    }
  }
  /* Copy an array of values into this Array's data */
  template <class I> inline _HOSTDEV void operator=(I const *rhs) {
    for (ulong i=0; i<totElems; i++) {
      data[i] = rhs[i];
    }
  }

  /* COMPARISON */
  inline _HOSTDEV int dimsMatch(ulong const ndims, ulong const dimSizes[]) const {
    if (this->ndims != ndims) {
      return -1;
    }
    for (int i=0; i<ndims; i++) {
      if (this->dimSizes[i] != dimSizes[i]) {
        return -1;
      }
    }
    return 0;
  }

  /* ACCESSORS */
  inline _HOSTDEV int get_ndims() const {
    return ndims;
  }
  inline _HOSTDEV ulong get_totElems() const {
    return totElems;
  }
  inline _HOSTDEV ulong const *get_dimSizes() const {
    return dimSizes;
  }
  inline _HOSTDEV T *get_data() const {
    return data;
  }

  /* INFORM */
  inline void print_ndims() const {
    std::cout << "Number of Dimensions: " << ndims << "\n";
  }
  inline void print_totElems() const {
    std::cout << "Total Number of Elements: " << totElems << "\n";
  }
  inline void print_dimSizes() const {
    std::cout << "Dimension Sizes: ";
    for (int i=0; i<ndims; i++) {
      std::cout << dimSizes[i] << ", ";
    }
    std::cout << "\n";
  }
  inline void print_data() const {
    if (ndims == 1) {
      for (ulong i=0; i<dimSizes[0]; i++) {
        std::cout << std::setw(12) << (*this)(i) << "\n";
      }
    } else if (ndims == 2) {
      for (ulong j=0; j<dimSizes[0]; j++) {
        for (ulong i=0; i<dimSizes[1]; i++) {
          std::cout << std::setw(12) << (*this)(i,j) << " ";
        }
        std::cout << "\n";
      }
    } else if (ndims == 0) {
      std::cout << "Empty Array\n\n";
    } else {
      for (ulong i=0; i<totElems; i++) {
        std::cout << std::setw(12) << data[i] << " ";
      }
      std::cout << "\n";
    }
    std::cout << "\n";
  }

  /* OPERATOR<<
  Print the array. If it's 2-D, print a pretty looking matrix */
  inline friend std::ostream &operator<<(std::ostream& os, Array const &v) {
    os << "Number of Dimensions: " << v.ndims << "\n";
    os << "Total Number of Elements: " << v.totElems << "\n";
    os << "Dimension Sizes: ";
    for (int i=0; i<v.ndims; i++) {
      os << v.dimSizes[i] << ", ";
    }
    os << "\n";
    if (v.ndims == 1) {
      for (ulong i=0; i<v.dimSizes[0]; i++) {
        os << std::setw(12) << v(i) << "\n";
      }
    } else if (v.ndims == 2) {
      for (ulong j=0; j<v.dimSizes[1]; j++) {
        for (ulong i=0; i<v.dimSizes[0]; i++) {
          os << std::setw(12) << v(i,j) << " ";
        }
        os << "\n";
      }
    } else if (v.ndims == 0) {
      os << "Empty Array\n\n";
    } else {
      for (ulong i=0; i<v.totElems; i++) {
        os << v.data[i] << " ";
      }
      os << "\n";
    }
    os << "\n";
    return os;
  }

};

#endif

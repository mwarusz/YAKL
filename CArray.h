
#pragma once

template <class T, int rank, int myMem> class Array<T,rank,myMem,styleC> {
public:

  size_t offsets  [rank];  // Precomputed dimension offsets for efficient data access into a 1-D pointer
  size_t dimension[rank];  // Sizes of the 8 possible dimensions
  T      * myData;      // Pointer to the flattened internal data
  bool   owned;         // Whether is is owned (owned = allocated,ref_counted,deallocated) or not
  #ifdef YAKL_DEBUG
    std::string myname; // Label for debug printing. Only stored if debugging is turned on
  #endif


  // Start off all constructors making sure the pointers are null
  YAKL_INLINE void nullify() {
    owned = true;
    myData   = nullptr;
    refCount = nullptr;
  }

  /* CONSTRUCTORS
  You can declare the array empty or with up to 8 dimensions
  Like kokkos, you need to give a label for the array for debug printing
  Always nullify before beginning so that myData == nullptr upon init. This allows the
  setup() functions to keep from deallocating myData upon initialization, since
  you don't know what "myData" will be when the object is created.
  */
  YAKL_INLINE Array() {
    nullify();
  }
  YAKL_INLINE Array(char const * label) {
    nullify();
    #ifdef YAKL_DEBUG
      myname = std::string(label);
    #endif
  }
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Owned constructors
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  Array(char const * label, size_t const d1) {
    #ifdef YAKL_DEBUG
      if( rank != 1 ) { throw "ERROR: Calling invalid constructor on rank 1 Array"; }
    #endif
    nullify();
    setup(label,d1);
  }
  Array(char const * label, size_t const d1, size_t const d2) {
    #ifdef YAKL_DEBUG
      if( rank != 2 ) { throw "ERROR: Calling invalid constructor on rank 2 Array"; }
    #endif
    nullify();
    setup(label,d1,d2);
  }
  Array(char const * label, size_t const d1, size_t const d2, size_t const d3) {
    #ifdef YAKL_DEBUG
      if( rank != 3 ) { throw "ERROR: Calling invalid constructor on rank 3 Array"; }
    #endif
    nullify();
    setup(label,d1,d2,d3);
  }
  Array(char const * label, size_t const d1, size_t const d2, size_t const d3, size_t const d4) {
    #ifdef YAKL_DEBUG
      if( rank != 4 ) { throw "ERROR: Calling invalid constructor on rank 4 Array"; }
    #endif
    nullify();
    setup(label,d1,d2,d3,d4);
  }
  Array(char const * label, size_t const d1, size_t const d2, size_t const d3, size_t const d4, size_t const d5) {
    #ifdef YAKL_DEBUG
      if( rank != 5 ) { throw "ERROR: Calling invalid constructor on rank 5 Array"; }
    #endif
    nullify();
    setup(label,d1,d2,d3,d4,d5);
  }
  Array(char const * label, size_t const d1, size_t const d2, size_t const d3, size_t const d4, size_t const d5, size_t const d6) {
    #ifdef YAKL_DEBUG
      if( rank != 6 ) { throw "ERROR: Calling invalid constructor on rank 6 Array"; }
    #endif
    nullify();
    setup(label,d1,d2,d3,d4,d5,d6);
  }
  Array(char const * label, size_t const d1, size_t const d2, size_t const d3, size_t const d4, size_t const d5, size_t const d6, size_t const d7) {
    #ifdef YAKL_DEBUG
      if( rank != 7 ) { throw "ERROR: Calling invalid constructor on rank 7 Array"; }
    #endif
    nullify();
    setup(label,d1,d2,d3,d4,d5,d6,d7);
  }
  Array(char const * label, size_t const d1, size_t const d2, size_t const d3, size_t const d4, size_t const d5, size_t const d6, size_t const d7, size_t const d8) {
    #ifdef YAKL_DEBUG
      if( rank != 8 ) { throw "ERROR: Calling invalid constructor on rank 8 Array"; }
    #endif
    nullify();
    setup(label,d1,d2,d3,d4,d5,d6,d7,d8);
  }
  template <class INT, typename std::enable_if< std::is_integral<INT>::value , int >::type = 0>
  Array(char const * label, std::vector<INT> const dims) {
    #ifdef YAKL_DEBUG
      if ( dims.size() <= rank ) { throw "ERROR: dims < rank"; }
      if ( rank < 1 || rank > 8 ) { throw "ERROR: Invalid rank, must be between 1 and 8"; }
    #endif
    nullify();
         if ( rank == 1 ) { setup(label,dims[0]); }
    else if ( rank == 2 ) { setup(label,dims[0],dims[1]); }
    else if ( rank == 3 ) { setup(label,dims[0],dims[1],dims[2]); }
    else if ( rank == 4 ) { setup(label,dims[0],dims[1],dims[2],dims[3]); }
    else if ( rank == 5 ) { setup(label,dims[0],dims[1],dims[2],dims[3],dims[4]); }
    else if ( rank == 6 ) { setup(label,dims[0],dims[1],dims[2],dims[3],dims[4],dims[5]); }
    else if ( rank == 7 ) { setup(label,dims[0],dims[1],dims[2],dims[3],dims[4],dims[5],dims[6]); }
    else if ( rank == 8 ) { setup(label,dims[0],dims[1],dims[2],dims[3],dims[4],dims[5],dims[6],dims[7]); }
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Non-owned constructors
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  Array(char const * label, T * data, size_t const d1) {
    #ifdef YAKL_DEBUG
      if ( rank != 1 ) { throw "ERROR: Calling invalid constructor on rank 1 Array"; }
    #endif
    nullify();
    owned = false;
    setup(label,d1);
    myData = data;
  }
  Array(char const * label, T * data, size_t const d1, size_t const d2) {
    #ifdef YAKL_DEBUG
      if ( rank != 2 ) { throw "ERROR: Calling invalid constructor on rank 2 Array"; }
    #endif
    nullify();
    owned = false;
    setup(label,d1,d2);
    myData = data;
  }
  Array(char const * label, T * data, size_t const d1, size_t const d2, size_t const d3) {
    #ifdef YAKL_DEBUG
      if ( rank != 3 ) { throw "ERROR: Calling invalid constructor on rank 3 Array"; }
    #endif
    nullify();
    owned = false;
    setup(label,d1,d2,d3);
    myData = data;
  }
  Array(char const * label, T * data, size_t const d1, size_t const d2, size_t const d3, size_t const d4) {
    #ifdef YAKL_DEBUG
      if ( rank != 4 ) { throw "ERROR: Calling invalid constructor on rank 4 Array"; }
    #endif
    nullify();
    owned = false;
    setup(label,d1,d2,d3,d4);
    myData = data;
  }
  Array(char const * label, T * data, size_t const d1, size_t const d2, size_t const d3, size_t const d4, size_t const d5) {
    #ifdef YAKL_DEBUG
      if ( rank != 5 ) { throw "ERROR: Calling invalid constructor on rank 5 Array"; }
    #endif
    nullify();
    owned = false;
    setup(label,d1,d2,d3,d4,d5);
    myData = data;
  }
  Array(char const * label, T * data, size_t const d1, size_t const d2, size_t const d3, size_t const d4, size_t const d5, size_t const d6) {
    #ifdef YAKL_DEBUG
      if ( rank != 6 ) { throw "ERROR: Calling invalid constructor on rank 6 Array"; }
    #endif
    nullify();
    owned = false;
    setup(label,d1,d2,d3,d4,d5,d6);
    myData = data;
  }
  Array(char const * label, T * data, size_t const d1, size_t const d2, size_t const d3, size_t const d4, size_t const d5, size_t const d6, size_t const d7) {
    #ifdef YAKL_DEBUG
      if ( rank != 7 ) { throw "ERROR: Calling invalid constructor on rank 7 Array"; }
    #endif
    nullify();
    owned = false;
    setup(label,d1,d2,d3,d4,d5,d6,d7);
    myData = data;
  }
  Array(char const * label, T * data, size_t const d1, size_t const d2, size_t const d3, size_t const d4, size_t const d5, size_t const d6, size_t const d7, size_t const d8) {
    #ifdef YAKL_DEBUG
      if ( rank != 8 ) { throw "ERROR: Calling invalid constructor on rank 8 Array"; }
    #endif
    nullify();
    owned = false;
    setup(label,d1,d2,d3,d4,d5,d6,d7,d8);
    myData = data;
  }
  template <class INT, typename std::enable_if< std::is_integral<INT>::value , int >::type = 0>
  Array(char const * label, T * data, std::vector<INT> const dims) {
    #ifdef YAKL_DEBUG
      if ( dims.size() <= rank ) { throw "ERROR: dims < rank"; }
      if ( rank < 1 || rank > 8 ) { throw "ERROR: Invalid rank, must be between 1 and 8"; }
    #endif
    nullify();
    owned = false;
         if ( rank == 1 ) { setup(label,dims[0]); }
    else if ( rank == 2 ) { setup(label,dims[0],dims[1]); }
    else if ( rank == 3 ) { setup(label,dims[0],dims[1],dims[2]); }
    else if ( rank == 4 ) { setup(label,dims[0],dims[1],dims[2],dims[3]); }
    else if ( rank == 5 ) { setup(label,dims[0],dims[1],dims[2],dims[3],dims[4]); }
    else if ( rank == 6 ) { setup(label,dims[0],dims[1],dims[2],dims[3],dims[4],dims[5]); }
    else if ( rank == 7 ) { setup(label,dims[0],dims[1],dims[2],dims[3],dims[4],dims[5],dims[6]); }
    else if ( rank == 8 ) { setup(label,dims[0],dims[1],dims[2],dims[3],dims[4],dims[5],dims[6],dims[7]); }
    myData = data;
  }


  /*
  COPY CONSTRUCTORS / FUNCTIONS
  This shares the pointers with another Array and increments the refCounter
  */
  Array(Array const &rhs) {
    // constructor, so no need to deallocate
    nullify();
    owned    = rhs.owned;
    for (int i=0; i<rank; i++) {
      offsets  [i] = rhs.offsets  [i];
      dimension[i] = rhs.dimension[i];
    }
    #ifdef YAKL_DEBUG
      myname = rhs.myname;
    #endif
    myData   = rhs.myData;
    refCount = rhs.refCount;
    if (owned) { (*refCount)++; }
  }


  Array & operator=(Array const &rhs) {
    if (this == &rhs) {
      return *this;
    }
    owned    = rhs.owned;
    deallocate();
    for (int i=0; i<rank; i++) {
      offsets  [i] = rhs.offsets  [i];
      dimension[i] = rhs.dimension[i];
    }
    #ifdef YAKL_DEBUG
      myname = rhs.myname;
    #endif
    myData   = rhs.myData;
    refCount = rhs.refCount;
    if (owned) { (*refCount)++; }

    return *this;
  }


  /*
  MOVE CONSTRUCTORS
  This steals the pointers form the rhs rather than sharing and sets rhs pointers to nullptr.
  Therefore, no need to increment refCout
  */
  Array(Array &&rhs) {
    // constructor, so no need to deallocate
    nullify();
    owned    = rhs.owned;
    for (int i=0; i<rank; i++) {
      offsets  [i] = rhs.offsets  [i];
      dimension[i] = rhs.dimension[i];
    }
    #ifdef YAKL_DEBUG
      myname = rhs.myname;
    #endif
    myData   = rhs.myData;
    refCount = rhs.refCount;

    rhs.myData   = nullptr;
    rhs.refCount = nullptr;
  }


  Array& operator=(Array &&rhs) {
    if (this == &rhs) { return *this; }
    owned    = rhs.owned;
    deallocate();
    for (int i=0; i<rank; i++) {
      offsets  [i] = rhs.offsets  [i];
      dimension[i] = rhs.dimension[i];
    }
    #ifdef YAKL_DEBUG
      myname = rhs.myname;
    #endif
    myData   = rhs.myData;
    refCount = rhs.refCount;

    rhs.myData   = nullptr;
    rhs.refCount = nullptr;

    return *this;
  }


  /*
  DESTRUCTOR
  Decrement the refCounter, and if it's zero, deallocate and nullify.  
  */
  ~Array() {
    deallocate();
  }


  /* ARRAY INDEXERS (FORTRAN index ordering)
  Return the element at the given index (either read-only or read-write)
  */
  YAKL_INLINE T &operator()(size_t const i0) const {
    #ifdef YAKL_DEBUG
      if ( rank != 1 ) { throw "ERROR: Calling invalid function on rank 1 Array"; }
      this->check_index(0,i0,0,dimension[0]-1,__FILE__,__LINE__);
    #endif
    size_t ind = i0;
    return myData[ind];
  }
  YAKL_INLINE T &operator()(size_t const i0, size_t const i1) const {
    #ifdef YAKL_DEBUG
      if ( rank != 2 ) { throw "ERROR: Calling invalid function on rank 2 Array"; }
      this->check_index(0,i0,0,dimension[0]-1,__FILE__,__LINE__);
      this->check_index(1,i1,0,dimension[1]-1,__FILE__,__LINE__);
    #endif
    size_t ind = i0*offsets[0] + i1;
    return myData[ind];
  }
  YAKL_INLINE T &operator()(size_t const i0, size_t const i1, size_t const i2) const {
    #ifdef YAKL_DEBUG
      if ( rank != 3 ) { throw "ERROR: Calling invalid function on rank 3 Array"; }
      this->check_index(0,i0,0,dimension[0]-1,__FILE__,__LINE__);
      this->check_index(1,i1,0,dimension[1]-1,__FILE__,__LINE__);
      this->check_index(2,i2,0,dimension[2]-1,__FILE__,__LINE__);
    #endif
    size_t ind = i0*offsets[0] + i1*offsets[1] + i2;
    return myData[ind];
  }
  YAKL_INLINE T &operator()(size_t const i0, size_t const i1, size_t const i2, size_t const i3) const {
    #ifdef YAKL_DEBUG
      if ( rank != 4 ) { throw "ERROR: Calling invalid function on rank 4 Array"; }
      this->check_index(0,i0,0,dimension[0]-1,__FILE__,__LINE__);
      this->check_index(1,i1,0,dimension[1]-1,__FILE__,__LINE__);
      this->check_index(2,i2,0,dimension[2]-1,__FILE__,__LINE__);
      this->check_index(3,i3,0,dimension[3]-1,__FILE__,__LINE__);
    #endif
    size_t ind = i0*offsets[0] + i1*offsets[1] + i2*offsets[2] + i3;
    return myData[ind];
  }
  YAKL_INLINE T &operator()(size_t const i0, size_t const i1, size_t const i2, size_t const i3, size_t const i4) const {
    #ifdef YAKL_DEBUG
      if ( rank != 5 ) { throw "ERROR: Calling invalid function on rank 5 Array"; }
      this->check_index(0,i0,0,dimension[0]-1,__FILE__,__LINE__);
      this->check_index(1,i1,0,dimension[1]-1,__FILE__,__LINE__);
      this->check_index(2,i2,0,dimension[2]-1,__FILE__,__LINE__);
      this->check_index(3,i3,0,dimension[3]-1,__FILE__,__LINE__);
      this->check_index(4,i4,0,dimension[4]-1,__FILE__,__LINE__);
    #endif
    size_t ind = i0*offsets[0] + i1*offsets[1] + i2*offsets[2] + i3*offsets[3] + i4;
    return myData[ind];
  }
  YAKL_INLINE T &operator()(size_t const i0, size_t const i1, size_t const i2, size_t const i3, size_t const i4, size_t const i5) const {
    #ifdef YAKL_DEBUG
      if ( rank != 6 ) { throw "ERROR: Calling invalid function on rank 6 Array"; }
      this->check_index(0,i0,0,dimension[0]-1,__FILE__,__LINE__);
      this->check_index(1,i1,0,dimension[1]-1,__FILE__,__LINE__);
      this->check_index(2,i2,0,dimension[2]-1,__FILE__,__LINE__);
      this->check_index(3,i3,0,dimension[3]-1,__FILE__,__LINE__);
      this->check_index(4,i4,0,dimension[4]-1,__FILE__,__LINE__);
      this->check_index(5,i5,0,dimension[5]-1,__FILE__,__LINE__);
    #endif
    size_t ind = i0*offsets[0] + i1*offsets[1] + i2*offsets[2] + i3*offsets[3] + i4*offsets[4] + i5;
    return myData[ind];
  }
  YAKL_INLINE T &operator()(size_t const i0, size_t const i1, size_t const i2, size_t const i3, size_t const i4, size_t const i5, size_t const i6) const {
    #ifdef YAKL_DEBUG
      if ( rank != 7 ) { throw "ERROR: Calling invalid function on rank 7 Array"; }
      this->check_index(0,i0,0,dimension[0]-1,__FILE__,__LINE__);
      this->check_index(1,i1,0,dimension[1]-1,__FILE__,__LINE__);
      this->check_index(2,i2,0,dimension[2]-1,__FILE__,__LINE__);
      this->check_index(3,i3,0,dimension[3]-1,__FILE__,__LINE__);
      this->check_index(4,i4,0,dimension[4]-1,__FILE__,__LINE__);
      this->check_index(5,i5,0,dimension[5]-1,__FILE__,__LINE__);
      this->check_index(6,i6,0,dimension[6]-1,__FILE__,__LINE__);
    #endif
    size_t ind = i0*offsets[0] + i1*offsets[1] + i2*offsets[2] + i3*offsets[3] + i4*offsets[4] + i5*offsets[5] + i6;
    return myData[ind];
  }
  YAKL_INLINE T &operator()(size_t const i0, size_t const i1, size_t const i2, size_t const i3, size_t const i4, size_t const i5, size_t const i6, size_t const i7) const {
    #ifdef YAKL_DEBUG
      if ( rank != 8 ) { throw "ERROR: Calling invalid function on rank 8 Array"; }
      this->check_index(0,i0,0,dimension[0]-1,__FILE__,__LINE__);
      this->check_index(1,i1,0,dimension[1]-1,__FILE__,__LINE__);
      this->check_index(2,i2,0,dimension[2]-1,__FILE__,__LINE__);
      this->check_index(3,i3,0,dimension[3]-1,__FILE__,__LINE__);
      this->check_index(4,i4,0,dimension[4]-1,__FILE__,__LINE__);
      this->check_index(5,i5,0,dimension[5]-1,__FILE__,__LINE__);
      this->check_index(6,i6,0,dimension[6]-1,__FILE__,__LINE__);
      this->check_index(7,i7,0,dimension[7]-1,__FILE__,__LINE__);
    #endif
    size_t ind = i0*offsets[0] + i1*offsets[1] + i2*offsets[2] + i3*offsets[3] + i4*offsets[4] + i5*offsets[5] + i6*offsets[6] + i7;
    return myData[ind];
  }


  inline void check_index(int const dim, long const ind, long const lb, long const ub, char const *file, int const line) const {
    if (ind < lb || ind > ub) {
      std::stringstream ss;
      #ifdef YAKL_DEBUG
        ss << "For Array labeled: " << myname << "\n";
      #endif
      ss << "Index " << dim << " of " << rank << " out of bounds\n";
      ss << "File, Line: " << file << ", " << line << "\n";
      ss << "Index: " << ind << ". Bounds: (" << lb << "," << ub << ")\n";
      throw std::out_of_range(ss.str());
    }
  }


  inline Array<T,rank,memHost,styleC> createHostCopy() const {
    Array<T,rank,memHost,styleC> ret;  // nullified + owned == true
    for (int i=0; i<rank; i++) {
      ret.offsets  [i] = offsets  [i];
      ret.dimension[i] = dimension[i];
    }
    #ifdef YAKL_DEBUG
      ret.myname = myname;
    #endif
    ret.allocate();
    if (myMem == memHost) {
      for (size_t i=0; i<totElems(); i++) { ret.myData[i] = myData[i]; }
    } else {
      #ifdef __USE_CUDA__
        cudaMemcpyAsync(ret.myData,myData,totElems()*sizeof(T),cudaMemcpyDeviceToHost,0);
        cudaDeviceSynchronize();
      #elif defined(__USE_HIP__)
        hipMemcpyAsync(ret.myData,myData,totElems()*sizeof(T),hipMemcpyDeviceToHost,0);
        hipDeviceSynchronize();
      #else
        for (size_t i=0; i<totElems(); i++) { ret.myData[i] = myData[i]; }
      #endif
    }
    return ret;
  }


  inline Array<T,rank,memDevice,styleC> createDeviceCopy() const {
    Array<T,rank,memDevice,styleC> ret;  // nullified + owned == true
    for (int i=0; i<rank; i++) {
      ret.offsets  [i] = offsets  [i];
      ret.dimension[i] = dimension[i];
    }
    #ifdef YAKL_DEBUG
      ret.myname = myname;
    #endif
    ret.allocate();
    if (myMem == memHost) {
      #ifdef __USE_CUDA__
        cudaMemcpyAsync(ret.myData,myData,totElems()*sizeof(T),cudaMemcpyHostToDevice,0);
        cudaDeviceSynchronize();
      #elif defined(__USE_HIP__)
        hipMemcpyAsync(ret.myData,myData,totElems()*sizeof(T),hipMemcpyHostToDevice,0);
        hipDeviceSynchronize();
      #else
        for (size_t i=0; i<totElems(); i++) { ret.myData[i] = myData[i]; }
      #endif
    } else {
      #ifdef __USE_CUDA__
        cudaMemcpyAsync(ret.myData,myData,totElems()*sizeof(T),cudaMemcpyDeviceToDevice,0);
        cudaDeviceSynchronize();
      #elif defined(__USE_HIP__)
        hipMemcpyAsync(ret.myData,myData,totElems()*sizeof(T),hipMemcpyDeviceToDevice,0);
        hipDeviceSynchronize();
      #else
        for (size_t i=0; i<totElems(); i++) { ret.myData[i] = myData[i]; }
      #endif
    }
    return ret;
  }


  inline void deep_copy_to(Array<T,rank,memHost,styleC> lhs) {
    if (myMem == memHost) {
      for (size_t i=0; i<totElems(); i++) { lhs.myData[i] = myData[i]; }
    } else {
      #ifdef __USE_CUDA__
        cudaMemcpyAsync(lhs.myData,myData,totElems()*sizeof(T),cudaMemcpyDeviceToHost,0);
      #elif defined(__USE_HIP__)
        hipMemcpyAsync(lhs.myData,myData,totElems()*sizeof(T),hipMemcpyDeviceToHost,0);
      #else
        for (size_t i=0; i<totElems(); i++) { lhs.myData[i] = myData[i]; }
      #endif
    }
  }


  inline void deep_copy_to(Array<T,rank,memDevice,styleC> lhs) {
    if (myMem == memHost) {
      #ifdef __USE_CUDA__
        cudaMemcpyAsync(lhs.myData,myData,totElems()*sizeof(T),cudaMemcpyHostToDevice,0);
      #elif defined(__USE_HIP__)
        hipMemcpyAsync(lhs.myData,myData,totElems()*sizeof(T),hipMemcpyHostToDevice,0);
      #else
        for (size_t i=0; i<totElems(); i++) { lhs.myData[i] = myData[i]; }
      #endif
    } else {
      #ifdef __USE_CUDA__
        cudaMemcpyAsync(lhs.myData,myData,totElems()*sizeof(T),cudaMemcpyDeviceToDevice,0);
      #elif defined(__USE_HIP__)
        hipMemcpyAsync(lhs.myData,myData,totElems()*sizeof(T),hipMemcpyDeviceToDevice,0);
      #else
        for (size_t i=0; i<totElems(); i++) { lhs.myData[i] = myData[i]; }
      #endif
    }
  }


  void setRandom() {
    Random rand;
    rand.fillArray(this->data(),this->totElems());
  }
  void setRandom(Random &rand) {
    rand.fillArray(this->data(),this->totElems());
  }


  /* ACCESSORS */
  YAKL_INLINE int get_rank() const {
    return rank;
  }
  YAKL_INLINE size_t get_totElems() const {
    size_t tot = dimension[0];
    for (int i=1; i<rank; i++) { tot *= dimension[i]; }
    return tot;
  }
  YAKL_INLINE size_t totElems() const {
    return get_totElems();
  }
  YAKL_INLINE size_t const *get_dimensions() const {
    return dimension;
  }
  YAKL_INLINE T *data() const {
    return myData;
  }
  YAKL_INLINE T *get_data() const {
    return myData;
  }
  YAKL_INLINE size_t extent( int const dim ) const {
    return dimension[dim];
  }
  YAKL_INLINE int extent_int( int const dim ) const {
    return (int) dimension[dim];
  }

  YAKL_INLINE int span_is_contiguous() const {
    return 1;
  }
  YAKL_INLINE int use_count() const {
    if (owned) {
      return *refCount;
    } else {
      return -1;
    }
  }
  YAKL_INLINE bool initialized() const {
    return myData != nullptr;
  }
  const char* label() const {
    #ifdef YAKL_DEBUG
      return myname.c_str();
    #else
      return "";
    #endif
  }


  /* INFORM */
  inline void print_data() const {
    #ifdef YAKL_DEBUG
      std::cout << "For Array labeled: " << myname << "\n";
    #endif
    if (rank == 1) {
      for (size_t i=0; i<dimension[0]; i++) {
        std::cout << std::setw(12) << (*this)(i) << "\n";
      }
    } else if (rank == 2) {
      for (size_t j=0; j<dimension[0]; j++) {
        for (size_t i=0; i<dimension[1]; i++) {
          std::cout << std::setw(12) << (*this)(i,j) << " ";
        }
        std::cout << "\n";
      }
    } else if (rank == 0) {
      std::cout << "Empty Array\n\n";
    } else {
      for (size_t i=0; i<totElems(); i++) {
        std::cout << std::setw(12) << myData[i] << " ";
      }
      std::cout << "\n";
    }
    std::cout << "\n";
  }


  /* OPERATOR<<
  Print the array. If it's 2-D, print a pretty looking matrix */
  inline friend std::ostream &operator<<(std::ostream& os, Array const &v) {
    #ifdef YAKL_DEBUG
      os << "For Array labeled: " << v.myname << "\n";
    #endif
    os << "Number of Dimensions: " << rank << "\n";
    os << "Total Number of Elements: " << v.totElems() << "\n";
    os << "Dimension Sizes: ";
    for (int i=0; i<rank; i++) {
      os << v.dimension[i] << ", ";
    }
    os << "\n";
    for (size_t i=0; i<v.totElems(); i++) {
      os << v.myData[i] << " ";
    }
    os << "\n";
    return os;
  }


  // This is stuff the user has no business messing with

  int *refCount; // Pointer shared by multiple copies of this Array to keep track of allcation / free

  // It would be dangerous for the user to call this directly rather than through the constructors, so we're "hiding" it :)
  inline void setup(char const * label, size_t d0, size_t d1=1, size_t d2=1, size_t d3=1, size_t d4=1, size_t d5=1, size_t d6=1, size_t d7=1) {
    #ifdef YAKL_DEBUG
      myname = std::string(label);
    #endif

    deallocate();

                     dimension[0] = d0;  
    if (rank >= 2) { dimension[1] = d1; }
    if (rank >= 3) { dimension[2] = d2; }
    if (rank >= 4) { dimension[3] = d3; }
    if (rank >= 5) { dimension[4] = d4; }
    if (rank >= 6) { dimension[5] = d5; }
    if (rank >= 7) { dimension[6] = d6; }
    if (rank >= 8) { dimension[7] = d7; }

    offsets[rank-1] = 1;
    for (int i=rank-2; i>=0; i--) {
      offsets[i] = offsets[i+1] * dimension[i+1];
    }
    allocate();
  }


  inline void allocate() {
    if (owned) {
      refCount = new int;
      *refCount = 1;
      if (myMem == memDevice) {
        myData = (T *) yaklAllocDevice( totElems()*sizeof(T) );
      } else {
        myData = (T *) yaklAllocHost  ( totElems()*sizeof(T) );
      }
    }
  }


  inline void deallocate() {
    if (owned) {
      if (refCount != nullptr) {
        (*refCount)--;

        if (*refCount == 0) {
          delete refCount;
          refCount = nullptr;
          if (myMem == memDevice) {
            yaklFreeDevice(myData);
          } else {
            yaklFreeHost  (myData);
          }
          myData = nullptr;
        }

      }
    }
  }


};

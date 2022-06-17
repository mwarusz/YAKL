
#pragma once

namespace yakl {

  class FunctorBuffer {
  protected:
    Gator               pool;
    std::vector<void *> pointer_list;

  public:

    FunctorBuffer() {}
    FunctorBuffer            (      FunctorBuffer && );
    FunctorBuffer &operator= (      FunctorBuffer && );
    FunctorBuffer            (const FunctorBuffer &  ) = delete;
    FunctorBuffer &operator= (const FunctorBuffer &  ) = delete;
    ~FunctorBuffer() { finalize(); }

    void init(std::function<void *( size_t )>       alloc   = [] (size_t bytes) -> void * { return ::malloc(bytes); },
              std::function<void( void * )>         dealloc = [] (void *ptr) { ::free(ptr); }                        ,
              std::function<void( void *, size_t )> zero    = [] (void *ptr, size_t bytes) {}  ) {
      size_t initialSize = 1024*1024*1;
      size_t growSize    = 1024*1024*1;
      size_t blockSize   = sizeof(size_t);
      pool.init( alloc , dealloc , zero , initialSize , growSize , blockSize , "FunctorBuffer pool" );
    }


    void finalize() { pool.finalize(); }


    void * alloc_functor( size_t bytes ) {
      void *ptr = pool.allocate( bytes , "FunctorBuffer" );
      pointer_list.push_back( ptr );
      return ptr;
    }


    void free_all_functors() {
      if (! pointer_list.empty()) {
        for (int i=pointer_list.size()-1; i >= 0; i--) {
          pool.free( pointer_list[i] , "FunctorBuffer" );
          pointer_list.pop_back();
        }
      }
    }

  };


  extern FunctorBuffer functor_buffer;

}



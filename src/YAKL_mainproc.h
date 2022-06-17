
#pragma once

namespace yakl {

  // Determine if this is the main process in the case of multiple MPI tasks
  // This is nearly always used just to avoid printing to stdout or stderr from all MPI tasks
  inline bool yakl_mainproc() {
    // Only actually check if the user says MPI is available. Otherwise, always return true
    #ifdef HAVE_MPI
      int is_initialized;
      MPI_Initialized(&is_initialized);
      if (!is_initialized) {
        return true;
      } else {
        int myrank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        return myrank == 0;
      }
    #else
      return true;
    #endif
  }

}



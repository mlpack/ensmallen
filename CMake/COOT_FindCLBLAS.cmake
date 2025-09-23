# - Find clBLAS (includes and library)
# This module defines
#  CLBLAS_INCLUDE_DIR
#  CLBLAS_LIBRARIES
#  CLBLAS_FOUND
# also defined, but not for general use are
#  CLBLAS_LIBRARY, where to find the library.

find_path(CLBLAS_INCLUDE_DIR clBLAS.h
/usr/include/
/usr/local/include/
)

set(CLBLAS_NAMES ${CLBLAS_NAMES} clBLAS)
find_library(CLBLAS_LIBRARY
  NAMES ${CLBLAS_NAMES}
  PATHS /usr/lib64/ /usr/local/lib64/ /usr/lib /usr/local/lib
  )

if (CLBLAS_LIBRARY AND CLBLAS_INCLUDE_DIR)
    set(CLBLAS_LIBRARIES ${CLBLAS_LIBRARY})
    set(CLBLAS_FOUND "YES")
else ()
  set(CLBLAS_FOUND "NO")
endif ()

if (CLBLAS_FOUND)
   if (NOT CLBLAS_FIND_QUIETLY)
      message(STATUS "Found a clBLAS library: ${CLBLAS_LIBRARIES}")
   endif ()
else ()
    if (CLBLAS_FIND_REQUIRED)
      message(FATAL_ERROR "Could not find a clBLAS library")
   endif ()
endif ()

# Deprecated declarations.
set (NATIVE_CLBLAS_INCLUDE_PATH ${CLBLAS_INCLUDE_DIR} )
get_filename_component (NATIVE_CLBLAS_LIB_PATH ${CLBLAS_LIBRARY} PATH)

mark_as_advanced(
  CLBLAS_LIBRARY
  CLBLAS_INCLUDE_DIR
  )

# - Find clBlast (includes and library)
# This module defines
#  CLBLAST_INCLUDE_DIR
#  CLBLAST_LIBRARIES
#  CLBLAST_FOUND
# also defined, but not for general use are
#  CLBLAST_LIBRARY, where to find the library.

find_path(CLBLAST_INCLUDE_DIR clblast.h
/usr/include/
/usr/local/include/
)

set(CLBLAST_NAMES ${CLBLAST_NAMES} clblast)
find_library(CLBLAST_LIBRARY
  NAMES ${CLBLAST_NAMES}
  PATHS /usr/lib64/ /usr/local/lib64/ /usr/lib /usr/local/lib
  )

if (CLBLAST_LIBRARY AND CLBLAST_INCLUDE_DIR)
    set(CLBLAST_LIBRARIES ${CLBLAST_LIBRARY})
    set(CLBLAST_FOUND "YES")
else ()
  set(CLBLAST_FOUND "NO")
endif ()

if (CLBLAST_FOUND)
   if (NOT CLBLAST_FIND_QUIETLY)
      message(STATUS "Found a clBlast library: ${CLBLAST_LIBRARIES}")
   endif ()
else ()
    if (CLBLAST_FIND_REQUIRED)
      message(FATAL_ERROR "Could not find a clBlast library")
   endif ()
endif ()

# Deprecated declarations.
set (NATIVE_CLBLAST_INCLUDE_PATH ${CLBLAST_INCLUDE_DIR} )
get_filename_component (NATIVE_CLBLAST_LIB_PATH ${CLBLAST_LIBRARY} PATH)

mark_as_advanced(
  CLBLAST_LIBRARY
  CLBLAST_INCLUDE_DIR
  )

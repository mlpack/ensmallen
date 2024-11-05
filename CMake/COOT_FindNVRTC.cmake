# - Find clBlast (includes and library)
# This module defines
#  CLBLAST_INCLUDE_DIR
#  CLBLAST_LIBRARIES
#  CLBLAST_FOUND
# also defined, but not for general use are
#  CLBLAST_LIBRARY, where to find the library.

set(NVRTC_NAMES ${NVRTC_NAMES} nvrtc)
find_library(NVRTC_LIBRARY
  NAMES ${NVRTC_NAMES}
  PATHS /usr/lib64/ /usr/local/lib64/ /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu/
  )

if (NVRTC_LIBRARY)
  set(NVRTC_LIBRARIES ${NVRTC_LIBRARY})
  set(NVRTC_FOUND "YES")
else ()
  set(NVRTC_FOUND "NO")
endif ()

if (NVRTC_FOUND)
  if (NOT NVRTC_FIND_QUIETLY)
    message(STATUS "Found NVRTC library: ${NVRTC_LIBRARIES}")
   endif ()
else ()
  if (NVRTC_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find NVRTC library")
   endif ()
endif ()

# Deprecated declarations.
get_filename_component (NATIVE_NVRTC_LIB_PATH ${NVRTC_LIBRARY} PATH)

mark_as_advanced(NVRTC_LIBRARY)

# - Find Bandicoot
# Find Bandicoot: GPU accelerator add-on for the Armadillo C++ linear algebra
# library
#
# Using Bandicoot:
#  find_package(Bandicoot REQUIRED)
#  include_directories(${BANDICOOT_INCLUDE_DIRS})
#  add_executable(foo foo.cc)
#  target_link_libraries(foo ${BANDICOOT_LIBRARIES})
# This module sets the following variables:
#  BANDICOOT_FOUND - set to true if the library is found
#  BANDICOOT_INCLUDE_DIRS - list of required include directories
#  BANDICOOT_LIBRARIES - list of libraries to be linked
#  BANDICOOT_VERSION_MAJOR - major version number
#  BANDICOOT_VERSION_MINOR - minor version number
#  BANDICOOT_VERSION_PATCH - patch version number
#  BANDICOOT_VERSION_STRING - version number as a string (ex: "1.0.4")
#  BANDICOOT_VERSION_NOTE - name of the version (ex: "unstable development version")

find_path(BANDICOOT_INCLUDE_DIR
  NAMES bandicoot
  PATHS "$ENV{ProgramFiles}/Bandicoot/include"
  )

if(BANDICOOT_INCLUDE_DIR)
  # Extract version information.
  file(READ "${BANDICOOT_INCLUDE_DIR}/bandicoot_bits/coot_version.hpp" _bandicoot_HEADER_CONTENTS)
  string(REGEX REPLACE ".*#define COOT_VERSION_MAJOR ([0-9]+).*" "\\1" BANDICOOT_VERSION_MAJOR "${_bandicoot_HEADER_CONTENTS}")
  string(REGEX REPLACE ".*#define COOT_VERSION_MINOR ([0-9]+).*" "\\1" BANDICOOT_VERSION_MINOR "${_bandicoot_HEADER_CONTENTS}")
  string(REGEX REPLACE ".*#define COOT_VERSION_PATCH ([0-9]+).*" "\\1" BANDICOOT_VERSION_PATCH "${_bandicoot_HEADER_CONTENTS}")
  string(REGEX REPLACE ".*#define COOT_VERSION_NOTE\ +\"([0-9a-zA-Z\ _-]+)\".*" "\\1" BANDICOOT_VERSION_NOTE "${_bandicoot_HEADER_CONTENTS}")

  set(BANDICOOT_VERSION_STRING "${BANDICOOT_VERSION_MAJOR}.${BANDICOOT_VERSION_MINOR}.${BANDICOOT_VERSION_PATCH}")
endif ()

# Determine what support libraries are being used, and whether or not we need to
# link against them.  We need to look in config.hpp.
set(SUPPORT_INCLUDE_DIRS "")
set(SUPPORT_LIBRARIES "")
set(COOT_NEED_LIBRARY true) # Assume true.
if(EXISTS "${BANDICOOT_INCLUDE_DIR}/bandicoot_bits/config.hpp")
  file(READ "${BANDICOOT_INCLUDE_DIR}/bandicoot_bits/config.hpp" _bandicoot_CONFIG_CONTENTS)
  # COOT_USE_WRAPPER
  string(REGEX MATCH "\r?\n[\t ]*#define[ \t]+COOT_USE_WRAPPER[ \t]*\r?\n" COOT_USE_WRAPPER "${_bandicoot_CONFIG_CONTENTS}")

  # COOT_USE_OPENCL
  string(REGEX MATCH "\r?\n[\t ]*#if[\t ]+!defined[(]COOT_USE_OPENCL[)][\t ]*\r?\n[\t
  ]*#define[ \t]+COOT_USE_OPENCL[ \t]*\r?\n" COOT_USE_OPENCL "${_bandicoot_CONFIG_CONTENTS}")

  # COOT_USE_CUDA
  string(REGEX MATCH "\r?\n[\t ]*#if[\t ]+!defined[(]COOT_USE_CUDA[)][\t ]*\r?\n[\t
  ]*#define[ \t]+COOT_USE_CUDA[ \t]*\r?\n" COOT_USE_CUDA "${_bandicoot_CONFIG_CONTENTS}")

  # COOT_USE_LAPACK
  string(REGEX MATCH "\r?\n[\t ]*#if[\t ]+!defined[(]COOT_USE_LAPACK[)][\t ]*\r?\n[\t ]*#define[ \t]+COOT_USE_LAPACK[ \t]*\r?\n" COOT_USE_LAPACK "${_bandicoot_CONFIG_CONTENTS}")

  # COOT_USE_BLAS
  string(REGEX MATCH "\r?\n[\t ]*#if[\t ]+!defined[(]COOT_USE_BLAS[)][\t ]*\r?\n[\t ]*#define[ \t]+COOT_USE_BLAS[ \t]*\r?\n" COOT_USE_BLAS "${_bandicoot_CONFIG_CONTENTS}")

  # If we aren't wrapping, things get a little more complex.
  if(NOT COOT_USE_WRAPPER)
    set(COOT_NEED_LIBRARY false)
    message(STATUS "COOT_USE_WRAPPER is not defined, so all dependencies of "
                   "Bandicoot must be manually linked.")

    set(HAVE_OPENCL false)
    set(HAVE_CUDA   false)
    set(HAVE_LAPACK false)
    set(HAVE_BLAS   false)

    # Search for OpenCL.
    if (NOT "${COOT_USE_OPENCL}" STREQUAL "" AND NOT HAVE_OPENCL)
      set(OpenCL_FIND_QUIETLY true)
      include(FindOpenCL)

      if (OpenCL_FOUND)
        message(STATUS "OpenCL includes: ${OpenCL_INCLUDE_DIRS}")
        message(STATUS "OpenCL libraries: ${OpenCL_LIBRARIES}")

        set(SUPPORT_INCLUDE_DIRS "${SUPPORT_INCLUDE_DIRS}"
            "${OpenCL_INCLUDE_DIRS}")
        set(SUPPORT_LIBRARIES "${SUPPORT_LIBRARIES}" "${OpenCL_LIBRARIES}")
        set(HAVE_OPENCL true)
      endif ()

      # Search for clBLAS.
      set(CLBLAS_FIND_QUIETLY true)
      include(COOT_FindCLBLAS)

      if (CLBLAS_FOUND)
        message(STATUS "clBLAS includes: ${CLBLAS_INCLUDE_DIR}")
        message(STATUS "clBLAS libraries: ${CLBLAS_LIBRARIES}")

        set(SUPPORT_INCLUDE_DIRS "${SUPPORT_INCLUDE_DIRS}"
            "${CLBLAS_INCLUDE_DIR}")
        set(SUPPORT_LIBRARIES "${SUPPORT_LIBRARIES}" "${CLBLAS_LIBRARIES}")
        set(HAVE_CLBLAS true)
      endif ()

      # Search for clBlast.
      set(CLBLAST_FIND_QUIETLY true)
      include(COOT_FindCLBlast)

      if (CLBLAST_FOUND)
        message(STATUS "clBlast includes: ${CLBLAST_INCLUDE_DIR}")
        message(STATUS "clBlast libraries: ${CLBLAST_LIBRARIES}")

        set(SUPPORT_INCLUDE_DIRS "${SUPPORT_INCLUDE_DIRS}"
            "${CLBLAST_INCLUDE_DIR}")
        set(SUPPORT_LIBRARIES "${SUPPORT_LIBRARIES}" "${CLBLAST_LIBRARIES}")
        set(HAVE_CLBLAST true)
      endif ()
    endif ()

    # Search for CUDA.
    if (NOT "${COOT_USE_CUDA}" STREQUAL "" AND NOT HAVE_CUDA)
      # FindCUDA is deprecated since version 3.10 and replaced with
      # FindCUDAToolkit wich was added in CMake 3.17.
      message(STATUS "${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}")
      if ("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}" LESS "3.67")
        set(CUDA_FIND_QUIETLY true)
        find_package(CUDA)

        if (CUDA_FOUND)
	  message(STATUS "CUDA includes: ${CUDA_INCLUDE_DIRS}")
	  message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")

          # We also need NVRTC and also libcuda itself, which the old FindCUDA package do not find.
          find_library(CUDA_cuda_LIBRARY cuda
              HINTS ${CUDA_TOOLKIT_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}/lib ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
          find_library(CUDA_nvrtc_LIBRARY nvrtc
              HINTS ${CUDA_TOOLKIT_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}/lib ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

          include(COOT_FindNVRTC)

          if (NVRTC_FOUND)
            message(STATUS "NVRTC libraries: ${NVRTC_LIBRARIES}")
            set(SUPPORT_LIBRARIES "${SUPPORT_LIBRARIES}" "${NVRTC_LIBRARIES}")
          endif ()

          set(SUPPORT_INCLUDE_DIRS "${SUPPORT_INCLUDE_DIRS}"
              "${CUDA_INCLUDE_DIRS}")
          set(SUPPORT_LIBRARIES "${SUPPORT_LIBRARIES}"
              "${CUDA_LIBRARIES}"
              "${CUDA_nvrtc_LIBRARY}"
              "${CUDA_CUDA_LIBRARY}"
              "${CUDA_CUBLAS_LIBRARIES}"
              "${CUDA_curand_LIBRARY}"
              "${CUDA_cusolver_LIBRARY}")
          set(CUDA_INCLUDE_DIRS "")
          set(HAVE_CUDA true)

        endif ()
      else ()
        set(CUDA_TOOLKIT_FIND_QUIETLY true)
        find_package(CUDAToolkit REQUIRED)

        if (CUDAToolkit_FOUND)
	  message(STATUS "CUDA includes: ${CUDAToolkit_INCLUDE_DIRS}")
	  message(STATUS "CUDA libraries: ${CUDAToolkit_LIBRARY_DIR}")

	  set(CUDA_LIBRARIES CUDA::cudart CUDA::cuda_driver)
          set(CUDA_CUBLAS_LIBRARIES CUDA::cublas)
          set(CUDA_curand_LIBRARY CUDA::curand)
          set(CUDA_cusolver_LIBRARY CUDA::cusolver)
          set(CUDA_nvrtc_LIBRARY CUDA::nvrtc)

          set(SUPPORT_INCLUDE_DIRS "${SUPPORT_INCLUDE_DIRS}"
              "${CUDAToolkit_INCLUDE_DIRS}")
          set(SUPPORT_LIBRARIES "${SUPPORT_LIBRARIES}"
              CUDA_LIBRARIES
	      CUDA_CUBLAS_LIBRARIES
	      CUDA_curand_LIBRARY
	      CUDA_cusolver_LIBRARY
	      CUDA_nvrtc_LIBRARY)
          set(HAVE_CUDA true)
        endif()
      endif ()
    endif ()

    # Search for LAPACK/BLAS (or replacement).
    if ((NOT "${COOT_USE_LAPACK}" STREQUAL "") AND
      (NOT "${COOT_USE_BLAS}" STREQUAL ""))
      # In order of preference: MKL, ACML, OpenBLAS, ATLAS
      set(MKL_FIND_QUIETLY true)
      include(ARMA_FindMKL)
      set(ACMLMP_FIND_QUIETLY true)
      include(ARMA_FindACMLMP)
      set(ACML_FIND_QUIETLY true)
      include(ARMA_FindACML)

      if (MKL_FOUND)
        message(STATUS "Using MKL for LAPACK/BLAS: ${MKL_LIBRARIES}")

        set(SUPPORT_LIBRARIES "${SUPPORT_LIBRARIES}" "${MKL_LIBRARIES}")
        set(HAVE_LAPACK true)
        set(HAVE_BLAS   true)
      elseif (ACMLMP_FOUND)
        message(STATUS "Using multi-core ACML libraries for LAPACK/BLAS:
            ${ACMLMP_LIBRARIES}")

        set(SUPPORT_LIBRARIES "${SUPPORT_LIBRARIES}" "${ACMLMP_LIBRARIES}")
        set(HAVE_LAPACK true)
        set(HAVE_BLAS   true)
      elseif (ACML_FOUND)
        message(STATUS "Using ACML for LAPACK/BLAS: ${ACML_LIBRARIES}")

        set(SUPPORT_LIBRARIES "${SUPPORT_LIBRARIES}" "${ACML_LIBRARIES}")
        set(HAVE_LAPACK true)
        set(HAVE_BLAS   true)
      endif ()
    endif ()

    # If we haven't found BLAS, try.
    if (NOT "${COOT_USE_BLAS}" STREQUAL "" AND NOT HAVE_BLAS)
      # Search for BLAS.
      set(OpenBLAS_FIND_QUIETLY false)
      include(ARMA_FindOpenBLAS)
      set(CBLAS_FIND_QUIETLY true)
      include(ARMA_FindCBLAS)
      set(BLAS_FIND_QUIETLY true)
      include(ARMA_FindBLAS)

      if (OpenBLAS_FOUND)
        # Warn if ATLAS is found also.
        if (CBLAS_FOUND)
          message(STATUS "Warning: both OpenBLAS and ATLAS have been found; "
              "ATLAS will not be used.")
        endif ()
        message(STATUS "Using OpenBLAS for BLAS: ${OpenBLAS_LIBRARIES}")

        set(SUPPORT_LIBRARIES "${SUPPORT_LIBRARIES}" "${OpenBLAS_LIBRARIES}")
        set(HAVE_BLAS true)
      elseif (CBLAS_FOUND)
        message(STATUS "Using ATLAS for BLAS: ${CBLAS_LIBRARIES}")

        set(SUPPORT_LIBRARIES "${SUPPORT_LIBRARIES}" "${CBLAS_LIBRARIES}")
        set(SUPPORT_INCLUDE_DIRS "${SUPPORT_INCLUDE_DIRS}"
            "${CBLAS_INCLUDE_DIR}")
        set(HAVE_BLAS true)
      elseif (BLAS_FOUND)
        message(STATUS "Using standard BLAS: ${BLAS_LIBRARIES}")

        set(SUPPORT_LIBRARIES "${SUPPORT_LIBRARIES}" "${BLAS_LIBRARIES}")
        set(HAVE_BLAS true)
      endif ()
    endif ()

    # If we haven't found LAPACK, try.
    if (NOT "${COOT_USE_LAPACK}" STREQUAL "" AND NOT HAVE_LAPACK)
      # Search for LAPACK.
      set(CLAPACK_FIND_QUIETLY true)
      include(ARMA_FindCLAPACK)
      set(LAPACK_FIND_QUIETLY true)
      include(ARMA_FindLAPACK)

      # Only use ATLAS if OpenBLAS isn't being used.
      if (CLAPACK_FOUND AND NOT OpenBLAS_FOUND)
        message(STATUS "Using ATLAS for LAPACK: ${CLAPACK_LIBRARIES}")

        set(SUPPORT_LIBRARIES "${SUPPORT_LIBRARIES}" "${CLAPACK_LIBRARIES}")
        set(SUPPORT_INCLUDE_DIRS "${SUPPORT_INCLUDE_DIRS}"
            "${CLAPACK_INCLUDE_DIR}")
        set(HAVE_LAPACK true)
      elseif (LAPACK_FOUND)
        message(STATUS "Using standard LAPACK: ${LAPACK_LIBRARIES}")

        set(SUPPORT_LIBRARIES "${SUPPORT_LIBRARIES}" "${LAPACK_LIBRARIES}")
        set(HAVE_LAPACK true)
      endif ()
    endif ()

    if (NOT "${COOT_USE_LAPACK}" STREQUAL "" AND NOT HAVE_LAPACK)
      message(FATAL_ERROR "Cannot find LAPACK library, but COOT_USE_LAPACK is "
          "set. Try specifying LAPACK libraries manually by setting the "
          "LAPACK_LIBRARY variable.")
    endif ()

    if (NOT "${COOT_USE_BLAS}" STREQUAL "" AND NOT HAVE_BLAS)
      message(FATAL_ERROR "Cannot find BLAS library, but COOT_USE_BLAS is set. "
          "Try specifying BLAS libraries manually by setting the BLAS_LIBRARY "
          "variable.")
    endif ()

  endif()
else()
  message(STATUS "${BANDICOOT_INCLUDE_DIR}/bandicoot_bits/config.hpp not "
      "found!  Cannot determine what to link against.")
endif()

if (COOT_NEED_LIBRARY)
  # UNIX paths are standard, no need to write.
  find_library(BANDICOOT_LIBRARY
    NAMES bandicoot
    PATHS "$ENV{ProgramFiles}/Bandicoot/lib"  "$ENV{ProgramFiles}/Bandicoot/lib64" "$ENV{ProgramFiles}/Bandicoot"
    )

  # Checks 'REQUIRED', 'QUIET' and versions.
  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(Bandicoot
    REQUIRED_VARS BANDICOOT_LIBRARY BANDICOOT_INCLUDE_DIR
    VERSION_VAR BANDICOOT_VERSION_STRING)
else ()
  # Checks 'REQUIRED', 'QUIET' and versions.
  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(Bandicoot
    REQUIRED_VARS BANDICOOT_INCLUDE_DIR
    VERSION_VAR BANDICOOT_VERSION_STRING)
endif ()

if (BANDICOOT_FOUND)
  # Also include support include directories.
  set(BANDICOOT_INCLUDE_DIRS ${BANDICOOT_INCLUDE_DIR} ${SUPPORT_INCLUDE_DIRS})
  # Also include support libraries to link against.
  if (COOT_NEED_LIBRARY)
    set(BANDICOOT_LIBRARIES ${BANDICOOT_LIBRARY} ${SUPPORT_LIBRARIES})
  else ()
    set(BANDICOOT_LIBRARIES ${SUPPORT_LIBRARIES})
  endif ()
  message(STATUS "Bandicoot libraries: ${BANDICOOT_LIBRARIES}")
  message(STATUS "Bandicoot includes: ${BANDICOOT_INCLUDE_DIR}")
endif ()

# Hide internal variables
mark_as_advanced(
  BANDICOOT_INCLUDE_DIR
  BANDICOOT_LIBRARIES)

if (BANDICOOT_FOUND AND NOT TARGET Bandicoot:Bandicoot)
  add_library(Bandicoot::Bandicoot INTERFACE IMPORTED)
  set_target_properties(Bandicoot::Bandicoot PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${BANDICOOT_INCLUDE_DIR}"
      INTERFACE_LINK_LIBRARIES "${BANDICOOT_LIBRARIES}")
endif()

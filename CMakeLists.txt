# ensmallen CMake configuration.  This project has no configurable options---it
# just installs the headers to the install location, and optionally builds the
# test program.
cmake_minimum_required(VERSION 2.8.10)
project(ensmallen C CXX)

option(USE_OPENMP "If available, use OpenMP for parallelization." ON)
option(SHOW_COMPILE_WARNINGS "Verify compilation is free from warnings." OFF) 

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMake")

# Ensure we have C++11 features.  Since we support CMake < 3.1, this needs a
# little bit of special handling.
if ((${CMAKE_MAJOR_VERSION} LESS 3 OR
    (${CMAKE_MAJOR_VERSION} EQUAL 3 AND ${CMAKE_MINOR_VERSION} LESS 1))
    AND NOT FORCE_CXX11)
  # Older versions of CMake do not support target_compile_features(), so we have
  # to use something kind of hacky.
  include(CMake/CXX11.cmake)
  check_for_cxx11_compiler(HAS_CXX11)
  if(NOT HAS_CXX11)
    message(FATAL_ERROR "No C++11 compiler available!")
  endif()
  enable_cxx11()
else()
  # set required standard to c++11
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif ()

# Detect OpenMP support in a compiler. If the compiler supports OpenMP, flags
# to compile with OpenMP are returned and added for compilation.
if (USE_OPENMP)
  find_package(OpenMP)
endif ()

if (OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else ()
  # Disable warnings for all the unknown OpenMP pragmas.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
endif ()

# Verify the code conforms to ISO standards regarding C++ as
# indicated in the GCC options
# https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html
# Moreover, treat any compilation warning as an error.
if ( SHOW_COMPILE_WARNINGS )
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wpedantic -Wextra -Werror")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wpedantic -Wextra -Werror")
endif ()

# The only dependency we need is Armadillo.
#
# We keep the minimum version in sync with mlpack, otherwise we could have
# irritating compatibility issues.
find_package(Armadillo 6.500.0 REQUIRED)
include_directories(BEFORE "${ARMADILLO_INCLUDE_DIR}")
include_directories(BEFORE "${CMAKE_SOURCE_DIR}/include/")

# Install the headers to the correct location.
install(DIRECTORY "${CMAKE_SOURCE_DIR}/include/ensmallen_bits"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/include"
        PATTERN "*~" EXCLUDE
        PATTERN "*.sw*" EXCLUDE)
install(FILES ${CMAKE_SOURCE_DIR}/include/ensmallen.hpp
        DESTINATION "${CMAKE_INSTALL_PREFIX}/include")

enable_testing()

add_subdirectory(tests)

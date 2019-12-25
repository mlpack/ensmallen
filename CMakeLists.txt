# ensmallen CMake configuration.  This project has no configurable options---it
# just installs the headers to the install location, and optionally builds the
# test program.
cmake_minimum_required(VERSION 3.3.2)
project(ensmallen 
        VERSION 2.10.5
        LANGUAGES C CXX)

# Configurable options for CMake
option(USE_OPENMP "If available, use OpenMP for parallelization." ON)
option(BUILD_TESTS "Build tests." ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMake")

# Set required C++ standard to C++11 
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create library target
add_library(ensmallen INTERFACE)
target_include_directories(ensmallen INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# Find OpenMP and link it.
if(USE_OPENMP)
  if(NOT TARGET OpenMP::OpenMP_CXX)
    find_package(Threads REQUIRED)
    add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
    set_property(TARGET OpenMP::OpenMP_CXX
        PROPERTY INTERFACE_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})
    # Only works if the same flag is passed to the linker; use CMake 3.9+ otherwise (Intel, AppleClang)
    set_property(TARGET OpenMP::OpenMP_CXX
        PROPERTY INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_FLAGS} Threads::Threads)
  endif()
  target_link_libraries(ensmallen INTERFACE OpenMP::OpenMP_CXX)
endif()

# Find Armadillo and link it.
find_package(Armadillo 8.400.0 REQUIRED)
target_link_libraries(ensmallen INTERFACE Armadillo::Armadillo)

# Install the headers to the correct location.
install(DIRECTORY "${CMAKE_SOURCE_DIR}/include/ensmallen_bits"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/include"
        PATTERN "*~" EXCLUDE
        PATTERN "*.sw*" EXCLUDE)
install(FILES ${CMAKE_SOURCE_DIR}/include/ensmallen.hpp
        DESTINATION "${CMAKE_INSTALL_PREFIX}/include")

enable_testing()

if (BUILD_TESTS)
  add_subdirectory(tests)
endif()

# ensmallen CMake configuration.  This project has no configurable options---it
# just installs the headers to the install location, and optionally builds the
# test program.
cmake_minimum_required(VERSION 3.3.2)
project(ensmallen
        LANGUAGES C CXX)

# Configurable options for CMake.
option(USE_OPENMP "If available, use OpenMP for parallelization." ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMake")

# Set required C++ standard to C++11.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Extract version from sources.
set(ENSMALLEN_VERSION_FILE_NAME "${PROJECT_SOURCE_DIR}/include/ensmallen_bits/ens_version.hpp")

if(NOT EXISTS ${ENSMALLEN_VERSION_FILE_NAME})
  message(FATAL_ERROR "Can't read ${ENSMALLEN_VERSION_FILE_NAME}")
endif()

file(READ ${ENSMALLEN_VERSION_FILE_NAME} ENSMALLEN_VERSION_FILE_CONTENTS)
string(REGEX REPLACE ".*#define ENS_VERSION_MAJOR ([0-9]+).*" "\\1" ENSMALLEN_VERSION_MAJOR "${ENSMALLEN_VERSION_FILE_CONTENTS}")
string(REGEX REPLACE ".*#define ENS_VERSION_MINOR ([0-9]+).*" "\\1" ENSMALLEN_VERSION_MINOR "${ENSMALLEN_VERSION_FILE_CONTENTS}")
string(REGEX REPLACE ".*#define ENS_VERSION_PATCH ([0-9]+).*" "\\1" ENSMALLEN_VERSION_PATCH "${ENSMALLEN_VERSION_FILE_CONTENTS}")

message(STATUS "Configuring ensmallen ${ENSMALLEN_VERSION_MAJOR}.${ENSMALLEN_VERSION_MINOR}.${ENSMALLEN_VERSION_PATCH}")
set(VERSION "${ENSMALLEN_VERSION_MAJOR}.${ENSMALLEN_VERSION_MINOR}.${ENSMALLEN_VERSION_PATCH}")

# Create library target.
add_library(ensmallen INTERFACE)
target_include_directories(ensmallen INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# Set warning flags for target.
if(MSVC)
  target_compile_options(ensmallen INTERFACE $<BUILD_INTERFACE:/Wall>)
else()
  target_compile_options(ensmallen INTERFACE $<BUILD_INTERFACE:-Wall -Wpedantic -Wunused-parameter>)
endif()

# Find OpenMP and link it.
if(USE_OPENMP)
  if(NOT TARGET OpenMP::OpenMP_CXX)
    find_package(Threads REQUIRED)
    add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
    set_property(TARGET OpenMP::OpenMP_CXX
        PROPERTY INTERFACE_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})
    # Only works if the same flag is passed to the linker; use CMake 3.9+ otherwise (Intel, AppleClang).
    set_property(TARGET OpenMP::OpenMP_CXX
        PROPERTY INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_FLAGS} Threads::Threads)
  endif()
  target_link_libraries(ensmallen INTERFACE OpenMP::OpenMP_CXX)
endif()

# Find Armadillo and link it.
find_package(Armadillo 9.800.0 REQUIRED)
target_link_libraries(ensmallen INTERFACE Armadillo::Armadillo)

# Set helper variables for creating the version, config and target files.
include(CMakePackageConfigHelpers)
set(ENSMALLEN_CMAKE_DIR "lib/cmake/ensmallen" CACHE STRING
      "Installation directory for cmake files, relative to ${CMAKE_INSTALL_PREFIX}.")
set(VERSION_CONFIG "${PROJECT_BINARY_DIR}/ensmallen-config-version.cmake")
set(PROJECT_CONFIG "${PROJECT_BINARY_DIR}/ensmallen-config.cmake")
set(TARGETS_EXPORT_NAME ensmallen-targets)

# Generate the version, config and target files into the build directory.
write_basic_package_version_file(${VERSION_CONFIG}
    VERSION ${VERSION}
    COMPATIBILITY AnyNewerVersion)
configure_package_config_file(${PROJECT_SOURCE_DIR}/CMake/ensmallen-config.cmake.in
    ${PROJECT_CONFIG}
    INSTALL_DESTINATION ${ENSMALLEN_CMAKE_DIR})
export(TARGETS ensmallen NAMESPACE ensmallen::
    FILE ${PROJECT_BINARY_DIR}/${TARGETS_EXPORT_NAME}.cmake)

# Install version, config and target files.
install(FILES ${PROJECT_CONFIG} ${VERSION_CONFIG}
    DESTINATION ${ENSMALLEN_CMAKE_DIR})
install(EXPORT ${TARGETS_EXPORT_NAME} DESTINATION ${ENSMALLEN_CMAKE_DIR}
    NAMESPACE ensmallen::)

# Export the targets and install the header files.
install(TARGETS ensmallen EXPORT ${TARGETS_EXPORT_NAME} DESTINATION lib)
install(DIRECTORY "${CMAKE_SOURCE_DIR}/include/ensmallen_bits"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/include"
        PATTERN "*~" EXCLUDE
        PATTERN "*.sw*" EXCLUDE)
install(FILES ${CMAKE_SOURCE_DIR}/include/ensmallen.hpp
        DESTINATION "${CMAKE_INSTALL_PREFIX}/include")

# Enable testing and build tests.
enable_testing()
add_subdirectory(tests)

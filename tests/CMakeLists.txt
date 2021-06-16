# The tests that need to be compiled.
set(ENSMALLEN_TESTS_SOURCES
    main.cpp
    moead_test.cpp
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
add_executable(ensmallen_tests ${ENSMALLEN_TESTS_SOURCES})
target_link_libraries(ensmallen_tests PRIVATE ensmallen)

# Copy test data into place.
add_custom_command(TARGET ensmallen_tests
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/data/
      ${CMAKE_BINARY_DIR}/data/
)

enable_testing()
add_test(NAME ensmallen_tests COMMAND ensmallen_tests
WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
project(ensmallen_tests CXX)

# mlpack test cases.
set(ENSMALLEN_TESTS_SOURCES
    main.cpp
    rmsprop_test.cpp
)

add_executable(${PROJECT_NAME} ${ENSMALLEN_TESTS_SOURCES})

enable_testing()
add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
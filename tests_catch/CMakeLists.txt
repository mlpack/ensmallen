project(ensmallen_tests CXX)

# mlpack test cases.
set(ENSMALLEN_TESTS_SOURCES
    main.cpp
    sgd_test.cpp
)

add_executable(${PROJECT_NAME} ${ENSMALLEN_TESTS_SOURCES})

target_link_libraries(${PROJECT_NAME} ${ARMADILLO_LIBRARIES})

enable_testing()
add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
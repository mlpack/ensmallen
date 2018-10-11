project(ensmallen_tests CXX)

set(ENSMALLEN_TESTS_SOURCES
    main.cpp
    ada_delta_test.cpp
    ada_grad_test.cpp
    adam_test.cpp
    aug_lagrangian_test.cpp
    bigbatch_sgd_test.cpp
    cmaes_test.cpp
    cne_test.cpp
    function_test.cpp
    gradient_descent_test.cpp
    katyusha_test.cpp
    lbfgs_test.cpp
    line_search_test.cpp
    lrsdp_test.cpp
    momentum_sgd_test.cpp
    nesterov_momentum_sgd_test.cpp
    # parallel_sgd_test.cpp
    proximal_test.cpp
    rmsprop_test.cpp
    sa_test.cpp
    sarah_test.cpp
    # scd_test.cpp
    sdp_primal_dual_test.cpp
    sgdr_test.cpp
    sgd_test.cpp
    smorms3_test.cpp
    snapshot_ensembles.cpp
    spalera_sgd_test.cpp
    svrg_test.cpp
)

add_executable(${PROJECT_NAME} ${ENSMALLEN_TESTS_SOURCES})

target_link_libraries(${PROJECT_NAME} ${ARMADILLO_LIBRARIES})

enable_testing()
add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})

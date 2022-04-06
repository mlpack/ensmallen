# The tests that need to be compiled.
set(ENSMALLEN_TESTS_SOURCES
    main.cpp
    ada_belief_test.cpp
    ada_bound_test.cpp
    ada_delta_test.cpp
    ada_grad_test.cpp
    ada_sqrt_test.cpp
    adam_test.cpp
    aug_lagrangian_test.cpp
    bigbatch_sgd_test.cpp
    callbacks_test.cpp
    cmaes_test.cpp
    cne_test.cpp
    de_test.cpp
    demon_adam_test.cpp
    demon_sgd_test.cpp
    eve_test.cpp
    frankwolfe_test.cpp
    ftml_test.cpp
    function_test.cpp
    gradient_descent_test.cpp
    grid_search_test.cpp
    iqn_test.cpp
    katyusha_test.cpp
    lbfgs_test.cpp
    line_search_test.cpp
    lookahead_test.cpp
    lrsdp_test.cpp
    moead_test.cpp
    momentum_sgd_test.cpp
    nesterov_momentum_sgd_test.cpp
    nsga2_test.cpp
    parallel_sgd_test.cpp
    proximal_test.cpp
    pso_test.cpp
    quasi_hyperbolic_momentum_sgd_test.cpp
    rmsprop_test.cpp
    sa_test.cpp
    sarah_test.cpp
    scd_test.cpp
    sdp_primal_dual_test.cpp
    sgdr_test.cpp
    sgd_test.cpp
    smorms3_test.cpp
    snapshot_ensembles.cpp
    spalera_sgd_test.cpp
    spsa_test.cpp
    svrg_test.cpp
    swats_test.cpp
    wn_grad_test.cpp
    yogi_test.cpp
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
add_executable(ensmallen_tests EXCLUDE_FROM_ALL ${ENSMALLEN_TESTS_SOURCES})
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

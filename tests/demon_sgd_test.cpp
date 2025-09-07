/**
 * @file demon_sgd_test.cpp
 * @author Marcus Edel
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#if defined(ENS_USE_COOT)
  #include <armadillo>
  #include <bandicoot>
#endif
#include <ensmallen.hpp>
#include "catch.hpp"
#include "test_function_tools.hpp"
#include "test_types.hpp"

using namespace ens;
using namespace ens::test;

TEMPLATE_TEST_CASE("DemonSGD_LogisticRegressionFunction", "[DemonSGD]",
    ENS_ALL_TEST_TYPES)
{
  DemonSGD optimizer(32.0 /* huge step size needed! */, 16, 0.9, 10000000, 1e-9,
      true, true, true);
  // It could take several tries to get everything to converge.
  LogisticRegressionFunctionTest<TestType>(optimizer,
      Tolerances<TestType>::LRTrainAcc,
      Tolerances<TestType>::LRTestAcc,
      10);
}

#ifdef ENS_HAVE_COOT

TEMPLATE_TEST_CASE("DemonSGD_LogisticRegressionFunction", "[DemonSGD]",
    coot::mat, coot::fmat)
{
  DemonSGD optimizer(32.0 /* huge step size needed! */, 16, 0.9, 1000000, 1e-9,
      true, true, true);
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
      optimizer, 0.003, 0.006, 10);
}

#endif

/**
 * @file spsa_test.cpp
 * @author N Rajiv Vaidyanathan
 * @author Marcus Edel
 *
 * Test file for the SPSA optimizer.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <ensmallen.hpp>
#include "catch.hpp"
#include "test_function_tools.hpp"

using namespace arma;
using namespace ens;
using namespace ens::test;

TEMPLATE_TEST_CASE("SPSA_SphereFunction", "[SPSA]", ENS_TEST_TYPES)
{
  SPSA optimizer(0.1, 0.102, 0.16, 0.3, 100000, 0);
  FunctionTest<SphereFunctionType<TestType, arma::Row<size_t>>, TestType>(
      optimizer,
      Tolerances<TestType>::LargeObj,
      Tolerances<TestType>::LargeCoord);
}

TEMPLATE_TEST_CASE("SPSA_MatyasFunction", "[SPSA]", ENS_TEST_TYPES)
{
  SPSA optimizer(0.1, 0.102, 0.16, 0.3, 100000, 0);
  FunctionTest<MatyasFunction, TestType>(
      optimizer,
      Tolerances<TestType>::LargeObj,
      Tolerances<TestType>::LargeCoord);
}

TEMPLATE_TEST_CASE("SPSA_LogisticRegressionFunction", "[SPSA]", ENS_TEST_TYPES)
{
  // We allow many trials, because SPSA is definitely not guaranteed to
  // converge.
  SPSA optimizer(0.5, 0.102, 0.002, 0.3, 5000, 1e-8);
  LogisticRegressionFunctionTest<TestType, arma::Row<size_t>>(
      optimizer,
      Tolerances<TestType>::LRTrainAcc,
      Tolerances<TestType>::LRTestAcc,
      25);
}

#ifdef USE_COOT

TEMPLATE_TEST_CASE("SPSA_SphereFunction", "[SPSA]", coot::mat, coot::fmat)
{
  SPSA optimizer(0.1, 0.102, 0.16, 0.3, 100000, 0);
  FunctionTest<SphereFunctionType<TestType, coot::Row<size_t>>, TestType>(
      optimizer, 1.0, 0.1);
}

TEMPLATE_TEST_CASE("SPSA_MatyasFunction", "[SPSA]", coot::mat, coot::fmat)
{
  SPSA optimizer(0.1, 0.102, 0.16, 0.3, 100000, 0);
  FunctionTest<MatyasFunction, TestType>(optimizer, 0.1, 0.01);
}

TEMPLATE_TEST_CASE("SPSA_LogisticRegressionFunction", "[SPSA]", coot::mat)
{
  // We allow 10 trials, because SPSA is definitely not guaranteed to
  // converge.
  SPSA optimizer(0.5, 0.102, 0.002, 0.3, 5000, 1e-8);
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
      optimizer, 0.003, 0.006, 10);
}

#endif

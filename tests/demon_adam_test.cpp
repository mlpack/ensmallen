/**
 * @file demon_adam_test.cpp
 * @author Marcus Edel
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <ensmallen.hpp>
#include "catch.hpp"
#include "test_function_tools.hpp"

using namespace ens;
using namespace ens::test;

TEMPLATE_TEST_CASE("DemonAdamLogisticRegressionTest", "[DemonAdam]", arma::mat)
{
  DemonAdam optimizer(0.2, 32, 0.9, 0.9, 0.999, 1e-8,
      10000, 1e-9, true, true, true);
  LogisticRegressionFunctionTest<TestType>(optimizer, 0.003, 0.006, 6);
}

TEMPLATE_TEST_CASE("DemonAdaMaxLogisticRegressionTest", "[DemonAdam]", arma::mat)
{
  DemonAdamType<AdaMaxUpdate> optimizer(0.5, 10, 0.9, 0.9, 0.999, 1e-8,
      10000, 1e-9, true, true, true);
  LogisticRegressionFunctionTest<TestType>(optimizer, 0.003, 0.006, 6);
}

TEMPLATE_TEST_CASE("DemonAdamSphereFunctionTest", "[DemonAdam]", arma::mat, arma::fmat)
{
  DemonAdam optimizer(0.5, 2, 0.9);
  FunctionTest<SphereFunction<TestType, arma::Row<size_t>>, TestType>(
      optimizer, 1.0, 0.1);
}

TEMPLATE_TEST_CASE("DemonAdamMatyasFunctionTest", "[DemonAdam]", arma::mat, arma::fmat)
{
  DemonAdam optimizer(0.5, 1, 0.9);
  FunctionTest<MatyasFunction, TestType>(optimizer, 0.1, 0.01);
}

TEMPLATE_TEST_CASE("DemonAdamSphereFunctionTestSpMat", "[DemonAdam]", arma::sp_mat)
{
  DemonAdam optimizer(0.5, 2, 0.9);
  FunctionTest<SphereFunction<TestType, arma::Row<size_t>>, TestType>(
      optimizer, 1.0, 0.1);
}

#ifdef USE_COOT

TEMPLATE_TEST_CASE("DemonAdamLogisticRegressionTest", "[DemonAdam]", coot::mat)
{
  DemonAdam optimizer(0.2, 32, 0.9, 0.9, 0.999, 1e-8,
      10000, 1e-9, true, true, true);
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(optimizer, 0.003, 0.006, 6);
}

TEMPLATE_TEST_CASE("DemonAdaMaxLogisticRegressionTest", "[DemonAdam]", coot::mat)
{
  DemonAdamType<AdaMaxUpdate> optimizer(0.5, 10, 0.9, 0.9, 0.999, 1e-8,
      10000, 1e-9, true, true, true);
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(optimizer, 0.003, 0.006, 6);
}

TEMPLATE_TEST_CASE("DemonAdamSphereFunctionTest", "[DemonAdam]", coot::mat, coot::fmat)
{
  DemonAdam optimizer(0.5, 2, 0.9);
  FunctionTest<SphereFunction<TestType, coot::Row<size_t>>, TestType>(
      optimizer, 1.0, 0.1);
}

TEMPLATE_TEST_CASE("DemonAdamMatyasFunctionTest", "[DemonAdam]", coot::mat, coot::fmat)
{
  DemonAdam optimizer(0.5, 1, 0.9);
  FunctionTest<MatyasFunction, TestType>(optimizer, 0.1, 0.01);
}

#endif

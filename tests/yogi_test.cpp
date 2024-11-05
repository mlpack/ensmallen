/**
 * @file yogi_test.cpp
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

TEMPLATE_TEST_CASE("YogiSphereFunctionTest", "[Yogi]", arma::mat, arma::fmat)
{
  Yogi optimizer(0.5, 2, 0.7, 0.999, 1e-8, 500000, 1e-3, false);

  FunctionTest<SphereFunction<TestType, arma::Row<size_t>>, TestType>(
      optimizer, 0.5, 0.1);
}

TEMPLATE_TEST_CASE("YogiMcCormickFunctionTest", "[Yogi]", arma::mat, arma::fmat)
{
  Yogi optimizer(0.5, 1, 0.7, 0.999, 1e-8, 500000, 1e-5, false);
  FunctionTest<McCormickFunction, TestType>(optimizer, 0.5, 0.1);
}

TEMPLATE_TEST_CASE("YogiLogisticRegressionTest", "[Yogi]", arma::mat, arma::fmat)
{
  Yogi optimizer;
  LogisticRegressionFunctionTest<TestType>(optimizer, 0.003, 0.006);
}

#ifdef USE_COOT

TEMPLATE_TEST_CASE("YogiSphereFunctionTest", "[Yogi]", coot::mat, coot::fmat)
{
  Yogi optimizer(0.5, 2, 0.7, 0.999, 1e-8, 500000, 1e-3, false);

  FunctionTest<SphereFunction<TestType, coot::Row<size_t>>, TestType>(
      optimizer, 0.5, 0.1);
}

TEMPLATE_TEST_CASE("YogiMcCormickFunctionTest", "[Yogi]", coot::mat)
{
  Yogi optimizer(0.5, 1, 0.7, 0.999, 1e-8, 500000, 1e-5, false);
  FunctionTest<McCormickFunction, TestType>(optimizer, 0.5, 0.1);
}

TEMPLATE_TEST_CASE("YogiLogisticRegressionTest", "[Yogi]", coot::mat)
{
  Yogi optimizer;
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(optimizer, 0.003, 0.006);
}

#endif

/**
 * @file wn_grad_test.cpp
 * @author Marcus Edel
 *
 * Test file for the WNGrad optimizer.
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

TEMPLATE_TEST_CASE("WNGrad_LogisticRegressionFunction", "[WNGrad]",
    ENS_ALL_TEST_TYPES)
{
  WNGrad optimizer(0.56, 1, 500000, 1e-9, true);
  LogisticRegressionFunctionTest<TestType, arma::Row<size_t>>(optimizer);
}

TEMPLATE_TEST_CASE("WNGrad_SphereFunction", "[WNGrad]", ENS_ALL_TEST_TYPES)
{
  WNGrad optimizer(1.12, 2, 500000, 1e-9, true);
  FunctionTest<SphereFunction, TestType>(
      optimizer,
      Tolerances<TestType>::LargeObj,
      Tolerances<TestType>::LargeCoord);
}

// The Styblinski-Tang function is too difficult to make converge for WNGrad in
// low precision.
TEMPLATE_TEST_CASE("WNGrad_StyblinskiTangFunction", "[WNGrad]",
    ENS_FULLPREC_TEST_TYPES, ENS_SPARSE_TEST_TYPES)
{
  WNGrad optimizer(1.12, 2, 500000, 1e-9, true);
  FunctionTest<StyblinskiTangFunction, TestType>(
      optimizer,
      5 * Tolerances<TestType>::LargeObj,
      5 * Tolerances<TestType>::LargeCoord);
}

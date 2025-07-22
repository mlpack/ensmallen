/**
 * @file ada_grad_test.cpp
 * @author Abhinav Moudgil
 * @author Marcus Edel
 * @author Conrad Sanderson
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

TEMPLATE_TEST_CASE("AdaGrad_LogisticRegressionFunction", "[AdaGrad]",
    ENS_ALL_TEST_TYPES)
{
  AdaGrad adagrad(0.99, 32, 10 * Tolerances<TestType>::Obj, 500000,
      Tolerances<TestType>::Obj, true);
  LogisticRegressionFunctionTest<TestType, arma::Row<size_t>>(adagrad);
}

#ifdef USE_COOT

TEMPLATE_TEST_CASE("AdaGrad_LogisticRegressionFunction", "[AdaGrad]",
    coot::mat, coot::fmat)
{
  AdaGrad adagrad(0.99, 32, 1e-8, 500000, 1e-9, true);
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(adagrad);
}

#endif

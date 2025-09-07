/**
 * @file ada_delta_test.cpp
 * @author Marcus Edel
 * @author Vasanth Kalingeri
 * @author Abhinav Moudgil
 * @author Conrad Sanderson
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

TEMPLATE_TEST_CASE("AdaDelta_LogisticRegressionFunction", "[AdaDelta]",
    ENS_ALL_TEST_TYPES)
{
  typedef typename TestType::elem_type ElemType;

  // Use a large epsilon if we are using FP16, to avoid underflow in the first
  // iterations.
  AdaDelta adaDelta(1.0, 32, 0.95, sizeof(ElemType) == 2 ? 1e-4 : 1e-6);
  LogisticRegressionFunctionTest<TestType, arma::Row<size_t>>(adaDelta);
}

#ifdef ENS_HAVE_COOT

TEMPLATE_TEST_CASE("AdaDelta_LogisticRegressionFunction", "[AdaDelta]",
    coot::mat, coot::fmat)
{
  AdaDelta adaDelta;
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
      adaDelta, 0.003, 0.006, 1);
}

#endif

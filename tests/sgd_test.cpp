/**
 * @file sgd_test.cpp
 * @author Ryan Curtin
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

using namespace arma;
using namespace ens;
using namespace ens::test;

template<typename MatType>
void SGDGeneralizedRosenbrockTest()
{
  typedef typename MatType::elem_type ElemType;

  // Loop over several variants.
  for (size_t i = 10; i < 50; i += 5)
  {
    // Create the generalized Rosenbrock function.
    GeneralizedRosenbrockFunction<MatType> f(i);

    // Allow a few trials.
    for (size_t trial = 0; trial < 5; ++trial)
    {
      StandardSGD s(0.001, 1, 0, 1e-15, true);

      MatType coordinates = f.GetInitialPoint();
      float result = s.Optimize(f, coordinates);

      if (trial != 4)
      {
        if (result != Approx(0.0).margin(1e-5))
          continue;
        for (size_t j = 0; j < i; ++j)
        {
          if (ElemType(coordinates(j)) != Approx(1.0).epsilon(1e-3))
            continue;
        }
      }

      REQUIRE(result == Approx(0.0).margin(1e-5));
      for (size_t j = 0; j < i; ++j)
        REQUIRE(ElemType(coordinates(j)) == Approx(1.0).epsilon(1e-3));
      break;
    }
  }
}

template<typename MatType, typename LabelsType>
void SGDLogisticRegressionTest()
{
  MatType data, testData, shuffledData;
  LabelsType responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);
  LogisticRegressionFunction<MatType, LabelsType> lr(
      shuffledData, shuffledResponses, 0.5);

  StandardSGD sgd;
  MatType coordinates = lr.GetInitialPoint();
  sgd.Optimize(lr, coordinates);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses, coordinates);

  REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses,
      coordinates);
  REQUIRE(testAcc == Approx(100.0).epsilon(0.006)); // 0.6% error tolerance.
}

TEMPLATE_TEST_CASE("SGDGeneralizedRosenbrock", "[SGDTest]",
    arma::mat, arma::fmat)
{
  SGDGeneralizedRosenbrockTest<TestType>();
}

TEMPLATE_TEST_CASE("SGDLogisticRegressionTest", "[SGDTest]",
    arma::mat)
{
  SGDLogisticRegressionTest<TestType, arma::Row<size_t>>();
}

#ifdef USE_COOT

TEMPLATE_TEST_CASE("SGDGeneralizedRosenbrock", "[SGDTest]",
    coot::mat, coot::fmat)
{
  SGDGeneralizedRosenbrockTest<TestType>();
}

TEMPLATE_TEST_CASE("SGDLogisticRegressionTest", "[SGDTest]",
    coot::mat)
{
  SGDLogisticRegressionTest<TestType, coot::Row<size_t>>();
}

#endif

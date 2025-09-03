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
#if defined(ENS_USE_COOT)
  #include <armadillo>
  #include <bandicoot>
#endif
#include <ensmallen.hpp>
#include "catch.hpp"
#include "test_function_tools.hpp"

using namespace arma;
using namespace ens;
using namespace ens::test;

template<typename MatType>
void SGDGeneralizedRosenbrockTest(const size_t variants = 50)
{
  typedef typename MatType::elem_type ElemType;

  // Loop over several variants.
  for (size_t i = 10; i < variants; i += 5)
  {
    // Create the generalized Rosenbrock function.
    GeneralizedRosenbrockFunction f(i);

    // Allow a few trials.
    for (size_t trial = 0; trial < 5; ++trial)
    {
      StandardSGD s(0.001, 1, 0, 1e-15, true);

      MatType coordinates = f.GetInitialPoint<MatType>();
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

template<typename MatType>
void SGDLogisticRegressionTest()
{
  MatType data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);
  LogisticRegressionFunction<MatType> lr(shuffledData, shuffledResponses, 0.5);

  StandardSGD sgd(0.32);
  MatType coordinates = lr.GetInitialPoint();
  sgd.Optimize(lr, coordinates);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses, coordinates);

  REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses,
      coordinates);
  REQUIRE(testAcc == Approx(100.0).epsilon(0.006)); // 0.6% error tolerance.
}

TEMPLATE_TEST_CASE("SGD_GeneralizedRosenbrockFunction", "[SGD]",
    arma::mat, arma::fmat)
{
  SGDGeneralizedRosenbrockTest<TestType>();
}

TEMPLATE_TEST_CASE("SGD_LogisticRegressionFunction", "[SGD]",
    arma::mat)
{
  SGDLogisticRegressionTest<TestType>();
}

#ifdef ENS_HAVE_COOT

TEMPLATE_TEST_CASE("SGD_GeneralizedRosenbrockFunction", "[SGD]",
    coot::mat, coot::fmat)
{
  SGDGeneralizedRosenbrockTest<TestType>(15);
}

TEMPLATE_TEST_CASE("SGD_LogisticRegressionFunction", "[SGD]",
    coot::mat)
{
  SGDLogisticRegressionTest<TestType>();
}

#endif

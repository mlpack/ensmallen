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
#include "test_types.hpp"

using namespace arma;
using namespace ens;
using namespace ens::test;

template<typename MatType>
void SGDGeneralizedRosenbrockTest(const size_t variants = 50)
{
  typedef typename MatType::elem_type ElemType;

  const double objTol = 100 * Tolerances<MatType>::Obj;
  const double coordTol = 100 * Tolerances<MatType>::Coord;

  // Loop over several variants.
  for (size_t i = 10; i < variants; i += 5)
  {
    // Create the generalized Rosenbrock function.
    GeneralizedRosenbrockFunctionType<MatType> f(i);

    // Allow a few trials.
    for (size_t trial = 0; trial < 5; ++trial)
    {
      StandardSGD s(0.001, 1, 1000000, Tolerances<MatType>::Obj / 100, true);

      MatType coordinates = f.GetInitialPoint();
      float result = s.Optimize(f, coordinates);

      if (trial != 4)
      {
        if (result != Approx(0.0).margin(objTol))
          continue;
        for (size_t j = 0; j < i; ++j)
        {
          if (ElemType(coordinates(j)) != Approx(1.0).epsilon(coordTol))
            continue;
        }
      }

      REQUIRE(result == Approx(0.0).margin(objTol));
      for (size_t j = 0; j < i; ++j)
        REQUIRE(ElemType(coordinates(j)) == Approx(1.0).epsilon(coordTol));
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
  LogisticRegressionFunction<MatType> lr(
      shuffledData, shuffledResponses, 0.5);

  StandardSGD sgd;
  MatType coordinates = lr.GetInitialPoint();
  sgd.Optimize(lr, coordinates);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses, coordinates);

  REQUIRE(acc == Approx(100.0).epsilon(Tolerances<MatType>::LRTrainAcc));

  const double testAcc = lr.ComputeAccuracy(testData, testResponses,
      coordinates);
  REQUIRE(testAcc == Approx(100.0).epsilon(Tolerances<MatType>::LRTestAcc));
}

TEMPLATE_TEST_CASE("SGD_GeneralizedRosenbrockFunction", "[SGD]", ENS_TEST_TYPES)
{
  SGDGeneralizedRosenbrockTest<TestType>();
}

TEMPLATE_TEST_CASE("SGD_LogisticRegressionFunction", "[SGD]", ENS_TEST_TYPES)
{
  SGDLogisticRegressionTest<TestType, arma::Row<size_t>>();
}

#ifdef USE_COOT

TEMPLATE_TEST_CASE("SGD_GeneralizedRosenbrockFunction", "[SGD]",
    coot::mat, coot::fmat)
{
  SGDGeneralizedRosenbrockTest<TestType>(15);
}

TEMPLATE_TEST_CASE("SGD_LogisticRegressionFunction", "[SGD]",
    coot::mat)
{
  SGDLogisticRegressionTest<TestType, coot::Row<size_t>>();
}

#endif

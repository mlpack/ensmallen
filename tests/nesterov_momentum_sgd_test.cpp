/**
 * @file nesterov_momentum_sgd_test.cpp
 * @author Sourabh Varshney
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

using namespace ens;
using namespace ens::test;

TEMPLATE_TEST_CASE("NesterovMomentumSGD_SGDTestFunction",
    "[NesterovMomentumSGD]", arma::mat, arma::fmat)
{
  SGDTestFunction f;
  NesterovMomentumUpdate nesterovMomentumUpdate(0.9);
  NesterovMomentumSGD s(0.0003, 1, 2500000, 1e-9, true, nesterovMomentumUpdate,
      NoDecay(), true, true);

  TestType coordinates = f.GetInitialPoint<TestType>();
  double result = s.Optimize(f, coordinates);

  REQUIRE(result == Approx(-1.0).margin(0.01));
  REQUIRE(coordinates(0) == Approx(0.0).margin(3e-3));
  REQUIRE(coordinates(1) == Approx(0.0).margin(1e-6));
  REQUIRE(coordinates(2) == Approx(0.0).margin(1e-6));
}

TEMPLATE_TEST_CASE("NesterovMomentumSGD_GeneralizedRosenbrockFunction",
    "[NesterovMomentumSGD]", arma::mat)
{
  // Loop over several variants.
  for (size_t i = 10; i < 50; i += 5)
  {
    // Create the generalized Rosenbrock function.
    GeneralizedRosenbrockFunction f(i);
    NesterovMomentumUpdate nesterovMomentumUpdate(0.9);
    NesterovMomentumSGD s(0.0001, 1, 0, 1e-15, true, nesterovMomentumUpdate,
        NoDecay(), true, true);

    TestType coordinates = f.GetInitialPoint();
    double result = s.Optimize(f, coordinates);

    REQUIRE(result == Approx(0.0).margin(1e-4));
    for (size_t j = 0; j < i; ++j)
      REQUIRE(coordinates(j) == Approx(1.0).epsilon(0.003));
  }
}

TEMPLATE_TEST_CASE("NesterovMomentumSGD_GeneralizedRosenbrockFunction",
    "[NesterovMomentumSGD]", arma::fmat)
{
  // Loop over several variants.
  for (size_t i = 10; i < 50; i += 5)
  {
    // Create the generalized Rosenbrock function.
    GeneralizedRosenbrockFunction f(i);
    NesterovMomentumUpdate nesterovMomentumUpdate(0.9);
    NesterovMomentumSGD s(0.00015, 1, 0, 1e-10, true, nesterovMomentumUpdate);

    size_t trial = 0;
    float result = std::numeric_limits<float>::max();
    TestType coordinates;
    while (trial++ < 8 && result > 0.1)
    {
      coordinates = f.GetInitialPoint<TestType>();
      result = s.Optimize(f, coordinates);
    }

    REQUIRE(result == Approx(0.0).margin(0.02));
    for (size_t j = 0; j < i; ++j)
      REQUIRE(coordinates(j) == Approx(1.0).margin(0.05));
  }
}

TEMPLATE_TEST_CASE("NesterovMomentumSGD_GeneralizedRosenbrockFunction",
    "[NesterovMomentumSGD]", arma::sp_mat)
{
  // Loop over several variants.
  for (size_t i = 10; i < 50; i += 5)
  {
    // Create the generalized Rosenbrock function.
    GeneralizedRosenbrockFunction f(i);
    NesterovMomentumUpdate nesterovMomentumUpdate(0.9);
    NesterovMomentumSGD s(0.0001, 1, 0, 1e-15, true, nesterovMomentumUpdate);

    TestType coordinates = f.GetInitialPoint<TestType>();
    double result = s.Optimize(f, coordinates);

    REQUIRE(result == Approx(0.0).margin(1e-4));
    for (size_t j = 0; j < i; ++j)
      REQUIRE(coordinates(j) == Approx(1.0).epsilon(0.003));
  }
}

#ifdef ENS_USE_COOT

TEMPLATE_TEST_CASE("NesterovMomentum_GeneralizedRosenbrockFunction",
    "[NesterovMomentumSGD]", coot::mat)
{
  typedef typename TestType::elem_type ElemType;

  // Create the generalized Rosenbrock function.
  GeneralizedRosenbrockFunction f(10);
  NesterovMomentumUpdate nesterovMomentumUpdate(0.9);
  NesterovMomentumSGD s(0.0001, 1, 0, 1e-15, true, nesterovMomentumUpdate,
      NoDecay(), true, true);

  TestType coordinates = f.GetInitialPoint();
  ElemType result = s.Optimize(f, coordinates);

  REQUIRE(result == Approx(0.0).margin(1e-4));
  for (size_t j = 0; j < 10; ++j)
    REQUIRE(ElemType(coordinates(j)) == Approx(1.0).epsilon(0.003));
}

TEMPLATE_TEST_CASE("NesterovMomentumSGD_GeneralizedRosenbrockFunction",
    "[NesterovMomentumSGD]", coot::fmat)
{
  typedef typename TestType::elem_type ElemType;

  // Create the generalized Rosenbrock function.
  GeneralizedRosenbrockFunctionType<TestType, coot::Row<size_t>> f(10);
  NesterovMomentumUpdate nesterovMomentumUpdate(0.9);
  NesterovMomentumSGD s(0.00015, 1, 0, 1e-10, true, nesterovMomentumUpdate);

  size_t trial = 0;
  ElemType result = std::numeric_limits<ElemType>::max();
  TestType coordinates;
  while (trial++ < 8 && result > 0.1)
  {
    coordinates = f.GetInitialPoint();
    result = s.Optimize(f, coordinates);
  }

  REQUIRE(result == Approx(0.0).margin(0.02));
  for (size_t j = 0; j < 10; ++j)
    REQUIRE(ElemType(coordinates(j)) == Approx(1.0).margin(0.05));
}

#endif

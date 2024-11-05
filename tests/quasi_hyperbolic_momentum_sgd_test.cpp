/**
 * @file quasi_hyperbolic_momentum_sgd_test.cpp
 * @author Niteya Shah
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

TEMPLATE_TEST_CASE("QHSphereFunction", "[QHMomentumSGD]",
    arma::mat, arma::fmat)
{
  QHUpdate update(0.4, 0.9);
  QHSGD s(0.002, 1, 2500000, 1e-9, true, update, NoDecay(), true, true);
  FunctionTest<SphereFunction<TestType, arma::Row<size_t>>, TestType>(
      s, 0.03, 0.003);
}

TEST_CASE("QHSpMatTestSphereFunction", "[QHMomentumSGD]")
{
  QHUpdate update(0.9, 0.9);
  QHSGD s(0.002, 1, 2500000, 1e-15, true, update);
  s.ExactObjective() = true;
  FunctionTest<SphereFunction<>, arma::sp_mat>(s, 0.03, 0.003);
}

TEMPLATE_TEST_CASE("QHSGDSGDGeneralizedRosenbrockTest", "[QHMomentumSGD]",
    arma::mat)
{
  for (size_t i = 10; i < 50; i += 5)
  {
    // Create the generalized Rosenbrock function.
    GeneralizedRosenbrockFunction<TestType, arma::Row<size_t>> f(i);
    QHUpdate update(0.9, 0.99);
    QHSGD s(0.0005, 1, 2500000, 1e-15, true, update, NoDecay(), true, true);

    TestType coordinates = f.GetInitialPoint();
    double result = s.Optimize(f, coordinates);

    REQUIRE(result == Approx(0.0).margin(1e-4));
    for (size_t j = 0; j < i; ++j)
      REQUIRE(coordinates(j) == Approx(1.0).epsilon(1e-4));
  }
}

#ifdef USE_COOT

TEMPLATE_TEST_CASE("QHSphereFunction", "[QHMomentumSGD]",
    coot::mat, coot::fmat)
{
  QHUpdate update(0.4, 0.9);
  QHSGD s(0.002, 1, 2500000, 1e-9, true, update, NoDecay(), true, true);
  FunctionTest<SphereFunction<TestType, coot::Row<size_t>>, TestType>(
      s, 0.03, 0.003);
}

TEMPLATE_TEST_CASE("QHSGDSGDGeneralizedRosenbrockTest", "[QHMomentumSGD]",
    coot::mat)
{
  typedef typename TestType::elem_type ElemType;

  for (size_t i = 10; i < 50; i += 5)
  {
    // Create the generalized Rosenbrock function.
    GeneralizedRosenbrockFunction<TestType, coot::Row<size_t>> f(i);
    QHUpdate update(0.9, 0.99);
    QHSGD s(0.0005, 1, 2500000, 1e-15, true, update, NoDecay(), true, true);

    TestType coordinates = f.GetInitialPoint();
    double result = s.Optimize(f, coordinates);

    REQUIRE(result == Approx(0.0).margin(1e-4));
    for (size_t j = 0; j < i; ++j)
      REQUIRE(ElemType(coordinates(j)) == Approx(1.0).epsilon(1e-4));
  }
}

#endif

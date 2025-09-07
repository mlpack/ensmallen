/**
 * @file demon_adam_test.cpp
 * @author Marcus Edel
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

TEMPLATE_TEST_CASE("DemonAdam_LogisticRegressionFunction", "[DemonAdam]",
    ENS_ALL_TEST_TYPES)
{
  DemonAdam optimizer(6.4, 32, 0.9, 0.9, 0.999, Tolerances<TestType>::Obj,
      10000, Tolerances<TestType>::Obj / 10, true, true, true);
  // This may require a few attempts to get right.
  LogisticRegressionFunctionTest<TestType>(optimizer,
      Tolerances<TestType>::LRTrainAcc,
      Tolerances<TestType>::LRTestAcc,
      6);
}

TEMPLATE_TEST_CASE("DemonAdaMax_LogisticRegressionFunction", "[DemonAdam]",
    ENS_ALL_TEST_TYPES)
{
  DemonAdamType<AdaMaxUpdate> optimizer(5.0, 10, 0.9, 0.9, 0.999,
      Tolerances<TestType>::Obj, 10000, Tolerances<TestType>::Obj / 10, true,
      true, true);
  // This may require a few attempts to get right.
  LogisticRegressionFunctionTest<TestType>(optimizer,
      Tolerances<TestType>::LRTrainAcc,
      Tolerances<TestType>::LRTestAcc,
      6);
}

TEMPLATE_TEST_CASE("DemonAdam_SphereFunction", "[DemonAdam]",
    ENS_ALL_TEST_TYPES, ENS_SPARSE_TEST_TYPES)
{
  DemonAdam optimizer(1.0, 2, 0.9);
  FunctionTest<SphereFunction, TestType>(
      optimizer,
      10 * Tolerances<TestType>::LargeObj,
      Tolerances<TestType>::LargeCoord);
}

TEMPLATE_TEST_CASE("DemonAdam_MatyasFunction", "[DemonAdam]",
    ENS_ALL_TEST_TYPES)
{
  DemonAdam optimizer(0.5, 1, 0.9);
  FunctionTest<MatyasFunction, TestType>(optimizer, 0.1, 0.01);
}

#ifdef ENS_HAVE_COOT

TEMPLATE_TEST_CASE("DemonAdam_LogisticRegressionFunction", "[DemonAdam]",
    coot::mat)
{
  DemonAdam optimizer(6.4, 32, 0.9, 0.9, 0.999, 1e-8,
      10000, 1e-9, true, true, true);
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
      optimizer, 0.003, 0.006, 6);
}

TEMPLATE_TEST_CASE("DemonAdaMax_LogisticRegressionFunction", "[DemonAdam]",
    coot::mat)
{
  DemonAdamType<AdaMaxUpdate> optimizer(5.0, 10, 0.9, 0.9, 0.999, 1e-8,
      10000, 1e-9, true, true, true);
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
      optimizer, 0.003, 0.006, 6);
}

TEMPLATE_TEST_CASE("DemonAdam_SphereFunction", "[DemonAdam]",
    coot::mat, coot::fmat)
{
  DemonAdam optimizer(1.0, 2, 0.9);
  FunctionTest<SphereFunction, TestType>(optimizer, 1.0, 0.1);
}

TEMPLATE_TEST_CASE("DemonAdam_MatyasFunction", "[DemonAdam]",
    coot::mat, coot::fmat)
{
  DemonAdam optimizer(0.5, 1, 0.9);
  FunctionTest<MatyasFunction, TestType>(optimizer, 0.1, 0.01);
}

#endif

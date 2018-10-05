// Copyright (c) 2018 ensmallen developers.
// 
// Licensed under the 3-clause BSD license (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.opensource.org/licenses/BSD-3-Clause

#include <ensmallen.hpp>
#include "catch.hpp"

using namespace std;
using namespace ens;


// #include <mlpack/core.hpp>
// #include <mlpack/core/optimizers/scd/scd.hpp>
// #include <mlpack/core/optimizers/scd/descent_policies/greedy_descent.hpp>
// #include <mlpack/core/optimizers/scd/descent_policies/cyclic_descent.hpp>
// #include <mlpack/core/optimizers/parallel_sgd/sparse_test_function.hpp>
// #include <mlpack/methods/logistic_regression/logistic_regression_function.hpp>
// #include <mlpack/methods/softmax_regression/softmax_regression_function.hpp>
// 
// using namespace mlpack;
// using namespace mlpack::math;
// using namespace mlpack::optimization;
// using namespace mlpack::optimization::test;
// using namespace mlpack::regression;

/**
 * Test the correctness of the SCD implementation by using a dataset with a
 * precalculated minima.
 */
TEST_CASE("PreCalcSCDTest","[SCDTest]")
{
  arma::mat predictors("0 0 0.4; 0 0 0.6; 0 0.3 0; 0.2 0 0; 0.2 -0.5 0;");
  arma::Row<size_t> responses("1  1  0;");

  LogisticRegressionFunction<arma::mat> f(predictors, responses, 0.0001);

  SCD<> s(0.02, 60000, 1e-5);
  arma::mat iterate = f.InitialPoint();

  double objective = s.Optimize(f, iterate);

  REQUIRE(objective <= 0.055);
}

/**
 * Test the correctness of the SCD implemenation by using the sparse test
 * function, with dijoint features which optimize to a precalculated minima.
 */
TEST_CASE("DisjointFeatureTest","[SCDTest]")
{
  // The test function for parallel SGD should work with SCD, as the gradients
  // of the individual functions are projections into the ith dimension.
  SparseTestFunction f;
  SCD<> s(0.4);

  arma::mat iterate = f.GetInitialPoint();

  double result = s.Optimize(f, iterate);

  // The final value of the objective function should be close to the optimal
  // value, that is the sum of values at the vertices of the parabolas.
  REQUIRE(result == Approx(123.75).epsilon(0.0001));

  // The co-ordinates should be the vertices of the parabolas.
  REQUIRE(iterate[0] == Approx(2.0).epsilon(0.0002));
  REQUIRE(iterate[1] == Approx(1.0).epsilon(0.0002));
  REQUIRE(iterate[2] == Approx(1.5).epsilon(0.0002));
  REQUIRE(iterate[3] == Approx(4.0).epsilon(0.0002));
}

/**
 * Test the greedy descent policy.
 */
TEST_CASE("GreedyDescentTest","[SCDTest]")
{
  // In the sparse test function, the given point has the maximum gradient at
  // the feature with index 2.
  arma::mat point("1; 2; 3; 4;");

  SparseTestFunction f;

  GreedyDescent descentPolicy;

  REQUIRE_EQUAL(descentPolicy.DescentFeature(0, point, f) == 2);

  // Changing the point under consideration, so that the maximum gradient is at
  // index 1.
  point[1] = 10;

  REQUIRE(descentPolicy.DescentFeature(0, point, f) == 1);
}

/**
 * Test the cyclic descent policy.
 */
TEST_CASE("CyclicDescentTest","[SCDTest]")
{
  const size_t features = 10;
  struct DummyFunction
  {
    static size_t NumFeatures()
    {
      return features;
    }
  };

  DummyFunction dummy;

  CyclicDescent descentPolicy;

  for (size_t i = 0; i < 15; ++i)
  {
    REQUIRE(descentPolicy.DescentFeature(i, arma::mat(), dummy) == (i % features));
  }
}

/**
 * Test the random descent policy.
 */
TEST_CASE("RandomDescentTest","[SCDTest]")
{
  const size_t features = 10;
  struct DummyFunction
  {
    static size_t NumFeatures()
    {
      return features;
    }
  };

  DummyFunction dummy;

  CyclicDescent descentPolicy;

  for (size_t i = 0; i < 100; ++i)
  {
    size_t j = descentPolicy.DescentFeature(i, arma::mat(), dummy);
    REQUIRE(j < features);
    REQUIRE(j >= 0);
  }
}

/**
 * Test that LogisticRegressionFunction::PartialGradient() works as expected.
 */
TEST_CASE("LogisticRegressionFunctionPartialGradientTest","[SCDTest]")
{
  // Evaluate the gradient and feature gradient and equate.
  arma::mat predictors("0 0 0.4; 0 0 0.6; 0 0.3 0; 0.2 0 0; 0.2 -0.5 0;");
  arma::Row<size_t> responses("1  1  0;");

  LogisticRegressionFunction<arma::mat> f(predictors, responses, 0.0001);

  arma::mat testPoint(1, f.NumFeatures(), arma::fill::randu);

  arma::mat testGradient;

  f.Gradient(testPoint, testGradient);

  for (size_t i = 0; i < f.NumFeatures(); ++i)
  {
    arma::sp_mat fGrad;
    f.PartialGradient(testPoint, i, fGrad);

    CheckMatrices(testGradient.col(i), arma::mat(fGrad.col(i)));
  }
}

/**
 * Test that SoftmaxRegressionFunction::PartialGradient() works as expected.
 */
TEST_CASE(SoftmaxRegressionFunctionPartialGradientTest,"[SCDTest]")
{
  const size_t points = 1000;
  const size_t inputSize = 10;
  const size_t numClasses = 5;

  // Initialize a random dataset.
  arma::mat data;
  data.randu(inputSize, points);

  // Create random class labels.
  arma::Row<size_t> labels(points);
  for (size_t i = 0; i < points; i++)
    labels(i) = RandInt(0, numClasses);

  // 2 objects for 2 terms in the cost function. Each term contributes towards
  // the gradient and thus need to be checked independently.
  SoftmaxRegressionFunction srf(data, labels, numClasses, 0);

  // Create a random set of parameters.
  arma::mat parameters;
  parameters.randu(numClasses, inputSize);

  // Get gradients for the current parameters.
  arma::mat gradient;
  srf.Gradient(parameters, gradient);

  // For each parameter.
  for (size_t j = 0; j < inputSize; j++)
  {
    // Get the gradient for this feature.
    arma::sp_mat fGrad;

    srf.PartialGradient(parameters, j, fGrad);

    CheckMatrices(gradient.col(j), arma::mat(fGrad.col(j)));
  }
}

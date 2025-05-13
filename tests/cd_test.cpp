/**
 * @file scd_test.cpp
 * @author Shikhar Bhardwaj
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

using namespace std;
using namespace ens;
using namespace ens::test;

/**
 * Test the correctness of the CD implementation by using a dataset with a
 * precalculated minima.
 */
TEMPLATE_TEST_CASE("CD_LogisticRegressionFunction", "[CD]", arma::mat)
{
  TestType predictors("0 0 0.4; 0 0 0.6; 0 0.3 0; 0.2 0 0; 0.2 -0.5 0;");
  arma::Row<size_t> responses("1  1  0;");

  LogisticRegressionFunction<TestType> f(predictors, responses, 0.0001);

  CD<> s(0.02, 60000, 1e-5);
  TestType iterate = f.InitialPoint();

  double objective = s.Optimize(f, iterate);

  REQUIRE(objective <= 0.055);
}

/**
 * Test the correctness of the CD implemenation by using the sparse test
 * function, with disjoint features which optimize to a precalculated minima.
 */
TEMPLATE_TEST_CASE("CD_SparseTestFunction", "[CD]", arma::mat)
{
  // The test function for parallel SGD should work with CD, as the gradients
  // of the individual functions are projections into the ith dimension.
  CD<> s(0.4);
  FunctionTest<SparseTestFunction, TestType>(s, 0.01, 0.001);
}

/**
 * Test the correctness of the CD implemenation by using the sparse test
 * function, with disjoint features which optimize to a precalculated minima.
 * Use arma::fmat.
 */
TEMPLATE_TEST_CASE("CD_SparseTestFunction", "[CD]", arma::fmat)
{
  // The test function for parallel SGD should work with CD, as the gradients
  // of the individual functions are projections into the ith dimension.
  CD<> s(0.4);
  FunctionTest<SparseTestFunction, TestType>(s, 0.2, 0.02);
}

/**
 * Test the correctness of the CD implemenation by using the sparse test
 * function, with disjoint features which optimize to a precalculated minima.
 * Use arma::sp_mat.
 */
TEMPLATE_TEST_CASE("CD_SparseTestFunction", "[CD]", arma::sp_mat)
{
  // The test function for parallel SGD should work with CD, as the gradients
  // of the individual functions are projections into the ith dimension.
  CD<> s(0.4);
  FunctionTest<SparseTestFunction, arma::sp_mat>(s, 0.01, 0.001);
}

/**
 * Test the greedy descent policy.
 */
TEMPLATE_TEST_CASE("CD_GreedyDescent", "[CD]", arma::mat)
{
  // In the sparse test function, the given point has the maximum gradient at
  // the feature with index 2.
  TestType point("1; 2; 3; 4;");

  SparseTestFunction f;

  REQUIRE(GreedyDescent::DescentFeature<SparseTestFunction,
                                        TestType,
                                        TestType>(0, point, f) == 2);

  // Changing the point under consideration, so that the maximum gradient is at
  // index 1.
  point(1) = 10;

  REQUIRE(GreedyDescent::DescentFeature<SparseTestFunction,
                                        TestType,
                                        TestType>(0, point, f) == 1);
}

/**
 * Test the cyclic descent policy.
 */
TEMPLATE_TEST_CASE("CD_CyclicDescent", "[CD]", arma::mat)
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

  for (size_t i = 0; i < 15; ++i)
  {
    REQUIRE(CyclicDescent::DescentFeature<DummyFunction, TestType, TestType>(
        i, TestType(), dummy) == (i % features));
  }
}

/**
 * Test the random descent policy.
 */
TEMPLATE_TEST_CASE("CD_RandomDescent", "[CD]", arma::mat)
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

  for (size_t i = 0; i < 100; ++i)
  {
    size_t j = CyclicDescent::DescentFeature<DummyFunction,
                                             TestType,
                                             TestType>(i, TestType(), dummy);
    REQUIRE(j < features);
    REQUIRE(j >= 0);
  }
}

/**
 * Test that LogisticRegressionFunction::PartialGradient() works as expected.
 */
TEMPLATE_TEST_CASE("CD_LogisticRegressionFunctionPartialGradient", "[CD]",
    arma::mat)
{
  typedef typename TestType::elem_type ElemType;

  // Evaluate the gradient and feature gradient and equate.
  TestType predictors("0 0 0.4; 0 0 0.6; 0 0.3 0; 0.2 0 0; 0.2 -0.5 0;");
  arma::Row<size_t> responses("1  1  0;");

  LogisticRegressionFunction<TestType> f(predictors, responses, 0.0001);

  TestType testPoint(1, f.NumFeatures(), arma::fill::randu);

  TestType testGradient;

  f.Gradient(testPoint, testGradient);

  for (size_t i = 0; i < f.NumFeatures(); ++i)
  {
    arma::SpMat<ElemType> fGrad;
    f.PartialGradient(testPoint, i, fGrad);

    CheckMatrices(TestType(testGradient.col(i)), TestType(fGrad.col(i)));
  }
}

/**
 * Test that SoftmaxRegressionFunction::PartialGradient() works as expected.
 */
TEMPLATE_TEST_CASE("CD_SoftmaxRegressionFunctionPartialGradient", "[CD]",
    arma::mat)
{
  typedef typename TestType::elem_type ElemType;

  const size_t points = 1000;
  const size_t inputSize = 10;
  const size_t numClasses = 5;

  // Initialize a random dataset.
  TestType data;
  data.randu(inputSize, points);

  // Create random class labels.
  arma::Row<size_t> labels = arma::randi<arma::Row<size_t> >(
      points, arma::distr_param(0, numClasses - 1));

  // 2 objects for 2 terms in the cost function. Each term contributes towards
  // the gradient and thus need to be checked independently.
  SoftmaxRegressionFunction srf(data, labels, numClasses, 0);

  // Create a random set of parameters.
  TestType parameters;
  parameters.randu(numClasses, inputSize);

  // Get gradients for the current parameters.
  TestType gradient;
  srf.Gradient(parameters, gradient);

  // For each parameter.
  for (size_t j = 0; j < inputSize; j++)
  {
    // Get the gradient for this feature.
    arma::SpMat<ElemType> fGrad;

    srf.PartialGradient(parameters, j, fGrad);

    CheckMatrices(TestType(gradient.col(j)), TestType(fGrad.col(j)));
  }
}
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

TEMPLATE_TEST_CASE("PreCalcSCDTest", "[SCD]", arma::mat, arma::fmat)
{
  TestType predictors("0 0 0.4; 0 0 0.6; 0 0.3 0; 0.2 0 0; 0.2 -0.5 0;");
  arma::Row<size_t> responses("1  1  0;");

  LogisticRegressionFunction<TestType, arma::Row<size_t>> f(
      predictors, responses, 0.0001);

  SCD<> s(0.02, 60000, 1e-5);
  TestType iterate = f.GetInitialPoint();

  double objective = s.Optimize(f, iterate);

  REQUIRE(objective <= 0.055);
}

/* TEST_CASE("DisjointFeatureTest", "[SCDTest]") */
/* { */
/*   // The test function for parallel SGD should work with SCD, as the gradients */
/*   // of the individual functions are projections into the ith dimension. */
/*   SCD<> s(0.4); */
/*   FunctionTest<SparseTestFunction>(s, 0.01, 0.001); */
/* } */

/* TEST_CASE("DisjointFeatureFMatTest", "[SCDTest]") */
/* { */
/*   // The test function for parallel SGD should work with SCD, as the gradients */
/*   // of the individual functions are projections into the ith dimension. */
/*   SCD<> s(0.4); */
/*   FunctionTest<SparseTestFunction, arma::fmat>(s, 0.2, 0.02); */
/* } */

/* TEST_CASE("DisjointFeatureSpMatTest", "[SCDTest]") */
/* { */
/*   // The test function for parallel SGD should work with SCD, as the gradients */
/*   // of the individual functions are projections into the ith dimension. */
/*   SCD<> s(0.4); */
/*   FunctionTest<SparseTestFunction, arma::sp_mat>(s, 0.01, 0.001); */
/* } */

/* TEST_CASE("GreedyDescentTest", "[SCDTest]") */
/* { */
/*   // In the sparse test function, the given point has the maximum gradient at */
/*   // the feature with index 2. */
/*   arma::mat point("1; 2; 3; 4;"); */

/*   SparseTestFunction f; */

/*   GreedyDescent descentPolicy; */

/*   REQUIRE(descentPolicy.DescentFeature<SparseTestFunction, */
/*                                        arma::mat, */
/*                                        arma::mat>(0, point, f) == 2); */

/*   // Changing the point under consideration, so that the maximum gradient is at */
/*   // index 1. */
/*   point(1) = 10; */

/*   REQUIRE(descentPolicy.DescentFeature<SparseTestFunction, */
/*                                        arma::mat, */
/*                                        arma::mat>(0, point, f) == 1); */
/* } */

/* TEST_CASE("CyclicDescentTest", "[SCDTest]") */
/* { */
/*   const size_t features = 10; */
/*   struct DummyFunction */
/*   { */
/*     static size_t NumFeatures() */
/*     { */
/*       return features; */
/*     } */
/*   }; */

/*   DummyFunction dummy; */

/*   CyclicDescent descentPolicy; */

/*   for (size_t i = 0; i < 15; ++i) */
/*   { */
/*     REQUIRE(descentPolicy.DescentFeature<DummyFunction, arma::mat, arma::mat>( */
/*         i, arma::mat(), dummy) == (i % features)); */
/*   } */
/* } */

/* TEST_CASE("RandomDescentTest", "[SCDTest]") */
/* { */
/*   const size_t features = 10; */
/*   struct DummyFunction */
/*   { */
/*     static size_t NumFeatures() */
/*     { */
/*       return features; */
/*     } */
/*   }; */

/*   DummyFunction dummy; */

/*   CyclicDescent descentPolicy; */

/*   for (size_t i = 0; i < 100; ++i) */
/*   { */
/*     size_t j = descentPolicy.DescentFeature<DummyFunction, */
/*                                             arma::mat, */
/*                                             arma::mat>(i, arma::mat(), dummy); */
/*     REQUIRE(j < features); */
/*     REQUIRE(j >= 0); */
/*   } */
/* } */

/* TEST_CASE("LogisticRegressionFunctionPartialGradientTest", "[SCDTest]") */
/* { */
/*   // Evaluate the gradient and feature gradient and equate. */
/*   arma::mat predictors("0 0 0.4; 0 0 0.6; 0 0.3 0; 0.2 0 0; 0.2 -0.5 0;"); */
/*   arma::Row<size_t> responses("1  1  0;"); */

/*   LogisticRegressionFunction<arma::mat> f(predictors, responses, 0.0001); */

/*   arma::mat testPoint(1, f.NumFeatures(), arma::fill::randu); */

/*   arma::mat testGradient; */

/*   f.Gradient(testPoint, testGradient); */

/*   for (size_t i = 0; i < f.NumFeatures(); ++i) */
/*   { */
/*     arma::sp_mat fGrad; */
/*     f.PartialGradient(testPoint, i, fGrad); */

/*     CheckMatrices(arma::mat(testGradient.col(i)), arma::mat(fGrad.col(i))); */
/*   } */
/* } */

/* TEST_CASE("SoftmaxRegressionFunctionPartialGradientTest", "[SCDTest]") */
/* { */
/*   const size_t points = 1000; */
/*   const size_t inputSize = 10; */
/*   const size_t numClasses = 5; */

/*   // Initialize a random dataset. */
/*   arma::mat data; */
/*   data.randu(inputSize, points); */

/*   // Create random class labels. */
/*   arma::Row<size_t> labels = arma::randi<arma::Row<size_t> >( */
/*       points, arma::distr_param(0, numClasses - 1)); */

/*   // 2 objects for 2 terms in the cost function. Each term contributes towards */
/*   // the gradient and thus need to be checked independently. */
/*   SoftmaxRegressionFunction srf(data, labels, numClasses, 0); */

/*   // Create a random set of parameters. */
/*   arma::mat parameters; */
/*   parameters.randu(numClasses, inputSize); */

/*   // Get gradients for the current parameters. */
/*   arma::mat gradient; */
/*   srf.Gradient(parameters, gradient); */

/*   // For each parameter. */
/*   for (size_t j = 0; j < inputSize; j++) */
/*   { */
/*     // Get the gradient for this feature. */
/*     arma::sp_mat fGrad; */

/*     srf.PartialGradient(parameters, j, fGrad); */

/*     CheckMatrices(arma::mat(gradient.col(j)), arma::mat(fGrad.col(j))); */
/*   } */
/* } */

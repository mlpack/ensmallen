/**
 * @file test_function_tools.hpp
 * @author Marcus Edel
 * @author Ryan Curtin
 * @author Conrad Sanderson
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_TESTS_TEST_FUNCTION_TOOLS_HPP
#define ENSMALLEN_TESTS_TEST_FUNCTION_TOOLS_HPP

#include "catch.hpp"

/**
 * Create the data for the a logistic regression test.
 *
 * @param data Matrix object to store the data into.
 * @param testData Matrix object to store the test data into.
 * @param shuffledData Matrix object to store the shuffled data into.
 * @param responses Matrix object to store the overall responses into.
 * @param testResponses Matrix object to store the test responses into.
 * @param shuffledResponses Matrix object to store the shuffled responses into.
 */
template<typename MatType>
inline void LogisticRegressionTestData(MatType& data,
                                       MatType& testData,
                                       MatType& shuffledData,
                                       arma::Row<size_t>& responses,
                                       arma::Row<size_t>& testResponses,
                                       arma::Row<size_t>& shuffledResponses)
{
  // Generate a two-Gaussian dataset.
  data = MatType(3, 1000);
  responses = arma::Row<size_t>(1000);
  for (size_t i = 0; i < 500; ++i)
  {
    // The first Gaussian is centered at (1, 1, 1) and has covariance I.
    data.col(i) = arma::randn<arma::Col<typename MatType::elem_type>>(3) +
        arma::Col<typename MatType::elem_type>("1.0 1.0 1.0");
    responses(i) = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    // The second Gaussian is centered at (9, 9, 9) and has covariance I.
    data.col(i) = arma::randn<arma::Col<typename MatType::elem_type>>(3) +
        arma::Col<typename MatType::elem_type>("9.0 9.0 9.0");
    responses(i) = 1;
  }

  // Shuffle the dataset.
  arma::uvec indices = arma::shuffle(arma::linspace<arma::uvec>(0,
      data.n_cols - 1, data.n_cols));
  shuffledData = MatType(3, 1000);
  shuffledResponses = arma::Row<size_t>(1000);
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    shuffledData.col(i) = data.col(indices(i));
    shuffledResponses(i) = responses[indices(i)];
  }

  // Create a test set.
  testData = MatType(3, 1000);
  testResponses = arma::Row<size_t>(1000);
  for (size_t i = 0; i < 500; ++i)
  {
    testData.col(i) = arma::randn<arma::Col<typename MatType::elem_type>>(3) +
        arma::Col<typename MatType::elem_type>("1.0 1.0 1.0");
    testResponses(i) = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    testData.col(i) = arma::randn<arma::Col<typename MatType::elem_type>>(3) +
        arma::Col<typename MatType::elem_type>("9.0 9.0 9.0");
    testResponses(i) = 1;
  }
}

// Check the values of two matrices.
template<typename MatType>
inline void CheckMatrices(const MatType& a,
                          const MatType& b,
                          double tolerance = 1e-5)
{
  REQUIRE(a.n_rows == b.n_rows);
  REQUIRE(a.n_cols == b.n_cols);

  for (size_t i = 0; i < a.n_elem; ++i)
  {
    if (std::abs(a(i)) < tolerance / 2)
      REQUIRE(b(i) == Approx(0.0).margin(tolerance / 2.0));
    else
      REQUIRE(a(i) == Approx(b(i)).epsilon(tolerance));
  }
}

template<typename FunctionType, typename OptimizerType, typename PointType>
bool TestOptimizer(FunctionType& f,
                   OptimizerType& optimizer,
                   PointType& point,
                   const PointType& expectedResult,
                   const double coordinateMargin,
                   const double expectedObjective,
                   const double objectiveMargin,
                   const bool mustSucceed = true)
{
  const double objective = optimizer.Optimize(f, point);

  if (mustSucceed)
  {
    REQUIRE(objective == Approx(expectedObjective).margin(objectiveMargin));
    for (size_t i = 0; i < point.n_elem; ++i)
    {
      REQUIRE(point[i] == Approx(expectedResult[i]).margin(coordinateMargin));
    }
  }
  else
  {
    if (objective != Approx(expectedObjective).margin(objectiveMargin))
      return false;

    for (size_t i = 0; i < point.n_elem; ++i)
    {
      if (point[i] != Approx(expectedResult[i]).margin(coordinateMargin))
        return false;
    }
  }

  return true;
}

// This runs a test multiple times, but does not do any special behavior between
// runs.
template<typename FunctionType, typename OptimizerType, typename PointType>
void MultipleTrialOptimizerTest(FunctionType& f,
                                OptimizerType& optimizer,
                                PointType& initialPoint,
                                const PointType& expectedResult,
                                const double coordinateMargin,
                                const double expectedObjective,
                                const double objectiveMargin,
                                const size_t trials = 1)
{
  for (size_t t = 0; t < trials; ++t)
  {
    PointType coordinates(initialPoint);

    // Only force success on the last trial.
    bool result = TestOptimizer(f, optimizer, coordinates, expectedResult,
        coordinateMargin, expectedObjective, objectiveMargin,
        (t == (trials - 1)));
    if (result && t != (trials - 1))
    {
      // Just make sure at least something was tested for reporting purposes.
      REQUIRE(result == true);
      return;
    }
  }
}

template<typename FunctionType,
         typename MatType = arma::mat,
         typename OptimizerType = ens::StandardSGD>
void FunctionTest(OptimizerType& optimizer,
                  const double objectiveMargin = 0.01,
                  const double coordinateMargin = 0.001,
                  const size_t trials = 1)
{
  FunctionType f;
  MatType initialPoint = f.template GetInitialPoint<MatType>();
  MatType expectedResult = f.template GetFinalPoint<MatType>();

  MultipleTrialOptimizerTest(f, optimizer, initialPoint, expectedResult,
      coordinateMargin, f.GetFinalObjective(), objectiveMargin, trials);
}

template<typename MatType = arma::mat, typename OptimizerType>
void LogisticRegressionFunctionTest(OptimizerType& optimizer,
                                    const double trainAccuracyTolerance,
                                    const double testAccuracyTolerance,
                                    const size_t trials = 1)
{
  // We have to generate new data for each trial, so we can't use
  // MultipleTrialOptimizerTest().
  MatType data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  for (size_t i = 0; i < trials; ++i)
  {
    LogisticRegressionTestData(data, testData, shuffledData,
        responses, testResponses, shuffledResponses);
    ens::test::LogisticRegression<MatType> lr(shuffledData, shuffledResponses,
        0.5);

    MatType coordinates = lr.GetInitialPoint();

    optimizer.Optimize(lr, coordinates);

    const double acc = lr.ComputeAccuracy(data, responses, coordinates);
    const double testAcc = lr.ComputeAccuracy(testData, testResponses,
        coordinates);

    // Provide a shortcut to try again if we're not on the last trial.
    if (i != (trials - 1))
    {
      if (acc != Approx(100.0).epsilon(trainAccuracyTolerance))
        continue;
      if (testAcc != Approx(100.0).epsilon(testAccuracyTolerance))
        continue;
    }

    REQUIRE(acc == Approx(100.0).epsilon(trainAccuracyTolerance));
    REQUIRE(testAcc == Approx(100.0).epsilon(testAccuracyTolerance));
    break;
  }
}

#endif

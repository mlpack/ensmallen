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
#include "test_types.hpp"

namespace ens {
namespace test {

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
template<typename MatType, typename LabelsType>
inline void LogisticRegressionTestData(
    MatType& data,
    MatType& testData,
    LabelsType& responses,
    LabelsType& testResponses,
    const typename std::enable_if_t<!IsSparseMatrixType<MatType>::value>* = 0)
{
  // Generate a two-Gaussian dataset.
  data.set_size(3, 1000);
  responses.set_size(1000);

  // The first Gaussian is centered at (1, 1, 1) and has covariance I.
  data.cols(0, 499) = randn<MatType>(3, 500) + 1;
  responses.subvec(0, 499).zeros();

  // The second Gaussian is centered at (9, 9, 9) and has covariance I.
  data.cols(500, 999) = randn<MatType>(3, 500) + 9;
  responses.subvec(500, 999).ones();

  // Create a test set.
  testData.set_size(3, 1000);
  testResponses.set_size(1000);

  testData.cols(0, 499) = randn<MatType>(3, 500) + 1;
  testResponses.subvec(0, 499).zeros();
  testData.cols(500, 999) = randn<MatType>(3, 500) + 9;
  testResponses.subvec(500, 999).ones();
}

template<typename MatType, typename LabelsType>
inline void LogisticRegressionTestData(
    MatType& data,
    MatType& testData,
    LabelsType& responses,
    LabelsType& testResponses,
    const typename std::enable_if_t<IsSparseMatrixType<MatType>::value>* = 0)
{
  arma::Mat<typename MatType::elem_type> tmpData, tmpTestData;
  arma::Row<typename MatType::elem_type> tmpResponses, tmpTestResponses;

  // Sparse matrices don't support the necessary functionality with randn<>.
  LogisticRegressionTestData(tmpData, tmpTestData, tmpResponses,
      tmpTestResponses);

  data = conv_to<MatType>::from(tmpData);
  responses = conv_to<LabelsType>::from(tmpResponses);
  testData = conv_to<MatType>::from(tmpTestData);
  testResponses = conv_to<LabelsType>::from(tmpTestResponses);
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

  typedef typename PointType::elem_type eT;

  if (mustSucceed)
  {
    REQUIRE(objective == Approx(expectedObjective).margin(objectiveMargin));
    for (size_t i = 0; i < point.n_elem; ++i)
    {
      REQUIRE(eT(point[i]) ==
          Approx(expectedResult[i]).margin(coordinateMargin));
    }
  }
  else
  {
    if (objective != Approx(expectedObjective).margin(objectiveMargin))
      return false;

    for (size_t i = 0; i < point.n_elem; ++i)
    {
      if (eT(point[i]) != Approx(expectedResult[i]).margin(coordinateMargin))
        return false;
    }
  }

  return true;
}

// This runs a test multiple times, but does not do any special behavior between
// runs.
template<typename FunctionType, typename OptimizerType, typename PointType>
void MultipleTrialOptimizerTest(
    FunctionType& f,
    OptimizerType& optimizer,
    PointType& initialPoint,
    const PointType& expectedResult,
    const typename PointType::elem_type coordinateMargin,
    const typename PointType::elem_type expectedObjective,
    const typename PointType::elem_type objectiveMargin,
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
                  const typename MatType::elem_type objectiveMargin =
                      typename MatType::elem_type(0.01),
                  const typename MatType::elem_type coordinateMargin =
                      typename MatType::elem_type(0.001),
                  const size_t trials = 1)
{
  FunctionType f;
  MatType initialPoint = f.template GetInitialPoint<MatType>();
  MatType expectedResult = f.template GetFinalPoint<MatType>();

  MultipleTrialOptimizerTest(f, optimizer, initialPoint, expectedResult,
      coordinateMargin, typename MatType::elem_type(f.GetFinalObjective()),
      objectiveMargin, trials);
}

template<typename MatType = arma::mat, typename LabelsType = arma::Row<size_t>,
    typename OptimizerType>
void LogisticRegressionFunctionTest(
    OptimizerType& optimizer,
    const double trainAccuracyTolerance = Tolerances<MatType>::LRTrainAcc,
    const double testAccuracyTolerance = Tolerances<MatType>::LRTestAcc,
    const size_t trials = 1)
{
  // We have to generate new data for each trial, so we can't use
  // MultipleTrialOptimizerTest().
  MatType data, testData;
  LabelsType responses, testResponses;

  for (size_t i = 0; i < trials; ++i)
  {
    LogisticRegressionTestData(data, testData, responses, testResponses);

    MatType data2 = data;
    LabelsType responses2 = responses;
    ens::test::LogisticRegressionFunction<MatType> lr(data2, responses2, 0.5);
    lr.Shuffle(); // We didn't shuffle the data earlier.

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

} // namespace test
} // namespace ens

#endif

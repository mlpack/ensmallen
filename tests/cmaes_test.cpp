/**
 * @file cmaes_test.cpp
 * @author Marcus Edel
 * @author Kartik Nighania
 * @author Conrad Sanderson
 * @author John Phan
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

template<typename MatType = arma::mat, typename OptimizerType>
void LogisticRegressionFunctionTestReal(OptimizerType& optimizer,
                                        const double trainAccuracyTolerance,
                                        const double testAccuracyTolerance,
                                        const std::string dataset_path,
                                        const size_t trials = 1,
                                        const double split_ratio = 0.8)
{
  MatType data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  // Load the dataset
  if (data.load(dataset_path, arma::csv_ascii) == false)
  {
    FAIL("couldn't load data");
    return;
  }
  
  data = data.t();
  responses = arma::conv_to<arma::Row<size_t>>::from(data.row(0));
  data = data.rows(1, data.n_rows-1);

  arma::uvec indices = arma::shuffle(arma::linspace<arma::uvec>(0, 
      data.n_cols - 1, data.n_cols)); 
  size_t idx = data.n_cols * split_ratio;

  shuffledData = MatType(data.n_rows, idx);
  shuffledResponses = arma::Row<size_t>(idx);

  testData = MatType(data.n_rows, data.n_cols-idx);
  testResponses = arma::Row<size_t>(data.n_cols-idx);

  for (size_t i = 0; i < idx; ++i)
  {
    shuffledData.col(i) = data.col(indices(i));
    shuffledResponses.col(i) = responses.col(indices(i));
  }
  
  for (size_t i = idx; i < data.n_cols; ++i)
  {
    testData.col(i-idx) = data.col(indices(i));
    testResponses.col(i-idx) = responses.col(indices(i));
  }

  ens::test::LogisticRegression<MatType> lr(data, responses, 0.5);

  for (size_t i = 0; i < trials; ++i)
  {
    MatType coordinates = lr.GetInitialPoint();
    optimizer.Optimize(lr, coordinates);

    const double acc = lr.ComputeAccuracy(shuffledData, shuffledResponses, 
        coordinates);
    const double testAcc = lr.ComputeAccuracy(testData, testResponses,
        coordinates);

    // Provide a shortcut to try again if we're not on the last trial.
    if (i != (trials - 1))
    {
      if (acc != Approx(60.393258427).epsilon(trainAccuracyTolerance))
        continue;
      if (testAcc != Approx(66.4804).epsilon(testAccuracyTolerance))
        continue;
    }
    REQUIRE(acc == Approx(60.393258427).epsilon(trainAccuracyTolerance));
    REQUIRE(testAcc == Approx(66.4804).epsilon(testAccuracyTolerance));
    break;
  }
}

template<typename MatType = arma::mat, typename OptimizerType>
void LogisticRegressionFunctionTestRealM(OptimizerType& optimizer,
                                        const std::string dataset_path,
                                        const size_t dim = 200,
                                        const size_t trials = 1,
                                        const double split_ratio = 0.8)
{
  MatType data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  // Load the dataset
  if (data.load(dataset_path, arma::csv_ascii) == false)
  {
    FAIL("couldn't load data");
    return;
  }
  
  data = data.t();
  size_t n_row = data.n_rows;
  std::cout << data.n_rows << " " << data.n_cols << std::endl;
  responses = arma::conv_to<arma::Row<size_t>>::from(data.row(0));
  data = data.rows(1, std::min(dim, n_row)-1);
  arma::uvec indices = arma::shuffle(arma::linspace<arma::uvec>(0, 
      data.n_cols - 1, data.n_cols)); 
  size_t idx = data.n_cols * split_ratio;

  shuffledData = MatType(data.n_rows, idx);
  shuffledResponses = arma::Row<size_t>(idx);

  testData = MatType(data.n_rows, data.n_cols-idx);
  testResponses = arma::Row<size_t>(data.n_cols-idx);

  for (size_t i = 0; i < idx; ++i)
  {
    shuffledData.col(i) = data.col(indices(i));
    shuffledResponses.col(i) = responses.col(indices(i));
  }
  
  for (size_t i = idx; i < data.n_cols; ++i)
  {
    testData.col(i-idx) = data.col(indices(i));
    testResponses.col(i-idx) = responses.col(indices(i));
  }

  ens::test::LogisticRegression<MatType> lr(data, responses, 0.5);
  arma::wall_clock clock;
  double bestT = 0.0; double bestAcc = 0.0; double bestTestAcc = 0.0;
  for (size_t i = 0; i < trials; ++i)
  {
    double lapsT = 0.0;
    MatType coordinates = lr.GetInitialPoint();
    clock.tic();
    optimizer.Optimize(lr, coordinates);
    lapsT += clock.toc();
    const double acc = lr.ComputeAccuracy(shuffledData, shuffledResponses, 
        coordinates);
    const double testAcc = lr.ComputeAccuracy(testData, testResponses,
        coordinates);

    bestAcc = acc; bestTestAcc = testAcc; bestT = lapsT;    
    // Provide a shortcut to try again if we're not on the last trial.
    if (acc < bestAcc)
      continue;
    if (testAcc < bestTestAcc)
      continue;
    bestAcc = acc; bestTestAcc = testAcc; bestT = lapsT;    
  }
  std::cout << "following results: " << std::endl;
  std::cout << bestAcc << " " << bestTestAcc << " " << bestT << std::endl;
}
/**
 * Run CMA-ES with the full selection policy on logistic regression and
 * make sure the results are acceptable.
 */
TEST_CASE("CMAESLogisticRegressionTest", "[CMAESTest]")
{
  CMAES<> cmaes(0, -1, 1, 32, 200, 1e-3);
  LogisticRegressionFunctionTest(cmaes, 0.003, 0.006, 5);
}

/**
 * Run CMA-ES with the random selection policy on logistic regression and
 * make sure the results are acceptable.
 */
TEST_CASE("ApproxCMAESLogisticRegressionTest", "[CMAESTest]")
{
  ApproxCMAES cmaes(0, -1, 1, 32, 200, 1e-3);
  LogisticRegressionFunctionTest(cmaes, 0.003, 0.006, 5);
}

/**
 * Run Active CMA-ES with the full selection policy on logistic regression and
 * make sure the results are acceptable.
 */
TEST_CASE("ActiveCMAESLogisticRegressionTest", "[CMAESTest]")
{
  ActiveCMAES cmaes(0, -1, 1, 32, 200, 1e-3);
  LogisticRegressionFunctionTest(cmaes, 0.003, 0.006, 5);
}

/**
 * Run Sep-CMA-ES with the full selection policy on logistic regression and
 * make sure the results are acceptable.
 */
TEST_CASE("SepCMAESLogisticRegressionTest", "[CMAESTest]")
{
  SepCMAES cmaes(0, -1, 1, 32, 200, 1e-3);
  LogisticRegressionFunctionTest(cmaes, 0.003, 0.006, 5);
}

/**
 * Run Vd-CMA-ES with the full selection policy on logistic regression and
 * make sure the results are acceptable.
 */
TEST_CASE("VDCMAESLogisticRegressionTest", "[CMAESTest]")
{
  VDCMAES cmaes(0, -1, 1, 32, 200, 1e-3);
  LogisticRegressionFunctionTest(cmaes, 0.003, 0.006, 5);
}

/**
 * Run CMA-ES with the full selection policy on titanic dataset and 
 * using logistic regression algorithm
 * 
 * See more on dataset at https://www.kaggle.com/competitions/titanic/data
 * Make sure the results are acceptable.
 */
TEST_CASE("CMAESTitanicTest", "[CMAESTest]")
{
  CMAES<> cmaes(0, -1, 1, 32, 200, 1e-3);
  LogisticRegressionFunctionTestReal(cmaes, 0.003, 0.006, "data/titanic.csv", 5);
}

/**
 * Run CMA-ES with the random selection policy on titanic dataset and 
 * using logistic regression algorithm
 * 
 * See more on dataset at https://www.kaggle.com/competitions/titanic/data
 * Make sure the results are acceptable.
 */
TEST_CASE("ApproxCMAESTitanicTest", "[CMAESTest]")
{
  ApproxCMAES cmaes(0, -1, 1, 32, 200, 1e-3);
  LogisticRegressionFunctionTestReal(cmaes, 0.003, 0.006, "data/titanic.csv", 5);
}

/**
 * Run Active CMA-ES with the full selection policy on titanic dataset and 
 * using logistic regression algorithm
 * 
 * See more on dataset at https://www.kaggle.com/competitions/titanic/data
 * Make sure the results are acceptable.
 */
TEST_CASE("ActiveCMAESTitanicTest", "[CMAESTest]")
{
  ActiveCMAES cmaes(0, -1, 1, 32, 200, 1e-3);
  LogisticRegressionFunctionTestReal(cmaes, 0.003, 0.006, "data/titanic.csv", 5);
}

/**
 * Run Sep-CMA-ES with the full selection policy on titanic dataset and 
 * using logistic regression algorithm
 * 
 * See more on dataset at https://www.kaggle.com/competitions/titanic/data
 * Make sure the results are acceptable.
 */
TEST_CASE("SepCMAESTitanicTest", "[CMAESTest]")
{
  SepCMAES cmaes(0, -1, 1, 32, 200, 1e-3);
  LogisticRegressionFunctionTestReal(cmaes, 0.003, 0.006, "data/titanic.csv", 5);
}

/**
 * Run Vd-CMA-ES with the full selection policy on titanic dataset and 
 * using logistic regression algorithm
 * 
 * See more on dataset at https://www.kaggle.com/competitions/titanic/data
 * Make sure the results are acceptable.
 */
TEST_CASE("VDCMAESTitanicTest", "[CMAESTest]")
{
  VDCMAES cmaes(0, -1, 1, 32, 200, 1e-3);
  LogisticRegressionFunctionTestReal(cmaes, 0.003, 0.006, "data/titanic.csv", 5);
}


// Benchmarking purposes
TEST_CASE("CMAESGisetteTest", "[CMAESTest]")
{
  CMAES<> cmaes(0, -1, 1, 32, 200, 1e-3);
  LogisticRegressionFunctionTestRealM(cmaes, "data/gisette.csv", 5000, 1);
}
TEST_CASE("ApproxCMAESGisetteTest", "[CMAESTest]")
{
  ApproxCMAES cmaes(0, -1, 1, 32, 200, 1e-3);
  LogisticRegressionFunctionTestRealM(cmaes, "data/gisette.csv", 5000, 1);
}
TEST_CASE("ActiveCMAESGisetteTest", "[CMAESTest]")
{
  ActiveCMAES cmaes(0, -1, 1, 32, 200, 1e-3);
  LogisticRegressionFunctionTestRealM(cmaes, "data/gisette.csv", 5000, 1);
}
TEST_CASE("ActiveApproxCMAESGisetteTest", "[CMAESTest]")
{
  ActiveApproxCMAES cmaes(0, -1, 1, 32, 200, 1e-3);
  LogisticRegressionFunctionTestRealM(cmaes, "data/gisette.csv", 5000, 1);
}
TEST_CASE("SepCMAESGisetteTest", "[CMAESTest]")
{
  SepCMAES cmaes(0, -1, 1, 32, 200, 1e-3);
  LogisticRegressionFunctionTestRealM(cmaes, "data/gisette.csv", 5000, 1);
}
TEST_CASE("VDCMAESGisetteTest", "[CMAESTest]")
{
  VDCMAES cmaes(0, -1, 1, 32, 200, 1e-3);
  LogisticRegressionFunctionTestRealM(cmaes, "data/gisette.csv", 5000, 1);
}
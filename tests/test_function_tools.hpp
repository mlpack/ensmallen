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

#include <ensmallen.hpp>
#include "catch.hpp"

using namespace ens;
using namespace ens::test;

/**
 * The National Institute of Standards and Technology has released a set of
 * problems to test non-linear least squares solvers. More information about the
 * background on these problems and suggested evaluation methodology can be
 * found at:
 *
 *    http://www.itl.nist.gov/div898/strd/nls/nls_info.shtml
 *
 * The problem data themselves can be found at:
 *
 *    http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml
 *
 * The problems are divided into three levels of difficulty, Lower,
 * Average and Higher. For each problem there are two starting guesses,
 * the first one far away from the global minimum and the second
 * closer to it.
 *
 * @tparam OptimizerType The optimizer type used to optimize the
 *    Non-linear Least Square problems.
 */
template<typename OptimizerType>
class NISTProblems
{
 public:
  /**
   * NISTProblems object instantiation with the given optimizer.
   *
   * @param optimizer The optimizer that is used to solve the NIST problems.
   */
  NISTProblems(OptimizerType& optimizer) : optimizer(optimizer)
  {
    /* Nothing to do here. */
  }

  /**
   * Evaluate the optimizer on the NIST Non-linear Least Square problems.
   *
   * @return Number of successfully solved starting values and the log relative
   *    error.
   */
  arma::vec Evaluate()
  {
    arma::vec results(3 + 3);

    // Level of difficulty: Lower.
    arma::vec lowerResults = arma::zeros<arma::vec>(2);
    lowerResults += Evaluate<Misra1a>("data/misra1a.csv",
        arma::mat("2.3894212918E+02, 5.5015643181E-04"));
    lowerResults += Evaluate<Chwirut2>("data/chwirut2.csv",
        arma::mat("1.6657666537E-01, 5.1653291286E-03, 1.2150007096E-02"));
    lowerResults += Evaluate<Chwirut1>("data/chwirut1.csv",
        arma::mat("1.9027818370E-01, 6.1314004477E-03, 1.0530908399E-02"));
    lowerResults += Evaluate<Lanczos3>("data/lanczos3.csv",
        arma::mat("8.6816414977E-02, 9.5498101505E-01, 8.4400777463E-01, \
                   2.9515951832E+00, 1.5825685901E+00, 4.9863565084E+00"));
    lowerResults += Evaluate<Gauss1>("data/gauss1.csv",
        arma::mat("9.8778210871E+01, 1.0497276517E-02, 1.0048990633E+02, \
                   6.7481111276E+01, 2.3129773360E+01, 7.1994503004E+01, \
                   1.7899805021E+02, 1.8389389025E+01"));
    lowerResults += Evaluate<Gauss2>("data/gauss2.csv",
        arma::mat("9.9018328406E+01, 1.0994945399E-02, 1.0188022528E+02, \
                   1.0703095519E+02, 2.3578584029E+01, 7.2045589471E+01, \
                   1.5327010194E+02, 1.9525972636E+01"));
    lowerResults += Evaluate<DanWood>("data/danWood.csv",
        arma::mat("7.6886226176E-01, 3.8604055871E+00"));
    lowerResults += Evaluate<Misra1b>("data/misra1b.csv",
        arma::mat("3.3799746163E+02, 3.9039091287E-04"));

    // Level of difficulty: Average.
    arma::vec averageResults = arma::zeros<arma::vec>(2);
    averageResults += Evaluate<Kirby2>("data/kirby2.csv",
        arma::mat("1.6745063063E+00, -1.3927397867E-01, 2.5961181191E-03, \
                   -1.7241811870E-03, 2.1664802578E-05"));
    averageResults += Evaluate<Hahn1>("data/hahn1.csv",
        arma::mat("1.0776351733E+00, -1.2269296921E-01, 4.0863750610E-03, \
                   -1.4262662514E-06, -5.7609940901E-03, 2.4053735503E-04, \
                   -1.2314450199E-07"));
    averageResults += Evaluate<Nelson>("data/nelson.csv",
        arma::mat("2.5906836021E+00, 5.6177717026E-09, -5.7701013174E-02"));
    averageResults += Evaluate<MGH17>("data/mgh17.csv",
        arma::mat("3.7541005211E-01, 1.9358469127E+00, -1.4646871366E+00, \
                   1.2867534640E-02, 2.2122699662E-02"));
    averageResults += Evaluate<Lanczos1>("data/lanczos1.csv",
        arma::mat("9.5100000027E-02, 1.0000000001E+00, 8.6070000013E-01, \
                   3.0000000002E+00, 1.5575999998E+00, 5.0000000001E+00"));
    averageResults += Evaluate<Lanczos2>("data/lanczos2.csv",
        arma::mat("9.6251029939E-02, 1.0057332849E+00, 8.6424689056E-01, \
                   3.0078283915E+00, 1.5529016879E+00, 5.0028798100E+00"));
    averageResults += Evaluate<Gauss3>("data/gauss3.csv",
        arma::mat("9.8940368970E+01, 1.0945879335E-02, 1.0069553078E+02, \
                   1.1163619459E+02, 2.3300500029E+01, 7.3705031418E+01, \
                   1.4776164251E+02, 1.9668221230E+01"));
    averageResults += Evaluate<Misra1c>("data/misra1c.csv",
        arma::mat("6.3642725809E+02, 2.0813627256E-04"));
    averageResults += Evaluate<Misra1d>("data/misra1d.csv",
        arma::mat("4.3736970754E+02, 3.0227324449E-04"));
    averageResults += Evaluate<Roszman1>("data/roszman1.csv",
        arma::mat("2.0196866396E-01, -6.1953516256E-06, 1.2044556708E+03, \
                   -1.8134269537E+02"));
    averageResults += Evaluate<ENSO>("data/enso.csv",
        arma::mat("1.0510749193E+01, 3.0762128085E+00, 5.3280138227E-01, \
                   4.4311088700E+01, -1.6231428586E+00, 5.2554493756E-01, \
                   2.6887614440E+01, 2.1232288488E-01, 1.4966870418E+00"));

    // Level of difficulty: Higher.
    arma::vec higherResults = arma::zeros<arma::vec>(2);
    higherResults += Evaluate<MGH09>("data/mgh09.csv",
        arma::mat("1.9280693458E-01, 1.9128232873E-01, 1.2305650693E-01, \
                   1.3606233068E-01"));
    higherResults += Evaluate<Thurber>("data/thurber.csv",
        arma::mat("1.2881396800E+03, 1.4910792535E+03, 5.8323836877E+02, \
                   7.5416644291E+01, 9.6629502864E-01, 3.9797285797E-01, \
                   4.9727297349E-02"));
    higherResults += Evaluate<BoxBOD>("data/boxBOD.csv",
        arma::mat("2.1380940889E+02, 5.4723748542E-01"));
    higherResults += Evaluate<Rat42>("data/rat42.csv",
        arma::mat("7.2462237576E+01, 2.6180768402E+00, 6.7359200066E-02"));
    higherResults += Evaluate<MGH10>("data/mgh10.csv",
        arma::mat("5.6096364710E-03, 6.1813463463E+03, 3.4522363462E+02"));
    higherResults += Evaluate<Eckerle4>("data/eckerle4.csv",
        arma::mat("1.5543827178E+00, 4.0888321754E+00, 4.5154121844E+02"));
    higherResults += Evaluate<Rat43>("data/rat43.csv",
        arma::mat("6.9964151270E+02, 5.2771253025E+00, 7.5962938329E-01, \
                   1.2792483859E+00"));
    higherResults += Evaluate<Bennett5>("data/bennett5.csv",
        arma::mat("-2.5235058043E+03, 4.6736564644E+010, 9.3218483193E-01"));

    results(0) = lowerResults(0);
    results(1) = lowerResults(1) / lowerResults(0);

    results(2) = averageResults(0);
    results(3) = averageResults(0) > 0 ? averageResults(1) /
        averageResults(0) : 0;

    results(4) = higherResults(0);
    results(5) = higherResults(0) > 0 ? higherResults(1) /
        higherResults(0) : 0;

    Info << "Level of Difficulty: Lower solved: " << lowerResults(0)
        << " LRE: " << lowerResults(1) << std::endl;
    Info << "Level of Difficulty: Average solved: " << averageResults(0)
        << " LRE: " << averageResults(1) << std::endl;
    Info << "Level of Difficulty: Higher solved: " << higherResults(0)
        << " LRE: " << higherResults(1) << std::endl;

    return results;
  }

 private:
  //! The locally stored optimizer.
  OptimizerType optimizer;

  /**
   * Evaluate the optimizer on the given function.
   *
   * @tparam FuctionType The function type of the Non-linear Least Square
   *    problem.
   * @param filename The filename that holds the Non-linear Least Squares
   *    problem predictions and responses.
   * @param expected The expected output.
   * @return Number of successfully solved starting values and the log relative
   *    error.
   */
  template<typename FuctionType>
  arma::vec Evaluate(const std::string& filename, const arma::mat& expected)
  {
    arma::vec results = arma::zeros<arma::vec>(2);
    size_t result = 0;

    arma::mat data;
    data.load(filename, arma::csv_ascii);
    inplace_trans(data);

    const arma::mat predictors = data.rows(1, data.n_rows - 1);
    const arma::rowvec responses = data.row(0);

    NIST<FuctionType> function(predictors, responses);

    for (size_t i = 0; i < function.Function().Functions(); ++i)
    {
      arma::mat coordinates = function.GetInitialPoint(i);
      optimizer.Optimize(function, coordinates);

      const double lre = LogRelativeError(expected, coordinates);
      if (lre > 4)
      {
        results(0)++;
        results(1) += lre;
      }
    }

    const double lre = (results(0) > 0 ? (results(1) / results(0)) : 0);

    Info << "Problem: " << filename << " solved: " << results(0) << " of "
        << function.Function().Functions() << " LRE: " << lre << std::endl;

    return results;
  }

  /**
   * Compute the log relative error by comparing each component of the solution
   * with the ground truth, and taking the minimum.
   *
   * @param expected The expected output.
   * @param actual The actual calculated output by the optimizer.
   * @return The log relative error.
   */
  double LogRelativeError(const arma::mat& expected, const arma::mat& actual)
  {
    double lre = 12;
    for (size_t i = 0; i < expected.n_elem; ++i)
    {
      const double tlre = -std::log10(std::fabs(expected(i) - actual(i)) /
          std::fabs(expected(i)));

      lre = std::min(lre, std::max(0.0, std::min(11.0, tlre)));
    }

    return lre;
  }

};

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

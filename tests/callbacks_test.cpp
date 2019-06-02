/**
 * @file callbacks_test.cpp
 * @author Marcus Edel
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
using namespace ens::callbacks::traits;

/**
 * Utility class with no functions.
 */
class EmptyTestFunction { };

/**
 * Utility class with Evaluate().
 */
class EvaluateTestFunction
{
 public:
  template<typename OptimizerType, typename FunctionType, typename MatType>
  void Evaluate(OptimizerType& /* optimizer */,
                FunctionType& /* function */,
                const MatType& /* coordinates */,
                const double /* objective */)
  { }
};

/**
 * Utility class with Gradient().
 */
class GradientTestFunction
{
 public:
  template<typename OptimizerType,
           typename FunctionType,
           typename MatType,
           typename GradType>
  void Gradient(OptimizerType& /* optimizer */,
                FunctionType& /* function */,
                const MatType& /* coordinates */,
                const GradType& /* objective */)
  { }
};

/**
 * Utility class with BeginEpoch().
 */
class BeginEpochTestFunction
{
 public:
  template<typename OptimizerType, typename FunctionType, typename MatType>
  void BeginEpoch(OptimizerType& /* optimizer */,
                FunctionType& /* function */,
                const MatType& /* coordinates */,
                const size_t /* epoch */,
                const double /* objective */)
  { }
};

/**
 * Utility class with EndEpoch().
 */
class EndEpochTestFunction
{
 public:
  template<typename OptimizerType, typename FunctionType, typename MatType>
  void EndEpoch(OptimizerType& /* optimizer */,
                FunctionType& /* function */,
                const MatType& /* coordinates */,
                const size_t /* epoch */,
                const double /* objective */)
  { }
};

/**
 * Utility class with Evaluate(), Gradient(), BeginEpoch(), EndEpoch().
 */
class CompleteTestFunction
{
 public:
  template<typename OptimizerType, typename FunctionType, typename MatType>
  void Evaluate(OptimizerType& /* optimizer */,
                FunctionType& /* function */,
                const MatType& /* coordinates */,
                const double /* objective */)
  { }

  template<typename OptimizerType,
           typename FunctionType,
           typename MatType,
           typename GradType>
  void Gradient(OptimizerType& /* optimizer */,
                FunctionType& /* function */,
                const MatType& /* coordinates */,
                const GradType& /* objective */)
  { }

  template<typename OptimizerType, typename FunctionType, typename MatType>
  void BeginEpoch(OptimizerType& /* optimizer */,
                FunctionType& /* function */,
                const MatType& /* coordinates */,
                const size_t /* epoch */,
                const double /* objective */)
  { }

  template<typename OptimizerType, typename FunctionType, typename MatType>
  void EndEpoch(OptimizerType& /* optimizer */,
                FunctionType& /* function */,
                const MatType& /* coordinates */,
                const size_t /* epoch */,
                const double /* objective */)
  { }
};

/**
 * Make sure that an empty class doesn't have any callbacks.
 */
TEST_CASE("CallbacksEmptyTest", "[CallbacksTest]")
{
  const bool hasEvaluate =  HasEvaluate<EmptyTestFunction,
      TypedForms<StandardSGD, EmptyTestFunction,
      arma::mat>::template EvaluateForm>::value;

  const bool hasGradient =  HasGradient<EmptyTestFunction,
      TypedForms<StandardSGD, EmptyTestFunction,
      arma::mat, arma::mat>::template GradientForm>::value;

  const bool hasBeginEpoch =  HasBeginEpoch<EmptyTestFunction,
      TypedForms<StandardSGD, EmptyTestFunction,
      arma::mat>::template BeginEpochForm>::value;

  const bool hasEndEpoch =  HasEndEpoch<EmptyTestFunction,
      TypedForms<StandardSGD, EmptyTestFunction,
      arma::mat>::template BeginEpochForm>::value;

  REQUIRE(hasEvaluate == false);
  REQUIRE(hasGradient == false);
  REQUIRE(hasBeginEpoch == false);
  REQUIRE(hasEndEpoch == false);
}

/**
 * Make sure we don't add any functions if we only have Evaluate().
 */
TEST_CASE("CallbacksEvaluateOnlyTest", "[CallbacksTest]")
{
  const bool hasEvaluate =  HasEvaluate<EvaluateTestFunction,
      TypedForms<StandardSGD, EvaluateTestFunction,
      arma::mat>::template EvaluateForm>::value;

  const bool hasGradient =  HasGradient<EvaluateTestFunction,
      TypedForms<StandardSGD, EvaluateTestFunction,
      arma::mat, arma::mat>::template GradientForm>::value;

  const bool hasBeginEpoch =  HasBeginEpoch<EvaluateTestFunction,
      TypedForms<StandardSGD, EvaluateTestFunction,
      arma::mat>::template BeginEpochForm>::value;

  const bool hasEndEpoch =  HasEndEpoch<EvaluateTestFunction,
      TypedForms<StandardSGD, EvaluateTestFunction,
      arma::mat>::template BeginEpochForm>::value;

  REQUIRE(hasEvaluate == true);
  REQUIRE(hasGradient == false);
  REQUIRE(hasBeginEpoch == false);
  REQUIRE(hasEndEpoch == false);
}

/**
 * Make sure we don't add any functions if we only have Gradient().
 */
TEST_CASE("CallbacksGradientOnlyTest", "[CallbacksTest]")
{
  const bool hasEvaluate =  HasEvaluate<GradientTestFunction,
      TypedForms<StandardSGD, GradientTestFunction,
      arma::mat>::template EvaluateForm>::value;

  const bool hasGradient =  HasGradient<GradientTestFunction,
      TypedForms<StandardSGD, GradientTestFunction,
      arma::mat, arma::mat>::template GradientForm>::value;

  const bool hasBeginEpoch =  HasBeginEpoch<GradientTestFunction,
      TypedForms<StandardSGD, GradientTestFunction,
      arma::mat>::template BeginEpochForm>::value;

  const bool hasEndEpoch =  HasEndEpoch<GradientTestFunction,
      TypedForms<StandardSGD, GradientTestFunction,
      arma::mat>::template BeginEpochForm>::value;

  REQUIRE(hasEvaluate == false);
  REQUIRE(hasGradient == true);
  REQUIRE(hasBeginEpoch == false);
  REQUIRE(hasEndEpoch == false);
}

/**
 * Make sure we don't add any functions if we only have BeginEpoch().
 */
TEST_CASE("CallbacksBeginEpochOnlyTest", "[CallbacksTest]")
{
  const bool hasEvaluate =  HasEvaluate<BeginEpochTestFunction,
      TypedForms<StandardSGD, BeginEpochTestFunction,
      arma::mat>::template EvaluateForm>::value;

  const bool hasGradient =  HasGradient<BeginEpochTestFunction,
      TypedForms<StandardSGD, BeginEpochTestFunction,
      arma::mat, arma::mat>::template GradientForm>::value;

  const bool hasBeginEpoch =  HasBeginEpoch<BeginEpochTestFunction,
      TypedForms<StandardSGD, BeginEpochTestFunction,
      arma::mat>::template BeginEpochForm>::value;

  const bool hasEndEpoch =  HasEndEpoch<BeginEpochTestFunction,
      TypedForms<StandardSGD, BeginEpochTestFunction,
      arma::mat>::template BeginEpochForm>::value;

  REQUIRE(hasEvaluate == false);
  REQUIRE(hasGradient == false);
  REQUIRE(hasBeginEpoch == true);
  REQUIRE(hasEndEpoch == false);
}

/**
 * Make sure we don't add any functions if we only have EndEpoch().
 */
TEST_CASE("CallbacksEndEpochOnlyTest", "[CallbacksTest]")
{
  const bool hasEvaluate =  HasEvaluate<EndEpochTestFunction,
      TypedForms<StandardSGD, EndEpochTestFunction,
      arma::mat>::template EvaluateForm>::value;

  const bool hasGradient =  HasGradient<EndEpochTestFunction,
      TypedForms<StandardSGD, EndEpochTestFunction,
      arma::mat, arma::mat>::template GradientForm>::value;

  const bool hasBeginEpoch =  HasBeginEpoch<EndEpochTestFunction,
      TypedForms<StandardSGD, EndEpochTestFunction,
      arma::mat>::template BeginEpochForm>::value;

  const bool hasEndEpoch =  HasEndEpoch<EndEpochTestFunction,
      TypedForms<StandardSGD, EndEpochTestFunction,
      arma::mat>::template BeginEpochForm>::value;

  REQUIRE(hasEvaluate == false);
  REQUIRE(hasGradient == false);
  REQUIRE(hasBeginEpoch == false);
  REQUIRE(hasEndEpoch == true);
}

/**
 * Make sure we have all callbacks.
 */
TEST_CASE("CallbacksCompleteFunctionsTest", "[CallbacksTest]")
{
  const bool hasEvaluate =  HasEvaluate<CompleteTestFunction,
      TypedForms<StandardSGD, CompleteTestFunction,
      arma::mat>::template EvaluateForm>::value;

  const bool hasGradient =  HasGradient<CompleteTestFunction,
      TypedForms<StandardSGD, CompleteTestFunction,
      arma::mat, arma::mat>::template GradientForm>::value;

  const bool hasBeginEpoch =  HasBeginEpoch<CompleteTestFunction,
      TypedForms<StandardSGD, CompleteTestFunction,
      arma::mat>::template BeginEpochForm>::value;

  const bool hasEndEpoch =  HasEndEpoch<CompleteTestFunction,
      TypedForms<StandardSGD, CompleteTestFunction,
      arma::mat>::template BeginEpochForm>::value;

  REQUIRE(hasEvaluate == true);
  REQUIRE(hasGradient == true);
  REQUIRE(hasBeginEpoch == true);
  REQUIRE(hasEndEpoch == true);
}

/**
 * Make sure the EarlyStopAtMinLoss callback will stop the optimization process.
 */
TEST_CASE("EarlyStopAtMinLossCallbackTest", "[CallbacksTest]")
{
  SGDTestFunction f;
  arma::mat coordinates = f.GetInitialPoint();

  // Instantiate the optimizer with a number of iterations that will take a
  // long time to finish.
  StandardSGD s(0.0003, 1, 10000000000, -10, true);

  // The optimization process should return in one second.
  const double result = s.Optimize(f, coordinates, EarlyStopAtMinLoss(100));

  REQUIRE(s.Terminate());
  REQUIRE(result == Approx(-1.0).epsilon(0.0005));
  REQUIRE(coordinates[0] == Approx(0.0).margin(1e-3));
  REQUIRE(coordinates[1] == Approx(0.0).margin(1e-7));
  REQUIRE(coordinates[2] == Approx(0.0).margin(1e-7));
}

/**
 * Make sure the PrintLoss callback will print the loss to the specified
 * output stream.
 */
TEST_CASE("PrintLossCallbackTest", "[CallbacksTest]")
{
  SGDTestFunction f;
  arma::mat coordinates = f.GetInitialPoint();

  StandardSGD s(0.0003, 1, 10, 1e-9, true);

  std::stringstream stream;
  s.Optimize(f, coordinates, PrintLoss(stream));

  REQUIRE(stream.str().length() > 0);
}

/**
 * Make sure the ProgressBar callback will show the progress on the specified
 * output stream.
 */
TEST_CASE("ProgressBarCallbackTest", "[CallbacksTest]")
{
  SGDTestFunction f;
  arma::mat coordinates = f.GetInitialPoint();

  StandardSGD s(0.0003, 1, 10, 1e-9, true);

  std::stringstream stream;
  s.Optimize(f, coordinates, ProgressBar(10, stream));

  REQUIRE(stream.str().length() > 0);
}

/**
 * Make sure the StoreBestCoordinates callback will store the best coordinates
 * and objetive.
 */
TEST_CASE("StoreBestCoordinatesCallbackTest", "[CallbacksTest]")
{
  SGDTestFunction f;
  arma::mat coordinates = f.GetInitialPoint();

  StandardSGD s(0.0003, 1, 5000000, 1e-9, true);

  StoreBestCoordinates<decltype(coordinates)> cb;
  const double result = s.Optimize(f, coordinates, cb);

  REQUIRE(cb.BestObjective() <= result);
  REQUIRE(cb.BestObjective() == Approx(-1.0).epsilon(0.0005));
  REQUIRE(cb.BestCoordinates()[0] == Approx(0.0).margin(1e-3));
  REQUIRE(cb.BestCoordinates()[1] == Approx(0.0).margin(1e-7));
}

/**
 * Make sure the TimerStop callback will stop the optimization process.
 */
TEST_CASE("TimerStopCallbackTest", "[CallbacksTest]")
{
  SGDTestFunction f;
  arma::mat coordinates = f.GetInitialPoint();

  // Instantiate the optimizer with a number of iterations that will take a
  // long time to finish.
  StandardSGD s(0.0003, 1, 10000000000, -10, true);

  arma::wall_clock timer;
  timer.tic();

  // The optimization process should return in one second.
  const double result = s.Optimize(f, coordinates, TimerStop(1));

  // Add some time to account for the function to return.
  REQUIRE(timer.toc() < 2);
}

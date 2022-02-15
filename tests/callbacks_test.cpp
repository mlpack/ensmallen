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
#include "test_function_tools.hpp"

using namespace ens;
using namespace ens::test;
using namespace ens::callbacks::traits;

/**
 * Utility class with Evaluate(), Gradient(), BeginEpoch(), EndEpoch(),
 * BeginOptimization(), EndOptimization(), EvaluateConstraint(),
 * GradientConstraint(), StepTaken.
 */
class CompleteCallbackTestFunction
{
 public:
  CompleteCallbackTestFunction() :
      calledEvaluate(false),
      calledGradient(false),
      calledBeginEpoch(false),
      calledEndEpoch(false),
      calledBeginOptimization(false),
      calledEndOptimization(false),
      calledEvaluateConstraint(false),
      calledGradientConstraint(false),
      calledStepTaken(false),
      calledGenerationalStepTaken(false)
  { }

  template<typename OptimizerType, typename FunctionType, typename MatType>
  void Evaluate(OptimizerType& /* optimizer */,
                FunctionType& /* function */,
                const MatType& /* coordinates */,
                const double /* objective */)
  { calledEvaluate = true; }

  template<typename OptimizerType,
           typename FunctionType,
           typename MatType,
           typename GradType>
  void Gradient(OptimizerType& /* optimizer */,
                FunctionType& /* function */,
                const MatType& /* coordinates */,
                GradType& /* objective */)
  { calledGradient = true; }

  template<typename OptimizerType, typename FunctionType, typename MatType>
  void BeginEpoch(OptimizerType& /* optimizer */,
                  FunctionType& /* function */,
                  const MatType& /* coordinates */,
                  const size_t /* epoch */,
                  const double /* objective */)
  { calledBeginEpoch = true; }

  template<typename OptimizerType, typename FunctionType, typename MatType>
  void EndEpoch(OptimizerType& /* optimizer */,
                FunctionType& /* function */,
                const MatType& /* coordinates */,
                const size_t /* epoch */,
                const double /* objective */)
  { calledEndEpoch = true; }

  template<typename OptimizerType, typename FunctionType, typename MatType>
  void BeginOptimization(OptimizerType& /* optimizer */,
                         FunctionType& /* function */,
                         MatType& /* coordinates */)
  { calledBeginOptimization = true; }

  template<typename OptimizerType, typename FunctionType, typename MatType>
  void EndOptimization(OptimizerType& /* optimizer */,
                       FunctionType& /* function */,
                       MatType& /* coordinates */)
  { calledEndOptimization = true; }

  template<typename OptimizerType, typename FunctionType, typename MatType>
  void EvaluateConstraint(OptimizerType& /* optimizer */,
                          FunctionType& /* function */,
                          const MatType& /* coordinates */,
                          const size_t /* constraint */,
                          const double /* constraintValue */)
  { calledEvaluateConstraint = true; }

  template<typename OptimizerType,
           typename FunctionType,
           typename MatType,
           typename GradType>
  void GradientConstraint(OptimizerType& /* optimizer */,
                          FunctionType& /* function */,
                          const MatType& /* coordinates */,
                          const size_t /* constraint */,
                          GradType& /* gradient */)
  { calledGradientConstraint = true; }

  template<typename OptimizerType, typename FunctionType, typename MatType>
  void StepTaken(OptimizerType& /* optimizer */,
                 FunctionType& /* function */,
                 MatType& /* coordinates */)
  { calledStepTaken = true; }

    template<typename OptimizerType,
             typename FunctionType,
             typename MatType,
             typename ObjectivesVecType,
             typename IndicesType>
    void GenerationalStepTaken(OptimizerType& /* optimizer */,
                               FunctionType& /* function */,
                               MatType& /* coordinates */,
                               ObjectivesVecType& /* objectives */,
                               IndicesType& /* frontIndices */)
    { calledGenerationalStepTaken = true; }

  bool calledEvaluate;
  bool calledGradient;
  bool calledBeginEpoch;
  bool calledEndEpoch;
  bool calledBeginOptimization;
  bool calledEndOptimization;
  bool calledEvaluateConstraint;
  bool calledGradientConstraint;
  bool calledStepTaken;
  bool calledGenerationalStepTaken;
};

template<typename OptimizerType>
void CallbacksFullFunctionTest(OptimizerType& optimizer,
                               bool calledEvaluate,
                               bool calledGradient,
                               bool calledBeginEpoch,
                               bool calledEndEpoch,
                               bool calledBeginOptimization,
                               bool calledEndOptimization,
                               bool calledEvaluateConstraint,
                               bool calledGradientConstraint,
                               bool calledStepTaken)
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);
  LogisticRegression<> lr(shuffledData, shuffledResponses, 0.5);

  CompleteCallbackTestFunction cb;

  arma::mat coordinates = lr.GetInitialPoint();
  optimizer.Optimize(lr, coordinates, cb);

  REQUIRE(cb.calledEvaluate == calledEvaluate);
  REQUIRE(cb.calledGradient == calledGradient);
  REQUIRE(cb.calledBeginEpoch == calledBeginEpoch);
  REQUIRE(cb.calledEndEpoch == calledEndEpoch);
  REQUIRE(cb.calledBeginOptimization == calledBeginOptimization);
  REQUIRE(cb.calledEndOptimization == calledEndOptimization);
  REQUIRE(cb.calledEvaluateConstraint == calledEvaluateConstraint);
  REQUIRE(cb.calledGradientConstraint == calledGradientConstraint);
  REQUIRE(cb.calledStepTaken == calledStepTaken);
}

template<typename OptimizerType>
void CallbacksFullMultiobjectiveFunctionTest(OptimizerType& optimizer,
                                             bool calledEvaluate,
                                             bool calledGradient,
                                             bool calledBeginEpoch,
                                             bool calledEndEpoch,
                                             bool calledBeginOptimization,
                                             bool calledEndOptimization,
                                             bool calledEvaluateConstraint,
                                             bool calledGradientConstraint,
                                             bool calledStepTaken,
                                             bool calledGenerationalStepTaken)
{
  SchafferFunctionN1<arma::mat> SCH;

  typedef decltype(SCH.objectiveA) ObjectiveTypeA;
  typedef decltype(SCH.objectiveB) ObjectiveTypeB;

  CompleteCallbackTestFunction cb;

  arma::mat coordinates = SCH.GetInitialPoint();
  std::tuple<ObjectiveTypeA, ObjectiveTypeB> objectives = SCH.GetObjectives();

  optimizer.Optimize(objectives, coordinates, cb);

  REQUIRE(cb.calledEvaluate == calledEvaluate);
  REQUIRE(cb.calledGradient == calledGradient);
  REQUIRE(cb.calledBeginEpoch == calledBeginEpoch);
  REQUIRE(cb.calledEndEpoch == calledEndEpoch);
  REQUIRE(cb.calledBeginOptimization == calledBeginOptimization);
  REQUIRE(cb.calledEndOptimization == calledEndOptimization);
  REQUIRE(cb.calledEvaluateConstraint == calledEvaluateConstraint);
  REQUIRE(cb.calledGradientConstraint == calledGradientConstraint);
  REQUIRE(cb.calledStepTaken == calledStepTaken);
  REQUIRE(cb.calledGenerationalStepTaken == calledGenerationalStepTaken);
}

template<typename OptimizerType>
void EarlyStopCallbacksLambdaFunctionTest(OptimizerType& optimizer)
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);

  LogisticRegression<> lr(shuffledData, shuffledResponses, 0.5);
  arma::mat coordinates = lr.GetInitialPoint();

  EarlyStopAtMinLoss cb(
      [&](const arma::mat& /* coordinates */)
      {
        return lr.ComputeAccuracy(testData, testResponses,
                                  coordinates);
      });

  optimizer.Optimize(lr, coordinates, cb);
}

TEST_CASE("EarlyStopAtMinLossLambdaCallbackTest", "[CallbacksTest]")
{
  SMORMS3 smorms3;
  EarlyStopCallbacksLambdaFunctionTest(smorms3);
}

TEST_CASE("EarlyStopAtMinLossCustomLambdaTest", "[CallbacksTest]")
{
  // Use the 50-dimensional Rosenbrock function.
  GeneralizedRosenbrockFunction f(50);
  // Start at some really large point.
  arma::mat coordinates = f.GetInitialPoint();
  coordinates.fill(100.0);

  EarlyStopAtMinLoss cb(
      [&](const arma::mat& coordinates)
      {
        // Terminate if any coordinate has a value less than 10.
        double minValue = arma::abs(coordinates).min();
        return (minValue < 10.0) ?
          std::numeric_limits<double>::max() : minValue;
      });

  SMORMS3 smorms3;
  smorms3.Optimize(f, coordinates, cb);

  // Make sure that we did not get to the optimum.
  for (size_t i = 0; i < coordinates.n_elem; ++i)
    REQUIRE(std::abs(coordinates[i]) >= 3.0);
}

/**
 * Make sure we invoke all callbacks (AdaBound).
 */
TEST_CASE("AdaBoundCallbacksFullFunctionTest", "[CallbacksTest]")
{
  AdaBound optimizer(0.001, 2, 0.1, 1e-3, 0.9, 0.999, 1e-8, 1000,
      1e-3, false);
  CallbacksFullFunctionTest(optimizer, true, true, true, true, true, true,
      false, false, true);
}

/**
 * Make sure we invoke all callbacks (AdaDelta).
 */
TEST_CASE("AdaDeltaCallbacksFullFunctionTest", "[CallbacksTest]")
{
  AdaDelta optimizer(1.0, 1, 0.99, 1e-8, 2000, 1e-9, true);
  CallbacksFullFunctionTest(optimizer, true, true, true, true, true, true,
      false, false, true);
}

/**
 * Make sure we invoke all callbacks (AdaGrad).
 */
TEST_CASE("AdaGradCallbacksFullFunctionTest", "[CallbacksTest]")
{
  AdaGrad optimizer(0.99, 1, 1e-8, 2000, 1e-9, true);
  CallbacksFullFunctionTest(optimizer, true, true, true, true, true, true,
      false, false, true);
}

/**
 * Make sure we invoke all callbacks (Adam).
 */
TEST_CASE("AdamCallbacksFullFunctionTest", "[CallbacksTest]")
{
  Adam optimizer(0.5, 2, 0.7, 0.999, 1e-8, 2000, 1e-3, false);
  CallbacksFullFunctionTest(optimizer, true, true, true, true, true, true,
      false, false, true);
}

/**
 * Make sure we invoke all callbacks (BigBatchSGD).
 */
TEST_CASE("BigBatchSGDCallbacksFullFunctionTest", "[CallbacksTest]")
{
  BBS_BB optimizer(1, 0.01, 0.1, 2000, 1e-4);
  CallbacksFullFunctionTest(optimizer, true, true, true, true, true, true,
      false, false, true);
}

/**
 * Make sure we invoke all callbacks (CMAES).
 */
TEST_CASE("CMAESCallbacksFullFunctionTest", "[CallbacksTest]")
{
  CMAES<> optimizer(0, -1, 1, 32, 3, 1e-3);
  CallbacksFullFunctionTest(optimizer, true, false, false, false, true, true,
      false, false, true);
}

/**
 * Make sure we invoke all callbacks (CNE).
 */
TEST_CASE("CNECallbacksFullFunctionTest", "[CallbacksTest]")
{
  CNE optimizer(200, 6, 0.2, 0.2, 0.2, 1e-5);
  CallbacksFullFunctionTest(optimizer, true, false, false, false, true, true,
      false, false, true);
}

/**
 * Make sure we invoke all callbacks (DE).
 */
TEST_CASE("DECallbacksFullFunctionTest", "[CallbacksTest]")
{
  DE optimizer(200, 6, 0.6, 0.8, 1e-5);
  CallbacksFullFunctionTest(optimizer, true, false, false, false, true, true,
      false, false, true);
}

/**
 * Make sure we invoke all callbacks (Eve).
 */
TEST_CASE("EveCallbacksFullFunctionTest", "[CallbacksTest]")
{
  Eve optimizer(1e-3, 1, 0.9, 0.999, 0.999, 1e-8, 10000, 2000, 1e-9, true);
  CallbacksFullFunctionTest(optimizer, true, true, true, true, true, true,
      false, false, true);
}

/**
 * Make sure we invoke all callbacks (FTML).
 */
TEST_CASE("FTMLCallbacksFullFunctionTest", "[CallbacksTest]")
{
  FTML optimizer(0.001, 1, 0.9, 0.999, 1e-8, 2000, 1e-5, true);
  CallbacksFullFunctionTest(optimizer, true, true, true, true, true, true,
      false, false, true);
}

/**
 * Make sure we invoke all callbacks (GradientDescent).
 */
TEST_CASE("GradientDescentCallbacksFullFunctionTest", "[CallbacksTest]")
{
  GradientDescent optimizer(0.001, 3, 1e-15);
  CallbacksFullFunctionTest(optimizer, true, true, false, false, true, true,
      false, false, true);
}

/**
 * Make sure we invoke all callbacks (IQN).
 */
TEST_CASE("IQNCallbacksFullFunctionTest", "[CallbacksTest]")
{
  IQN optimizer(0.01, 1, 3, 1e-3);
  CallbacksFullFunctionTest(optimizer, true, true, false, false, true, true,
      false, false, true);
}

/**
 * Make sure we invoke all callbacks (Katyusha).
 */
TEST_CASE("KatyushaCallbacksFullFunctionTest", "[CallbacksTest]")
{
  Katyusha optimizer(1.0, 10.0, 1, 3, 0, 1e-10, true);
  CallbacksFullFunctionTest(optimizer, true, true, false, false, true, true,
      false, false, true);
}

/**
 * Make sure we invoke all callbacks (NSGA2).
 */
TEST_CASE("NSGA2CallbacksFullFunctionTest", "[CallbackTest]")
{
  arma::vec lowerBound = {-1000};
  arma::vec upperBound = {1000};
  NSGA2 optimizer(20, 5000, 0.5, 0.5, 1e-3, 1e-6, lowerBound, upperBound);
  CallbacksFullMultiobjectiveFunctionTest(optimizer, false, false, false, false,
      true, true, false, false, false, true);
}

/**
 * Make sure we invoke all callbacks (MOEA/D-DE).
 */
TEST_CASE("MOEADCallbacksFullFunctionTest", "[CallbackTest]")
{
  arma::vec lowerBound = {-1000};
  arma::vec upperBound = {1000};
  DefaultMOEAD optimizer(150, 300, 1.0, 0.9, 20, 20, 0.5, 2, 1E-10, lowerBound, upperBound);
  CallbacksFullMultiobjectiveFunctionTest(optimizer, false, false, false, false,
      true, true, false, false, false, true);
}

/**
 * Make sure we invoke all callbacks (Lookahead).
 */
TEST_CASE("LookaheadCallbacksFullFunctionTest", "[CallbacksTest]")
{
  Adam adam(0.001, 1, 0.9, 0.999, 1e-8, 100, 1e-10, false, true);
  Lookahead<Adam> optimizer(adam, 0.5, 1000, 10, -10, NoDecay(),
      false, true);
  CallbacksFullFunctionTest(optimizer, true, true, true, true, true, true,
      false, false, true);
}

/**
 * Make sure we invoke all callbacks (Padam).
 */
TEST_CASE("PadamCallbacksFullFunctionTest", "[CallbacksTest]")
{
  Padam optimizer(1e-2, 1, 0.9, 0.99, 0.25, 1e-5, 1000);
  CallbacksFullFunctionTest(optimizer, true, true, true, true, true, true,
      false, false, true);
}

/**
 * Make sure we invoke all callbacks (QHAdam).
 */
TEST_CASE("QHAdamCallbacksFullFunctionTest", "[CallbacksTest]")
{
  QHAdam optimizer(0.02, 2, 0.6, 0.9, 0.9, 0.999, 1e-8, 1000, 1e-7, true);
  CallbacksFullFunctionTest(optimizer, true, true, true, true, true, true,
      false, false, true);
}

/**
 * Make sure we invoke all callbacks (RMSProp).
 */
TEST_CASE("RMSPropCallbacksFullFunctionTest", "[CallbacksTest]")
{
  RMSProp optimizer(1e-3, 1, 0.99, 1e-8, 1000, 1e-9, true);
  CallbacksFullFunctionTest(optimizer, true, true, true, true, true, true,
      false, false, true);
}
/**
 * Make sure we invoke all callbacks (SARAH).
 */
TEST_CASE("SARAHCallbacksFullFunctionTest", "[CallbacksTest]")
{
  SARAH optimizer(0.01, 2, 3, 0, 1e-5, true);
  CallbacksFullFunctionTest(optimizer, true, true, false, false, true, true,
      false, false, true);
}

/**
 * Make sure we invoke all callbacks (SCD).
 */
TEST_CASE("SCDCallbacksFullFunctionTest", "[CallbacksTest]")
{
  SCD<> optimizer(0.4, 4);
  CallbacksFullFunctionTest(optimizer, true, true, false, false, true, true,
      false, false, true);
}

/**
 * Make sure we invoke all callbacks (SGD).
 */
TEST_CASE("SGDCallbacksFullFunctionTest", "[CallbacksTest]")
{
  StandardSGD optimizer(0.0003, 1, 2000, 1e-9, true);
  CallbacksFullFunctionTest(optimizer, true, true, true, true, true, true,
      false, false, true);
}

/**
 * Make sure we invoke all callbacks (SGDR).
 */
TEST_CASE("SGDRCallbacksFullFunctionTest", "[CallbacksTest]")
{
  SGDR<> optimizer(50, 2.0, 1, 0.01, 2000, 1e-3);
  CallbacksFullFunctionTest(optimizer, true, true, true, true, true, true,
      false, false, true);
}

/**
 * Make sure we invoke all callbacks (SMORMS3).
 */
TEST_CASE("SMORMS3CallbacksFullFunctionTest", "[CallbacksTest]")
{
  SMORMS3 optimizer(0.001, 1, 1e-16, 1000, 1e-9, true);
  CallbacksFullFunctionTest(optimizer, true, true, true, true, true, true,
      false, false, true);
}

/**
 * Make sure we invoke all callbacks (SPALeRASGD).
 */
TEST_CASE("SPALeRASGDCallbacksFullFunctionTest", "[CallbacksTest]")
{
  SPALeRASGD<> optimizer(0.05, 30, 2000, 1e-4);
  CallbacksFullFunctionTest(optimizer, true, true, true, true, true, true,
      false, false, true);
}

/**
 * Make sure we invoke all callbacks (SPSA).
 */
TEST_CASE("SPSACallbacksFullFunctionTest", "[CallbacksTest]")
{
  SPSA optimizer(0.1, 0.102, 0.16, 0.3, 10, 0);
  CallbacksFullFunctionTest(optimizer, true, false, false, false, true, true,
      false, false, true);
}

/**
 * Make sure we invoke all callbacks (SVRG).
 */
TEST_CASE("SVRGCallbacksFullFunctionTest", "[CallbacksTest]")
{
  SVRG optimizer(0.005, 2, 4, 0, 1e-5, true);
  CallbacksFullFunctionTest(optimizer, true, true, false, false, true, true,
      false, false, true);
}

/**
 * Make sure we invoke all callbacks (SWATS).
 */
TEST_CASE("SWATSCallbacksFullFunctionTest", "[CallbacksTest]")
{
  SWATS optimizer(0.01, 10, 0.9, 0.999, 1e-6, 1000, 1e-9, true);
  CallbacksFullFunctionTest(optimizer, true, true, true, true, true, true,
      false, false, true);
}

/**
 * Make sure we invoke all callbacks (WNGrad).
 */
TEST_CASE("WNGradCallbacksFullFunctionTest", "[CallbacksTest]")
{
  WNGrad optimizer(0.56, 1, 1000, 1e-9, true);
  CallbacksFullFunctionTest(optimizer, true, true, true, true, true, true,
      false, false, true);
}

/**
 * Make sure we invoke all callbacks (ParallelSGD).
 */
TEST_CASE("ParallelSGDCallbacksFullFunctionTest", "[CallbacksTest]")
{
  ConstantStep decayPolicy(0.4);
  ParallelSGD<ConstantStep> optimizer(4, 2, 1e-5, true, decayPolicy);
  CallbacksFullFunctionTest(optimizer, true, true, false, false, true, true,
      false, false, true);
}

/**
 * Make sure we invoke all callbacks (LBestPSO).
 */
TEST_CASE("LBestPSOCallbacksFullFunctionTest", "[CallbacksTest]")
{
  LBestPSO optimizer;
  CallbacksFullFunctionTest(optimizer, true, false, false, false, true, true,
      false, false, true);
}

/**
 * Make sure we invoke all callbacks (L_BFGS).
 */
TEST_CASE("L_BFGSCallbacksFullFunctionTest", "[CallbacksTest]")
{
  L_BFGS optimizer(10, 4);
  CallbacksFullFunctionTest(optimizer, true, true, false, false, true, true,
      false, false, true);
}

/**
 * Make sure we invoke all callbacks (SA).
 */
TEST_CASE("SACallbacksFullFunctionTest", "[CallbacksTest]")
{
  ExponentialSchedule schedule;
  SA<> optimizer(schedule, 10, 1000., 1000, 100, 1e-11, 3, 1.5, 0.3, 0.3);
  CallbacksFullFunctionTest(optimizer, true, false, false, false, true, true,
      false, false, true);
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
  StandardSGD s(0.0003, 1, 2000000000, -10);
  s.ExactObjective() = true;

  // The optimization process should return in one second.
  const double result = s.Optimize(f, coordinates, EarlyStopAtMinLoss(100));

  REQUIRE(result == Approx(-1.0).epsilon(0.0005));
  REQUIRE(coordinates(0) == Approx(0.0).margin(1e-3));
  REQUIRE(coordinates(1) == Approx(0.0).margin(1e-7));
  REQUIRE(coordinates(2) == Approx(0.0).margin(1e-7));
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
 * and objective.
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
  REQUIRE(cb.BestCoordinates()(0) == Approx(0.0).margin(1e-3));
  REQUIRE(cb.BestCoordinates()(1) == Approx(0.0).margin(1e-7));
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
  Adam opt(0.5, 2, 0.7, 0.999, 1e-8, 2000000000, -100, false);
  arma::wall_clock timer;

  timer.tic();
  // The optimization process should return in one second.
  opt.Optimize(f, coordinates, TimerStop(0.5));
  // Add some time to account for the function to return.
  REQUIRE(timer.toc() < 2);
}

/**
 * Make sure the ProgressBar callback will show the progress on the specified
 * output stream if the MaxIterations parameter of the optimizer is 0.
 */
TEST_CASE("ProgressBarCallbackNoMaxIterationsTest", "[CallbacksTest]")
{
  SGDTestFunction f;
  arma::mat coordinates = f.GetInitialPoint();

  StandardSGD s(0.0003, 1, 0, DBL_MAX, true);

  std::stringstream stream;
  s.Optimize(f, coordinates, ProgressBar(10, stream));

  REQUIRE(stream.str().length() > 0);
}

/**
 * Make sure the ProgressBar callback will show the progress on the specified
 * output stream with the correct epoch number if the MaxIterations parameter
 * of the optimizer is 0.
 */
TEST_CASE("ProgressBarCallbackNoMaxIterationsEpochTest", "[CallbacksTest]")
{
  SGDTestFunction f;
  arma::mat coordinates = f.GetInitialPoint();

  StandardSGD s(0.0003, 1, 0, DBL_MAX, true);

  std::stringstream stream;
  s.Optimize(f, coordinates, ProgressBar(10, stream));
  REQUIRE(stream.str().find("Epoch 1") != std::string::npos);
  REQUIRE(stream.str().find("Epoch 1/") == std::string::npos);
}

/**
 * Make sure the ProgressBar callback will show the progress on the specified
 * output stream with the correct epoch number if the MaxIterations parameter
 * of the optimizer is not equal to 0.
 */
TEST_CASE("ProgressBarCallbackEpochTest", "[CallbacksTest]")
{
  SGDTestFunction f;
  arma::mat coordinates = f.GetInitialPoint();

  StandardSGD s(0.0003, 1, 1, 1e-9, true);

  std::stringstream stream;
  s.Optimize(f, coordinates, ProgressBar(10, stream));
  REQUIRE(stream.str().find("Epoch 1/1") != std::string::npos);
}

/**
 * Make sure the Report callback will show the report on the specified
 * output stream.
 */
TEST_CASE("ReportCallbackTest", "[CallbacksTest]")
{
  std::stringstream stream;

  SGDTestFunction f0;
  StandardSGD s(0.0003, 1, 10000, 1e-9, true);

  arma::mat coordinates = f0.GetInitialPoint();
  s.Optimize(f0, coordinates, Report(0.1, stream));
  REQUIRE(stream.str().length() > 0);

  stream.str("");
  RosenbrockWoodFunction f1;
  L_BFGS lbfgs;
  lbfgs.MaxIterations() = 100;

  coordinates = f1.GetInitialPoint();
  lbfgs.Optimize(f1, coordinates, Report(0.1, stream));
  REQUIRE(stream.str().length() > 0);

  stream.str("");
  SchafferFunctionN2 f2;
  CNE cne;
  cne.MaxGenerations() = 100;

  coordinates = f2.GetInitialPoint();
  cne.Optimize(f2, coordinates, Report(0.1, stream));
  REQUIRE(stream.str().length() > 0);

  stream.str("");
  AugLagrangianTestFunction f3;
  AugLagrangian aug;

  coordinates = f3.GetInitialPoint();
  aug.Optimize(f3, coordinates, Report(0.1, stream));
  REQUIRE(stream.str().length() > 0);
}

/**
 * Make sure the GradClipByNorm callback will clip the gradient.
 */
TEST_CASE("GradClipByNormCallbackTest", "[CallbacksTest]")
{
  SGDTestFunction f;
  arma::mat coordinates = f.GetInitialPoint();

  StandardSGD s(0.0003, 1, 10, 1e-9, true);

  std::stringstream stream;
  s.Optimize(f, coordinates, GradClipByNorm(0.5), Report(0.1, stream));

  // We don't store the gradient during the optimization process, so we use the
  // output of the Report callback function to check if the gradient is
  // clipped.
  std::string line;
  bool gradientInfo = false;
  double gradient = 1;
  while (std::getline(stream, line, '\n'))
  {
    if (gradientInfo)
    {
      size_t iter;
      double loss, lossChange, stepSize, totalTime;

      std::stringstream stream(line);
      stream >> iter >> loss >> lossChange >> gradient >> stepSize >> totalTime;
      break;
    }

    gradientInfo = line.find("|gradient|") != std::string::npos;
  }

  REQUIRE(gradient == 0.5);
}
/**
 * Make sure the GradClipByValue callback will clip the gradient.
 */
TEST_CASE("GradClipByValueCallbackTest", "[CallbacksTest]")
{
  SGDTestFunction f;
  arma::mat coordinates = f.GetInitialPoint();

  StandardSGD s(0.0003, 1, 10, 1e-9, true);

  std::stringstream stream;
  s.Optimize(f, coordinates, GradClipByValue(0, 0), Report(0.1, stream));

  // We don't store the gradient during the optimization process, so we use the
  // output of the Report callback function to check if the gradient is
  // clipped.
  std::string line;
  bool gradientInfo = false;
  double gradient = 1;
  while (std::getline(stream, line, '\n'))
  {
    if (gradientInfo)
    {
      size_t iter;
      double loss, lossChange, stepSize, totalTime;

      std::stringstream stream(line);
      stream >> iter >> loss >> lossChange >> gradient >> stepSize >> totalTime;
      break;
    }

    gradientInfo = line.find("|gradient|") != std::string::npos;
  }

  REQUIRE(gradient == 0);
}

/**
 * @file progress_bar.hpp
 * @author Marcus Edel
 *
 * Implementation of a simple progress bar callback function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_CALLBACKS_PROGRESS_BAR_HPP
#define ENSMALLEN_CALLBACKS_PROGRESS_BAR_HPP

#include <ensmallen_bits/function.hpp>

namespace ens {

/**
 * A simple progress bar, based on the maximum number of optimizer iterations,
 * batch-size, number of functions and the StepTaken callback function.
 */
class ProgressBar
{
 public:
  /**
   * Set up the progress bar callback class with the given width and output
   * stream.
   *
   * @param widthIn Width of the bar.
   * @param ostream Ostream which receives output from this object.
   */
  ProgressBar(const size_t widthIn = 70,
              std::ostream& output = arma::get_cout_stream()) :
      width(100.0 / widthIn),
      output(output),
      objective(0),
      epochs(0),
      epochSize(0),
      step(1),
      steps(0),
      newEpoch(false),
      epoch(1)

  { /* Nothing to do here. */ }

  /**
   * Callback function called at the begin of the optimization process.
   *
   * @param optimizer The optimizer used to update the function.
   * @param function Function to optimize.
   * @param coordinates Starting point.
   */
  template<typename OptimizerType, typename FunctionType, typename MatType>
  void BeginOptimization(OptimizerType& optimizer,
                         FunctionType& function,
                         MatType& /* coordinates */)
  {
    static_assert(traits::HasBatchSizeSignature<
      OptimizerType>::value,
      "The OptimizerType does not have a correct definition of BatchSize(). "
      "Please check that the OptimizerType fully satisfies the requirements "
      "of the ProgressBar API; see the callbacks documentation for more "
      "details.");

    static_assert(traits::HasMaxIterationsSignature<
      OptimizerType>::value,
      "The OptimizerType does not have a correct definition of MaxIterations()."
      " Please check that the OptimizerType fully satisfies the requirements "
      "of the ProgressBar API; see the callbacks documentation for more "
      "details.");

    static_assert(traits::HasNumFunctionsSignature<
      FunctionType>::value,
      "The OptimizerType does not have a correct definition of NumFunctions(). "
      "Please check that the OptimizerType fully satisfies the requirements "
      "of the ProgressBar API; see the callbacks documentation for more "
      "details.");

    epochSize = function.NumFunctions() / optimizer.BatchSize();
    if (function.NumFunctions() % optimizer.BatchSize() > 0)
      epochSize++;

    epochs = optimizer.MaxIterations() / function.NumFunctions();
    if (optimizer.MaxIterations() % function.NumFunctions() > 0)
      epochs++;

    stepTimer.tic();
  }

  /**
   * Callback function called at the begin of a pass over the data.
   *
   * @param optimizer The optimizer used to update the function.
   * @param function Function to optimize.
   * @param coordinates Starting point.
   * @param epochIn The index of the current epoch.
   * @param objective Objective value of the current point.
   */
  template<typename OptimizerType, typename FunctionType, typename MatType>
  void BeginEpoch(OptimizerType& /* optimizer */,
                  FunctionType& /* function */,
                  const MatType& /* coordinates */,
                  const size_t epochIn,
                  const double /* objective */)
  {
    // Start the timer.
    epochTimer.tic();

    // Reset epoch parameter.
    objective = 0;
    step = 1;

    epoch = epochIn;
    newEpoch = true;
  }

  /**
   * Callback function called once a step is taken.
   *
   * @param optimizer The optimizer used to update the function.
   * @param function Function to optimize.
   * @param coordinates Starting point.
   * @param objective Objective value of the current point.
   */
  template<typename OptimizerType, typename FunctionType, typename MatType>
  void StepTaken(OptimizerType& /* optimizer */,
                 FunctionType& /* function */,
                 const MatType& /* coordinates */)
  {
    if (newEpoch)
    {
      output << "Epoch " << epoch;
      if (epochs > 0)
      {
        output << "/" << epochs;
      }
      output << '\n';
      newEpoch = false;
    }

    const size_t progress = ((double) step / epochSize) * 100;
    output << step++ << "/" << epochSize << " [";
    for (size_t i = 0; i < 100; i += width)
    {
      if (i < progress)
      {
        output << "=";
      }
      else if (i == progress)
      {
        output << ">";
      }
      else
      {
        output << ".";
      }
    }

    output << "] " << progress << "% - ETA: " << (size_t) (stepTimer.toc() *
        (epochSize - step + 1)) % 60 << "s - loss: " <<
        objective / (double) step <<  "\r";
    output.flush();

    stepTimer.tic();
  }

  /**
   * Callback function called at any call to Evaluate().
   *
   * @param optimizer The optimizer used to update the function.
   * @param function Function to optimize.
   * @param coordinates Starting point.
   * @param objectiveIn Objective value of the current point.
   */
  template<typename OptimizerType, typename FunctionType, typename MatType>
  void Evaluate(OptimizerType& optimizer,
                FunctionType& /* function */,
                const MatType& /* coordinates */,
                const double objectiveIn)
  {
    objective += objectiveIn / optimizer.BatchSize();
    steps++;
  }

  /**
   * Callback function called at the end of a pass over the data.
   *
   * @param optimizer The optimizer used to update the function.
   * @param function Function to optimize.
   * @param coordinates Starting point.
   * @param epoch The index of the current epoch.
   * @param objective Objective value of the current point.
   */
  template<typename OptimizerType, typename FunctionType, typename MatType>
  void EndEpoch(OptimizerType& /* optimizer */,
                FunctionType& /* function */,
                const MatType& /* coordinates */,
                const size_t /* epoch */,
                const double objective)
  {
    const size_t progress = ((double) (step - 1) / epochSize) * 100;
    output << step - 1 << "/" << epochSize << " [";
    for (size_t i = 0; i < 100; i += width)
    {
      if (i < progress)
      {
        output << "=";
      }
      else if (i == progress)
      {
        output << ">";
      }
      else
      {
        output << ".";
      }
    }
    const double epochTimerElapsed = epochTimer.toc();
    const size_t stepTime = epochTimerElapsed / (double) epochSize * 1000;
    output << "] " << progress << "% - " << epochTimerElapsed
        << "s/epoch; " << stepTime << "ms/step; loss: " << objective  <<  "\n";
    output.flush();
  }

 private:
  //! Length of a single step (1%).
  double width;

  //! The output stream that all data is to be sent to; example: std::cout.
  std::ostream& output;

  //! Objective over the current epoch.
  double objective;

  //! Total number of epochs
  size_t epochs;

  //! Number of steps per epoch.
  size_t epochSize;

  //! Current step number.
  size_t step;

  //! Number of steps taken.
  size_t steps;

  //! Indicates a new epoch.
  bool newEpoch;

  //! Locally-stored epoch.
  size_t epoch;

  //! Locally-stored step timer object.
  arma::wall_clock stepTimer;

  //! Locally-stored epoch timer object.
  arma::wall_clock epochTimer;
};

} // namespace ens

#endif

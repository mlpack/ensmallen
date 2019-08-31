/**
 * @file timer_stop.hpp
 * @author Marcus Edel
 *
 * Implementation of the timer stop callback function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_CALLBACKS_TIMER_STOP_HPP
#define ENSMALLEN_CALLBACKS_TIMER_STOP_HPP

namespace ens {

/**
 * Timer stop function, is based on the BeginOptimization callback function to
 * start the timer and the EndEpoch callback function to update the timer.
 */
class TimerStop
{
 public:
  /**
   * Set up the print loss callback class with the width and output stream.
   *
   * @param durationIn The duration of the timer in seconds.
   */
  TimerStop(const double durationIn) : duration(durationIn)
  { /* Nothing to do here. */ }

  /**
   * Callback function called at the start of the optimization process.
   *
   * @param optimizer The optimizer used to update the function.
   * @param function Function to optimize.
   * @param coordinates Starting point.
   */
  template<typename OptimizerType, typename FunctionType, typename MatType>
  void BeginOptimization(OptimizerType& /* optimizer */,
                         FunctionType& /* function */,
                         MatType& /* coordinates */)
  {
    // Start the timer.
    timer.tic();
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
  bool EndEpoch(OptimizerType& /* optimizer */,
                FunctionType& /* function */,
                const MatType& /* coordinates */,
                const size_t /* epoch */,
                const double /* objective */)
  {
    if (timer.toc() > duration)
    {
      Info << "Timer timeout reached; terminate optimization." << std::endl;
      return true;
    }

    return false;
  }

 private:
  //! The duration in seconds.
  double duration;

  //! Locally-stored timer object.
  arma::wall_clock timer;
};

} // namespace ens

#endif

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

namespace ens {

/**
 * A simple progress bar, based on the maximum number of optimizer iterations
 * and the EndEpoch callback function.
 */
class ProgressBar
{
 public:
  /**
   * Set up the progress bar callback class with the given width and output
   * stream.
   *
   * @param width Width of the bar.
   * @param ostream Ostream which receives output from this object.
   */
  ProgressBar(const size_t width = 70,
              std::ostream& output = arma::get_cout_stream()) :
      step(100.0 / width),
      output(output),
      previousProgress(0)
  { /* Nothing to do here. */ }

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
  void EndEpoch(OptimizerType& optimizer,
                FunctionType& /* function */,
                const MatType& /* coordinates */,
                const size_t epoch,
                const double /* objective */)
  {
    if (!optimizer->MaxIterations())
    {
      Warn << "Maximum number of iterations not defined (no limit),"
           << " no progress bar shown." << std::endl;
      return;
    }

    const size_t progress = ((double) epoch / optimizer->MaxIterations()) * 100;

    // Skip if the progress hasn't changed.
    if (progress <= previousProgress)
      return;

    output << "[";
    for (size_t i = 0; i < 100; i += step)
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
        output << " ";
      }
    }

    output << "] " << progress << " %\r";
    output.flush();

    previousProgress = progress;
  }

 private:
  //! Length of a single step (1%).
  double step;

  //! The output stream that all data is to be sent to; example: std::cout.
  std::ostream& output;

  //! Locally-stored previous progress.
  size_t previousProgress;
};

} // namespace ens

#endif

/**
 * @file print_loss.hpp
 * @author Marcus Edel
 *
 * Implementation of the print loss callback function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_CALLBACKS_PRINT_LOSS_HPP
#define ENSMALLEN_CALLBACKS_PRINT_LOSS_HPP

namespace ens {

/**
 * Print loss function, based on the EndEpoch callback function.
 */
class PrintLoss
{
 public:
  /**
   * Set up the print loss callback class with the width and output stream.
   *
   * @param ostream Ostream which receives output from this object.
   */
  PrintLoss(std::ostream& output = arma::get_cout_stream()) : output(output)
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
  bool EndEpoch(OptimizerType& /* optimizer */,
                FunctionType& /* function */,
                const MatType& /* coordinates */,
                const size_t /* epoch */,
                const double objective)
  {
    output << objective << std::endl;
    return false;
  }

 private:
  //! The output stream that all data is to be sent to; example: std::cout.
  std::ostream& output;
};

} // namespace ens

#endif

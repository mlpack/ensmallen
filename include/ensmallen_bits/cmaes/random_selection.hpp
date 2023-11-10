/**
 * @file random_selection.hpp
 * @author Marcus Edel
 *
 * Randomly select dataset points for use in the Evaluation step.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_CMAES_RANDOM_SELECTION_HPP
#define ENSMALLEN_CMAES_RANDOM_SELECTION_HPP

namespace ens {

/*
 * Randomly select dataset points for use in the Evaluation step.
 */
class RandomSelection
{
 public:
  /**
   * Constructor for the random selection strategy.
   *
   * @param fraction The dataset fraction used for the selection (Default 0.3).
   */
  RandomSelection(const double fraction = 0.3) : fraction(fraction)
  {
    // Nothing to do here.
  }

  //! Get the dataset fraction.
  double Fraction() const { return fraction; }
  //! Modify the dataset fraction.
  double& Fraction() { return fraction; }

  /**
   * Randomly select dataset points to calculate the objective function.
   *
   * @tparam SeparableFunctionType Type of the function to be evaluated.
   * @param function Function to optimize.
   * @param batchSize Batch size to use for each step.
   * @param terminate Whether optimization should be terminated after this call.
   * @param iterate starting point.
   */
  template<typename SeparableFunctionType,
           typename MatType,
           typename... CallbackTypes>
  double Select(SeparableFunctionType& function,
                const size_t batchSize,
                const MatType& iterate,
                bool& terminate,
                CallbackTypes&... callbacks)
  {
    // Find the number of functions to use.
    const size_t numFunctions = function.NumFunctions();

    typename MatType::elem_type objective = 0;
    for (size_t f = 0; f < std::floor(numFunctions * fraction); f += batchSize)
    {
      const size_t selection = arma::as_scalar(arma::randi<arma::uvec>(
          1, arma::distr_param(0, numFunctions - 1)));
      const size_t effectiveBatchSize = std::min(batchSize,
          numFunctions - selection);

      objective += function.Evaluate(iterate, selection, effectiveBatchSize);

      terminate |= Callback::Evaluate(*this, f, iterate, objective,
          callbacks...);
    }

    return objective;
  }

 private:
  //! Dataset fraction parameter.
  double fraction;
};

} // namespace ens

#endif

/**
 * @file full_selection.hpp
 * @author Marcus Edel
 *
 * Select the full dataset for use in the Evaluation step.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_CMAES_FULL_SELECTION_HPP
#define ENSMALLEN_CMAES_FULL_SELECTION_HPP

namespace ens {

/*
 * Select the full dataset for use in the Evaluation step.
 */
class FullSelection
{
 public:
  /**
   * Select the full dataset to calculate the objective function.
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
    for (size_t f = 0; f < numFunctions; f += batchSize)
    {
      const size_t effectiveBatchSize = std::min(batchSize, numFunctions - f);
      objective += function.Evaluate(iterate, f, effectiveBatchSize);

      terminate |= Callback::Evaluate(*this, f, iterate, objective,
          callbacks...);
    }

    return objective;
  }
};

} // namespace ens

#endif

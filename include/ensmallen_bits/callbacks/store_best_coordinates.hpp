/**
 * @file store_best_coordinates.hpp
 * @author Marcus Edel
 *
 * Implementation of the store best coordinates callback function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_CALLBACKS_STORE_BEST_COORDINATES_HPP
#define ENSMALLEN_CALLBACKS_STORE_BEST_COORDINATES_HPP

namespace ens {

/**
 * Store best coordinates function, based on the Evaluate callback function.
 *
 * @tparam MatType Type of the model coordinates (arma::colvec, arma::mat,
 *     arma::sp_mat or arma::cube).
 */
template<typename ModelMatType = arma::mat>
class StoreBestCoordinates
{
 public:
  /**
   * Set up the store best model class, which keeps the best-performing
   * coordinates and objective.
   */
  StoreBestCoordinates() : bestObjective(std::numeric_limits<double>::max())
  { /* Nothing to do here. */ }

  /**
   * Callback function called after any call to Evaluate().
   *
   * @param optimizer The optimizer used to update the function.
   * @param function Function to optimize.
   * @param coordinates Starting point.
   * @param objective Objective value of the current point.
   */
  template<typename OptimizerType, typename FunctionType, typename MatType>
  bool Evaluate(OptimizerType& /* optimizer */,
                FunctionType& /* function */,
                const MatType& coordinates,
                const double objective)
  {
    if (objective < bestObjective)
    {
      bestObjective = objective;
      bestCoordinates = coordinates;
    }
    return false;
  }

  //! Get the best coordinates.
  ModelMatType const& BestCoordinates() const { return bestCoordinates; }
  //! Modify the best coordinates.
  ModelMatType& BestCoordinates() { return bestCoordinates; }

  //! Get the best objective.
  double const& BestObjective() const { return bestObjective; }
  //! Modify the best objective.
  double& BestObjective() { return bestObjective; }

 private:
  //! Locally-stored best objective.
  double bestObjective;

  //! Locally-stored best model coordinates.
  ModelMatType bestCoordinates;
};

} // namespace ens

#endif

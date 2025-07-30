/**
 * @file grid_search.hpp
 * @author Kirill Mishchenko
 *
 * Grid-search optimization.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_GRID_SEARCH_GRID_SEARCH_HPP
#define ENSMALLEN_GRID_SEARCH_GRID_SEARCH_HPP

namespace ens {

/**
 * An optimizer that finds the minimum of a given function by iterating through
 * points on a multidimensional grid.
 *
 * GridSearch can optimize categorical functions.  For more details, see the
 * documentation on function types included with this distribution or on the
 * ensmallen website.
 */
class GridSearch
{
 public:
  /**
   * Optimize (minimize) the given function by iterating through the all
   * possible combinations of values for the parameters specified in
   * datasetInfo.
   *
   * @tparam FunctionType Type of function to optimize.
   * @tparam MatType Type of matrix to optimize with.
   * @param function Function to optimize.
   * @param bestParameters Variable for storing results.
   * @param categoricalDimensions Set of dimension types.  If a value is true,
   *     then that dimension is a categorical dimension.
   * @param numCategories Number of categories in each categorical dimension.
   * @return Objective value of the final point.
   */
  template<typename FunctionType, typename MatType>
  typename MatType::elem_type Optimize(
      FunctionType& function,
      MatType& bestParameters,
      const std::vector<bool>& categoricalDimensions,
      const arma::Row<size_t>& numCategories);

 private:
  /**
   * Iterate through the last (parameterValueCollections.size() - i) dimensions
   * of the grid and change the arguments bestObjective and bestParameters if
   * there is something better. The values for the first i dimensions
   * (parameters) are specified in the first i rows of the currentParameters
   * argument.
   */
  template<typename FunctionType, typename MatType>
  void Optimize(
      FunctionType& function,
      typename MatType::elem_type& bestObjective,
      MatType& bestParameters,
      MatType& currentParameters,
      const std::vector<bool>& categoricalDimensions,
      const arma::Row<size_t>& numCategories,
      size_t i);
};

} // namespace ens

// Include implementation
#include "grid_search_impl.hpp"

#endif

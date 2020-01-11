/**
 * @file grid_search_impl.hpp
 * @author Kirill Mishchenko
 *
 * Implementation of the grid-search optimization.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_GRID_SEARCH_GRID_SEARCH_IMPL_HPP
#define ENSMALLEN_GRID_SEARCH_GRID_SEARCH_IMPL_HPP

#include <limits>
#include <ensmallen_bits/function.hpp>

namespace ens {

template<typename FunctionType, typename MatType>
typename MatType::elem_type GridSearch::Optimize(
    FunctionType& function,
    MatType& bestParameters,
    const std::vector<bool>& categoricalDimensions,
    const arma::Row<size_t>& numCategories)
{
  for (size_t i = 0; i < categoricalDimensions.size(); ++i)
  {
    if (!categoricalDimensions[i])
    {
      std::ostringstream oss;
      oss << "GridSearch::Optimize(): the dimension " << i
          << " is not categorical" << std::endl;
      throw std::invalid_argument(oss.str());
    }
  }

  // Convenience typedefs.
  typedef typename MatType::elem_type ElemType;

  ElemType bestObjective = std::numeric_limits<ElemType>::max();
  bestParameters.set_size(categoricalDimensions.size(), 1);
  MatType currentParameters(categoricalDimensions.size(), 1);

  /* Initialize best parameters for the case (very unlikely though) when no set
   * of parameters gives an objective value better than
   * std::numeric_limits<double>::max() */
  for (size_t i = 0; i < categoricalDimensions.size(); ++i)
    bestParameters(i, 0) = 0;

  Optimize(function, bestObjective, bestParameters, currentParameters,
      categoricalDimensions, numCategories, 0);

  return bestObjective;
}

template<typename FunctionType, typename MatType>
void GridSearch::Optimize(
    FunctionType& function,
    typename MatType::elem_type& bestObjective,
    MatType& bestParameters,
    MatType& currentParameters,
    const std::vector<bool>& categoricalDimensions,
    const arma::Row<size_t>& numCategories,
    size_t i)
{
  // Convenience typedefs.
  typedef typename MatType::elem_type ElemType;
  typedef typename MatTypeTraits<MatType>::BaseMatType BaseMatType;

  // Make sure we have the methods that we need.  No restrictions on the matrix
  // type are needed.
  traits::CheckArbitraryFunctionTypeAPI<FunctionType, BaseMatType>();

  if (i < categoricalDimensions.size())
  {
    for (size_t j = 0; j < numCategories(i); ++j)
    {
      currentParameters(i) = j;
      Optimize(function, bestObjective, bestParameters, currentParameters,
          categoricalDimensions, numCategories, i + 1);
    }
  }
  else
  {
    ElemType objective = function.Evaluate((BaseMatType&) currentParameters);
    if (objective < bestObjective)
    {
      bestObjective = objective;
      bestParameters = currentParameters;
    }
  }
}

} // namespace ens

#endif

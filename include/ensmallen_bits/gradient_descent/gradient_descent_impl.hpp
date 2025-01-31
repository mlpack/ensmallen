/**
 * @file gradient_descent_impl.hpp
 * @author Sumedh Ghaisas
 *
 * Simple gradient descent implementation.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_GRADIENT_DESCENT_GRADIENT_DESCENT_IMPL_HPP
#define ENSMALLEN_GRADIENT_DESCENT_GRADIENT_DESCENT_IMPL_HPP

// In case it hasn't been included yet.
#include "gradient_descent.hpp"

#include <ensmallen_bits/function.hpp>

namespace ens {

//! Constructor.
inline GradientDescent::GradientDescent(
    const double stepSize,
    const size_t maxIterations,
    const double tolerance) :
    stepSize(stepSize),
    maxIterations(maxIterations),
    tolerance(tolerance)
{ /* Nothing to do. */ }

//! Optimize the function (minimize).
template<typename FunctionType,
         typename MatType,
         typename GradType,
         typename... CallbackTypes>
typename std::enable_if<IsArmaType<GradType>::value,
typename MatType::elem_type>::type
GradientDescent::Optimize(FunctionType& function,
                          MatType& iterateIn,
                          CallbackTypes&&... callbacks)
{
  // Convenience typedefs.
  typedef typename MatType::elem_type ElemType;
  typedef typename MatTypeTraits<MatType>::BaseMatType BaseMatType;
  typedef typename MatTypeTraits<GradType>::BaseMatType BaseGradType;

  // Use the Function<> wrapper type to provide additional functionality.
  typedef Function<FunctionType, BaseMatType, BaseGradType> FullFunctionType;
  FullFunctionType& f(static_cast<FullFunctionType&>(function));

  // Make sure we have the methods that we need.
  traits::CheckFunctionTypeAPI<FullFunctionType, BaseMatType, BaseGradType>();
  RequireFloatingPointType<BaseMatType>();
  RequireFloatingPointType<BaseGradType>();
  RequireSameInternalTypes<BaseMatType, BaseGradType>();

  // To keep track of where we are and how things are going.
  ElemType overallObjective = std::numeric_limits<ElemType>::max();
  ElemType lastObjective = std::numeric_limits<ElemType>::max();

  BaseMatType& iterate = (BaseMatType&) iterateIn;
  BaseGradType gradient(iterate.n_rows, iterate.n_cols);

  // Controls early termination of the optimization process.
  bool terminate = false;

  // Now iterate!
  Callback::BeginOptimization(*this, f, iterate, callbacks...);
  for (size_t i = 1; i != maxIterations && !terminate; ++i)
  {
    overallObjective = f.EvaluateWithGradient(iterate, gradient);

    terminate |= Callback::EvaluateWithGradient(*this, f, iterate,
        overallObjective, gradient, callbacks...);

    // Output current objective function.
    Info << "Gradient Descent: iteration " << i << ", objective "
        << overallObjective << "." << std::endl;

    if (std::isnan(overallObjective) || std::isinf(overallObjective))
    {
      Warn << "Gradient Descent: converged to " << overallObjective
          << "; terminating" << " with failure.  Try a smaller step size?"
          << std::endl;

      Callback::EndOptimization(*this, f, iterate, callbacks...);
      return overallObjective;
    }

    if (std::abs(lastObjective - overallObjective) < tolerance)
    {
      Info << "Gradient Descent: minimized within tolerance "
          << tolerance << "; " << "terminating optimization." << std::endl;

      Callback::EndOptimization(*this, f, iterate, callbacks...);
      return overallObjective;
    }

    // Reset the counter variables.
    lastObjective = overallObjective;

    // And update the iterate.
    iterate -= stepSize * gradient;
    terminate |= Callback::StepTaken(*this, f, iterate, callbacks...);
  }

  Info << "Gradient Descent: maximum iterations (" << maxIterations
      << ") reached; " << "terminating optimization." << std::endl;

  Callback::EndOptimization(*this, f, iterate, callbacks...);
  return overallObjective;
}

template<typename FunctionType,
         typename MatType,
         typename GradType,
         typename... CallbackTypes>
typename std::enable_if<IsArmaType<GradType>::value,
typename MatType::elem_type>::type
GradientDescent::Optimize(
    FunctionType& function,
    MatType& iterate,
    const std::vector<bool>& categoricalDimensions,
    const arma::Row<size_t>& numCategories,
    CallbackTypes&&... callbacks)
{
  if (categoricalDimensions.size() != iterate.n_rows)
  {
    std::ostringstream oss;
    oss << "GradientDescent::Optimize(): expected information about "
        << iterate.n_rows << " dimensions in categoricalDimensions, "
        << "but got " << categoricalDimensions.size();
    throw std::invalid_argument(oss.str());
  }

  if (numCategories.n_elem != iterate.n_rows)
  {
    std::ostringstream oss;
    oss << "GradientDescent::Optimize(): expected numCategories to have length "
        << "equal to number of dimensions (" << iterate.n_rows << ") but it has"
        << " length " << numCategories.n_elem;
    throw std::invalid_argument(oss.str());
  }

  for (size_t i = 0; i < categoricalDimensions.size(); ++i)
  {
    if (categoricalDimensions[i])
    {
      std::ostringstream oss;
      oss << "GradientDescent::Optimize(): the dimension " << i
          << "is not numeric" << std::endl;
      throw std::invalid_argument(oss.str());
    }
  }

  return Optimize(function, iterate, callbacks...);
}

} // namespace ens

#endif

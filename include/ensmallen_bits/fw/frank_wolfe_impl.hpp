/**
 * @file frank_wolfe_impl.hpp
 * @author Chenzhe Diao
 *
 * Frank-Wolfe Algorithm.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_FW_FRANK_WOLFE_IMPL_HPP
#define ENSMALLEN_FW_FRANK_WOLFE_IMPL_HPP

// In case it hasn't been included yet.
#include "frank_wolfe.hpp"

#include <ensmallen_bits/function.hpp>

namespace ens {

//! Constructor of the FrankWolfe class.
template<
    typename LinearConstrSolverType,
    typename UpdateRuleType>
FrankWolfe<LinearConstrSolverType, UpdateRuleType>::
FrankWolfe(const LinearConstrSolverType linearConstrSolver,
           const UpdateRuleType updateRule,
           const size_t maxIterations,
           const double tolerance) :
    linearConstrSolver(linearConstrSolver),
    updateRule(updateRule),
    maxIterations(maxIterations),
    tolerance(tolerance)
{ /* Nothing to do*/ }


//! Optimize the function (minimize).
template<
    typename LinearConstrSolverType,
    typename UpdateRuleType>
template<typename FunctionType, typename MatType, typename GradType,
         typename... CallbackTypes>
typename std::enable_if<IsArmaType<GradType>::value,
typename MatType::elem_type>::type
FrankWolfe<LinearConstrSolverType, UpdateRuleType>::Optimize(
  FunctionType& function,
  MatType& iterateIn,
  CallbackTypes&&... callbacks)
{
  // Convenience typedefs.
  typedef typename MatType::elem_type ElemType;
  typedef typename MatTypeTraits<MatType>::BaseMatType BaseMatType;
  typedef typename MatTypeTraits<GradType>::BaseMatType BaseGradType;

  typedef Function<FunctionType, BaseMatType, BaseGradType> FullFunctionType;
  FullFunctionType& f = static_cast<FullFunctionType&>(function);

  // Make sure we have all necessary functions.
  traits::CheckFunctionTypeAPI<FullFunctionType, BaseMatType, BaseGradType>();
  RequireFloatingPointType<BaseMatType>();
  RequireFloatingPointType<BaseGradType>();
  RequireSameInternalTypes<BaseMatType, BaseGradType>();

  BaseMatType& iterate = (BaseMatType&) iterateIn;

  // To keep track of the function value.
  ElemType currentObjective = std::numeric_limits<ElemType>::max();

  BaseGradType gradient(iterate.n_rows, iterate.n_cols);
  BaseMatType s(iterate.n_rows, iterate.n_cols);
  BaseMatType iterateNew(iterate.n_rows, iterate.n_cols);
  double gap = 0;

  // Controls early termination of the optimization process.
  bool terminate = false;

  Callback::BeginOptimization(*this, f, iterate, callbacks...);
  for (size_t i = 1; i != maxIterations && !terminate; ++i)
  {
    currentObjective = f.EvaluateWithGradient(iterate, gradient);

    terminate |= Callback::EvaluateWithGradient(*this, f, iterate,
        currentObjective, gradient, callbacks...);

    // Output current objective function.
    Info << "FrankWolfe::Optimize(): iteration " << i << ", objective "
        << currentObjective << "." << std::endl;

    // Solve linear constrained problem, solution saved in s.
    linearConstrSolver.Optimize(gradient, s, callbacks...);

    // Check duality gap for return condition.
    gap = std::fabs(dot(iterate - s, gradient));
    if (gap < tolerance)
    {
      Info << "FrankWolfe::Optimize(): minimized within tolerance "
          << tolerance << "; " << "terminating optimization." << std::endl;

      Callback::EndOptimization(*this, f, iterate, callbacks...);
      return currentObjective;
    }

    // Update solution, save in iterateNew.
    updateRule.template Update<FunctionType, BaseMatType, BaseGradType>(f,
        iterate, s, iterateNew, i);

    iterate = std::move(iterateNew);
    terminate |= Callback::StepTaken(*this, f, iterate, callbacks...);
  }

  Info << "FrankWolfe::Optimize(): maximum iterations (" << maxIterations
      << ") reached; " << "terminating optimization." << std::endl;

  Callback::EndOptimization(*this, f, iterate, callbacks...);
  return currentObjective;
} // Optimize()

} // namespace ens

#endif

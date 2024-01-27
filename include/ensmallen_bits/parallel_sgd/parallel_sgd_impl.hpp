/**
 * @file parallel_sgd_impl.hpp
 * @author Shikhar Bhardwaj
 *
 * Implementation of Parallel Stochastic Gradient Descent.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PARALLEL_SGD_IMPL_HPP
#define ENSMALLEN_PARALLEL_SGD_IMPL_HPP

// In case it hasn't been included yet.
#include "parallel_sgd.hpp"

#include <ensmallen_bits/function.hpp>

namespace ens {

// Utility function to update a location of a dense matrix or other type using
// an atomic section.
template<typename MatType>
inline void UpdateLocation(MatType& iterate,
                           const size_t row,
                           const size_t col,
                           const typename MatType::elem_type value)
{
  ENS_PRAGMA_OMP_ATOMIC
  iterate(row, col) -= value;
}

// Utility function to update a location of a sparse matrix using a critical
// section.
template<typename eT>
inline void UpdateLocation(arma::SpMat<eT>& iterate,
                           const size_t row,
                           const size_t col,
                           const eT value)
{
  ENS_PRAGMA_OMP_CRITICAL_NAMED
  {
    iterate(row, col) -= value;
  }
}

template <typename DecayPolicyType>
ParallelSGD<DecayPolicyType>::ParallelSGD(
    const size_t maxIterations,
    const size_t threadShareSize,
    const double tolerance,
    const bool shuffle,
    const DecayPolicyType& decayPolicy) :
    maxIterations(maxIterations),
    threadShareSize(threadShareSize),
    tolerance(tolerance),
    shuffle(shuffle),
    decayPolicy(decayPolicy)
{ /* Nothing to do. */ }

template <typename DecayPolicyType>
template <typename SparseFunctionType,
          typename MatType,
          typename GradType,
          typename... CallbackTypes>
typename std::enable_if<IsArmaType<GradType>::value,
typename MatType::elem_type>::type ParallelSGD<DecayPolicyType>::Optimize(
    SparseFunctionType& function,
    MatType& iterateIn,
    CallbackTypes&&... callbacks)
{
  // Convenience typedefs.
  typedef typename MatType::elem_type ElemType;
  typedef typename MatTypeTraits<MatType>::BaseMatType BaseMatType;
  typedef typename MatTypeTraits<GradType>::BaseMatType BaseGradType;

  // Check that we have all the functions that we need.
  traits::CheckSparseFunctionTypeAPI<SparseFunctionType, BaseMatType,
      BaseGradType>();
  RequireFloatingPointType<BaseMatType>();
  RequireFloatingPointType<BaseGradType>();
  RequireSameInternalTypes<BaseMatType, BaseGradType>();

  BaseMatType& iterate = (BaseMatType&) iterateIn;

  ElemType overallObjective = DBL_MAX;
  ElemType lastObjective;

  // Controls early termination of the optimization process.
  bool terminate = false;

  // The order in which the functions will be visited.
  // TODO: maybe use function.Shuffle() instead?
  arma::Col<size_t> visitationOrder = arma::linspace<arma::Col<size_t>>(0,
      (function.NumFunctions() - 1), function.NumFunctions());

  // Iterate till the objective is within tolerance or the maximum number of
  // allowed iterations is reached. If maxIterations is 0, this will iterate
  // till convergence.
  Callback::BeginOptimization(*this, function, iterate, callbacks...);
  for (size_t i = 1; i != maxIterations && !terminate; ++i)
  {
    // Calculate the overall objective.
    lastObjective = overallObjective;

    overallObjective = function.Evaluate(iterate);

    terminate |= Callback::Evaluate(*this, function, iterate, overallObjective,
        callbacks...);

    // Output current objective function.
    Info << "Parallel SGD: iteration " << i << ", objective "
      << overallObjective << "." << std::endl;

    if (std::isnan(overallObjective) || std::isinf(overallObjective))
    {
      Warn << "Parallel SGD: converged to " << overallObjective
        << "; terminating with failure. Try a smaller step size?"
        << std::endl;

      Callback::EndOptimization(*this, function, iterate, callbacks...);
      return overallObjective;
    }

    if (std::abs(lastObjective - overallObjective) < tolerance)
    {
      Info << "SGD: minimized within tolerance " << tolerance << "; "
        << "terminating optimization." << std::endl;

      Callback::EndOptimization(*this, function, iterate, callbacks...);
      return overallObjective;
    }

    // Get the stepsize for this iteration
    double stepSize = decayPolicy.StepSize(i);

    // Shuffle for uniform sampling of functions by each thread.
    if (shuffle)
    {
      // Determine order of visitation.
      visitationOrder = arma::shuffle(visitationOrder);
    }

    ENS_PRAGMA_OMP_PARALLEL
    {
      // Each processor gets a subset of the instances.
      // Each subset is of size threadShareSize
      size_t threadId = 0;
      #ifdef ENS_USE_OPENMP
        threadId = omp_get_thread_num();
      #endif

      for (size_t j = threadId * threadShareSize;
          j < (threadId + 1) * threadShareSize && j < visitationOrder.n_elem;
          ++j)
      {
        // Each instance affects only some components of the decision variable.
        // So the gradient is sparse.
        BaseGradType gradient;

        // Evaluate the sparse gradient.
        // TODO: support for batch size > 1 could be really useful.
        function.Gradient(iterate, visitationOrder[j], gradient, 1);

        terminate |= Callback::Gradient(*this, function, iterate, gradient,
            callbacks...);

        // Update the decision variable with non-zero components of the
        // gradient.
        for (size_t i = 0; i < gradient.n_cols; ++i)
        {
          // Iterate over the non-zero elements.
          const typename GradType::iterator curEnd = gradient.end_col(i);
          for (typename GradType::iterator cur = gradient.begin_col(i);
              cur != curEnd; ++cur)
          {
            const ElemType value = (*cur);
            const arma::uword row = cur.row();

            // Call out to utility function to use the right type of OpenMP
            // lock.
            UpdateLocation(iterate, row, i, stepSize * value);
          }
        }
        terminate |= Callback::StepTaken(*this, function, iterate,
            callbacks...);
      }
    }
  }

  Info << "\nParallel SGD terminated with objective : " << overallObjective
      << "." << std::endl;

  Callback::EndOptimization(*this, function, iterate, callbacks...);
  return overallObjective;
}

} // namespace ens

#endif

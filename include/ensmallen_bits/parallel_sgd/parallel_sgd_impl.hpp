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
  ENS_PRAGMA_OMP_CRITICAL
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
template <typename SparseFunctionType, typename MatType, typename GradType>
typename MatType::elem_type ParallelSGD<DecayPolicyType>::Optimize(
    SparseFunctionType& function,
    MatType& iterate)
{
  typedef typename MatType::elem_type ElemType;

  // Check that we have all the functions that we need.
  //traits::CheckSparseFunctionTypeAPI<SparseFunctionType, MatType, GradType>();

  ElemType overallObjective = DBL_MAX;
  ElemType lastObjective;

  // The order in which the functions will be visited.
  // TODO: maybe use function.Shuffle() instead?
  arma::Col<size_t> visitationOrder = arma::linspace<arma::Col<size_t>>(0,
      (function.NumFunctions() - 1), function.NumFunctions());

  // Iterate till the objective is within tolerance or the maximum number of
  // allowed iterations is reached. If maxIterations is 0, this will iterate
  // till convergence.
  for (size_t i = 1; i != maxIterations; ++i)
  {
    // Calculate the overall objective.
    lastObjective = overallObjective;

    overallObjective = function.Evaluate(iterate);

    // Output current objective function.
    Info << "Parallel SGD: iteration " << i << ", objective "
      << overallObjective << "." << std::endl;

    if (std::isnan(overallObjective) || std::isinf(overallObjective))
    {
      Warn << "Parallel SGD: converged to " << overallObjective
        << "; terminating with failure. Try a smaller step size?"
        << std::endl;
      return overallObjective;
    }

    if (std::abs(lastObjective - overallObjective) < tolerance)
    {
      Info << "SGD: minimized within tolerance " << tolerance << "; "
        << "terminating optimization." << std::endl;
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
        GradType gradient;

        // Evaluate the sparse gradient.
        // TODO: support for batch size > 1 could be really useful.
        function.Gradient(iterate, visitationOrder[j], gradient, 1);

        // Update the decision variable with non-zero components of the
        // gradient.
        for (size_t i = 0; i < gradient.n_cols; ++i)
        {
          // Iterate over the non-zero elements.
          for (typename GradType::iterator cur = gradient.begin_col(i);
              cur != gradient.end_col(i); ++cur)
          {
            const ElemType value = (*cur);
            const arma::uword row = cur.row();

            // Call out to utility function to use the right type of OpenMP
            // lock.
            UpdateLocation(iterate, row, i, stepSize * value);
          }
        }
      }
    }
  }

  Info << "\nParallel SGD terminated with objective : " << overallObjective
      << "." << std::endl;
  return overallObjective;
}

} // namespace ens

#endif

/**
 * @file ada_sqrt_update.hpp
 * @author Marcus Edel
 *
 * AdaSqrt update for Stochastic Gradient Descent.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_ADA_SQRT_ADA_SQRT_UPDATE_HPP
#define ENSMALLEN_ADA_SQRT_ADA_SQRT_UPDATE_HPP

namespace ens {

/**
 * Implementation of the AdaSqrt update policy. AdaSqrt update policy chooses
 * learning rate dynamically by adapting to the data and iteration.
 *
 * For more information, see the following.
 *
 * @code
 * @misc{hu2019secondorder,
 *   title  = {Second-order Information in First-order Optimization Methods},
 *   author = {Yuzheng Hu and Licong Lin and Shange Tang},
 *   year   = {2019},
 *   eprint = {1912.09926},
 * }
 * @endcode
 *
 */
class AdaSqrtUpdate
{
 public:
  /**
   * Construct the AdaSqrt update policy with given epsilon parameter.
   *
   * @param epsilon The epsilon value used to initialise the squared gradient
   *        parameter.
   */
  AdaSqrtUpdate(const double epsilon = 1e-8) : epsilon(epsilon)
  {
    // Nothing to do.
  }

  //! Get the value used to initialise the squared gradient parameter.
  double Epsilon() const { return epsilon; }
  //! Modify the value used to initialise the squared gradient parameter.
  double& Epsilon() { return epsilon; }

  /**
   * The UpdatePolicyType policy classes must contain an internal 'Policy'
   * template class with two template arguments: MatType and GradType.  This is
   * instantiated at the start of the optimization, and holds parameters
   * specific to an individual optimization.
   */
  template<typename MatType, typename GradType>
  class Policy
  {
   public:
    /**
     * This constructor is called by the SGD optimizer before the start of the
     * iteration update process. In AdaSqrt update policy, squared gradient
     * matrix is initialized to the zeros matrix with the same size as gradient
     * matrix (see ens::SGD<>).
     *
     * @param parent Instantiated parent class.
     * @param rows Number of rows in the gradient matrix.
     * @param cols Number of columns in the gradient matrix.
     */
    Policy(AdaSqrtUpdate& parent, const size_t rows, const size_t cols) :
        parent(parent),
        squaredGradient(rows, cols),
        iteration(0)
    {
      // Initialize an empty matrix for sum of squares of parameter gradient.
      squaredGradient.zeros();
    }

    /**
     * Update step for SGD. The AdaSqrt update adapts the learning rate by
     * performing larger updates for more sparse parameters and smaller updates
     * for less sparse parameters.
     *
     * @param iterate Parameters that minimize the function.
     * @param stepSize Step size to be used for the given iteration.
     * @param gradient The gradient matrix.
     */
    void Update(MatType& iterate,
                const double stepSize,
                const GradType& gradient)
    {
      ++iteration;

      squaredGradient += arma::square(gradient);

      iterate -= stepSize * std::sqrt(iteration) * gradient /
          (squaredGradient + parent.epsilon);
    }

   private:
    // Instantiated parent class.
    AdaSqrtUpdate& parent;
    // The squared gradient matrix.
    GradType squaredGradient;
    // The number of iterations.
    size_t iteration;
  };

 private:
  // The epsilon value used to initialise the squared gradient parameter.
  double epsilon;
};

} // namespace ens

#endif

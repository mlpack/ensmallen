/**
 * @file smorms3_update.hpp
 * @author Vivek Pal
 *
 * SMORMS3 update for Stochastic Gradient Descent.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_SMORMS3_SMORMS3_UPDATE_HPP
#define ENSMALLEN_SMORMS3_SMORMS3_UPDATE_HPP

namespace ens {

/**
 * SMORMS3 is an optimizer that estimates a safe and optimal distance based on
 * curvature and normalizing the stepsize in the parameter space. It is a hybrid
 * of RMSprop and Yann LeCun’s method in "No more pesky learning rates".
 *
 * For more information, see the following.
 *
 * @code
 * @misc{Funk2015,
 *   author = {Simon Funk},
 *   title  = {RMSprop loses to SMORMS3 - Beware the Epsilon!},
 *   year   = {2015}
 *   url    = {http://sifter.org/~simon/journal/20150420.html}
 * }
 * @endcode
 */

class SMORMS3Update
{
 public:
  /**
   * Construct the SMORMS3 update policy with given epsilon parameter.
   *
   * @param epsilon Value used to initialise the mean squared gradient
   *        parameter.
   */
  SMORMS3Update(const double epsilon = 1e-16) : epsilon(epsilon)
  { /* Do nothing. */ }

  //! Get the value used to initialise the mean squared gradient parameter.
  double Epsilon() const { return epsilon; }
  //! Modify the value used to initialise the mean squared gradient parameter.
  double& Epsilon() { return epsilon; }

  /**
   * The UpdatePolicyType policy classes must contain an internal 'Policy'
   * template class with two template arguments: MatType and GradType.  This is
   * instantiated at the start of the optimization.
   */
  template<typename MatType, typename GradType>
  class Policy
  {
   public:
    /**
     * This is called by the optimizer method before the start of the iteration
     * update process.
     *
     * @param parent Instantiated parent class.
     * @param rows Number of rows in the gradient matrix.
     * @param cols Number of columns in the gradient matrix.
     */
    Policy(SMORMS3Update& parent, const size_t rows, const size_t cols) :
        parent(parent)
    {
      // Initialise the parameters mem, g and g2.
      mem.ones(rows, cols);
      g.zeros(rows, cols);
      g2.zeros(rows, cols);
    }

    /**
     * Update step for SMORMS3.
     *
     * @param iterate Parameter that minimizes the function.
     * @param stepSize Step size to be used for the given iteration.
     * @param gradient The gradient matrix.
     */
    void Update(MatType& iterate,
                const double stepSize,
                const GradType& gradient)
    {
      // Update the iterate.
      MatType r = 1 / (mem + 1);

      g = (1 - r) % g;
      g += r % gradient;

      g2 = (1 - r) % g2;
      g2 += r % (gradient % gradient);

      MatType x = (g % g) / (g2 + parent.epsilon);

      x.transform( [stepSize](typename MatType::elem_type &v)
          { return std::min(v, (typename MatType::elem_type) stepSize); } );

      iterate -= gradient % x / (arma::sqrt(g2) + parent.epsilon);

      mem %= (1 - x);
      mem += 1;
    }

   private:
    // Instantiated parent object.
    SMORMS3Update& parent;
    // Memory parameter.
    MatType mem;
    // Gradient estimate parameter.
    GradType g;
    // Squared gradient estimate parameter.
    GradType g2;
  };

 private:
  //! The value used to initialise the mean squared gradient parameter.
  double epsilon;
};

} // namespace ens

#endif

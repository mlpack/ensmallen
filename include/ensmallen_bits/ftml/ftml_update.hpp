/**
 * @file ftml_update.hpp
 * @author Marcus Edel
 *
 * FTML update for Follow the Moving Leader.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_FTML_FTML_UPDATE_HPP
#define ENSMALLEN_FTML_FTML_UPDATE_HPP

namespace ens {

/**
 * Follow the Moving Leader (FTML) is an optimizer where recent samples are
 * weighted more heavily in each iteration, so FTML can adapt more quickly to
 * changes.
 *
 * For more information, see the following.
 *
 * @code
 * @inproceedings{Zheng2017,
 *   author    = {Shuai Zheng and James T. Kwok},
 *   title     = {Follow the Moving Leader in Deep Learning},
 *   year      = {2017}
 *   booktitle = {Proceedings of the 34th International Conference on Machine
 *                Learning},
 *   pages     = {4110--4119},
 *   series    = {Proceedings of Machine Learning Research},
 *   publisher = {PMLR},
 * }
 * @endcode
 */
class FTMLUpdate
{
 public:
  /**
   * Construct the FTML update policy with given epsilon parameter.
   *
   * @param epsilon Epsilon is the minimum allowed gradient.
   * @param beta1 Exponential decay rate for the first moment estimates.
   * @param beta2 Exponential decay rate for the weighted infinity norm
   *        estimates.
   */
  FTMLUpdate(const double epsilon = 1e-8,
             const double beta1 = 0.9,
             const double beta2 = 0.999) :
      epsilon(epsilon),
      beta1(beta1),
      beta2(beta2)
  { /* Do nothing. */ }

  //! Get the value used to initialise the squared gradient parameter.
  double Epsilon() const { return epsilon; }
  //! Modify the value used to initialise the squared gradient parameter.
  double& Epsilon() { return epsilon; }

  //! Get the smoothing parameter.
  double Beta1() const { return beta1; }
  //! Modify the smoothing parameter.
  double& Beta1() { return beta1; }

  //! Get the second moment coefficient.
  double Beta2() const { return beta2; }
  //! Modify the second moment coefficient.
  double& Beta2() { return beta2; }

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
     * This constructor is called by the SGD Optimize() method before the start
     * of the iteration update process.
     *
     * @param parent AdamUpdate object.
     * @param rows Number of rows in the gradient matrix.
     * @param cols Number of columns in the gradient matrix.
     */
    Policy(FTMLUpdate& parent, const size_t rows, const size_t cols) :
        parent(parent)
    {
      v.zeros(rows, cols);
      z.zeros(rows, cols);
      d.zeros(rows, cols);
    }

    /**
     * Update step for FTML.
     *
     * @param iterate Parameter that minimizes the function.
     * @param stepSize Step size to be used for the given iteration.
     * @param gradient The gradient matrix.
     */
    void Update(MatType& iterate,
                const double stepSize,
                const GradType& gradient)
    {
      // Increment the iteration counter variable.
      ++iteration;

      // And update the iterate.
      v *= parent.beta2;
      v += (1 - parent.beta2) * (gradient % gradient);

      const double biasCorrection1 = 1.0 - std::pow(parent.beta1, iteration);
      const double biasCorrection2 = 1.0 - std::pow(parent.beta2, iteration);

      MatType sigma = -parent.beta1 * d;
      d = biasCorrection1 / stepSize *
        (arma::sqrt(v / biasCorrection2) + parent.epsilon);
      sigma += d;

      z *= parent.beta1;
      z += (1 - parent.beta1) * gradient - sigma % iterate;
      iterate = -z / d;
    }

   private:
    // Reference to instantiated parent object.
    FTMLUpdate& parent;

    // The exponential moving average of gradient values.
    GradType v;

    // The exponential moving average of squared gradient values.
    GradType z;

    // Parameter update term.
    MatType d;

    // The number of iterations.
    size_t iteration;
  };

 private:
  // The epsilon value used to initialise the squared gradient parameter.
  double epsilon;

  // The smoothing parameter.
  double beta1;

  // The second moment coefficient.
  double beta2;
};

} // namespace ens

#endif

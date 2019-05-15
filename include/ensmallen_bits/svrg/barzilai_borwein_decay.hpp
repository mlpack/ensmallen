/**
 * @file barzilai_borwein_decay.hpp
 * @author Marcus Edel
 *
 * Barzilai-Borwein decay policy.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_SVRG_BARZILIA_BORWEIN_DECAY_HPP
#define ENSMALLEN_SVRG_BARZILIA_BORWEIN_DECAY_HPP

namespace ens {

/**
 * Barzilai-Borwein decay policy for Stochastic variance reduced gradient
 * (SVRG).
 *
 * For more information, please refer to:
 *
 * @code
 * @incollection{Tan2016,
 *   title     = {Barzilai-Borwein Step Size for Stochastic Gradient Descent},
 *   author    = {Tan, Conghui and Ma, Shiqian and Dai, Yu-Hong
 *                and Qian, Yuqiu},
 *   booktitle = {Advances in Neural Information Processing Systems 29},
 *   editor    = {D. D. Lee and M. Sugiyama and U. V. Luxburg and I. Guyon
 *                and R. Garnett},
 *   pages     = {685--693},
 *   year      = {2016},
 *   publisher = {Curran Associates, Inc.}
 * }
 * @endcode
 */
class BarzilaiBorweinDecay
{
 public:
  /*
   * Construct the Barzilai-Borwein decay policy.
   *
   * @param maxStepSize The maximum step size.
   * @param eps The eps coefficient to avoid division by zero (numerical
   *    stability).
   */
  BarzilaiBorweinDecay(const double maxStepSize = DBL_MAX,
                       const double epsilon = 1e-7) :
      epsilon(epsilon),
      maxStepSize(maxStepSize)
  { /* Nothing to do. */}

  //! Get the numerical stability parameter.
  double Epsilon() const { return epsilon; }
  //! Modify the numerical stability parameter.
  double& Epsilon() { return epsilon; }

  //! Get the maximum step size.
  double MaxStepSize() const { return maxStepSize; }
  //! Modify the maximum step size.
  double& MaxStepSize() { return maxStepSize; }

  /**
   * The DecayPolicyType policy classes must contain an internal 'Policy'
   * template class with two template arguments: MatType and GradType.  This is
   * initialized at the start of the optimization, and holds parameters specific
   * to an individual optimization.
   */
  template<typename MatType, typename GradType>
  class Policy
  {
   public:
    /**
     * This constructor is called by the SGD Optimize() method before the start
     * of the iteration update process.
     */
    Policy(BarzilaiBorweinDecay& parent) : parent(parent) { /* Do nothing. */ }

    /**
     * Barzilai-Borwein update step for SVRG.
     *
     * @param iterate The current function parameter at time t.
     * @param iterate0 The last function parameters at time t - 1.
     * @param gradient The current gradient matrix at time t.
     * @param fullGradient The computed full gradient.
     * @param numBatches The number of batches.
     * @param stepSize Step size to be used for the given iteration.
     */
    void Update(const MatType& iterate,
                const MatType& iterate0,
                const GradType& /* gradient */,
                const GradType& fullGradient,
                const size_t numBatches,
                double& stepSize)
    {
      if (!fullGradient0.is_empty())
      {
        // Step size selection based on Barzilai-Borwein (BB).
        stepSize = std::pow(arma::norm(iterate - iterate0), 2.0) /
            (arma::dot(iterate - iterate0, fullGradient - fullGradient0) +
             parent.epsilon) / (double) numBatches;

        stepSize = std::min(stepSize, parent.maxStepSize);
      }

      fullGradient0 = std::move(fullGradient);
    }

   private:
    //! Reference to instantiated parent object.
    BarzilaiBorweinDecay& parent;

    //! Locally-stored full gradient.
    GradType fullGradient0;
  };

  //! The value used for numerical stability.
  double epsilon;

  //! The maximum step size.
  double maxStepSize;
};

} // namespace ens

#endif // ENSMALLEN_SVRG_BARZILIA_BORWEIN_DECAY_HPP

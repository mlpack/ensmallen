/**
 * @file spalera_stepsize.hpp
 * @author Marcus Edel
 *
 * Definition of the SPALeRA stepsize technique as described in:
 * "Stochastic Gradient Descent: Going As Fast As Possible But Not Faster" by
 * A. Schoenauer Sebag et al.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_SPALERA_SGD_SPALERA_STEPSIZE_HPP
#define ENSMALLEN_SPALERA_SGD_SPALERA_STEPSIZE_HPP

namespace ens {

/**
 * Definition of the SPALeRA stepize technique, which implementes a change
 * detection mechanism with an agnostic adaptation scheme.
 *
 * For more information, please refer to:
 *
 * @code
 * @misc{Schoenauer2017,
 *   title   = {Stochastic Gradient Descent:
 *              Going As Fast As Possible But Not Faster},
 *   author  = {Schoenauer-Sebag, Alice; Schoenauer, Marc; Sebag, Michele},
 *   journal = {CoRR},
 *   year    = {2017},
 *   url     = {https://arxiv.org/abs/1709.01427},
 * }
 * @endcode
 */
class SPALeRAStepsize
{
 public:
  /**
   * Construct the SPALeRAStepsize object with the given parameters.
   * The defaults here are not necessarily good for the given
   * problem, so it is suggested that the values used be tailored to the task at
   * hand.
   *
   * @param alpha Memory parameter of the agnostic learning rate adaptation.
   * @param epsilon Numerical stability parameter.
   * @param adaptRate Agnostic learning rate update rate.
   */
  SPALeRAStepsize(const double alpha = 0.001,
                  const double epsilon = 1e-6,
                  const double adaptRate = 3.10e-8) :
      alpha(alpha),
      epsilon(epsilon),
      adaptRate(adaptRate),
      lambda(0)
  {
    /* Nothing to do here. */
  }

  //! Get the agnostic learning rate adaptation parameter.
  double Alpha() const { return alpha; }
  //! Modify the agnostic learning rate adaptation parameter.
  double& Alpha() { return alpha; }

  //! Get the numerical stability parameter.
  double Epsilon() const { return epsilon; }
  //! Modify the numerical stability parameter.
  double& Epsilon() { return epsilon; }

  //! Get the agnostic learning rate update rate.
  double AdaptRate() const { return adaptRate; }
  //! Modify the agnostic learning rate update rate.
  double& AdaptRate() { return adaptRate; }

  //! Get the Page-Hinkley update parameter lambda.
  double Lambda() const { return lambda; }
  //! Modify the Page-Hinkley update parameter lambda.
  double& Lambda() { return lambda; }

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
     * @param lambda Page-Hinkley update parameter.
     */
    Policy(SPALeRAStepsize& parent,
           const size_t rows,
           const size_t cols,
           const double lambda) :
        parent(parent),
        mu0(0),
        un(0),
        mn(0),
        relaxedObjective(0),
        phCounter(0),
        eveCounter(0)
    {
      learningRates.ones(rows, cols);
      relaxedSums.zeros(rows, cols);

      parent.lambda = lambda;
    }

    /**
     * This function is called in each iteration.
     *
     * @param stepSize Step size to be used for the given iteration.
     * @param objective The current function loss.
     * @param batchSize Batch size to be used for the given iteration.
     * @param numFunctions The number of functions.
     * @param iterate Parameters that minimize the function.
     * @param gradient The gradient matrix.
     *
     * @return Stop or continue the learning process.
     */
    bool Update(const double stepSize,
                const typename MatType::elem_type objective,
                const size_t batchSize,
                const size_t numFunctions,
                MatType& iterate,
                const GradType& gradient)
    {
      // The ratio of mini-batch size to training set size; needed for the
      // Page-Hinkley relaxed objective computations.
      const double mbRatio = batchSize / (double) numFunctions;

      // Page-Hinkley iteration, check if we have to reset the parameter and
      // adjust the step size.
      if (phCounter > (1 / mbRatio))
      {
        relaxedObjective = (1 - mbRatio) * relaxedObjective + mbRatio *
            objective;
      }
      else
      {
        relaxedObjective = phCounter * relaxedObjective + objective;
        relaxedObjective /= (phCounter + 1);
      }

      // Update the mu0 parameter.
      mu0 = phCounter * mu0 + relaxedObjective;
      mu0 = mu0 / (phCounter + 1);

      // Update the un parameter.
      un += relaxedObjective - mu0;

      // Updating the mn parameter.
      if (un < mn)
        mn = un;

      // If the condition is true we reset the parameter and update parameter.
      if ((un - mn) > parent.lambda)
      {
        // Backtracking, reset the parameter.
        iterate = previousIterate;

        // Dividing learning rates by 2 as proposed in:
        // Stochastic Gradient Descent: Going As Fast As Possible But Not
        // Faster.
        learningRates /= 2;

        if (arma::any(arma::vectorise(learningRates) <= 1e-15))
        {
          // Stop because learning rate too low.
          return false;
        }

        // Reset evaluation and Page-Hinkley counter parameter.
        mu0 = un = mn = relaxedObjective = phCounter = eveCounter = 0;
      }
      else
      {
        const double paramMean = (parent.alpha / (2 - parent.alpha) *
            (1 - std::pow(1 - parent.alpha, 2 * (eveCounter + 1)))) /
            iterate.n_elem;

        const double paramStd = (parent.alpha / std::sqrt(iterate.n_elem)) /
            std::sqrt(iterate.n_elem);

        const typename MatType::elem_type normGradient =
            std::sqrt(arma::accu(arma::pow(gradient, 2)));

        relaxedSums *= (1 - parent.alpha);
        if (normGradient > parent.epsilon)
          relaxedSums += gradient * (parent.alpha / normGradient);

        learningRates %= arma::exp((arma::pow(relaxedSums, 2) - paramMean) *
            (parent.adaptRate / paramStd));

        previousIterate = iterate;

        iterate -= stepSize * (learningRates % gradient);

        // Keep track of the the number of evaluations and Page-Hinkley steps.
        eveCounter++;
        phCounter++;
      }

      return true;
    }

   private:
    //! Instantiated parent object.
    SPALeRAStepsize& parent;

    //! Page-Hinkley update parameter.
    double mu0;

    //! Page-Hinkley update parameter.
    double un;

    //! Page-Hinkley update parameter.
    double mn;

    //! Page-Hinkley update parameter.
    typename MatType::elem_type relaxedObjective;

    //! Page-Hinkley step counter.
    size_t phCounter;

    //! Evaluations step counter.
    size_t eveCounter;

    //! Locally-stored parameter wise learning rates.
    MatType learningRates;

    //! Locally-stored parameter wise sums.
    MatType relaxedSums;

    //! Locally-stored previous parameter matrix (backtracking).
    MatType previousIterate;
  };

 private:

  //! Memory parameter of the agnostic learning rate adaptation.
  double alpha;

  //! Numerical stability parameter.
  double epsilon;

  //! Agnostic learning rate update rate.
  double adaptRate;

  //! Page-Hinkley update parameter.
  double lambda;
};

} // namespace ens

#endif // ENSMALLEN_SPALERA_SGD_SPALERA_STEPSIZE_HPP

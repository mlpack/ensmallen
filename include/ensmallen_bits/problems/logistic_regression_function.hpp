/**
 * @file logistic_regression_function.hpp
 * @author Sumedh Ghaisas
 *
 * Implementation of the logistic regression function, which is meant to be
 * optimized by a separate optimizer class that takes LogisticRegressionFunction
 * as its FunctionType class.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_LOGISTIC_REGRESSION_FUNCTION_HPP
#define ENSMALLEN_PROBLEMS_LOGISTIC_REGRESSION_FUNCTION_HPP

namespace ens {
namespace test {

/**
 * The log-likelihood function for the logistic regression objective function.
 * This is used by various ensmallen optimizers to train a logistic regression
 * model.
 */
template<typename MatType = arma::mat>
class LogisticRegressionFunction
{
 public:
  LogisticRegressionFunction(MatType& predictors,
                             arma::Row<size_t>& responses,
                             const double lambda = 0);

  LogisticRegressionFunction(MatType& predictors,
                             arma::Row<size_t>& responses,
                             MatType& initialPoint,
                             const double lambda = 0);

  //! Return the initial point for the optimization.
  const MatType& InitialPoint() const { return initialPoint; }
  //! Modify the initial point for the optimization.
  MatType& InitialPoint() { return initialPoint; }

  //! Return the regularization parameter (lambda).
  const double& Lambda() const { return lambda; }
  //! Modify the regularization parameter (lambda).
  double& Lambda() { return lambda; }

  //! Return the matrix of predictors.
  const MatType& Predictors() const { return predictors; }
  //! Return the vector of responses.
  const arma::Row<size_t>& Responses() const { return responses; }

  /**
   * Shuffle the order of function visitation.  This may be called by the
   * optimizer.
   */
  void Shuffle();

  /**
   * Evaluate the logistic regression log-likelihood function with the given
   * parameters.  Note that if a point has 0 probability of being classified
   * directly with the given parameters, then Evaluate() will return nan (this
   * is kind of a corner case and should not happen for reasonable models).
   *
   * The optimum (minimum) of this function is 0.0, and occurs when each point
   * is classified correctly with very high probability.
   *
   * @param parameters Vector of logistic regression parameters.
   */
  typename MatType::elem_type Evaluate(const MatType& parameters) const;

  /**
   * Evaluate the logistic regression log-likelihood function with the given
   * parameters using the given batch size from the given point index.  This is
   * useful for optimizers such as SGD, which require a separable objective
   * function.  Note that if the points have 0 probability of being classified
   * correctly with the given parameters, then Evaluate() will return nan (this
   * is kind of a corner case and should not happen for reasonable models).
   *
   * The optimum (minimum) of this function is 0.0, and occurs when the points
   * are classified correctly with very high probability.
   *
   * @param parameters Vector of logistic regression parameters.
   * @param begin Index of the starting point to use for objective function
   *     evaluation.
   * @param batchSize Number of points to be passed at a time to use for
   *     objective function evaluation.
   */
  typename MatType::elem_type Evaluate(const MatType& parameters,
                                       const size_t begin,
                                       const size_t batchSize = 1) const;

  /**
   * Evaluate the gradient of the logistic regression log-likelihood function
   * with the given parameters.
   *
   * @param parameters Vector of logistic regression parameters.
   * @param gradient Vector to output gradient into.
   */
  template<typename GradType>
  void Gradient(const MatType& parameters, GradType& gradient) const;

  /**
   * Evaluate the gradient of the logistic regression log-likelihood function
   * with the given parameters, for the given batch size from a given point the
   * in dataset. This is useful for optimizers such as SGD, which require a
   * separable objective function.
   *
   * @param parameters Vector of logistic regression parameters.
   * @param begin Index of the starting point to use for objective function
   *     gradient evaluation.
   * @param gradient Vector to output gradient into.
   * @param batchSize Number of points to be processed as a batch for objective
   *     function gradient evaluation.
   */
  template<typename GradType>
  void Gradient(const MatType& parameters,
                const size_t begin,
                GradType& gradient,
                const size_t batchSize = 1) const;

  /**
   * Evaluate the gradient of the logistic regression log-likelihood function
   * with the given parameters, and with respect to only one feature in the
   * dataset.  This is useful for optimizers such as SCD, which require
   * partial gradients.
   *
   * @param parameters Vector of logistic regression parameters.
   * @param j Index of the feature with respect to which the gradient is to
   *    be computed.
   * @param gradient Sparse matrix to output gradient into.
   */
  void PartialGradient(const MatType& parameters,
                       const size_t j,
                       arma::sp_mat& gradient) const;

  /**
   * Evaluate the objective function and gradient of the logistic regression
   * log-likelihood function simultaneously with the given parameters.
   */
  template<typename GradType>
  typename MatType::elem_type EvaluateWithGradient(
      const MatType& parameters,
      GradType& gradient) const;

  template<typename GradType>
  typename MatType::elem_type EvaluateWithGradient(
      const MatType& parameters,
      const size_t begin,
      GradType& gradient,
      const size_t batchSize = 1) const;

  //! Return the initial point for the optimization.
  const MatType& GetInitialPoint() const { return initialPoint; }

  //! Return the number of separable functions (the number of predictor points).
  size_t NumFunctions() const { return predictors.n_cols; }

  //! Return the number of features(add 1 for the intercept term).
  size_t NumFeatures() const { return predictors.n_rows + 1; }

  /**
   * Compute the accuracy of the model on the given predictors and responses,
   * optionally using the given decision boundary.  The responses should be
   * either 0 or 1.  Logistic regression returns a value between 0 and 1.  If
   * the value is greater than the decision boundary, the response is taken to
   * be 1; otherwise, it is 0.  By default, the decision boundary is 0.5.
   *
   * The accuracy is returned as a percentage, between 0 and 100.
   *
   * @param predictors Input predictors.
   * @param responses Vector of responses.
   * @param parameters Vector of logistic regression parameters.
   * @param decisionBoundary Decision boundary (default 0.5).
   * @return Percentage of responses that are predicted correctly.
   */
  double ComputeAccuracy(const MatType& predictors,
                         const arma::Row<size_t>& responses,
                         const MatType& parameters,
                         const double decisionBoundary = 0.5) const;

  /**
   * Classify the given points, returning the predicted labels for each point.
   * Optionally, specify the decision boundary; logistic regression returns a
   * value between 0 and 1.  If the value is greater than the decision boundary,
   * the response is taken to be 1; otherwise, it is 0.  By default the decision
   * boundary is 0.5.
   *
   * @param dataset Set of points to classify.
   * @param labels Predicted labels for each point.
   * @param parameters Vector of logistic regression parameters.
   * @param decisionBoundary Decision boundary (default 0.5).
   */
  void Classify(const MatType& dataset,
                arma::Row<size_t>& labels,
                const MatType& parameters,
                const double decisionBoundary = 0.5) const;

 private:
  //! The initial point, from which to start the optimization.
  MatType initialPoint;
  //! The matrix of data points (predictors).  This is an alias until shuffling
  //! is done.
  MatType& predictors;
  //! The vector of responses to the input data points.  This is an alias until
  //! shuffling is done.
  arma::Row<size_t>& responses;
  //! The regularization parameter for L2-regularization.
  double lambda;
};

// Convenience typedefs.
template<typename MatType = arma::mat>
using LogisticRegression = LogisticRegressionFunction<MatType>;

} // namespace test
} // namespace ens

// Include implementation.
#include "logistic_regression_function_impl.hpp"

#endif // ENSMALLEN_PROBLEMS_LOGISTIC_REGRESSION_FUNCTION_HPP

/**
 * @file logistic_regression_function.cpp
 * @author Sumedh Ghaisas
 *
 * Implementation of the LogisticRegressionFunction class.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_LOGISTIC_REGRESSION_FUNCTION_IMPL_HPP
#define ENSMALLEN_PROBLEMS_LOGISTIC_REGRESSION_FUNCTION_IMPL_HPP

// In case it hasn't been included yet.
#include "logistic_regression_function.hpp"

namespace ens {
namespace test {

template<typename MatType>
LogisticRegressionFunction<MatType>::LogisticRegressionFunction(
    MatType& predictors,
    arma::Row<size_t>& responses,
    const double lambda) :
    // We promise to be well-behaved... the elements won't be modified.
    predictors(predictors),
    responses(responses),
    lambda(lambda)
{
  initialPoint = arma::Row<typename MatType::elem_type>(predictors.n_rows + 1,
      arma::fill::zeros);

  // Sanity check.
  if (responses.n_elem != predictors.n_cols)
  {
    std::ostringstream oss;
    oss << "LogisticRegressionFunction::LogisticRegressionFunction(): "
        << "predictors matrix has " << predictors.n_cols << " points, but "
        << "responses vector has " << responses.n_elem << " elements (should be"
        << " " << predictors.n_cols << ")!" << std::endl;
    throw std::logic_error(oss.str());
  }
}

template<typename MatType>
LogisticRegressionFunction<MatType>::LogisticRegressionFunction(
    MatType& predictors,
    arma::Row<size_t>& responses,
    MatType& initialPoint,
    const double lambda) :
    initialPoint(initialPoint),
    predictors(predictors),
    responses(responses),
    lambda(lambda)
{
  // To check if initialPoint is compatible with predictors.
  if (initialPoint.n_rows != (predictors.n_rows + 1) ||
      initialPoint.n_cols != 1)
    this->initialPoint = arma::Row<typename MatType::elem_type>(
        predictors.n_rows + 1, arma::fill::zeros);
}

/**
 * Shuffle the datapoints.
 */
template<typename MatType>
void LogisticRegressionFunction<MatType>::Shuffle()
{
  MatType newPredictors;
  arma::Row<size_t> newResponses;

  arma::uvec ordering = arma::shuffle(arma::linspace<arma::uvec>(0,
      predictors.n_cols - 1, predictors.n_cols));

  newPredictors.set_size(predictors.n_rows, predictors.n_cols);
  for (size_t i = 0; i < predictors.n_cols; ++i)
    newPredictors.col(i) = predictors.col(ordering[i]);
  newResponses = responses.cols(ordering);

  // Take ownership of the new data.
  predictors = std::move(newPredictors);
  responses = std::move(newResponses);
}

/**
 * Evaluate the logistic regression objective function given the estimated
 * parameters.
 */
template<typename MatType>
typename MatType::elem_type LogisticRegressionFunction<MatType>::Evaluate(
    const MatType& parameters) const
{
  // The objective function is the log-likelihood function (w is the parameters
  // vector for the model; y is the responses; x is the predictors; sig() is the
  // sigmoid function):
  //   f(w) = sum(y log(sig(w'x)) + (1 - y) log(sig(1 - w'x))).
  // We want to minimize this function.  L2-regularization is just lambda
  // multiplied by the squared l2-norm of the parameters then divided by two.
  typedef typename MatType::elem_type ElemType;

  // For the regularization, we ignore the first term, which is the intercept
  // term and take every term except the last one in the decision variable.
  const ElemType regularization = 0.5 * lambda *
      arma::dot(parameters.tail_cols(parameters.n_elem - 1),
      parameters.tail_cols(parameters.n_elem - 1));

  // Calculate vectors of sigmoids.  The intercept term is parameters(0, 0) and
  // does not need to be multiplied by any of the predictors.
  const arma::Row<ElemType> sigmoid = 1.0 / (1.0 +
      arma::exp(-(parameters(0, 0) +
                parameters.tail_cols(parameters.n_elem - 1) * predictors)));

  // Assemble full objective function.  Often the objective function and the
  // regularization as given are divided by the number of features, but this
  // doesn't actually affect the optimization result, so we'll just ignore those
  // terms for computational efficiency.  Note that the conversion causes some
  // copy and slowdown, but this is so negligible compared to the rest of the
  // calculation it is not worth optimizing for.
  const ElemType result = arma::accu(arma::log(1.0 -
      arma::conv_to<arma::Row<ElemType>>::from(responses) + sigmoid %
      (2 * arma::conv_to<arma::Row<ElemType>>::from(responses) - 1.0)));

  // Invert the result, because it's a minimization.
  return regularization - result;
}

/**
 * Evaluate the logistic regression objective function given the estimated
 * parameters for a given batch from a given point.
 */
template<typename MatType>
typename MatType::elem_type LogisticRegressionFunction<MatType>::Evaluate(
    const MatType& parameters,
    const size_t begin,
    const size_t batchSize) const
{
  typedef typename MatType::elem_type ElemType;

  // Calculate the regularization term.
  const ElemType regularization = lambda *
      (batchSize / (2.0 * predictors.n_cols)) *
      arma::dot(parameters.tail_cols(parameters.n_elem - 1),
                parameters.tail_cols(parameters.n_elem - 1));

  // Calculate the sigmoid function values.
  const arma::Row<ElemType> sigmoid = 1.0 / (1.0 +
      arma::exp(-(parameters(0, 0) +
                  parameters.tail_cols(parameters.n_elem - 1) *
                      predictors.cols(begin, begin + batchSize - 1))));

  // Compute the objective for the given batch size from a given point.
  arma::Row<ElemType> respD = arma::conv_to<arma::Row<ElemType>>::from(
      responses.subvec(begin, begin + batchSize - 1));
  const ElemType result = arma::accu(arma::log(1.0 - respD + sigmoid %
      (2 * respD - 1.0)));

  // Invert the result, because it's a minimization.
  return regularization - result;
}

//! Evaluate the gradient of the logistic regression objective function.
template<typename MatType>
template<typename GradType>
void LogisticRegressionFunction<MatType>::Gradient(
    const MatType& parameters,
    GradType& gradient) const
{
  typedef typename MatType::elem_type ElemType;
  // Regularization term.
  MatType regularization;
  regularization = lambda * parameters.tail_cols(parameters.n_elem - 1);

  const arma::Row<ElemType> sigmoids = (1 / (1 + arma::exp(-parameters(0, 0)
      - parameters.tail_cols(parameters.n_elem - 1) * predictors)));

  gradient.set_size(arma::size(parameters));
  gradient[0] = -arma::accu(responses - sigmoids);
  gradient.tail_cols(parameters.n_elem - 1) = (sigmoids - responses) *
      predictors.t() + regularization;
}

//! Evaluate the gradient of the logistic regression objective function for a
//! given batch size.
template<typename MatType>
template<typename GradType>
void LogisticRegressionFunction<MatType>::Gradient(
                const MatType& parameters,
                const size_t begin,
                GradType& gradient,
                const size_t batchSize) const
{
  typedef typename MatType::elem_type ElemType;

  // Regularization term.
  MatType regularization;
  regularization = lambda * parameters.tail_cols(parameters.n_elem - 1)
      / predictors.n_cols * batchSize;

  const arma::Row<ElemType> exponents = parameters(0, 0) +
      parameters.tail_cols(parameters.n_elem - 1) *
      predictors.cols(begin, begin + batchSize - 1);
  // Calculating the sigmoid function values.
  const arma::Row<ElemType> sigmoids = 1.0 / (1.0 + arma::exp(-exponents));

  gradient.set_size(parameters.n_rows, parameters.n_cols);
  gradient[0] = -arma::accu(responses.subvec(begin, begin + batchSize - 1) -
      sigmoids);
  gradient.tail_cols(parameters.n_elem - 1) = (sigmoids -
      responses.subvec(begin, begin + batchSize - 1)) *
      predictors.cols(begin, begin + batchSize - 1).t() + regularization;
}

/**
 * Evaluate the partial gradient of the logistic regression objective
 * function with respect to the individual features in the parameter.
 */
template <typename MatType>
void LogisticRegressionFunction<MatType>::PartialGradient(
    const MatType& parameters,
    const size_t j,
    arma::sp_mat& gradient) const
{
  const arma::Row<typename MatType::elem_type> diffs = responses -
      (1 / (1 + arma::exp(-parameters(0, 0) -
                          parameters.tail_cols(parameters.n_elem - 1) *
                              predictors)));

  gradient.set_size(arma::size(parameters));

  if (j == 0)
  {
    gradient[j] = -arma::accu(diffs);
  }
  else
  {
    gradient[j] = arma::dot(-predictors.row(j - 1), diffs) + lambda *
      parameters(0, j);
  }
}

template<typename MatType>
template<typename GradType>
typename MatType::elem_type
LogisticRegressionFunction<MatType>::EvaluateWithGradient(
    const MatType& parameters,
    GradType& gradient) const
{
  typedef typename MatType::elem_type ElemType;

  // Regularization term.
  MatType regularization = lambda *
      parameters.tail_cols(parameters.n_elem - 1);

  const ElemType objectiveRegularization = lambda / 2.0 *
      arma::dot(parameters.tail_cols(parameters.n_elem - 1),
                parameters.tail_cols(parameters.n_elem - 1));

  // Calculate the sigmoid function values.
  const arma::Row<ElemType> sigmoids = 1.0 / (1.0 +
      arma::exp(-(parameters(0, 0) +
                  parameters.tail_cols(parameters.n_elem - 1) * predictors)));

  gradient.set_size(arma::size(parameters));
  gradient[0] = -arma::accu(responses - sigmoids);
  gradient.tail_cols(parameters.n_elem - 1) = (sigmoids - responses) *
      predictors.t() + regularization;

  // Now compute the objective function using the sigmoids.
  ElemType result = arma::accu(arma::log(1.0 -
      arma::conv_to<arma::Row<ElemType>>::from(responses) + sigmoids %
      (2 * arma::conv_to<arma::Row<ElemType>>::from(responses) - 1.0)));

  // Invert the result, because it's a minimization.
  return objectiveRegularization - result;
}

template<typename MatType>
template<typename GradType>
typename MatType::elem_type
LogisticRegressionFunction<MatType>::EvaluateWithGradient(
    const MatType& parameters,
    const size_t begin,
    GradType& gradient,
    const size_t batchSize) const
{
  typedef typename MatType::elem_type ElemType;

  // Regularization term.
  MatType regularization =
      lambda * parameters.tail_cols(parameters.n_elem - 1) / predictors.n_cols *
      batchSize;

  const ElemType objectiveRegularization = lambda *
      (batchSize / (2.0 * predictors.n_cols)) *
      arma::dot(parameters.tail_cols(parameters.n_elem - 1),
                parameters.tail_cols(parameters.n_elem - 1));

  // Calculate the sigmoid function values.
  const arma::Row<ElemType> sigmoids = 1.0 / (1.0 +
      arma::exp(-(parameters(0, 0) +
                  parameters.tail_cols(parameters.n_elem - 1) *
                      predictors.cols(begin, begin + batchSize - 1))));

  gradient.set_size(parameters.n_rows, parameters.n_cols);
  gradient[0] = -arma::accu(responses.subvec(begin, begin + batchSize - 1) -
      sigmoids);
  gradient.tail_cols(parameters.n_elem - 1) = (sigmoids -
      responses.subvec(begin, begin + batchSize - 1)) *
      predictors.cols(begin, begin + batchSize - 1).t() + regularization;

  // Now compute the objective function using the sigmoids.
  arma::Row<ElemType> respD = arma::conv_to<arma::Row<ElemType>>::from(
      responses.subvec(begin, begin + batchSize - 1));
  const ElemType result = arma::accu(arma::log(1.0 - respD + sigmoids %
      (2 * respD - 1.0)));

  // Invert the result, because it's a minimization.
  return objectiveRegularization - result;
}

template<typename MatType>
void LogisticRegressionFunction<MatType>::Classify(
    const MatType& dataset,
    arma::Row<size_t>& labels,
    const MatType& parameters,
    const double decisionBoundary) const
{
  // Calculate sigmoid function for each point.  The (1.0 - decisionBoundary)
  // term correctly sets an offset so that floor() returns 0 or 1 correctly.
  labels = arma::conv_to<arma::Row<size_t>>::from((1.0 /
      (1.0 + arma::exp(-parameters(0) -
      parameters.tail_cols(parameters.n_elem - 1) * dataset))) +
      (1.0 - decisionBoundary));
}

template<typename MatType>
double LogisticRegressionFunction<MatType>::ComputeAccuracy(
    const MatType& predictors,
    const arma::Row<size_t>& responses,
    const MatType& parameters,
    const double decisionBoundary) const
{
  // Predict responses using the current model.
  arma::Row<size_t> tempResponses;
  Classify(predictors, tempResponses, parameters, decisionBoundary);

  // Count the number of responses that were correct.
  size_t count = 0;
  for (size_t i = 0; i < responses.n_elem; i++)
  {
    if (responses(i) == tempResponses(i))
      count++;
  }

  return (double) (count * 100) / responses.n_elem;
}

} // namespace test
} // namespace ens

#endif

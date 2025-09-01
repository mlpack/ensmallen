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
template<typename LabelsType>
LogisticRegressionFunction<MatType>::LogisticRegressionFunction(
    MatType& predictors,
    LabelsType& responsesIn,
    const double lambdaIn) :
    // We promise to be well-behaved... the elements won't be modified.
    predictors(predictors),
    // On old Armadillo versions, we cannot do both a sparse-to-dense conversion
    // and element type conversion in one shot.
    #if ARMA_VERSION_MAJOR < 12 || \
        (ARMA_VERSION_MAJOR == 12 && ARMA_VERSION_MINOR < 8)
    responses(conv_to<BaseRowType>::from(conv_to<typename ForwardType<
        LabelsType, ElemType>::bmat>::from(responsesIn))),
    #else
    responses(conv_to<MatType>::from(responsesIn)),
    #endif
    lambda(ElemType(lambdaIn)),
    halfLambda(ElemType(lambdaIn / 2.0))
{
  initialPoint = arma::Row<ElemType>(predictors.n_rows + 1, arma::fill::zeros);

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
template<typename LabelsType>
LogisticRegressionFunction<MatType>::LogisticRegressionFunction(
    MatType& predictors,
    LabelsType& responsesIn,
    MatType& initialPoint,
    const double lambdaIn) :
    initialPoint(initialPoint),
    predictors(predictors),
    // On old Armadillo versions, we cannot do both a sparse-to-dense conversion
    // and element type conversion in one shot.
    #if ARMA_VERSION_MAJOR < 12 || \
        (ARMA_VERSION_MAJOR == 12 && ARMA_VERSION_MINOR < 8)
    responses(conv_to<MatType>::from(conv_to<typename ForwardType<
        LabelsType, ElemType>::bmat>::from(responsesIn))),
    #else
    responses(conv_to<MatType>::from(responsesIn)),
    #endif
    lambda(ElemType(lambdaIn)),
    halfLambda(ElemType(lambdaIn / 2.0))
{
  // To check if initialPoint is compatible with predictors.
  if (initialPoint.n_rows != (predictors.n_rows + 1) ||
      initialPoint.n_cols != 1)
  {
    this->initialPoint = arma::Row<ElemType>(predictors.n_rows + 1,
        arma::fill::zeros);
  }
}

template<typename MatType>
void ShuffleImpl(MatType& predictors, MatType& responses,
    const typename std::enable_if_t<!IsSparseMatrixType<MatType>::value>* = 0)
{
  MatType allData = shuffle(join_cols(predictors, responses), 1);

  predictors = allData.rows(0, allData.n_rows - 2);
  responses = allData.row(allData.n_rows - 1);
}

template<typename MatType, typename BaseRowType>
void ShuffleImpl(MatType& predictors, BaseRowType& responses,
    const typename std::enable_if_t<IsSparseMatrixType<MatType>::value>* = 0)
{
  // For sparse data shuffle() is not available.
  arma::uvec ordering = shuffle(linspace<arma::uvec>(0, predictors.n_cols - 1,
      predictors.n_cols));

  predictors = predictors.cols(ordering);
  responses = responses.cols(ordering);
}

/**
 * Shuffle the datapoints.
 */
template<typename MatType>
void LogisticRegressionFunction<MatType>::Shuffle()
{
  ShuffleImpl<MatType>(predictors, responses);
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
  typedef typename ForwardType<MatType>::brow BaseRowType;

  // For the regularization, we ignore the first term, which is the intercept
  // term and take every term except the last one in the decision variable.
  const ElemType regularization = halfLambda *
      dot(parameters.tail_cols(parameters.n_elem - 1),
          parameters.tail_cols(parameters.n_elem - 1));

  // Calculate vectors of sigmoids.  The intercept term is parameters(0, 0) and
  // does not need to be multiplied by any of the predictors.
  const BaseRowType sigmoid = 1 / (1 + exp(-(parameters(0, 0) +
      parameters.tail_cols(parameters.n_elem - 1) * predictors)));

  // Assemble full objective function.  Often the objective function and the
  // regularization as given are divided by the number of features, but this
  // doesn't actually affect the optimization result, so we'll just ignore those
  // terms for computational efficiency.  Note that the conversion causes some
  // copy and slowdown, but this is so negligible compared to the rest of the
  // calculation it is not worth optimizing for.
  const ElemType result = accu(
      log(1 - responses + sigmoid % (2 * responses - 1)));

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
  typedef typename ForwardType<MatType>::brow BaseRowType;

  // Calculate the regularization term.
  const ElemType regularization = halfLambda *
      (batchSize / ElemType(predictors.n_cols)) *
      dot(parameters.tail_cols(parameters.n_elem - 1),
          parameters.tail_cols(parameters.n_elem - 1));

  // Calculate the sigmoid function values.
  const BaseRowType sigmoid = 1 / (1 + exp(-(parameters(0, 0) +
      parameters.tail_cols(parameters.n_elem - 1) *
      predictors.cols(begin, begin + batchSize - 1))));

  // Compute the objective for the given batch size from a given point.
  const ElemType result = accu(log(
      1 - responses.subvec(begin, begin + batchSize - 1) +
      sigmoid % (2 * responses.subvec(begin, begin + batchSize - 1) - 1)));

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
  // Regularization term.
  MatType regularization = lambda * parameters.tail_cols(parameters.n_elem - 1);

  const BaseRowType sigmoids = (1 / (1 + exp(-parameters(0, 0)
      - parameters.tail_cols(parameters.n_elem - 1) * predictors)));

  gradient.set_size(size(parameters));
  gradient[0] = -accu(responses - sigmoids);
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
  // Regularization term.
  MatType regularization = lambda * parameters.tail_cols(parameters.n_elem - 1)
      / predictors.n_cols * batchSize;

  const BaseRowType exponents = parameters(0, 0) +
      parameters.tail_cols(parameters.n_elem - 1) *
      predictors.cols(begin, begin + batchSize - 1);

  // Calculating the sigmoid function values.
  const BaseRowType sigmoids = 1 / (1 + exp(-exponents));

  gradient.set_size(parameters.n_rows, parameters.n_cols);
  gradient[0] = -accu(responses.subvec(begin, begin + batchSize - 1) -
      sigmoids);
  gradient.tail_cols(parameters.n_elem - 1) = (sigmoids -
      responses.subvec(begin, begin + batchSize - 1)) *
      predictors.cols(begin, begin + batchSize - 1).t() + regularization;
}

/**
 * Evaluate the partial gradient of the logistic regression objective
 * function with respect to the individual features in the parameter.
 */
template<typename MatType>
template<typename GradType>
void LogisticRegressionFunction<MatType>::PartialGradient(
    const MatType& parameters,
    const size_t j,
    GradType& gradient) const
{
  const BaseRowType diffs = responses - (1 / (1 + exp(-parameters(0, 0) -
      parameters.tail_cols(parameters.n_elem - 1) * predictors)));

  gradient.set_size(size(parameters));

  if (j == 0)
  {
    gradient[j] = -accu(diffs);
  }
  else
  {
    gradient[j] = dot(-predictors.row(j - 1), diffs) + lambda *
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
  // Regularization term.
  MatType regularization = lambda * parameters.tail_cols(parameters.n_elem - 1);

  const ElemType objectiveRegularization = halfLambda *
      dot(parameters.tail_cols(parameters.n_elem - 1),
          parameters.tail_cols(parameters.n_elem - 1));

  // Calculate the sigmoid function values.
  const BaseRowType sigmoids = 1 / (1 + exp(-(parameters(0, 0) +
      parameters.tail_cols(parameters.n_elem - 1) * predictors)));

  gradient.set_size(size(parameters));
  gradient[0] = -accu(responses - sigmoids);
  gradient.tail_cols(parameters.n_elem - 1) = (sigmoids - responses) *
      predictors.t() + regularization;

  // Now compute the objective function using the sigmoids.
  ElemType result = accu(log(1 - responses + sigmoids % (2 * responses - 1)));

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
  typedef typename ForwardType<MatType>::brow BaseRowType;

  // Regularization term.
  MatType regularization = lambda *
      parameters.tail_cols(parameters.n_elem - 1) / predictors.n_cols *
      batchSize;

  const ElemType objectiveRegularization = halfLambda *
      (batchSize / ElemType(predictors.n_cols)) *
      dot(parameters.tail_cols(parameters.n_elem - 1),
          parameters.tail_cols(parameters.n_elem - 1));

  // Calculate the sigmoid function values.
  const BaseRowType sigmoids = 1 / (1 + exp(-(parameters(0, 0) +
      parameters.tail_cols(parameters.n_elem - 1) *
      predictors.cols(begin, begin + batchSize - 1))));

  gradient.set_size(parameters.n_rows, parameters.n_cols);
  gradient[0] = -accu(responses.subvec(begin, begin + batchSize - 1) -
      sigmoids);
  gradient.tail_cols(parameters.n_elem - 1) = (sigmoids -
      responses.cols(begin, begin + batchSize - 1)) *
      predictors.cols(begin, begin + batchSize - 1).t() + regularization;

  // Now compute the objective function using the sigmoids.
  const ElemType result = accu(log(
      1 - responses.subvec(begin, begin + batchSize - 1) +
      sigmoids % (2 * responses.subvec(begin, begin + batchSize - 1) - 1)));

  // Invert the result, because it's a minimization.
  return objectiveRegularization - result;
}

template<typename MatType>
template<typename LabelsType>
void LogisticRegressionFunction<MatType>::Classify(
    const MatType& dataset,
    LabelsType& labels,
    const MatType& parameters,
    const double decisionBoundary) const
{
  // Calculate sigmoid function for each point.  The (1 - decisionBoundary)
  // term correctly sets an offset so that floor() returns 0 or 1 correctly.
  labels = conv_to<LabelsType>::from((1 / (1 + exp(-parameters(0) -
      parameters.tail_cols(parameters.n_elem - 1) * dataset))) +
      ElemType(1 - decisionBoundary));
}

template<typename MatType>
template<typename LabelsType>
double LogisticRegressionFunction<MatType>::ComputeAccuracy(
    const MatType& predictors,
    const LabelsType& responses,
    const MatType& parameters,
    const double decisionBoundary) const
{
  // Predict responses using the current model.
  LabelsType tempResponses;
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

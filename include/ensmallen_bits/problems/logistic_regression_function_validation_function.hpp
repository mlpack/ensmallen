/**
 * @file logistic_regression_function.hpp
 * @author Marcus Edel
 *
 * Minimal implementation of validation function for logisitic regression.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_PROBLEMS_LOGISTIC_REGRESSION_FUNCTION_VALIDATION_FUNCTION_HPP
#define ENSMALLEN_PROBLEMS_LOGISTIC_REGRESSION_FUNCTION_VALIDATION_FUNCTION_HPP

#include "logistic_regression_function.hpp"

namespace ens {
namespace test {

using namespace ens::test;

class LogisticRegressionValidationFunction
{
 public:
  LogisticRegressionValidationFunction(LogisticRegression<>& lrIn,
                                       arma::mat& coordinatesIn)
    : lr(lrIn), coordinates(coordinatesIn)
  {
  }

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
   */
  double Evaluate(arma::mat& predictors, arma::Row<size_t>& responses)
  {
    return lr.ComputeAccuracy(predictors, responses, coordinates);
  }

 private:
  LogisticRegression<> lr;
  arma::mat coordinates;
};

} // namespace test
} // namespace ens

#endif // ENSMALLEN_PROBLEMS_LOGISTIC_REGRESSION_FUNCTION_VALIDATION_FUNCTION_HPP

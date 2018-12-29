/**
 * @file nist.hpp
 * @author Marcus Edel
 *
 * Definition of the National Institute of Standards and Technology non-linear
 * least squares problems.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_NIST_HPP
#define ENSMALLEN_PROBLEMS_NIST_HPP

namespace ens {
namespace test {

/**
 * The National Institute of Standards and Technology has released a set of
 * problems to test non-linear least squares solvers. More information about the
 * background on these problems and suggested evaluation methodology can be
 * found at:
 *
 *    http://www.itl.nist.gov/div898/strd/nls/nls_info.shtml
 *
 * The problem data themselves can be found at:
 *
 *    http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml
 *
 * The problems are divided into three levels of difficulty, Lower,
 * Average and Higher. For each problem there are two starting guesses,
 * the first one far away from the global minimum and the second
 * closer to it.
 */
class Misra1a
{
 public:
  //! Initialize Misra1a.
  Misra1a() : initial("500 250; 0.0001 0.0005") {}

  /**
   * Evaluate the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @param result The calculated function result.
   */
  void Evaluate(const arma::mat& coordinates,
                const arma::mat& predictors,
                const arma::rowvec& responses,
                arma::mat& result) const
  {
    result = coordinates(0) *
        (1.0 - arma::exp(-1.0 * coordinates(1) * predictors)) - responses;
  }

  /**
   * Jacobian of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param jacobian The calculated jacobian matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   */
  void Jacobian(const arma::mat& coordinates,
                arma::mat& jacobian,
                const arma::mat& predictors,
                const arma::rowvec& responses) const
  {
    jacobian.zeros(coordinates.n_elem, predictors.n_cols);

    const arma::mat expr1 = arma::exp(-1.0 * coordinates(1) * predictors);
    const arma::mat expr2 = 1.0 - expr1;

    jacobian.row(0) = expr2;
    jacobian.row(1) = coordinates(0) * expr1 % predictors;
  }

  /**
   * Gradient of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param gradient The calculated gradient matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @return The calculated error.
   */
  double Gradient(const arma::mat& coordinates,
                  arma::mat& gradient,
                  const arma::mat& predictors,
                  const arma::rowvec& responses) const
  {
    arma::mat jacobian, result;
    Jacobian(coordinates, jacobian, predictors, responses);
    Evaluate(coordinates, predictors, responses, result);
    gradient = arma::trans(2.0 * result * jacobian.t());
    return arma::accu(arma::pow(result, 2.0));
  }

  /**
   * Get the starting point.
   *
   * @param index The index of the starting point.
   */
  arma::mat GetInitialPoint(const size_t index = 0) const
  {
    return initial.col(index);
  }

  //! Get the number of starting points.
  size_t Functions() const { return 2; }

  //! Locally stored initial starting points.
  arma::mat initial;
};

class Chwirut1
{
 public:
  //! Initialize Chwirut1.
  Chwirut1() : initial("0.1 0.15; 0.01 0.008; 0.02 0.010") {}

  /**
   * Evaluate the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @param result The calculated function result.
   */
  void Evaluate(const arma::mat& coordinates,
                const arma::mat& predictors,
                const arma::rowvec& responses,
                arma::mat& result) const
  {
    result = arma::exp(-1.0 * coordinates(0) * predictors) % (1.0 /
        (coordinates(1) + coordinates(2) * predictors)) - responses;
  }

  /**
   * Jacobian of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param jacobian The calculated jacobian matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   */
  void Jacobian(const arma::mat& coordinates,
                arma::mat& jacobian,
                const arma::mat& predictors,
                const arma::rowvec& responses) const
  {
    jacobian.zeros(coordinates.n_elem, predictors.n_cols);

    const arma::mat expr1 = arma::exp(-1.0 * coordinates(0) * predictors);
    const arma::mat expr2 = coordinates(1) + coordinates(2) * predictors;
    const arma::mat expr3 = expr1 % predictors;
    const arma::mat expr4 = arma::pow(expr2, 2.0);

    jacobian.row(0) = -1.0 * (expr3 % (1.0 / expr2));
    jacobian.row(1) = -1.0 * (expr1 % (1.0 / expr4));
    jacobian.row(2) = -1.0 * (expr3 % (1.0 / expr4));
  }

  /**
   * Gradient of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param gradient The calculated gradient matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @return The calculated error.
   */
  double Gradient(const arma::mat& coordinates,
                  arma::mat& gradient,
                  const arma::mat& predictors,
                  const arma::rowvec& responses) const
  {
    arma::mat jacobian, result;
    Jacobian(coordinates, jacobian, predictors, responses);
    Evaluate(coordinates, predictors, responses, result);
    gradient = arma::trans(2.0 * result * jacobian.t());

    return arma::accu(arma::pow(result, 2.0));
  }

  /**
   * Get the starting point.
   *
   * @param index The index of the starting point.
   */
  arma::mat GetInitialPoint(const size_t index = 0) const
  {
    return initial.col(index);
  }

  //! Get the number of starting points.
  size_t Functions() const { return 2; }

  //! Locally stored initial starting points.
  arma::mat initial;
};

using Chwirut2 = Chwirut1;

class Lanczos3
{
 public:
  //! Initialize Lanczos3.
  Lanczos3() : initial("1.2 0.5; 0.3 0.7; 5.6 3.6; 5.5 4.2; 6.5 4; 7.6 6.3") {}

  /**
   * Evaluate the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @param result The calculated function result.
   */
  void Evaluate(const arma::mat& coordinates,
                const arma::mat& predictors,
                const arma::rowvec& responses,
                arma::mat& result) const
  {
   result = coordinates(0) * arma::exp(-1.0 * coordinates(1) * predictors) +
      coordinates(2) * arma::exp(-1.0 * coordinates(3) * predictors) +
      coordinates(4) * arma::exp(-1.0 * coordinates(5) * predictors) -
      responses;
  }

  /**
   * Jacobian of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param jacobian The calculated jacobian matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   */
  void Jacobian(const arma::mat& coordinates,
                arma::mat& jacobian,
                const arma::mat& predictors,
                const arma::rowvec& responses) const
  {
    jacobian.zeros(coordinates.n_elem, predictors.n_cols);

    const arma::mat expr1 = arma::exp(-1.0 * coordinates(1) * predictors);
    const arma::mat expr2 = arma::exp(-1.0 * coordinates(3) * predictors);
    const arma::mat expr3 = arma::exp(-1.0 * coordinates(5) * predictors);

    jacobian.row(0) = expr1;
    jacobian.row(1) = -1.0 * (coordinates(0) * (expr1 % predictors));
    jacobian.row(2) = expr2;
    jacobian.row(3) = -1.0 * (coordinates(2) * (expr2 % predictors));
    jacobian.row(4) = expr3;
    jacobian.row(5) = -1.0 * (coordinates(4) * (expr3 % predictors));
  }

  /**
   * Gradient of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param gradient The calculated gradient matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @return The calculated error.
   */
  double Gradient(const arma::mat& coordinates,
                  arma::mat& gradient,
                  const arma::mat& predictors,
                  const arma::rowvec& responses) const
  {
    arma::mat jacobian, result;
    Jacobian(coordinates, jacobian, predictors, responses);
    Evaluate(coordinates, predictors, responses, result);
    gradient = arma::trans(2.0 * result * jacobian.t());

    return arma::accu(arma::pow(result, 2.0));
  }

  /**
   * Get the starting point.
   *
   * @param index The index of the starting point.
   */
  arma::mat GetInitialPoint(const size_t index = 0) const
  {
    return initial.col(index);
  }

  //! Get the number of starting points.
  size_t Functions() const { return 2; }

  //! Locally stored initial starting points.
  arma::mat initial;
};

class Gauss1
{
 public:
  //! Initialize Gauss1.
  Gauss1() : initial("97.0 94.0; 0.009 0.0105; 100.0 99.0; 65.0 63.0; \
                      20.0 25.0; 70.0 71.0; 178.0 180.0; 16.5 20.0") {}

  /**
   * Evaluate the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @param result The calculated function result.
   */
  void Evaluate(const arma::mat& coordinates,
                const arma::mat& predictors,
                const arma::rowvec& responses,
                arma::mat& result) const
  {
    result = coordinates(0) * arma::exp(-1.0 * coordinates(1) * predictors) +
        coordinates(2) * arma::exp(-1.0 * arma::pow(predictors -
        coordinates(3), 2.0) / std::pow(coordinates(4), 2.0)) + coordinates(5) *
        arma::exp(-1.0 * arma::pow(predictors - coordinates(6), 2.0) /
        std::pow(coordinates(7), 2.0)) - responses;
  }

  /**
   * Jacobian of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param jacobian The calculated jacobian matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   */
  void Jacobian(const arma::mat& coordinates,
                arma::mat& jacobian,
                const arma::mat& predictors,
                const arma::rowvec& responses) const
  {
    jacobian.zeros(coordinates.n_elem, predictors.n_cols);

    const arma::mat expr1 = arma::exp(-1.0 * coordinates(1) * predictors);
    const arma::mat expr2 = predictors - coordinates(3);
    const arma::mat expr3 = arma::pow(expr2, 2.0);
    const double expr4 = std::pow(coordinates(4), 2.0);
    const arma::mat expr5 = arma::exp(-1.0 * expr3 / expr4);
    const arma::mat expr6 = predictors - coordinates(6);
    const arma::mat expr7 = arma::pow(expr6, 2.0);
    const double expr8 = std::pow(coordinates(7), 2.0);
    const arma::mat expr9 = arma::exp(-1.0 * expr7 / expr8);

    jacobian.row(0) = expr1;
    jacobian.row(1) = -1.0 * (coordinates(0) * (expr1 % predictors));
    jacobian.row(2) = expr5;
    jacobian.row(3) = coordinates(2) * (expr5 % (2.0 * expr2 / expr4));
    jacobian.row(4) = coordinates(2) * (expr5 % (expr3 * (2.0 * coordinates(4))
        / std::pow(expr4, 2.0)));
    jacobian.row(5) = expr9;
    jacobian.row(6) = coordinates(5) * (expr9 % (2.0 * expr6 / expr8));
    jacobian.row(7) = coordinates(5) * (expr9 % (expr7 * (2.0 * coordinates(7))
        / std::pow(expr8, 2.0)));
  }

  /**
   * Gradient of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param gradient The calculated gradient matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @return The calculated error.
   */
  double Gradient(const arma::mat& coordinates,
                  arma::mat& gradient,
                  const arma::mat& predictors,
                  const arma::rowvec& responses) const
  {
    arma::mat jacobian, result;
    Jacobian(coordinates, jacobian, predictors, responses);
    Evaluate(coordinates, predictors, responses, result);
    gradient = arma::trans(2.0 * result * jacobian.t());

    return arma::accu(arma::pow(result, 2.0));
  }

  /**
   * Get the starting point.
   *
   * @param index The index of the starting point.
   */
  arma::mat GetInitialPoint(const size_t index = 0) const
  {
    return initial.col(index);
  }

  //! Get the number of starting points.
  size_t Functions() const { return 2; }

  //! Locally stored initial starting points.
  arma::mat initial;
};

class Gauss2
{
 public:
  //! Initialize Gauss2.
  Gauss2() : initial("96.0 98.0; 0.009 0.0105; 103.0 103.0; 106.0 105.0; \
                      18.0 20.0; 72.0 73.0; 151.0 150.0; 18.0 20.0") {}

  /**
   * Evaluate the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @param result The calculated function result.
   */
  void Evaluate(const arma::mat& coordinates,
                const arma::mat& predictors,
                const arma::rowvec& responses,
                arma::mat& result) const
  {
   function.Evaluate(coordinates, predictors, responses, result);
  }

  /**
   * Jacobian of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param jacobian The calculated jacobian matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   */
  void Jacobian(const arma::mat& coordinates,
                arma::mat& jacobian,
                const arma::mat& predictors,
                const arma::rowvec& responses) const
  {
    function.Jacobian(coordinates, jacobian, predictors, responses);
  }

  /**
   * Gradient of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param gradient The calculated gradient matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @return The calculated error.
   */
  double Gradient(const arma::mat& coordinates,
                  arma::mat& gradient,
                  const arma::mat& predictors,
                  const arma::rowvec& responses) const
  {
    return function.Gradient(coordinates, gradient, predictors, responses);
  }

  /**
   * Get the starting point.
   *
   * @param index The index of the starting point.
   */
  arma::mat GetInitialPoint(const size_t index = 0) const
  {
    return initial.col(index);
  }

  //! Get the number of starting points.
  size_t Functions() const { return 2; }

  arma::mat initial;

  //! Locally stored initial starting points.
  Gauss1 function;
};

class DanWood
{
 public:
  //! Initialize DanWood.
  DanWood() : initial("1 0.7; 5 4") {}

  /**
   * Evaluate the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @param result The calculated function result.
   */
  void Evaluate(const arma::mat& coordinates,
                const arma::mat& predictors,
                const arma::rowvec& responses,
                arma::mat& result) const
  {
    result = coordinates(0) * (arma::pow(predictors, coordinates(1))) -
        responses;
  }

  /**
   * Jacobian of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param jacobian The calculated jacobian matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   */
  void Jacobian(const arma::mat& coordinates,
                arma::mat& jacobian,
                const arma::mat& predictors,
                const arma::rowvec& responses) const
  {
    jacobian.zeros(coordinates.n_elem, predictors.n_cols);

    const arma::mat expr1 = arma::pow(predictors, coordinates(1));
    jacobian.row(0) = expr1;
    jacobian.row(1) = coordinates(0) * (expr1 % arma::log(predictors));
  }

  /**
   * Gradient of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param gradient The calculated gradient matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @return The calculated error.
   */
  double Gradient(const arma::mat& coordinates,
                  arma::mat& gradient,
                  const arma::mat& predictors,
                  const arma::rowvec& responses) const
  {
    arma::mat jacobian, result;
    Jacobian(coordinates, jacobian, predictors, responses);
    Evaluate(coordinates, predictors, responses, result);
    gradient = arma::trans(2.0 * result * jacobian.t());

    return arma::accu(arma::pow(result, 2.0));
  }

  /**
   * Get the starting point.
   *
   * @param index The index of the starting point.
   */
  arma::mat GetInitialPoint(const size_t index = 0) const
  {
    return initial.col(index);
  }

  //! Get the number of starting points.
  size_t Functions() const { return 2; }

  //! Locally stored initial starting points.
  arma::mat initial;
};

class Misra1b
{
 public:
  //! Initialize Misra1b.
  Misra1b() : initial("500 300; 0.0001 0.0002") {}

  /**
   * Evaluate the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @param result The calculated function result.
   */
  void Evaluate(const arma::mat& coordinates,
                const arma::mat& predictors,
                const arma::rowvec& responses,
                arma::mat& result) const
  {
    result = coordinates(0) * (1.0 - arma::pow(1.0 + coordinates(1) *
        predictors / 2.0, -2.0)) - responses;
  }

  /**
   * Jacobian of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param jacobian The calculated jacobian matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   */
  void Jacobian(const arma::mat& coordinates,
                arma::mat& jacobian,
                const arma::mat& predictors,
                const arma::rowvec& responses) const
  {
    jacobian.zeros(coordinates.n_elem, predictors.n_cols);

    const arma::mat expr1 = 1.0 + coordinates(1) * predictors / 2.0;

    jacobian.row(0) = 1.0 - arma::pow(expr1, -2.0);
    jacobian.row(1) = -1.0 * (coordinates(0) * arma::pow(expr1, -3.0) %
        (-2.0 * (predictors / 2.0)));
  }

  /**
   * Gradient of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param gradient The calculated gradient matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @return The calculated error.
   */
  double Gradient(const arma::mat& coordinates,
                  arma::mat& gradient,
                  const arma::mat& predictors,
                  const arma::rowvec& responses) const
  {
    arma::mat jacobian, result;
    Jacobian(coordinates, jacobian, predictors, responses);
    Evaluate(coordinates, predictors, responses, result);
    gradient = arma::trans(2.0 * result * jacobian.t());

    return arma::accu(arma::pow(result, 2.0));
  }

  /**
   * Get the starting point.
   *
   * @param index The index of the starting point.
   */
  arma::mat GetInitialPoint(const size_t index = 0) const
  {
    return initial.col(index);
  }

  //! Get the number of starting points.
  size_t Functions() const { return 2; }

  //! Locally stored initial starting points.
  arma::mat initial;
};

class Kirby2
{
 public:
  //! Initialize Kirby2.
  Kirby2() : initial("2 1.5; -0.1 -0.15; 0.003 0.0025; \
                      -0.001 -0.0015; 0.00001 0.00002") {}

  /**
   * Evaluate the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @param result The calculated function result.
   */
  void Evaluate(const arma::mat& coordinates,
                const arma::mat& predictors,
                const arma::rowvec& responses,
                arma::mat& result) const
  {
    result = coordinates(0) + coordinates(1) * predictors + coordinates(2) *
        arma::pow(predictors, 2.0) % (1.0 / (1.0 + coordinates(3) *
        predictors + coordinates(4) * arma::pow(predictors, 2.0))) - responses;
  }

  /**
   * Jacobian of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param jacobian The calculated jacobian matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   */
  void Jacobian(const arma::mat& coordinates,
                arma::mat& jacobian,
                const arma::mat& predictors,
                const arma::rowvec& responses) const
  {
    jacobian.zeros(coordinates.n_elem, predictors.n_cols);

    const arma::mat expr1 = predictors % predictors;
    const arma::mat expr2 = coordinates(0) + coordinates(1) * predictors +
        coordinates(2) * expr1;
    const arma::mat expr3 = 1.0 + coordinates(3) * predictors +
        coordinates(4) * expr1;
    const arma::mat expr4 = expr3 % expr3;

    jacobian.row(0) = 1.0 / expr3;
    jacobian.row(1) = predictors % jacobian.row(0);
    jacobian.row(2) = expr1 % jacobian.row(0);
    jacobian.row(3) = -1.0 * (expr2 % predictors % (1.0 / expr4));
    jacobian.row(4) = -1.0 * (expr2 % expr1 % (1.0 / expr4));
  }

  /**
   * Gradient of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param gradient The calculated gradient matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @return The calculated error.
   */
  double Gradient(const arma::mat& coordinates,
                  arma::mat& gradient,
                  const arma::mat& predictors,
                  const arma::rowvec& responses) const
  {
    arma::mat jacobian, result;
    Jacobian(coordinates, jacobian, predictors, responses);

    Evaluate(coordinates, predictors, responses, result);
    gradient = arma::trans(2.0 * result * jacobian.t());

    return arma::accu(arma::pow(result, 2.0));
  }

  /**
   * Get the starting point.
   *
   * @param index The index of the starting point.
   */
  arma::mat GetInitialPoint(const size_t index = 0) const
  {
    return initial.col(index);
  }

  //! Get the number of starting points.
  size_t Functions() const { return 2; }

  //! Locally stored initial starting points.
  arma::mat initial;
};

class Hahn1
{
 public:
  //! Initialize Hahn1.
  Hahn1() : initial("10 1; -1 -0.1; 0.05 0.005; -0.00001 -0.000001; \
                     -0.05 -0.005; 0.001 0.0001; -0.000001 -0.0000001") {}

  /**
   * Evaluate the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @param result The calculated function result.
   */
  void Evaluate(const arma::mat& coordinates,
                const arma::mat& predictors,
                const arma::rowvec& responses,
                arma::mat& result) const
  {
    result = (coordinates(0) + coordinates(1) * predictors + coordinates(2) *
        arma::pow(predictors, 2.0) + coordinates(3) *
        arma::pow(predictors, 3.0)) % (1.0 / (1.0 + coordinates(4) *
        predictors + coordinates(5) * arma::pow(predictors, 2.0) +
        coordinates(6) * arma::pow(predictors, 3.0))) - responses;
  }

  /**
   * Jacobian of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param jacobian The calculated jacobian matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   */
  void Jacobian(const arma::mat& coordinates,
                arma::mat& jacobian,
                const arma::mat& predictors,
                const arma::rowvec& responses) const
  {
    jacobian.zeros(coordinates.n_elem, predictors.n_cols);

    const arma::mat expr1 = predictors % predictors;
    const arma::mat expr2 = predictors % expr1;
    const arma::mat expr3 = coordinates(0) + coordinates(1) * predictors +
        coordinates(2) * expr1 + coordinates(3) * expr2;
    const arma::mat expr4 = 1.0 + coordinates(4) * predictors + coordinates(5) *
        expr1 + coordinates(6) * expr2;
    const arma::mat expr5 = expr4 % expr4;

    jacobian.row(0) = 1.0 / expr4;
    jacobian.row(1) = predictors % jacobian.row(0);
    jacobian.row(2) = expr1 % (1.0 / expr4);
    jacobian.row(3) = expr2 % (1.0 / expr4);
    jacobian.row(4) =-1.0 * (expr3 % predictors % (1.0 / expr5));
    jacobian.row(5) =-1.0 * (expr3 % expr1 % (1.0 / expr5));
    jacobian.row(6) =-1.0 * (expr3 % expr2 % (1.0 / expr5));
  }

  /**
   * Gradient of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param gradient The calculated gradient matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @return The calculated error.
   */
  double Gradient(const arma::mat& coordinates,
                  arma::mat& gradient,
                  const arma::mat& predictors,
                  const arma::rowvec& responses) const
  {
    arma::mat jacobian, result;
    Jacobian(coordinates, jacobian, predictors, responses);
    Evaluate(coordinates, predictors, responses, result);
    gradient = arma::trans(2.0 * result * jacobian.t());

    return arma::accu(arma::pow(result, 2.0));
  }

  /**
   * Get the starting point.
   *
   * @param index The index of the starting point.
   */
  arma::mat GetInitialPoint(const size_t index = 0) const
  {
    return initial.col(index);
  }

  //! Get the number of starting points.
  size_t Functions() const { return 2; }

  //! Locally stored initial starting points.
  arma::mat initial;
};

class Nelson
{
 public:
  //! Initialize Nelson.
  Nelson() : initial("2 2.5; 0.0001 0.000000005; -0.01 -0.05") {}

  /**
   * Evaluate the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @param result The calculated function result.
   */
  void Evaluate(const arma::mat& coordinates,
                const arma::mat& predictors,
                const arma::rowvec& responses,
                arma::mat& result) const
  {
    result = coordinates(0) - coordinates(1) * predictors.row(0) % arma::exp(
        -1.0 * coordinates(2) * predictors.row(1)) - arma::log(responses);
  }

  /**
   * Jacobian of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param jacobian The calculated jacobian matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   */
  void Jacobian(const arma::mat& coordinates,
                arma::mat& jacobian,
                const arma::mat& predictors,
                const arma::rowvec& responses) const
  {
    jacobian.zeros(coordinates.n_elem, predictors.n_cols);

    const arma::mat expr1 = coordinates(1) * predictors.row(0);
    const arma::mat expr2 = arma::exp(-1.0 * coordinates(2) *
        predictors.row(1));

    jacobian.row(0).ones();
    jacobian.row(1) = -1.0 * (predictors.row(0) % expr2);
    jacobian.row(2) = expr1 % (expr2 % predictors.row(1));
  }

  /**
   * Gradient of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param gradient The calculated gradient matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @return The calculated error.
   */
  double Gradient(const arma::mat& coordinates,
                  arma::mat& gradient,
                  const arma::mat& predictors,
                  const arma::rowvec& responses) const
  {
    arma::mat jacobian, result;
    Jacobian(coordinates, jacobian, predictors, responses);
    Evaluate(coordinates, predictors, responses, result);
    gradient = arma::trans(2.0 * result * jacobian.t());

    return arma::accu(arma::pow(result, 2.0));
  }

  /**
   * Get the starting point.
   *
   * @param index The index of the starting point.
   */
  arma::mat GetInitialPoint(const size_t index = 0) const
  {
    return initial.col(index);
  }

  //! Get the number of starting points.
  size_t Functions() const { return 2; }

  //! Locally stored initial starting points.
  arma::mat initial;
};

class MGH17
{
 public:
  //! Initialize MGH17.
  MGH17() : initial("50 0.5; 150 1.5; -100 -1; 1 0.01; 2 0.02") {}

  /**
   * Evaluate the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @param result The calculated function result.
   */
  void Evaluate(const arma::mat& coordinates,
                const arma::mat& predictors,
                const arma::rowvec& responses,
                arma::mat& result) const
  {
    result = coordinates(0) + coordinates(1) * arma::exp(-1.0 * predictors *
        coordinates(3)) + coordinates(2) * arma::exp(-1.0 * predictors *
        coordinates(4)) - responses;
  }

  /**
   * Jacobian of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param jacobian The calculated jacobian matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   */
  void Jacobian(const arma::mat& coordinates,
                arma::mat& jacobian,
                const arma::mat& predictors,
                const arma::rowvec& responses) const
  {
    jacobian.zeros(coordinates.n_elem, predictors.n_cols);

    const arma::mat expr1 = -1.0 * predictors;
    const arma::mat expr2 = arma::exp(expr1 * coordinates(3));
    const arma::mat expr3 = arma::exp(expr1 * coordinates(4));

    jacobian.row(0).ones();
    jacobian.row(1) = expr2;
    jacobian.row(2) = expr3;
    jacobian.row(3) = -1.0 * (coordinates(1) * (expr2 % predictors));
    jacobian.row(4) = -1.0 * (coordinates(2) * (expr3 % predictors));
  }

  /**
   * Gradient of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param gradient The calculated gradient matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @return The calculated error.
   */
  double Gradient(const arma::mat& coordinates,
                  arma::mat& gradient,
                  const arma::mat& predictors,
                  const arma::rowvec& responses) const
  {
    arma::mat jacobian, result;
    Jacobian(coordinates, jacobian, predictors, responses);
    Evaluate(coordinates, predictors, responses, result);
    gradient = arma::trans(2.0 * result * jacobian.t());

    return arma::accu(arma::pow(result, 2.0));
  }

  /**
   * Get the starting point.
   *
   * @param index The index of the starting point.
   */
  arma::mat GetInitialPoint(const size_t index = 0) const
  {
    return initial.col(index);
  }

  //! Get the number of starting points.
  size_t Functions() const { return 2; }

  //! Locally stored initial starting points.
  arma::mat initial;
};

class Lanczos1
{
 public:
  //! Initialize Lanczos1.
  Lanczos1() : initial("1.2 0.5; 0.3 0.7; 5.6 3.6; 5.5 4.2; 6.5 4; 7.6 6.3") {}

  /**
   * Evaluate the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @param result The calculated function result.
   */
  void Evaluate(const arma::mat& coordinates,
                const arma::mat& predictors,
                const arma::rowvec& responses,
                arma::mat& result) const
  {
    result = coordinates(0) * arma::exp(-1.0 * coordinates(1) * predictors) +
        coordinates(2) * arma::exp(-1.0 * coordinates(3) * predictors) +
        coordinates(4) * arma::exp(-1.0 * coordinates(5) * predictors) -
        responses;
  }

  /**
   * Jacobian of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param jacobian The calculated jacobian matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   */
  void Jacobian(const arma::mat& coordinates,
                arma::mat& jacobian,
                const arma::mat& predictors,
                const arma::rowvec& responses) const
  {
    jacobian.zeros(coordinates.n_elem, predictors.n_cols);

    const arma::mat expr1 = arma::exp(-1.0 * coordinates(1) * predictors);
    const arma::mat expr2 = arma::exp(-1.0 * coordinates(3) * predictors);
    const arma::mat expr3 = arma::exp(-1.0 * coordinates(5) * predictors);

    jacobian.row(0) = expr1;
    jacobian.row(1) = -1.0 * (coordinates(0) * (expr1 % predictors));
    jacobian.row(1) = expr2;
    jacobian.row(1) = -1.0 * (coordinates(2) * (expr2 % predictors));
    jacobian.row(1) = expr3;
    jacobian.row(1) = -1.0 * (coordinates(4) * (expr3 % predictors));
  }

  /**
   * Gradient of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param gradient The calculated gradient matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @return The calculated error.
   */
  double Gradient(const arma::mat& coordinates,
                  arma::mat& gradient,
                  const arma::mat& predictors,
                  const arma::rowvec& responses) const
  {
    arma::mat jacobian, result;
    Jacobian(coordinates, jacobian, predictors, responses);
    Evaluate(coordinates, predictors, responses, result);
    gradient = arma::trans(2.0 * result * jacobian.t());

    return arma::accu(arma::pow(result, 2.0));
  }

  /**
   * Get the starting point.
   *
   * @param index The index of the starting point.
   */
  arma::mat GetInitialPoint(const size_t index = 0) const
  {
    return initial.col(index);
  }

  //! Get the number of starting points.
  size_t Functions() const { return 2; }

  //! Locally stored initial starting points.
  arma::mat initial;
};

using Lanczos2 = Lanczos1;

class Gauss3
{
 public:
  //! Initialize Gauss3.
  Gauss3() : initial("94.9 96.0; 0.009 0.0096; 90.1 80.0; 113.0 110.0; \
                      20.0 25.0; 73.8 74.0; 140.0 139.0; 20.0 25.0") {}

  /**
   * Evaluate the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @param result The calculated function result.
   */
  void Evaluate(const arma::mat& coordinates,
                const arma::mat& predictors,
                const arma::rowvec& responses,
                arma::mat& result) const
  {
   function.Evaluate(coordinates, predictors, responses, result);
  }

  /**
   * Jacobian of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param jacobian The calculated jacobian matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   */
  void Jacobian(const arma::mat& coordinates,
                arma::mat& jacobian,
                const arma::mat& predictors,
                const arma::rowvec& responses) const
  {
    function.Jacobian(coordinates, jacobian, predictors, responses);
  }

  /**
   * Gradient of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param gradient The calculated gradient matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @return The calculated error.
   */
  double Gradient(const arma::mat& coordinates,
                  arma::mat& gradient,
                  const arma::mat& predictors,
                  const arma::rowvec& responses) const
  {
    return function.Gradient(coordinates, gradient, predictors, responses);
  }

  /**
   * Get the starting point.
   *
   * @param index The index of the starting point.
   */
  arma::mat GetInitialPoint(const size_t index = 0) const
  {
    return initial.col(index);
  }

  //! Get the number of starting points.
  size_t Functions() const { return 2; }

  //! Locally stored initial starting points.
  arma::mat initial;

  //! Locally stored function instantiation.
  Gauss1 function;
};

class Misra1c
{
 public:
  //! Initialize Misra1c.
  Misra1c() : initial("500 600; 0.0001 0.0002") {}

  /**
   * Evaluate the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @param result The calculated function result.
   */
  void Evaluate(const arma::mat& coordinates,
                const arma::mat& predictors,
                const arma::rowvec& responses,
                arma::mat& result) const
  {
    result = coordinates(0) * (1.0 - arma::pow(1.0 + 2.0 * coordinates(1) *
        predictors, -0.5)) - responses;
  }

  /**
   * Jacobian of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param jacobian The calculated jacobian matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   */
  void Jacobian(const arma::mat& coordinates,
                arma::mat& jacobian,
                const arma::mat& predictors,
                const arma::rowvec& responses) const
  {
    jacobian.zeros(coordinates.n_elem, predictors.n_cols);

    const arma::mat expr1 = 1.0 + 2.0 * coordinates(1) * predictors;

    jacobian.row(0) = 1.0 - arma::pow(expr1, -0.5);
    jacobian.row(1) = -1.0 * (coordinates(0) * arma::pow(expr1, -1.5) %
        (-0.5 * (2.0 * predictors)));
  }

  /**
   * Gradient of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param gradient The calculated gradient matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @return The calculated error.
   */
  double Gradient(const arma::mat& coordinates,
                  arma::mat& gradient,
                  const arma::mat& predictors,
                  const arma::rowvec& responses) const
  {
    arma::mat jacobian, result;
    Jacobian(coordinates, jacobian, predictors, responses);
    Evaluate(coordinates, predictors, responses, result);
    gradient = arma::trans(2.0 * result * jacobian.t());

    return arma::accu(arma::pow(result, 2.0));
  }

  /**
   * Get the starting point.
   *
   * @param index The index of the starting point.
   */
  arma::mat GetInitialPoint(const size_t index = 0) const
  {
    return initial.col(index);
  }

  //! Get the number of starting points.
  size_t Functions() const { return 2; }

  //! Locally stored initial starting points.
  arma::mat initial;
};

class Misra1d
{
 public:
  //! Initialize Misra1d.
  Misra1d() : initial("500 450; 0.0001 0.0003") {}

  /**
   * Evaluate the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @param result The calculated function result.
   */
  void Evaluate(const arma::mat& coordinates,
                const arma::mat& predictors,
                const arma::rowvec& responses,
                arma::mat& result) const
  {
    result = coordinates(0) * coordinates(1) * predictors %
        (arma::pow(1.0 + coordinates(1) * predictors, -1.0)) - responses;
  }

  /**
   * Jacobian of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param jacobian The calculated jacobian matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   */
  void Jacobian(const arma::mat& coordinates,
                arma::mat& jacobian,
                const arma::mat& predictors,
                const arma::rowvec& responses) const
  {
    jacobian.zeros(coordinates.n_elem, predictors.n_cols);

    const arma::mat expr1 = coordinates(0) * coordinates(1) * predictors;
    const arma::mat expr2 = coordinates(1) * predictors;
    const arma::mat expr3 = 1.0 + expr2;
    const arma::mat expr4 = arma::pow(expr3, -1.0);

    jacobian.row(0) = expr2 % expr4;
    jacobian.row(1) = coordinates(0) * predictors % expr4 + expr1 %
        (arma::pow(expr3, -2.0) % (-1.0 * predictors));
  }

  /**
   * Gradient of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param gradient The calculated gradient matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @return The calculated error.
   */
  double Gradient(const arma::mat& coordinates,
                  arma::mat& gradient,
                  const arma::mat& predictors,
                  const arma::rowvec& responses) const
  {
    arma::mat jacobian, result;
    Jacobian(coordinates, jacobian, predictors, responses);
    Evaluate(coordinates, predictors, responses, result);
    gradient = arma::trans(2.0 * result * jacobian.t());

    return arma::accu(arma::pow(result, 2.0));
  }

  /**
   * Get the starting point.
   *
   * @param index The index of the starting point.
   */
  arma::mat GetInitialPoint(const size_t index = 0) const
  {
    return initial.col(index);
  }

  //! Get the number of starting points.
  size_t Functions() const { return 2; }

  //! Locally stored initial starting points.
  arma::mat initial;
};

class Roszman1
{
 public:
  //! Initialize Roszman1.
  Roszman1() : initial("0.1 0.2; -0.00001 -0.000005; 1000 1200; -100 -150") {}

  /**
   * Evaluate the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @param result The calculated function result.
   */
  void Evaluate(const arma::mat& coordinates,
                const arma::mat& predictors,
                const arma::rowvec& responses,
                arma::mat& result) const
  {
    result = coordinates(0) - coordinates(1) * predictors - arma::atan(
        coordinates(2) / (predictors - coordinates(3))) / M_PI - responses;
  }

  /**
   * Jacobian of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param jacobian The calculated jacobian matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   */
  void Jacobian(const arma::mat& coordinates,
                arma::mat& jacobian,
                const arma::mat& predictors,
                const arma::rowvec& responses) const
  {
    jacobian.zeros(coordinates.n_elem, predictors.n_cols);

    const arma::mat expr1 = predictors - coordinates(3);
    const arma::mat expr2 = coordinates(2) / expr1;
    const arma::mat expr3 = 1.0 + arma::pow(expr2, 2.0);

    jacobian.row(0).ones();
    jacobian.row(1) = predictors;
    jacobian.row(2) = (1.0 / expr1) % (1.0 / expr3) / M_PI;
    jacobian.row(3) = coordinates(2) / arma::pow(expr1, 2.0) %
        (1.0 / expr3) / M_PI;
  }

  /**
   * Gradient of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param gradient The calculated gradient matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @return The calculated error.
   */
  double Gradient(const arma::mat& coordinates,
                  arma::mat& gradient,
                  const arma::mat& predictors,
                  const arma::rowvec& responses) const
  {
    arma::mat jacobian, result;
    Jacobian(coordinates, jacobian, predictors, responses);
    Evaluate(coordinates, predictors, responses, result);
    gradient = arma::trans(2.0 * result * jacobian.t());

    return arma::accu(arma::pow(result, 2.0));
  }

  /**
   * Get the starting point.
   *
   * @param index The index of the starting point.
   */
  arma::mat GetInitialPoint(const size_t index = 0) const
  {
    return initial.col(index);
  }

  //! Get the number of starting points.
  size_t Functions() const { return 2; }

  //! Locally stored initial starting points.
  arma::mat initial;
};

class ENSO
{
 public:
  //! Initialize ENSO.
  ENSO() : initial("11.0 10.0; 3.0 3.0; 0.5 0.5; 40.0 44.0; -0.7 -1.5; \
                    -1.3 0.5; 25.0 26.0; -0.3 -0.1; 1.4 1.5") {}

  /**
   * Evaluate the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @param result The calculated function result.
   */
  void Evaluate(const arma::mat& coordinates,
                const arma::mat& predictors,
                const arma::rowvec& responses,
                arma::mat& result) const
  {
    result = coordinates(0) +
        coordinates(1) * arma::cos(2.0 * M_PI * predictors / 12.0) +
        coordinates(2) * arma::sin(2.0 * M_PI * predictors / 12) +
        coordinates(4) * arma::cos(2.0 * M_PI * predictors / coordinates(3)) +
        coordinates(5) * arma::sin(2.0 * M_PI * predictors / coordinates(3)) +
        coordinates(7) * arma::cos(2.0 * M_PI * predictors / coordinates(6)) +
        coordinates(8) * arma::sin(2.0 * M_PI * predictors / coordinates(6)) -
        responses;
  }

  /**
   * Jacobian of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param jacobian The calculated jacobian matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   */
  void Jacobian(const arma::mat& coordinates,
                arma::mat& jacobian,
                const arma::mat& predictors,
                const arma::rowvec& responses) const
  {
    jacobian.zeros(coordinates.n_elem, predictors.n_cols);

    const arma::mat expr1 = 2.0 * M_PI * predictors;
    const arma::mat expr2 = expr1 / 12.0;
    const arma::mat expr3 = arma::cos(expr2);
    const arma::mat expr4 = arma::sin(expr2);
    const arma::mat expr5 = expr1 / coordinates(3);
    const arma::mat expr6 = arma::cos(expr5);
    const arma::mat expr7 = arma::sin(expr5);
    const arma::mat expr8 = expr1 / coordinates(6);
    const arma::mat expr9 = arma::cos(expr8);
    const arma::mat expr10 = arma::sin(expr8);
    const arma::mat expr11 = expr1 / (std::pow(coordinates(3), 2.0));
    const arma::mat expr12 = expr1 / (std::pow(coordinates(6), 2.0));

    jacobian.row(0).ones();
    jacobian.row(1) = expr3;
    jacobian.row(2) = expr4;
    jacobian.row(3) = coordinates(4) * (expr7 % expr11) -
        coordinates(5) * (expr6 % expr11);
    jacobian.row(4) = expr6;
    jacobian.row(5) = expr7;
    jacobian.row(6) = coordinates(7) * (expr9 % expr12) -
        coordinates(8) * (expr9 % expr12);
    jacobian.row(7) = expr9;
    jacobian.row(8) = expr10;
  }

  /**
   * Gradient of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param gradient The calculated gradient matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @return The calculated error.
   */
  double Gradient(const arma::mat& coordinates,
                  arma::mat& gradient,
                  const arma::mat& predictors,
                  const arma::rowvec& responses) const
  {
    arma::mat jacobian, result;
    Jacobian(coordinates, jacobian, predictors, responses);
    Evaluate(coordinates, predictors, responses, result);
    gradient = arma::trans(2.0 * result * jacobian.t());

    return arma::accu(arma::pow(result, 2.0));
  }

  /**
   * Get the starting point.
   *
   * @param index The index of the starting point.
   */
  arma::mat GetInitialPoint(const size_t index = 0) const
  {
    return initial.col(index);
  }

  //! Get the number of starting points.
  size_t Functions() const { return 2; }

  //! Locally stored initial starting points.
  arma::mat initial;
};

class MGH09
{
 public:
  //! Initialize MGH09.
  MGH09() : initial("25 0.25; 39 0.39; 41.5 0.415; 39 0.39") {}

  /**
   * Evaluate the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @param result The calculated function result.
   */
  void Evaluate(const arma::mat& coordinates,
                const arma::mat& predictors,
                const arma::rowvec& responses,
                arma::mat& result) const
  {
    result = coordinates(0) * (arma::pow(predictors, 2.0) + predictors *
        coordinates(1)) % (1.0 / (arma::pow(predictors, 2.0) + predictors *
        coordinates(2) + coordinates(3))) - responses;
  }

  /**
   * Jacobian of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param jacobian The calculated jacobian matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   */
  void Jacobian(const arma::mat& coordinates,
                arma::mat& jacobian,
                const arma::mat& predictors,
                const arma::rowvec& responses) const
  {
    jacobian.zeros(coordinates.n_elem, predictors.n_cols);

    const arma::mat expr1 = predictors % predictors;
    const arma::mat expr2 = expr1 + predictors * coordinates(1);
    const arma::mat expr3 = coordinates(0) * expr2;
    const arma::mat expr4 = expr1 + predictors * coordinates(2) +
        coordinates(3);
    const arma::mat expr5 = expr4 % expr4;

    jacobian.row(0) = expr2 % (1.0 / expr4);
    jacobian.row(1) = coordinates(0) * (predictors % (1.0 / expr4));
    jacobian.row(2) = -1.0 * (expr3 % (predictors % (1.0 / expr5)));
    jacobian.row(3) = -1.0 * (expr3 % (1.0 / expr5));
  }

  /**
   * Gradient of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param gradient The calculated gradient matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @return The calculated error.
   */
  double Gradient(const arma::mat& coordinates,
                  arma::mat& gradient,
                  const arma::mat& predictors,
                  const arma::rowvec& responses) const
  {
    arma::mat jacobian, result;
    Jacobian(coordinates, jacobian, predictors, responses);
    Evaluate(coordinates, predictors, responses, result);
    gradient = arma::trans(2.0 * result * jacobian.t());

    return arma::accu(arma::pow(result, 2.0));
  }

  /**
   * Get the starting point.
   *
   * @param index The index of the starting point.
   */
  arma::mat GetInitialPoint(const size_t index = 0) const
  {
    return initial.col(index);
  }

  //! Get the number of starting points.
  size_t Functions() const { return 2; }

  //! Locally stored initial starting points.
  arma::mat initial;
};

class Thurber
{
 public:
  //! Initialize Thurber.
  Thurber() : initial("1000 1300; 1000 1500; 400 500; 40 75; 0.7 1; \
                       0.3 0.4; 0.03 0.05") {}

  /**
   * Evaluate the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @param result The calculated function result.
   */
  void Evaluate(const arma::mat& coordinates,
                const arma::mat& predictors,
                const arma::rowvec& responses,
                arma::mat& result) const
  {
    result = (coordinates(0) + coordinates(1) * predictors +
        coordinates(2) * arma::pow(predictors, 2.0) + coordinates(3) *
        arma::pow(predictors, 3.0)) % (1.0 / (1.0 + coordinates(4) *
        predictors + coordinates(5) * arma::pow(predictors, 2.0) +
        coordinates(6) * arma::pow(predictors, 3.0))) - responses;
  }

  /**
   * Jacobian of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param jacobian The calculated jacobian matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   */
  void Jacobian(const arma::mat& coordinates,
                arma::mat& jacobian,
                const arma::mat& predictors,
                const arma::rowvec& responses) const
  {
    jacobian.zeros(coordinates.n_elem, predictors.n_cols);

    const arma::mat expr1 = coordinates(0) + coordinates(1) * predictors +
        coordinates(2) * arma::pow(predictors, 2.0) + coordinates(3) *
        arma::pow(predictors, 3.0);
    const arma::mat expr2 = 1.0 + coordinates(4) * predictors + coordinates(5) *
        arma::pow(predictors, 2.0) + coordinates(6) *
        arma::pow(predictors, 3.0);

    jacobian.row(0) = 1.0 / expr2;
    jacobian.row(1) = predictors % (1.0 / expr2);
    jacobian.row(2) = jacobian.row(1) % predictors;
    jacobian.row(3) = jacobian.row(2) % predictors;
    jacobian.row(4) = expr1 % (1.0 / (expr2 % expr2)) % predictors;
    jacobian.row(5) = jacobian.row(4) % predictors;
    jacobian.row(6) = jacobian.row(5) % predictors;
  }

  /**
   * Gradient of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param gradient The calculated gradient matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @return The calculated error.
   */
  double Gradient(const arma::mat& coordinates,
                  arma::mat& gradient,
                  const arma::mat& predictors,
                  const arma::rowvec& responses) const
  {
    arma::mat jacobian, result;
    Jacobian(coordinates, jacobian, predictors, responses);
    Evaluate(coordinates, predictors, responses, result);
    gradient = arma::trans(2.0 * result * jacobian.t());

    return arma::accu(arma::pow(result, 2.0));
  }

  /**
   * Get the starting point.
   *
   * @param index The index of the starting point.
   */
  arma::mat GetInitialPoint(const size_t index = 0) const
  {
    return initial.col(index);
  }

  //! Get the number of starting points.
  size_t Functions() const { return 2; }

  //! Locally stored initial starting points.
  arma::mat initial;
};

class BoxBOD
{
 public:
  //! Initialize BoxBOD.
  BoxBOD() : initial("1 100; 1 0.75") {}

  /**
   * Evaluate the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @param result The calculated function result.
   */
  void Evaluate(const arma::mat& coordinates,
                const arma::mat& predictors,
                const arma::rowvec& responses,
                arma::mat& result) const
  {
    result = coordinates(0) * (1.0 - arma::exp(-1.0 * coordinates(1) *
        predictors)) - responses;
  }

  /**
   * Jacobian of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param jacobian The calculated jacobian matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   */
  void Jacobian(const arma::mat& coordinates,
                arma::mat& jacobian,
                const arma::mat& predictors,
                const arma::rowvec& responses) const
  {
    jacobian.zeros(coordinates.n_elem, predictors.n_cols);

    jacobian.row(0) = (1.0 - arma::exp(-1.0 * coordinates(1) * predictors));
    jacobian.row(1) = coordinates(0) * predictors %
        arma::exp(-1.0 * coordinates(1) * predictors);
  }

  /**
   * Gradient of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param gradient The calculated gradient matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @return The calculated error.
   */
  double Gradient(const arma::mat& coordinates,
                  arma::mat& gradient,
                  const arma::mat& predictors,
                  const arma::rowvec& responses) const
  {
    arma::mat jacobian, result;
    Jacobian(coordinates, jacobian, predictors, responses);
    Evaluate(coordinates, predictors, responses, result);
    gradient = arma::trans(2.0 * result * jacobian.t());

    return arma::accu(arma::pow(result, 2.0));
  }

  /**
   * Get the starting point.
   *
   * @param index The index of the starting point.
   */
  arma::mat GetInitialPoint(const size_t index = 0) const
  {
    return initial.col(index);
  }

  //! Get the number of starting points.
  size_t Functions() const { return 2; }

  //! Locally stored initial starting points.
  arma::mat initial;
};

class Rat42
{
 public:
  //! Initialize Rat42.
  Rat42() : initial("100 75; 1 2.5; 0.1 0.07") {}

  /**
   * Evaluate the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @param result The calculated function result.
   */
  void Evaluate(const arma::mat& coordinates,
                const arma::mat& predictors,
                const arma::rowvec& responses,
                arma::mat& result) const
  {
    result = coordinates(0) / (1.0 + arma::exp(coordinates(1) - coordinates(2) *
        predictors)) - responses;
  }

  /**
   * Jacobian of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param jacobian The calculated jacobian matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   */
  void Jacobian(const arma::mat& coordinates,
                arma::mat& jacobian,
                const arma::mat& predictors,
                const arma::rowvec& responses) const
  {
    jacobian.zeros(coordinates.n_elem, predictors.n_cols);

    const arma::mat expr1 = arma::exp(coordinates(1) - coordinates(2) *
        predictors);
    const arma::mat expr2 = 1.0 + expr1;
    const arma::mat expr3 = expr2 % expr2;

    jacobian.row(0) = 1.0 / expr2;
    jacobian.row(1) = -1.0 * (coordinates(0) * (expr1 % (1.0 / expr3)));
    jacobian.row(1) = coordinates(0) * (expr1 % predictors) % (1.0 / expr3);
  }

  /**
   * Gradient of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param gradient The calculated gradient matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @return The calculated error.
   */
  double Gradient(const arma::mat& coordinates,
                  arma::mat& gradient,
                  const arma::mat& predictors,
                  const arma::rowvec& responses) const
  {
    arma::mat jacobian, result;
    Jacobian(coordinates, jacobian, predictors, responses);
    Evaluate(coordinates, predictors, responses, result);
    gradient = arma::trans(2.0 * result * jacobian.t());

    return arma::accu(arma::pow(result, 2.0));
  }

  /**
   * Get the starting point.
   *
   * @param index The index of the starting point.
   */
  arma::mat GetInitialPoint(const size_t index = 0) const
  {
    return initial.col(index);
  }

  //! Get the number of starting points.
  size_t Functions() const { return 2; }

  //! Locally stored initial starting points.
  arma::mat initial;
};

class MGH10
{
 public:
  //! Initialize MGH10.
  MGH10() : initial("2 0.02; 400000 4000; 25000 250") {}

  /**
   * Evaluate the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @param result The calculated function result.
   */
  void Evaluate(const arma::mat& coordinates,
                const arma::mat& predictors,
                const arma::rowvec& responses,
                arma::mat& result) const
  {
    result = coordinates(0) * arma::exp(coordinates(1) /
        (predictors + coordinates(2))) - responses;
  }

  /**
   * Jacobian of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param jacobian The calculated jacobian matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   */
  void Jacobian(const arma::mat& coordinates,
                arma::mat& jacobian,
                const arma::mat& predictors,
                const arma::rowvec& responses) const
  {
    jacobian.zeros(coordinates.n_elem, predictors.n_cols);

    const arma::mat expr1 = predictors + coordinates(2);
    const arma::mat expr2 = arma::exp(coordinates(1) / expr1);

    jacobian.row(0) = expr2;
    jacobian.row(1) = coordinates(0) * (expr2 % (1.0 / expr1));
    jacobian.row(2) = -1.0 * (coordinates(0) * (expr2 % (coordinates(1) *
        (1.0 / (expr1 % expr1)))));
  }

  /**
   * Gradient of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param gradient The calculated gradient matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @return The calculated error.
   */
  double Gradient(const arma::mat& coordinates,
                  arma::mat& gradient,
                  const arma::mat& predictors,
                  const arma::rowvec& responses) const
  {
    arma::mat jacobian, result;
    Jacobian(coordinates, jacobian, predictors, responses);
    Evaluate(coordinates, predictors, responses, result);
    gradient = arma::trans(2.0 * result * jacobian.t());

    return arma::accu(arma::pow(result, 2.0));
  }

  /**
   * Get the starting point.
   *
   * @param index The index of the starting point.
   */
  arma::mat GetInitialPoint(const size_t index = 0) const
  {
    return initial.col(index);
  }

  //! Get the number of starting points.
  size_t Functions() const { return 2; }

  //! Locally stored initial starting points.
  arma::mat initial;
};

class Eckerle4
{
 public:
  //! Initialize Eckerle4.
  Eckerle4() : initial("1 1.5; 10 5; 500 450") {}

  /**
   * Evaluate the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @param result The calculated function result.
   */
  void Evaluate(const arma::mat& coordinates,
                const arma::mat& predictors,
                const arma::rowvec& responses,
                arma::mat& result) const
  {
    result = (coordinates(0) / coordinates(1)) * arma::exp(-0.5 *
        arma::pow((predictors - coordinates(2)) / coordinates(1), 2.0)) -
        responses;
  }

  /**
   * Jacobian of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param jacobian The calculated jacobian matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   */
  void Jacobian(const arma::mat& coordinates,
                arma::mat& jacobian,
                const arma::mat& predictors,
                const arma::rowvec& responses) const
  {
    jacobian.zeros(coordinates.n_elem, predictors.n_cols);

    const double expr1 = coordinates(0) / coordinates(1);
    const arma::mat expr2 = predictors - coordinates(2);
    const arma::mat expr3 = expr2 / coordinates(1);
    const arma::mat expr4 = arma::exp(-0.5 * arma::pow(expr3, 2.0));
    const double expr5 = 1.0 / coordinates(1);
    const double expr6 = std::pow(coordinates(1), 2.0);

    jacobian.row(0) = expr5 * expr4;
    jacobian.row(1) = expr1 * (expr4 % (0.5 * (2.0 * (expr2 / expr6 % expr3))))
        - coordinates(0) / expr6 * expr4;
    jacobian.row(2) = expr1 * (expr4 % (0.5 * (2.0 * (expr5 * expr3))));
  }

  /**
   * Gradient of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param gradient The calculated gradient matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @return The calculated error.
   */
  double Gradient(const arma::mat& coordinates,
                  arma::mat& gradient,
                  const arma::mat& predictors,
                  const arma::rowvec& responses) const
  {
    arma::mat jacobian, result;
    Jacobian(coordinates, jacobian, predictors, responses);
    Evaluate(coordinates, predictors, responses, result);
    gradient = arma::trans(2.0 * result * jacobian.t());

    return arma::accu(arma::pow(result, 2.0));
  }

  /**
   * Get the starting point.
   *
   * @param index The index of the starting point.
   */
  arma::mat GetInitialPoint(const size_t index = 0) const
  {
    return initial.col(index);
  }

  //! Get the number of starting points.
  size_t Functions() const { return 2; }

  //! Locally stored initial starting points.
  arma::mat initial;
};

class Rat43
{
 public:
  //! Initialize Rat43.
  Rat43() : initial("100 700; 10 5; 1 0.75; 1 1.3") {}

  /**
   * Evaluate the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @param result The calculated function result.
   */
  void Evaluate(const arma::mat& coordinates,
                const arma::mat& predictors,
                const arma::rowvec& responses,
                arma::mat& result) const
  {
    result = coordinates(0) /
        arma::pow(1.0 + arma::exp(coordinates(1) - coordinates(2) *
        predictors), 1.0 / coordinates(3)) - responses;
  }

  /**
   * Jacobian of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param jacobian The calculated jacobian matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   */
  void Jacobian(const arma::mat& coordinates,
                arma::mat& jacobian,
                const arma::mat& predictors,
                const arma::rowvec& responses) const
  {
    jacobian.zeros(coordinates.n_elem, predictors.n_cols);

    const arma::mat expr1 = arma::exp(coordinates(1) - coordinates(2) *
        predictors);
    const arma::mat expr2 = 1.0 + expr1;
    const double expr3 = 1.0 / coordinates(3);
    const arma::mat expr4 = arma::pow(expr2, expr3);
    const arma::mat expr5 = arma::pow(expr2, expr3 - 1.0);
    const arma::mat expr6 = expr4 % expr4;

    jacobian.row(0) = 1.0 / expr4;
    jacobian.row(1) = -1.0 * (coordinates(0) * (expr5 % (expr3 * expr1 %
        predictors)) % (1.0 / expr6));
    jacobian.row(2) = coordinates(0) * (expr5 % (expr3 * expr1)) %
        (1.0 / expr6);
    jacobian.row(3) = coordinates(0) * (expr4 % arma::log(expr2) * (1.0 /
        (coordinates(3) * coordinates(3)))) % (1.0 / expr6);
  }

  /**
   * Gradient of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param gradient The calculated gradient matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @return The calculated error.
   */
  double Gradient(const arma::mat& coordinates,
                  arma::mat& gradient,
                  const arma::mat& predictors,
                  const arma::rowvec& responses) const
  {
    arma::mat jacobian, result;
    Jacobian(coordinates, jacobian, predictors, responses);
    Evaluate(coordinates, predictors, responses, result);
    gradient = arma::trans(2.0 * result * jacobian.t());

    return arma::accu(arma::pow(result, 2.0));
  }

  /**
   * Get the starting point.
   *
   * @param index The index of the starting point.
   */
  arma::mat GetInitialPoint(const size_t index = 0) const
  {
    return initial.col(index);
  }

  //! Get the number of starting points.
  size_t Functions() const { return 2; }

  //! Locally stored initial starting points.
  arma::mat initial;
};

class Bennett5
{
 public:
  //! Initialize Bennett5.
  Bennett5() : initial("-2000 -1500; 50 45; 0.8 0.85") {}

  /**
   * Evaluate the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @param result The calculated function result.
   */
  void Evaluate(const arma::mat& coordinates,
                const arma::mat& predictors,
                const arma::rowvec& responses,
                arma::mat& result) const
  {
    result = coordinates(0) * arma::pow(coordinates(1) + predictors,
        -1.0 / coordinates(2)) - responses;
  }

  /**
   * Jacobian of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param jacobian The calculated jacobian matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   */
  void Jacobian(const arma::mat& coordinates,
                arma::mat& jacobian,
                const arma::mat& predictors,
                const arma::rowvec& responses) const
  {
    jacobian.zeros(coordinates.n_elem, predictors.n_cols);

    const arma::mat expr1 = coordinates(1) + predictors;
    const double expr2 = -1.0 / coordinates(2);
    const arma::mat expr3 = arma::pow(expr1, expr2);

    jacobian.row(0) = expr3;
    jacobian.row(1) = coordinates(0) * arma::pow(expr1, (expr2 - 1)) * expr2;
    jacobian.row(2) = coordinates(0) * (expr3 % (arma::log(expr1) *
        (1.0 / std::pow(coordinates(2), 2.0))));
  }

  /**
   * Gradient of the function with the given parameter.
   *
   * @param coordinates The function coordinates.
   * @param gradient The calculated gradient matrix.
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   * @return The calculated error.
   */
  double Gradient(const arma::mat& coordinates,
                  arma::mat& gradient,
                  const arma::mat& predictors,
                  const arma::rowvec& responses) const
  {
    arma::mat jacobian, result;
    Jacobian(coordinates, jacobian, predictors, responses);
    Evaluate(coordinates, predictors, responses, result);
    gradient = arma::trans(2.0 * result * jacobian.t());

    return arma::accu(arma::pow(result, 2.0));
  }

  /**
   * Get the starting point.
   *
   * @param index The index of the starting point.
   */
  arma::mat GetInitialPoint(const size_t index = 0) const
  {
    return initial.col(index);
  }

  //! Get the number of starting points.
  size_t Functions() const { return 2; }

  //! Locally stored initial starting points.
  arma::mat initial;
};

template<class FunctionType>
class NIST
{
 public:
  /**
   * Create the NIST object with the given parameter.
   *
   * @param predictors Matrix of data points (X).
   * @param responses The measured data for each point in X (y).
   */
  NIST(const arma::mat& predictors, const arma::rowvec& responses) :
      predictors(predictors), responses(responses)
  {
    /* Nothing to do here. */
  }

  /**
   * Shuffle the order of function visitation. This may be called by the
   * optimizer.
   */
  void Shuffle()
  {
    /* Nothing to do here. */
  }

  //! Return 1 (the number of functions).
  size_t NumFunctions() const { return 1; }

  //! Get the starting point.
  arma::mat GetInitialPoint(const size_t index = 0) const
  {
    return function.GetInitialPoint(index);
  }

  /*
   * Evaluate a function for a particular batch-size.
   *
   * @param coordinates The function coordinates.
   * @param begin The first function.
   * @param batchSize Number of points to process.
   */
  double Evaluate(const arma::mat& coordinates,
                  const size_t /*begin*/,
                  const size_t /*batchSize*/) const
  {
    return Evaluate(coordinates);
  }

  /*
   * Evaluate a function with the given coordinates.
   *
   * @param coordinates The function coordinates.
   */
  double Evaluate(const arma::mat& coordinates) const
  {
    arma::mat result;
    function.Evaluate(coordinates, predictors, responses, result);
    return arma::accu(arma::pow(result, 2.0));
  }

  /**
   * Evaluate the gradient of the function with the given parameters.
   *
   * @param coordinates The function coordinates.
   * @param gradient Vector to output gradient into.
   */
  void Gradient(const arma::mat& coordinates, arma::mat& gradient) const
  {

    function.Gradient(coordinates, gradient, predictors, responses);
  }

  /*
   * Evaluate the gradient of a function for a particular batch-size.
   *
   * @param coordinates The function coordinates.
   * @param begin The first function.
   * @param gradient The function gradient.
   * @param batchSize Number of points to process.
   */
  void Gradient(const arma::mat& coordinates,
                const size_t /*begin*/,
                arma::mat& gradient,
                const size_t /*batchSize*/) const
  {
    function.Gradient(coordinates, gradient, predictors, responses);
  }

  /**
   * Evaluate the objective function and gradient of the function
   * simultaneously with the given parameters.
   *
   * @param coordinates The function coordinates.
   * @param gradient The function gradient.
   */
  double EvaluateWithGradient(const arma::mat& coordinates, arma::mat& gradient)
  {
    return function.Gradient(coordinates, gradient, predictors, responses);
  }

  /**
   * Evaluate the objective function and gradient of the function
   * simultaneously with the given parameters.
   *
   * @param coordinates The function coordinates.
   * @param begin The first function.
   * @param gradient The function gradient.
   * @param batchSize Number of points to process.
   */
  double EvaluateWithGradient(const arma::mat& coordinates,
                              const size_t begin,
                              arma::mat& gradient,
                              const size_t batchSize = 1)
  {
    return function.Gradient(coordinates, gradient, predictors, responses);
  }

  //! Return the function instantiation.
  FunctionType& Function() { return function; }

 private:
  //! Matrix of data points (X).
  arma::mat predictors;

  //! The measured data for each point in X (y).
  arma::rowvec responses;

  //! The instantiated objective function.
  FunctionType function;
};

} // namespace test
} // namespace ens

#endif // ENSMALLEN_PROBLEMS_NIST_HPP

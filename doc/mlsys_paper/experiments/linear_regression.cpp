/**
 * @file linear_regression.cpp
 * @author Ryan Curtin
 *
 * A simple implementation of the LinearRegressionFunction used as an example in
 * the paper, using EvaluateWithGradient().
 */
#include <ensmallen.hpp>

using namespace ens;

class LinearRegressionFunction
{
 public:
  LinearRegressionFunction(arma::mat& X, arma::rowvec& y) : X(X), y(y) { }

  double EvaluateWithGradient(const arma::mat& theta, arma::mat& gradient)
  {
    arma::rowvec v = (y - theta.t() * X);
    gradient = -(2 * v * X.t()).t();
    return arma::accu(v % v);
  }

 private:
  arma::mat& X;
  arma::rowvec& y;
};

int main(int argc, char** argv)
{
  if (argc < 3)
    throw std::invalid_argument("usage: program <dims> <points");

  int dims = atoi(argv[1]);
  int points = atoi(argv[2]);

  // This is just noise... the model will be worthless.  But that doesn't
  // actually matter.
  arma::mat X(dims, points, arma::fill::randu);
  arma::rowvec y(points, arma::fill::randu);

  // Add a slight pattern to the data...
  for (size_t i = 0; i < points; ++i)
  {
    double a = arma::as_scalar(arma::randu<arma::vec>(1));
    X(1, i) += a;
    y(i) += a;
  }

  LinearRegressionFunction lrf(X, y);
  L_BFGS lbfgs;
  lbfgs.MaxIterations() = 10;

  arma::vec theta(dims, arma::fill::randu);

  arma::wall_clock clock;

  clock.tic();
  lbfgs.Optimize(lrf, theta);
  std::cout << clock.toc() << std::endl;
}

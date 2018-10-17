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

  double EvaluateWithGradient(const arma::mat& theta,
                              const size_t begin,
                              arma::mat& gradient,
                              const size_t batchSize)
  {
    // Print entire objective every 10k points.
    if (begin % 8192 == 0)
      std::cout << std::pow(arma::norm(y - theta.t() * X), 2.0) << std::endl;

    arma::rowvec v = (y.subvec(begin, begin + batchSize - 1) - theta.t() *
        X.cols(begin, begin + batchSize - 1));
    gradient = -(2 * v * X.cols(begin, begin + batchSize - 1).t()).t();
    return arma::accu(v % v);
  }

  void Shuffle()
  {
    arma::uvec ordering = arma::shuffle(arma::linspace<arma::uvec>(0,
        X.n_cols - 1, X.n_cols));
    X = X.cols(ordering);
    y = y.cols(ordering);
  }

  size_t NumFunctions() const
  {
    return X.n_cols;
  }

 private:
  arma::mat& X;
  arma::rowvec& y;
};

int main(int argc, char** argv)
{
  if (argc < 3)
    throw std::invalid_argument("usage: program <points.csv> <responses.csv>");

  arma::mat X;
  X.load(argv[1]);
  X = X.t();

  arma::vec yIn;
  yIn.load(argv[2]);
  arma::rowvec y = yIn.t();

  arma::vec theta(X.n_rows, arma::fill::randu);
  arma::vec thetaOut(theta);

  LinearRegressionFunction lrf(X, y);

  // Optimize with several different optimizers.
  std::cout << "-- sgd --" << std::endl;
  StandardSGD(1e-10, 256, 5 * 515345, 0).Optimize(lrf, thetaOut = theta);
/*
  std::cout << "-- adam --" << std::endl;
  Adam(0.01, 256, 0.9, 0.999, 0, 5 * 515345).Optimize(lrf, thetaOut = theta);
  std::cout << "-- adagrad --" << std::endl;
  AdaGrad(0.01, 256, 0, 5 * 515345).Optimize(lrf, thetaOut = theta);
  std::cout << "-- smorms3 --" << std::endl;
  SMORMS3(0.001, 256, 0, 5 * 515345).Optimize(lrf, thetaOut = theta);
  std::cout << "-- spalera --" << std::endl;
  SPALeRASGD<>(0.01, 256, 5 * 515345, 0).Optimize(lrf, thetaOut = theta);
  std::cout << "-- rmsprop --" << std::endl;
  RMSProp(0.01, 256, 0.99, 0, 5 * 515345).Optimize(lrf, thetaOut = theta);
*/
}

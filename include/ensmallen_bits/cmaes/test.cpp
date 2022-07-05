#include <ensmallen.hpp>
#include <iostream>
class LinearRegressionFunction
{
 public:
  LinearRegressionFunction(arma::mat& dataIn,
                           arma::rowvec& responsesIn) :
      data(dataIn), responses(responsesIn) { }
  double Evaluate(const arma::mat& x, const size_t i, const size_t batchSize)
  {
    double objective = 0.0;
    for (int j = i; j < i + batchSize; ++j)
    {
      objective = objective + std::pow(arma::norm(responses(j) - x.t()*data.col(j)), 2.0);
    }
    return objective;
  }
  void Shuffle()
  {
    arma::uvec ordering = arma::shuffle(
        arma::linspace<arma::uvec>(0, data.n_cols - 1, data.n_cols));
    data = data.cols(ordering);
    responses = responses.cols(ordering);
  }
  size_t NumFunctions() { 
    return data.n_cols; 
  }
  private:
    arma::mat& data;
    arma::rowvec& responses;
};

int main(int argc, char ** argv)
{
  size_t iters; 
  std::cout << "Typing the number of iterations: " << std::endl; 
  std::cin >> iters; 
  ens::ActiveApproxCMAES cmaes(0, -10, 10, 32, 1000, 1e-5);
  double total_time = 0.0;
  for(size_t i = 0; i < iters; ++i){
    arma::wall_clock clock;
    arma::mat data(300, 10000, arma::fill::randn);
    arma::rowvec responses(10000, arma::fill::randn);
    arma::mat params(300, 1, arma::fill::randn);
    LinearRegressionFunction lrf(data, responses);
    clock.tic();
    cmaes.Optimize(lrf, params);
    total_time += clock.toc();
  }
  double aver_time = total_time/iters;
  std::cout << "The optimized linear regression model found by CMAES has the "
     << aver_time << std::endl;
}


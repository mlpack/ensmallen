#include <iostream>
#include <ensmallen.hpp>

//#define CATCH_CONFIG_MAIN  // catch.hpp will define main()
#define CATCH_CONFIG_RUNNER  // we will define main()
#include "catch.hpp"

int main(int argc, char** argv)
{
  /**
   * Uncomment these three lines if you want to test with different random seeds
   * each run.  This is good for ensuring that a test's tolerance is sufficient
   * across many different runs.
   */
  //size_t seed = std::time(NULL);
  //srand((unsigned int) seed);
  //arma::arma_rng::set_seed(seed);

  std::cout << "ensmallen version: " << ens::version::as_string() << std::endl;

  std::cout << "armadillo version: " << arma::arma_version::as_string() << std::endl;

  return Catch::Session().run(argc, argv);
}

#include <iostream>
#include <ensmallen.hpp>

//#define CATCH_CONFIG_MAIN  // catch.hpp will define main()
#define CATCH_CONFIG_RUNNER  // we will define main()
#include "catch.hpp"

int main(int argc, char** argv)
{
  std::cout << "ensmallen version: " << ens::ens_version::as_string() << std::endl;
  
  std::cout << "armadillo version: " << arma::arma_version::as_string() << std::endl;
  
  return Catch::Session().run(argc, argv);
}

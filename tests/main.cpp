/**
 * @file main.cpp
 * @author Conrad Sanderson
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <iostream>

#define COOT_DEFAULT_BACKEND CUDA_BACKEND
#define COOT_USE_U64S64
#define ENS_PRINT_INFO
#define ENS_PRINT_WARN
#include <ensmallen.hpp>

// We will define main().
#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

int main(int argc, char** argv)
{
  #ifdef USE_COOT
  coot::get_rt().init(true);
  #endif

  Catch::Session session;
  const int returnCode = session.applyCommandLine(argc, argv);
  // Check for a command line error.
  if (returnCode != 0)
    return returnCode;

  std::cout << "ensmallen version: " << ens::version::as_string() << std::endl;
  std::cout << "armadillo version: " << arma::arma_version::as_string() << std::endl;

  #ifdef USE_COOT
  std::cout << "bandicoot version: " << coot::coot_version::as_string() << std::endl;
  #endif

  // Use Catch2 command-line to set the random seed.
  // -rng-seed <'time'|number>
  // If a number is provided this is used directly as the seed. Alternatively
  // if the keyword 'time' is provided then the result of calling std::time(0)
  // is used.
  const size_t seed = session.config().rngSeed();
  std::cout << "random seed: " << seed << std::endl;
  srand((unsigned int) seed);
  arma::arma_rng::set_seed(seed);

  return session.run();
}

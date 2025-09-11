/**
 * @file sgdr_test.cpp
 * @author Marcus Edel
 * @author Conrad Sanderson
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#if defined(ENS_USE_COOT)
  #include <armadillo>
  #include <bandicoot>
#endif
#include <ensmallen.hpp>
#include "catch.hpp"
#include "test_function_tools.hpp"
#include "test_types.hpp"

using namespace ens;
using namespace ens::test;

TEMPLATE_TEST_CASE("SGDR_CyclicalResetTest", "[SGDR]", ENS_ALL_CPU_TEST_TYPES)
{
  const double stepSize = 0.5;
  TestType iterate;

  // Now run cyclical decay policy with a couple of multiplicators and initial
  // restarts.
  for (size_t restart = 5; restart < 100; restart += 10)
  {
    for (size_t mult = 2; mult < 5; ++mult)
    {
      double epochStepSize = stepSize;

      CyclicalDecay cyclicalDecay(restart, double(mult), stepSize);
      CyclicalDecay::Policy<TestType, TestType> p(cyclicalDecay);
      cyclicalDecay.EpochBatches() = (double) 1000 / 10;

      // Create all restart epochs.
      arma::Col<size_t> nextRestart(1000 / 10 /  mult);
      nextRestart(0) = restart;
      for (size_t j = 1; j < nextRestart.n_elem; ++j)
        nextRestart(j) = nextRestart(j - 1) * mult;

      for (size_t i = 0; i < 1000; ++i)
      {
        p.Update(iterate, epochStepSize, iterate);
        if (i <= restart || accu(find(nextRestart == i)) > 0)
        {
          REQUIRE(epochStepSize == stepSize);
        }
      }
    }
  }
}

TEMPLATE_TEST_CASE("SGDR_LogisticRegressionFunction", "[SGDR]",
    ENS_ALL_TEST_TYPES)
{
  // Run SGDR with a couple of batch sizes.
  for (size_t batchSize = 5; batchSize < 50; batchSize += 5)
  {
    SGDR<> sgdr(50, 2.0, batchSize, 0.005 * batchSize, 10000, 1e-3);
    LogisticRegressionFunctionTest<TestType>(sgdr);
  }
}

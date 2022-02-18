/**
 * @file snapshot_ensembles.cpp
 * @author Marcus Edel
 * @author Conrad Sanderson
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <ensmallen.hpp>
#include "catch.hpp"
#include "test_function_tools.hpp"

using namespace ens;
using namespace ens::test;

/*
 * Test that the step size resets after a specified number of epochs.
 */
TEST_CASE("SnapshotEnsemblesResetTest","[SnapshotEnsemblesTest]")
{
  const double stepSize = 0.5;
  arma::mat iterate;

  // Now run cyclical decay policy with a couple of multiplicators and initial
  // restarts.
  for (size_t restart = 5; restart < 100; restart += 10)
  {
    for (size_t mult = 2; mult < 5; ++mult)
    {
      double epochStepSize = stepSize;

      SnapshotEnsembles snapshotEnsembles(restart,
          double(mult), stepSize, 1000, 2);
      SnapshotEnsembles::Policy<arma::mat, arma::mat> p(snapshotEnsembles);

      snapshotEnsembles.EpochBatches() = 10 / (double)1000;
      // Create all restart epochs.
      arma::Col<size_t> nextRestart(1000 / 10 /  mult);
      nextRestart(0) = restart;
      for (size_t j = 1; j < nextRestart.n_elem; ++j)
        nextRestart(j) = nextRestart(j - 1) * mult;

      for (size_t i = 0; i < 1000; ++i)
      {
        p.Update(iterate, epochStepSize, iterate);
        if (i <= restart || arma::accu(arma::find(nextRestart == i)) > 0)
        {
          REQUIRE(epochStepSize == stepSize);
        }
      }

      REQUIRE(p.Snapshots().size() == 2);
    }
  }
}

/**
 * Run SGDR with snapshot ensembles on logistic regression and make sure the
 * results are acceptable.
 */
TEST_CASE("SnapshotEnsemblesLogisticRegressionTest","[SnapshotEnsemblesTest]")
{
  // Run SGDR with snapshot ensembles on a couple of batch sizes.
  for (size_t batchSize = 5; batchSize < 50; batchSize += 5)
  {
    SnapshotSGDR<> sgdr(50, 2.0, batchSize, 0.01, 10000, 1e-3);
    LogisticRegressionFunctionTest(sgdr, 0.003, 0.006);
  }
}

/**
 * Run SGDR with snapshot ensembles on logistic regression and make sure the
 * results are acceptable.  Use arma::fmat.
 */
TEST_CASE("SnapshotEnsemblesLogisticRegressionFMatTest","[SnapshotEnsemblesTest]")
{
  // Run SGDR with snapshot ensembles on a couple of batch sizes.
  for (size_t batchSize = 5; batchSize < 50; batchSize += 5)
  {
    SnapshotSGDR<> sgdr(50, 2.0, batchSize, 0.01, 10000, 1e-3);
    LogisticRegressionFunctionTest<arma::fmat>(sgdr, 0.03, 0.06, 3);
  }
}

#if ARMA_VERSION_MAJOR > 9 ||\
    (ARMA_VERSION_MAJOR == 9 && ARMA_VERSION_MINOR >= 400)

/**
 * Run SGDR with snapshot ensembles on logistic regression and make sure the
 * results are acceptable.  Use arma::sp_mat.
 */
TEST_CASE("SnapshotEnsemblesLogisticRegressionSpMatTest",
          "[SnapshotEnsemblesTest]")
{
  // Run SGDR with snapshot ensembles on a couple of batch sizes.
  for (size_t batchSize = 5; batchSize < 50; batchSize += 5)
  {
    SnapshotSGDR<> sgdr(50, 2.0, batchSize, 0.01, 10000, 1e-3);
    LogisticRegressionFunctionTest<arma::sp_mat>(sgdr, 0.003, 0.006);
  }
}

#endif

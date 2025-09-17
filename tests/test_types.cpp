/**
 * @file test_types.cpp
 * @author Ryan Curtin
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <ensmallen.hpp>
#include "test_types.hpp"

// Force instantiation of static tolerances members.
using namespace ens;
using namespace ens::test;

const double Tolerances<arma::mat>::Obj = 1e-8;
const double Tolerances<arma::mat>::Coord = 1e-4;
const double Tolerances<arma::mat>::LargeObj = 1e-3;
const double Tolerances<arma::mat>::LargeCoord = 1e-2;
const double Tolerances<arma::mat>::LRTrainAcc = 0.003;
const double Tolerances<arma::mat>::LRTestAcc = 0.006;

const float  Tolerances<arma::fmat>::Obj = 1e-4;
const float  Tolerances<arma::fmat>::Coord = 1e-2;
const float  Tolerances<arma::fmat>::LargeObj = 2e-3;
const float  Tolerances<arma::fmat>::LargeCoord = 2e-2;
const double Tolerances<arma::fmat>::LRTrainAcc = 0.003;
const double Tolerances<arma::fmat>::LRTestAcc = 0.006;

#if defined(ARMA_HAVE_FP16)
const arma::fp16 Tolerances<arma::mat>::Obj = arma::fp16(0.0001);
const arma::fp16 Tolerances<arma::mat>::Coord = arma::fp16(0.01);
const arma::fp16 Tolerances<arma::mat>::LargeObj = arma::fp16(0.03);
const arma::fp16 Tolerances<arma::mat>::LargeCoord = arma::fp16(0.1);
const double     Tolerances<arma::mat>::LRTrainAcc = 0.03;
const double     Tolerances<arma::mat>::LRTestAcc = 0.06;
#endif

const double Tolerances<arma::sp_mat>::Obj = 1e-8;
const double Tolerances<arma::sp_mat>::Coord = 1e-4;
const double Tolerances<arma::sp_mat>::LargeObj = 1e-3;
const double Tolerances<arma::sp_mat>::LargeCoord = 1e-2;
const double Tolerances<arma::sp_mat>::LRTrainAcc = 0.003;
const double Tolerances<arma::sp_mat>::LRTestAcc = 0.006;

#if defined(ENS_HAVE_COOT)
const double Tolerances<coot::mat>::Obj = 1e-8;
const double Tolerances<coot::mat>::Coord = 1e-4;
const double Tolerances<coot::mat>::LargeObj = 1e-3;
const double Tolerances<coot::mat>::LargeCoord = 1e-2;
const double Tolerances<coot::mat>::LRTrainAcc = 0.003;
const double Tolerances<coot::mat>::LRTestAcc = 0.006;

const float  Tolerances<coot::fmat>::Obj = 1e-4;
const float  Tolerances<coot::fmat>::Coord = 1e-2;
const float  Tolerances<coot::fmat>::LargeObj = 2e-3;
const float  Tolerances<coot::fmat>::LargeCoord = 2e-2;
const double Tolerances<coot::fmat>::LRTrainAcc = 0.003;
const double Tolerances<coot::fmat>::LRTestAcc = 0.006;
#endif

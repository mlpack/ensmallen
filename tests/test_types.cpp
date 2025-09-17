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

const double Tolerances<arma::mat>::Obj;
const double Tolerances<arma::mat>::Coord;
const double Tolerances<arma::mat>::LargeObj;
const double Tolerances<arma::mat>::LargeCoord;
const double Tolerances<arma::mat>::LRTrainAcc;
const double Tolerances<arma::mat>::LRTestAcc;

const float  Tolerances<arma::fmat>::Obj;
const float  Tolerances<arma::fmat>::Coord;
const float  Tolerances<arma::fmat>::LargeObj;
const float  Tolerances<arma::fmat>::LargeCoord;
const double Tolerances<arma::fmat>::LRTrainAcc;
const double Tolerances<arma::fmat>::LRTestAcc;

#if defined(ARMA_HAVE_FP16)
const arma::fp16 Tolerances<arma::mat>::Obj;
const arma::fp16 Tolerances<arma::mat>::Coord;
const arma::fp16 Tolerances<arma::mat>::LargeObj;
const arma::fp16 Tolerances<arma::mat>::LargeCoord;
const double     Tolerances<arma::mat>::LRTrainAcc;
const double     Tolerances<arma::mat>::LRTestAcc;
#endif

const double Tolerances<arma::sp_mat>::Obj;
const double Tolerances<arma::sp_mat>::Coord;
const double Tolerances<arma::sp_mat>::LargeObj;
const double Tolerances<arma::sp_mat>::LargeCoord;
const double Tolerances<arma::sp_mat>::LRTrainAcc;
const double Tolerances<arma::sp_mat>::LRTestAcc;

#if defined(ENS_HAVE_COOT)
const double Tolerances<coot::mat>::Obj;
const double Tolerances<coot::mat>::Coord;
const double Tolerances<coot::mat>::LargeObj;
const double Tolerances<coot::mat>::LargeCoord;
const double Tolerances<coot::mat>::LRTrainAcc;
const double Tolerances<coot::mat>::LRTestAcc;

const float  Tolerances<coot::fmat>::Obj = 1e-4;
const float  Tolerances<coot::fmat>::Coord = 1e-2;
const float  Tolerances<coot::fmat>::LargeObj = 2e-3;
const float  Tolerances<coot::fmat>::LargeCoord = 2e-2;
const double Tolerances<coot::fmat>::LRTrainAcc = 0.003;
const double Tolerances<coot::fmat>::LRTestAcc = 0.006;
#endif

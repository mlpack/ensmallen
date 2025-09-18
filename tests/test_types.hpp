/**
 * @file test_function_tools.hpp
 * @author Marcus Edel
 * @author Ryan Curtin
 * @author Conrad Sanderson
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_TESTS_TEST_TYPES_HPP
#define ENSMALLEN_TESTS_TEST_TYPES_HPP

#if defined(ENS_HAVE_COOT)
#define ENS_GPU_TEST_TYPES coot::mat, coot::fmat,
#else
#define ENS_GPU_TEST_TYPES
#endif

#if defined(ARMA_HAVE_FP16)
#define ENS_LOWPREC_TEST_TYPES arma::hmat,
#else
#define ENS_LOWPREC_TEST_TYPES
#endif

#define ENS_SPARSE_TEST_TYPES arma::sp_mat

#define ENS_TEST_TYPES arma::mat, arma::fmat

#define ENS_FULLPREC_CPU_TEST_TYPES ENS_TEST_TYPES
#define ENS_FULLPREC_TEST_TYPES ENS_GPU_TEST_TYPES ENS_TEST_TYPES

#define ENS_ALL_CPU_TEST_TYPES ENS_LOWPREC_TEST_TYPES ENS_TEST_TYPES
#define ENS_ALL_TEST_TYPES ENS_GPU_TEST_TYPES ENS_ALL_CPU_TEST_TYPES

namespace ens {
namespace test {

template<typename MatType>
struct Tolerances { };

// With C++17 we have inline static member initialization.  But ensmallen uses
// C++14, so we are stuck declaring the variables here, and defining them in
// test_types.cpp.

template<>
struct Tolerances<arma::mat>
{
  static const double Obj;
  static const double Coord;

  static const double LargeObj;
  static const double LargeCoord;

  // Tolerances for LogisticRegressionFunctionTest().
  static const double LRTrainAcc;
  static const double LRTestAcc;
};

template<>
struct Tolerances<arma::fmat>
{
  static const float Obj;
  static const float Coord;

  static const float LargeObj;
  static const float LargeCoord;

  // Tolerances for LogisticRegressionFunctionTest().
  static const double LRTrainAcc;
  static const double LRTestAcc;
};

#if defined(ARMA_HAVE_FP16)
template<>
struct Tolerances<arma::hmat>
{
  static const arma::fp16 Obj;
  static const arma::fp16 Coord;

  static const arma::fp16 LargeObj;
  static const arma::fp16 LargeCoord;

  // Tolerances for LogisticRegressionFunctionTest().
  static const double LRTrainAcc;
  static const double LRTestAcc;
};
#endif

template<>
struct Tolerances<arma::sp_mat>
{
  static const double Obj;
  static const double Coord;

  static const double LargeObj;
  static const double LargeCoord;

  // Tolerances for LogisticRegressionFunctionTest().
  static const double LRTrainAcc;
  static const double LRTestAcc;
};

#if defined(ENS_HAVE_COOT)
template<>
struct Tolerances<coot::mat>
{
  static const double Obj;
  static const double Coord;

  static const double LargeObj;
  static const double LargeCoord;

  // Tolerances for LogisticRegressionFunctionTest().
  static const double LRTrainAcc;
  static const double LRTestAcc;
};

template<>
struct Tolerances<coot::fmat>
{
  static const float Obj;
  static const float Coord;

  static const float LargeObj;
  static const float LargeCoord;

  // Tolerances for LogisticRegressionFunctionTest().
  static const double LRTrainAcc;
  static const double LRTestAcc;
};
#endif

} // namespace test
} // namespace ens

#endif

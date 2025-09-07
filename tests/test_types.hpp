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
#define ENS_GPU_TEST_TYPES coot::mat, coot:fmat,
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

template<>
struct Tolerances<arma::mat>
{
  constexpr static const double Obj = 1e-8;
  constexpr static const double Coord = 1e-4;

  constexpr static const double LargeObj = 1e-3;
  constexpr static const double LargeCoord = 1e-2;

  // Tolerances for LogisticRegressionFunctionTest().
  constexpr static const double LRTrainAcc = 0.003;
  constexpr static const double LRTestAcc = 0.006;
};

template<>
struct Tolerances<arma::fmat>
{
  constexpr static const float Obj = 1e-4;
  constexpr static const float Coord = 1e-2;

  constexpr static const float LargeObj = 2e-3;
  constexpr static const float LargeCoord = 2e-2;

  // Tolerances for LogisticRegressionFunctionTest().
  constexpr static const double LRTrainAcc = 0.003;
  constexpr static const double LRTestAcc = 0.006;
};

#if defined(ARMA_HAVE_FP16)
template<>
struct Tolerances<arma::hmat>
{
  constexpr static const arma::fp16 Obj = arma::fp16(0.0001);
  constexpr static const arma::fp16 Coord = arma::fp16(0.01);

  constexpr static const arma::fp16 LargeObj = arma::fp16(0.03);
  constexpr static const arma::fp16 LargeCoord = arma::fp16(0.1);

  // Tolerances for LogisticRegressionFunctionTest().
  constexpr static const double LRTrainAcc = 0.03;
  constexpr static const double LRTestAcc = 0.06;
};
#endif

template<>
struct Tolerances<arma::sp_mat>
{
  constexpr static const double Obj = 1e-8;
  constexpr static const double Coord = 1e-4;

  constexpr static const double LargeObj = 1e-3;
  constexpr static const double LargeCoord = 1e-2;

  // Tolerances for LogisticRegressionFunctionTest().
  constexpr static const double LRTrainAcc = 0.003;
  constexpr static const double LRTestAcc = 0.006;
};

#if defined(ENS_HAVE_COOT)
template<>
struct Tolerances<coot::mat>
{
  constexpr static const double Obj = 1e-8;
  constexpr static const double Coord = 1e-4;

  constexpr static const double LargeObj = 1e-3;
  constexpr static const double LargeCoord = 1e-2;

  // Tolerances for LogisticRegressionFunctionTest().
  constexpr static const double LRTrainAcc = 0.003;
  constexpr static const double LRTestAcc = 0.006;
};

template<>
struct Tolerances<coot::fmat>
{
  constexpr static const float Obj = 1e-4;
  constexpr static const float Coord = 1e-2;

  constexpr static const float LargeObj = 2e-3;
  constexpr static const float LargeCoord = 2e-2;

  // Tolerances for LogisticRegressionFunctionTest().
  constexpr static const double LRTrainAcc = 0.003;
  constexpr static const double LRTestAcc = 0.006;
};
#endif

} // namespace test
} // namespace ens

#endif

// Copyright (c) 2018 ensmallen developers.
//
// Licensed under the 3-clause BSD license (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.opensource.org/licenses/BSD-3-Clause

#include <ensmallen.hpp>
#include "catch.hpp"

using namespace ens;
using namespace ens::test;
using namespace ens::traits;

/**
 * Utility class with no functions.
 */
class EmptyTestFunction { };

/**
 * Utility class with Evaluate() but no Evaluate().
 */
class EvaluateTestFunction
{
 public:
  double Evaluate(const arma::mat& coordinates)
  {
    return arma::accu(coordinates);
  }

  double Evaluate(const arma::mat& coordinates,
                  const size_t begin,
                  const size_t batchSize)
  {
    return arma::accu(coordinates) + begin + batchSize;
  }
};

/**
 * Utility class with Gradient() but no Evaluate().
 */
class GradientTestFunction
{
 public:
  void Gradient(const arma::mat& coordinates, arma::mat& gradient)
  {
    gradient.ones(coordinates.n_rows, coordinates.n_cols);
  }

  void Gradient(const arma::mat& coordinates,
                const size_t /* begin */,
                arma::mat& gradient,
                const size_t /* batchSize */)
  {
    gradient.ones(coordinates.n_rows, coordinates.n_cols);
  }
};

/**
 * Utility class with Gradient() and Evaluate().
 */
class EvaluateGradientTestFunction
{
 public:
  double Evaluate(const arma::mat& coordinates)
  {
    return arma::accu(coordinates);
  }

  double Evaluate(const arma::mat& coordinates,
                  const size_t /* begin */,
                  const size_t /* batchSize */)
  {
    return arma::accu(coordinates);
  }

  void Gradient(const arma::mat& coordinates, arma::mat& gradient)
  {
    gradient.ones(coordinates.n_rows, coordinates.n_cols);
  }

  void Gradient(const arma::mat& coordinates,
                const size_t /* begin */,
                arma::mat& gradient,
                const size_t /* batchSize */)
  {
    gradient.ones(coordinates.n_rows, coordinates.n_cols);
  }
};

/**
 * Utility class with EvaluateWithGradient().
 */
class EvaluateWithGradientTestFunction
{
 public:
  double EvaluateWithGradient(const arma::mat& coordinates, arma::mat& gradient)
  {
    gradient.ones(coordinates.n_rows, coordinates.n_cols);
    return arma::accu(coordinates);
  }

  double EvaluateWithGradient(const arma::mat& coordinates,
                              const size_t /* begin */,
                              arma::mat& gradient,
                              const size_t /* batchSize */)
  {
    gradient.ones(coordinates.n_rows, coordinates.n_cols);
    return arma::accu(coordinates);
  }
};

/**
 * Utility class with all three functions.
 */
class EvaluateAndWithGradientTestFunction
{
 public:
  double Evaluate(const arma::mat& coordinates)
  {
    return arma::accu(coordinates);
  }

  double Evaluate(const arma::mat& coordinates,
                  const size_t begin,
                  const size_t batchSize)
  {
    return arma::accu(coordinates) + batchSize + begin;
  }

  void Gradient(const arma::mat& coordinates, arma::mat& gradient)
  {
    gradient.ones(coordinates.n_rows, coordinates.n_cols);
  }

  void Gradient(const arma::mat& coordinates,
                const size_t /* begin */,
                arma::mat& gradient,
                const size_t /* batchSize */)
  {
    gradient.ones(coordinates.n_rows, coordinates.n_cols);
  }

  double EvaluateWithGradient(const arma::mat& coordinates, arma::mat& gradient)
  {
    gradient.ones(coordinates.n_rows, coordinates.n_cols);
    return arma::accu(coordinates);
  }

  double EvaluateWithGradient(const arma::mat& coordinates,
                              const size_t /* begin */,
                              arma::mat& gradient,
                              const size_t /* batchSize */)
  {
    gradient.ones(coordinates.n_rows, coordinates.n_cols);
    return arma::accu(coordinates);
  }
};

/**
 * Utility class with const Evaluate() and non-const Gradient().
 */
class EvaluateAndNonConstGradientTestFunction
{
 public:
  double Evaluate(const arma::mat& coordinates) const
  {
    return arma::accu(coordinates);
  }

  void Gradient(const arma::mat& coordinates, arma::mat& gradient)
  {
    gradient.ones(coordinates.n_rows, coordinates.n_cols);
  }
};

/**
 * Utility class with const Evaluate() and non-const Gradient().
 */
class EvaluateAndStaticGradientTestFunction
{
 public:
  double Evaluate(const arma::mat& coordinates) const
  {
    return arma::accu(coordinates);
  }

  static void Gradient(const arma::mat& coordinates, arma::mat& gradient)
  {
    gradient.ones(coordinates.n_rows, coordinates.n_cols);
  }
};

/**
 * Make sure that an empty class doesn't have any methods added to it.
 */
TEST_CASE("AddEvaluateWithGradientEmptyTest", "[FunctionTest]")
{
  const bool hasEvaluate = HasEvaluate<Function<EmptyTestFunction>,
                                       EvaluateForm>::value;
  const bool hasGradient = HasGradient<Function<EmptyTestFunction>,
                                       GradientForm>::value;
  const bool hasEvaluateWithGradient =
      HasEvaluateWithGradient<Function<EmptyTestFunction>,
                              EvaluateWithGradientForm>::value;

  REQUIRE(hasEvaluate == false);
  REQUIRE(hasGradient == false);
  REQUIRE(hasEvaluateWithGradient == false);
}

/**
 * Make sure we don't add any functions if we only have Evaluate().
 */
TEST_CASE("AddEvaluateWithGradientEvaluateOnlyTest", "[FunctionTest]")
{
  const bool hasEvaluate = HasEvaluate<Function<EvaluateTestFunction>,
                                       EvaluateForm>::value;
  const bool hasGradient = HasGradient<Function<EvaluateTestFunction>,
                                       GradientForm>::value;
  const bool hasEvaluateWithGradient =
      HasEvaluateWithGradient<Function<EvaluateTestFunction>,
                              EvaluateWithGradientForm>::value;

  REQUIRE(hasEvaluate == true);
  REQUIRE(hasGradient == false);
  REQUIRE(hasEvaluateWithGradient == false);
}

/**
 * Make sure we don't add any functions if we only have Gradient().
 */
TEST_CASE("AddEvaluateWithGradientGradientOnlyTest", "[FunctionTest]")
{
  const bool hasEvaluate = HasEvaluate<Function<GradientTestFunction>,
                                       EvaluateForm>::value;
  const bool hasGradient = HasGradient<Function<GradientTestFunction>,
                                       GradientForm>::value;
  const bool hasEvaluateWithGradient =
      HasEvaluateWithGradient<Function<GradientTestFunction>,
                              EvaluateWithGradientForm>::value;

  REQUIRE(hasEvaluate == false);
  REQUIRE(hasGradient == true);
  REQUIRE(hasEvaluateWithGradient == false);
}

/**
 * Make sure we add EvaluateWithGradient() when we have both Evaluate() and
 * Gradient().
 */
TEST_CASE("AddEvaluateWithGradientBothTest", "[FunctionTest]")
{
  const bool hasEvaluate =
      HasEvaluate<Function<EvaluateGradientTestFunction>,
                           EvaluateForm>::value;
  const bool hasGradient =
      HasGradient<Function<EvaluateGradientTestFunction>,
                           GradientForm>::value;
  const bool hasEvaluateWithGradient =
      HasEvaluateWithGradient<Function<EvaluateGradientTestFunction>,
                              EvaluateWithGradientForm>::value;

  REQUIRE(hasEvaluate == true);
  REQUIRE(hasGradient == true);
  REQUIRE(hasEvaluateWithGradient == true);
}

/**
 * Make sure we add Evaluate() and Gradient() when we have only
 * EvaluateWithGradient().
 */
TEST_CASE("AddEvaluateWithGradientEvaluateWithGradientTest", "[FunctionTest]")
{
  const bool hasEvaluate =
      HasEvaluate<Function<EvaluateWithGradientTestFunction>,
                           EvaluateForm>::value;
  const bool hasGradient =
      HasGradient<Function<EvaluateWithGradientTestFunction>,
                           GradientForm>::value;
  const bool hasEvaluateWithGradient =
      HasEvaluateWithGradient<Function<EvaluateWithGradientTestFunction>,
                              EvaluateWithGradientForm>::value;

  REQUIRE(hasEvaluate == true);
  REQUIRE(hasGradient == true);
  REQUIRE(hasEvaluateWithGradient == true);
}

/**
 * Make sure we add no methods when we already have all three.
 */
TEST_CASE("AddEvaluateWithGradientAllThreeTest", "[FunctionTest]")
{
  const bool hasEvaluate =
      HasEvaluate<Function<EvaluateAndWithGradientTestFunction>,
                           EvaluateForm>::value;
  const bool hasGradient =
      HasGradient<Function<EvaluateAndWithGradientTestFunction>,
                           GradientForm>::value;
  const bool hasEvaluateWithGradient =
      HasEvaluateWithGradient<Function<EvaluateAndWithGradientTestFunction>,
                              EvaluateWithGradientForm>::value;

  REQUIRE(hasEvaluate == true);
  REQUIRE(hasGradient == true);
  REQUIRE(hasEvaluateWithGradient == true);
}

TEST_CASE("LogisticRegressionEvaluateWithGradientTest", "[FunctionTest]")
{
  const bool hasEvaluate =
      HasEvaluate<Function<LogisticRegressionFunction<>>,
                           EvaluateConstForm>::value;
  const bool hasGradient =
      HasGradient<Function<LogisticRegressionFunction<>>,
                           GradientConstForm>::value;
  const bool hasEvaluateWithGradient =
      HasEvaluateWithGradient<Function<LogisticRegressionFunction<>>,
                              EvaluateWithGradientConstForm>::value;

  REQUIRE(hasEvaluate == true);
  REQUIRE(hasGradient == true);
  REQUIRE(hasEvaluateWithGradient == true);
}

TEST_CASE("SDPTest", "[FunctionTest]")
{
  typedef AugLagrangianFunction<LRSDPFunction<SDP<arma::mat>>> FunctionType;

  const bool hasEvaluate =
      HasEvaluate<Function<FunctionType>, EvaluateConstForm>::value;
  const bool hasGradient =
      HasGradient<Function<FunctionType>, GradientConstForm>::value;
  const bool hasEvaluateWithGradient =
      HasEvaluateWithGradient<Function<FunctionType>,
                              EvaluateWithGradientConstForm>::value;

  REQUIRE(hasEvaluate == true);
  REQUIRE(hasGradient == true);
  REQUIRE(hasEvaluateWithGradient == true);
}

/**
 * Make sure that an empty class doesn't have any methods added to it.
 */
TEST_CASE("AddDecomposableEvaluateWithGradientEmptyTest", "[FunctionTest]")
{
  const bool hasEvaluate = HasEvaluate<Function<EmptyTestFunction>,
                                       DecomposableEvaluateForm>::value;
  const bool hasGradient = HasGradient<Function<EmptyTestFunction>,
                                       DecomposableGradientForm>::value;
  const bool hasEvaluateWithGradient =
      HasEvaluateWithGradient<Function<EmptyTestFunction>,
                              DecomposableEvaluateWithGradientForm>::value;

  REQUIRE(hasEvaluate == false);
  REQUIRE(hasGradient == false);
  REQUIRE(hasEvaluateWithGradient == false);
}

/**
 * Make sure we don't add any functions if we only have Evaluate().
 */
TEST_CASE("AddDecomposableEvaluateWithGradientEvaluateOnlyTest", "[FunctionTest]")
{
  const bool hasEvaluate = HasEvaluate<Function<EvaluateTestFunction>,
                                       DecomposableEvaluateForm>::value;
  const bool hasGradient = HasGradient<Function<EvaluateTestFunction>,
                                       DecomposableGradientForm>::value;
  const bool hasEvaluateWithGradient =
      HasEvaluateWithGradient<Function<EvaluateTestFunction>,
                              DecomposableEvaluateWithGradientForm>::value;

  REQUIRE(hasEvaluate == true);
  REQUIRE(hasGradient == false);
  REQUIRE(hasEvaluateWithGradient == false);
}

/**
 * Make sure we don't add any functions if we only have Gradient().
 */
TEST_CASE("AddDecomposableEvaluateWithGradientGradientOnlyTest", "[FunctionTest]")
{
  const bool hasEvaluate = HasEvaluate<Function<GradientTestFunction>,
                                       DecomposableEvaluateForm>::value;
  const bool hasGradient = HasGradient<Function<GradientTestFunction>,
                                       DecomposableGradientForm>::value;
  const bool hasEvaluateWithGradient =
      HasEvaluateWithGradient<Function<GradientTestFunction>,
                              DecomposableEvaluateWithGradientForm>::value;

  REQUIRE(hasEvaluate == false);
  REQUIRE(hasGradient == true);
  REQUIRE(hasEvaluateWithGradient == false);
}

/**
 * Make sure we add EvaluateWithGradient() when we have both Evaluate() and
 * Gradient().
 */
TEST_CASE("AddDecomposableEvaluateWithGradientBothTest", "[FunctionTest]")
{
  const bool hasEvaluate =
      HasEvaluate<Function<EvaluateGradientTestFunction>,
                           DecomposableEvaluateForm>::value;
  const bool hasGradient =
      HasGradient<Function<EvaluateGradientTestFunction>,
                           DecomposableGradientForm>::value;
  const bool hasEvaluateWithGradient =
      HasEvaluateWithGradient<Function<EvaluateGradientTestFunction>,
                              DecomposableEvaluateWithGradientForm>::value;

  REQUIRE(hasEvaluate == true);
  REQUIRE(hasGradient == true);
  REQUIRE(hasEvaluateWithGradient == true);
}

/**
 * Make sure we add Evaluate() and Gradient() when we have only
 * EvaluateWithGradient().
 */
TEST_CASE("AddDecomposableEvaluateWGradientEvaluateWithGradientTest", "[FunctionTest]")
{
  const bool hasEvaluate =
      HasEvaluate<Function<EvaluateWithGradientTestFunction>,
                           DecomposableEvaluateForm>::value;
  const bool hasGradient =
      HasGradient<Function<EvaluateWithGradientTestFunction>,
                           DecomposableGradientForm>::value;
  const bool hasEvaluateWithGradient =
      HasEvaluateWithGradient<Function<EvaluateWithGradientTestFunction>,
                              DecomposableEvaluateWithGradientForm>::value;

  Function<EvaluateWithGradientTestFunction> f;
  arma::mat coordinates(10, 10, arma::fill::ones);
  arma::mat gradient;
  f.Gradient(coordinates, 0, gradient, 5);

  REQUIRE(hasEvaluate == true);
  REQUIRE(hasGradient == true);
  REQUIRE(hasEvaluateWithGradient == true);
}

/**
 * Make sure we add no methods when we already have all three.
 */
TEST_CASE("AddDecomposableEvaluateWithGradientAllThreeTest", "[FunctionTest]")
{
  const bool hasEvaluate =
      HasEvaluate<Function<EvaluateAndWithGradientTestFunction>,
                  DecomposableEvaluateForm>::value;
  const bool hasGradient =
      HasGradient<Function<EvaluateAndWithGradientTestFunction>,
                           DecomposableGradientForm>::value;
  const bool hasEvaluateWithGradient =
      HasEvaluateWithGradient<Function<EvaluateAndWithGradientTestFunction>,
                              DecomposableEvaluateWithGradientForm>::value;

  REQUIRE(hasEvaluate == true);
  REQUIRE(hasGradient == true);
  REQUIRE(hasEvaluateWithGradient == true);
}

/**
 * Make sure we can properly create EvaluateWithGradient() even when one of the
 * functions is non-const.
 */
TEST_CASE("AddEvaluateWithGradientMixedTypesTest", "[FunctionTest]")
{
  const bool hasEvaluate =
      HasEvaluate<Function<EvaluateAndNonConstGradientTestFunction>,
                  EvaluateConstForm>::value;
  const bool hasGradient =
      HasGradient<Function<EvaluateAndNonConstGradientTestFunction>,
                  GradientForm>::value;
  const bool hasEvaluateWithGradient =
      HasEvaluateWithGradient<Function<EvaluateAndNonConstGradientTestFunction>,
                              EvaluateWithGradientForm>::value;

  REQUIRE(hasEvaluate == true);
  REQUIRE(hasGradient == true);
  REQUIRE(hasEvaluateWithGradient == true);
}

/**
 * Make sure we can properly create EvaluateWithGradient() even when one of the
 * functions is static.
 */
TEST_CASE("AddEvaluateWithGradientMixedTypesStaticTest", "[FunctionTest]")
{
  const bool hasEvaluate =
      HasEvaluate<Function<EvaluateAndStaticGradientTestFunction>,
                  EvaluateConstForm>::value;
  const bool hasGradient =
      HasGradient<Function<EvaluateAndStaticGradientTestFunction>,
                  GradientStaticForm>::value;
  const bool hasEvaluateWithGradient =
      HasEvaluateWithGradient<Function<EvaluateAndStaticGradientTestFunction>,
                              EvaluateWithGradientConstForm>::value;

  REQUIRE(hasEvaluate == true);
  REQUIRE(hasGradient == true);
  REQUIRE(hasEvaluateWithGradient == true);
}

class A
{
 public:
  size_t NumFunctions() const;
  size_t NumFeatures() const;
  double Evaluate(const arma::mat&, const size_t, const size_t) const;
  void Gradient(const arma::mat&, const size_t, arma::mat&, const size_t) const;
  void Gradient(const arma::mat&, const size_t, arma::sp_mat&, const size_t)
      const;
  void PartialGradient(const arma::mat&, const size_t, arma::sp_mat&) const;
};

class B
{
 public:
  size_t NumFunctions();
  size_t NumFeatures();
  double Evaluate(const arma::mat&, const size_t, const size_t);
  void Gradient(const arma::mat&, const size_t, arma::mat&, const size_t);
  void Gradient(const arma::mat&, const size_t, arma::sp_mat&, const size_t);
  void PartialGradient(const arma::mat&, const size_t, arma::sp_mat&);
};

class C
{
 public:
  size_t NumConstraints() const;
  double Evaluate(const arma::mat&) const;
  void Gradient(const arma::mat&, arma::mat&) const;
  double EvaluateConstraint(const size_t, const arma::mat&) const;
  void GradientConstraint(const size_t, const arma::mat&, arma::mat&) const;
};

class D
{
 public:
  size_t NumConstraints();
  double Evaluate(const arma::mat&);
  void Gradient(const arma::mat&, arma::mat&);
  double EvaluateConstraint(const size_t, const arma::mat&);
  void GradientConstraint(const size_t, const arma::mat&, arma::mat&);
};


/**
 * Test the correctness of the static check for DecomposableFunctionType API.
 */
TEST_CASE("DecomposableFunctionTypeCheckTest", "[FunctionTest]")
{
  static_assert(CheckNumFunctions<A>::value,
      "CheckNumFunctions static check failed.");
  static_assert(CheckNumFunctions<B>::value,
      "CheckNumFunctions static check failed.");
  static_assert(!CheckNumFunctions<C>::value,
      "CheckNumFunctions static check failed.");
  static_assert(!CheckNumFunctions<D>::value,
      "CheckNumFunctions static check failed.");

  static_assert(CheckDecomposableEvaluate<A>::value,
      "CheckDecomposableEvaluate static check failed.");
  static_assert(CheckDecomposableEvaluate<B>::value,
      "CheckDecomposableEvaluate static check failed.");
  static_assert(!CheckDecomposableEvaluate<C>::value,
      "CheckDecomposableEvaluate static check failed.");
  static_assert(!CheckDecomposableEvaluate<D>::value,
      "CheckDecomposableEvaluate static check failed.");

  static_assert(CheckDecomposableGradient<A>::value,
      "CheckDecomposableGradient static check failed.");
  static_assert(CheckDecomposableGradient<B>::value,
      "CheckDecomposableGradient static check failed.");
  static_assert(!CheckDecomposableGradient<C>::value,
      "CheckDecomposableGradient static check failed.");
  static_assert(!CheckDecomposableGradient<D>::value,
      "CheckDecomposableGradient static check failed.");
}

/**
 * Test the correctness of the static check for LagrangianFunctionType API.
 */
TEST_CASE("LagrangianFunctionTypeCheckTest", "[FunctionTest]")
{
  static_assert(!CheckEvaluate<A>::value, "CheckEvaluate static check failed.");
  static_assert(!CheckEvaluate<B>::value, "CheckEvaluate static check failed.");
  static_assert(CheckEvaluate<C>::value, "CheckEvaluate static check failed.");
  static_assert(CheckEvaluate<D>::value, "CheckEvaluate static check failed.");

  static_assert(!CheckGradient<A>::value, "CheckGradient static check failed.");
  static_assert(!CheckGradient<B>::value, "CheckGradient static check failed.");
  static_assert(CheckGradient<C>::value, "CheckGradient static check failed.");
  static_assert(CheckGradient<D>::value, "CheckGradient static check failed.");

  static_assert(!CheckNumConstraints<A>::value,
      "CheckNumConstraints static check failed.");
  static_assert(!CheckNumConstraints<B>::value,
      "CheckNumConstraints static check failed.");
  static_assert(CheckNumConstraints<C>::value,
      "CheckNumConstraints static check failed.");
  static_assert(CheckNumConstraints<D>::value,
      "CheckNumConstraints static check failed.");

  static_assert(!CheckEvaluateConstraint<A>::value,
      "CheckEvaluateConstraint static check failed.");
  static_assert(!CheckEvaluateConstraint<B>::value,
      "CheckEvaluateConstraint static check failed.");
  static_assert(CheckEvaluateConstraint<C>::value,
      "CheckEvaluateConstraint static check failed.");
  static_assert(CheckEvaluateConstraint<D>::value,
      "CheckEvaluateConstraint static check failed.");

  static_assert(!CheckGradientConstraint<A>::value,
      "CheckGradientConstraint static check failed.");
  static_assert(!CheckGradientConstraint<B>::value,
      "CheckGradientConstraint static check failed.");
  static_assert(CheckGradientConstraint<C>::value,
      "CheckGradientConstraint static check failed.");
  static_assert(CheckGradientConstraint<D>::value,
      "CheckGradientConstraint static check failed.");
}

/**
 * Test the correctness of the static check for SparseFunctionType API.
 */
TEST_CASE("SparseFunctionTypeCheckTest", "[FunctionTest]")
{
  static_assert(CheckSparseGradient<A>::value,
      "CheckSparseGradient static check failed.");
  static_assert(CheckSparseGradient<B>::value,
      "CheckSparseGradient static check failed.");
  static_assert(!CheckSparseGradient<C>::value,
      "CheckSparseGradient static check failed.");
  static_assert(!CheckSparseGradient<D>::value,
      "CheckSparseGradient static check failed.");
}

/**
 * Test the correctness of the static check for SparseFunctionType API.
 */
TEST_CASE("ResolvableFunctionTypeCheckTest", "[FunctionTest]")
{
  static_assert(CheckNumFeatures<A>::value,
      "CheckNumFeatures static check failed.");
  static_assert(CheckNumFeatures<B>::value,
      "CheckNumFeatures static check failed.");
  static_assert(!CheckNumFeatures<C>::value,
      "CheckNumFeatures static check failed.");
  static_assert(!CheckNumFeatures<D>::value,
      "CheckNumFeatures static check failed.");

  static_assert(CheckPartialGradient<A>::value,
      "CheckPartialGradient static check failed.");
  static_assert(CheckPartialGradient<B>::value,
      "CheckPartialGradient static check failed.");
  static_assert(!CheckPartialGradient<C>::value,
      "CheckPartialGradient static check failed.");
  static_assert(!CheckPartialGradient<D>::value,
      "CheckPartialGradient static check failed.");
}

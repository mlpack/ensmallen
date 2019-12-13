/**
 * @file function_test.cpp
 * @author Ryan Curtin
 * @author Shikhar Bhardwaj
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
  const bool hasEvaluate = HasEvaluate<
      Function<EmptyTestFunction, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template EvaluateForm>::value;
  const bool hasGradient = HasGradient<
      Function<EmptyTestFunction, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template GradientForm>::value;
  const bool hasEvaluateWithGradient = HasEvaluateWithGradient<
      Function<EmptyTestFunction, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template
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
  const bool hasEvaluate = HasEvaluate<
      Function<EvaluateTestFunction, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template EvaluateForm>::value;
  const bool hasGradient = HasGradient<
      Function<EvaluateTestFunction, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template GradientForm>::value;
  const bool hasEvaluateWithGradient = HasEvaluateWithGradient<
      Function<EvaluateTestFunction, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template
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
  const bool hasEvaluate = HasEvaluate<
      Function<GradientTestFunction, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template EvaluateForm>::value;
  const bool hasGradient = HasGradient<
      Function<GradientTestFunction, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template GradientForm>::value;
  const bool hasEvaluateWithGradient = HasEvaluateWithGradient<
      Function<GradientTestFunction, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template
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
  const bool hasEvaluate = HasEvaluate<
      Function<EvaluateGradientTestFunction, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template EvaluateForm>::value;
  const bool hasGradient = HasGradient<
      Function<EvaluateGradientTestFunction, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template GradientForm>::value;
  const bool hasEvaluateWithGradient = HasEvaluateWithGradient<
      Function<EvaluateGradientTestFunction, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template
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
  const bool hasEvaluate = HasEvaluate<
      Function<EvaluateWithGradientTestFunction, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template EvaluateForm>::value;
  const bool hasGradient = HasGradient<
      Function<EvaluateWithGradientTestFunction, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template GradientForm>::value;
  const bool hasEvaluateWithGradient = HasEvaluateWithGradient<
      Function<EvaluateWithGradientTestFunction, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template
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
  const bool hasEvaluate = HasEvaluate<
      Function<EvaluateAndWithGradientTestFunction, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template EvaluateForm>::value;
  const bool hasGradient = HasGradient<
      Function<EvaluateAndWithGradientTestFunction, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template GradientForm>::value;
  const bool hasEvaluateWithGradient = HasEvaluateWithGradient<
      Function<EvaluateAndWithGradientTestFunction, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template
          EvaluateWithGradientForm>::value;

  REQUIRE(hasEvaluate == true);
  REQUIRE(hasGradient == true);
  REQUIRE(hasEvaluateWithGradient == true);
}

TEST_CASE("LogisticRegressionEvaluateWithGradientTest", "[FunctionTest]")
{
  const bool hasEvaluate = HasEvaluate<
      Function<LogisticRegressionFunction<>, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template EvaluateConstForm>::value;
  const bool hasGradient = HasGradient<
      Function<LogisticRegressionFunction<>, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template GradientConstForm>::value;
  const bool hasEvaluateWithGradient = HasEvaluateWithGradient<
      Function<LogisticRegressionFunction<>, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template
          EvaluateWithGradientConstForm>::value;

  REQUIRE(hasEvaluate == true);
  REQUIRE(hasGradient == true);
  REQUIRE(hasEvaluateWithGradient == true);
}

TEST_CASE("SDPTest", "[FunctionTest]")
{
  typedef AugLagrangianFunction<LRSDPFunction<SDP<arma::mat>>> FunctionType;

  const bool hasEvaluate = HasEvaluate<
      Function<FunctionType, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template EvaluateConstForm>::value;
  const bool hasGradient = HasGradient<
      Function<FunctionType, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template GradientConstForm>::value;
  const bool hasEvaluateWithGradient = HasEvaluateWithGradient<
      Function<FunctionType, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template
          EvaluateWithGradientConstForm>::value;

  REQUIRE(hasEvaluate == true);
  REQUIRE(hasGradient == true);
  REQUIRE(hasEvaluateWithGradient == true);
}

/**
 * Make sure that an empty class doesn't have any methods added to it.
 */
TEST_CASE("AddSeparableEvaluateWithGradientEmptyTest", "[FunctionTest]")
{
  const bool hasEvaluate = HasEvaluate<
      Function<EmptyTestFunction, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template
          SeparableEvaluateForm>::value;
  const bool hasGradient = HasGradient<
      Function<EmptyTestFunction, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template
          SeparableGradientForm>::value;
  const bool hasEvaluateWithGradient = HasEvaluateWithGradient<
      Function<EmptyTestFunction, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template
          SeparableEvaluateWithGradientForm>::value;

  REQUIRE(hasEvaluate == false);
  REQUIRE(hasGradient == false);
  REQUIRE(hasEvaluateWithGradient == false);
}

/**
 * Make sure we don't add any functions if we only have Evaluate().
 */
TEST_CASE("AddSeparableEvaluateWithGradientEvaluateOnlyTest",
          "[FunctionTest]")
{
  const bool hasEvaluate = HasEvaluate<
      Function<EvaluateTestFunction, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template
          SeparableEvaluateForm>::value;
  const bool hasGradient = HasGradient<
      Function<EvaluateTestFunction, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template
          SeparableGradientForm>::value;
  const bool hasEvaluateWithGradient = HasEvaluateWithGradient<
      Function<EvaluateTestFunction, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template
          SeparableEvaluateWithGradientForm>::value;

  REQUIRE(hasEvaluate == true);
  REQUIRE(hasGradient == false);
  REQUIRE(hasEvaluateWithGradient == false);
}

/**
 * Make sure we don't add any functions if we only have Gradient().
 */
TEST_CASE("AddSeparableEvaluateWithGradientGradientOnlyTest",
          "[FunctionTest]")
{
  const bool hasEvaluate = HasEvaluate<
      Function<GradientTestFunction, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template
          SeparableEvaluateForm>::value;
  const bool hasGradient = HasGradient<
      Function<GradientTestFunction, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template
          SeparableGradientForm>::value;
  const bool hasEvaluateWithGradient = HasEvaluateWithGradient<
      Function<GradientTestFunction, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template
          SeparableEvaluateWithGradientForm>::value;

  REQUIRE(hasEvaluate == false);
  REQUIRE(hasGradient == true);
  REQUIRE(hasEvaluateWithGradient == false);
}

/**
 * Make sure we add EvaluateWithGradient() when we have both Evaluate() and
 * Gradient().
 */
TEST_CASE("AddSeparableEvaluateWithGradientBothTest", "[FunctionTest]")
{
  const bool hasEvaluate = HasEvaluate<
      Function<EvaluateGradientTestFunction, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template
          SeparableEvaluateForm>::value;
  const bool hasGradient = HasGradient<
      Function<EvaluateGradientTestFunction, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template
          SeparableGradientForm>::value;
  const bool hasEvaluateWithGradient = HasEvaluateWithGradient<
      Function<EvaluateGradientTestFunction, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template
          SeparableEvaluateWithGradientForm>::value;

  REQUIRE(hasEvaluate == true);
  REQUIRE(hasGradient == true);
  REQUIRE(hasEvaluateWithGradient == true);
}

/**
 * Make sure we add Evaluate() and Gradient() when we have only
 * EvaluateWithGradient().
 */
TEST_CASE("AddSeparableEvaluateWGradientEvaluateWithGradientTest",
          "[FunctionTest]")
{
  const bool hasEvaluate = HasEvaluate<
      Function<EvaluateWithGradientTestFunction, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template
          SeparableEvaluateForm>::value;
  const bool hasGradient = HasGradient<
      Function<EvaluateWithGradientTestFunction, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template
          SeparableGradientForm>::value;
  const bool hasEvaluateWithGradient = HasEvaluateWithGradient<
      Function<EvaluateWithGradientTestFunction, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template
          SeparableEvaluateWithGradientForm>::value;

  Function<EvaluateWithGradientTestFunction, arma::mat, arma::mat> f;
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
TEST_CASE("AddSeparableEvaluateWithGradientAllThreeTest", "[FunctionTest]")
{
  const bool hasEvaluate = HasEvaluate<
      Function<EvaluateAndWithGradientTestFunction, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template
          SeparableEvaluateForm>::value;
  const bool hasGradient = HasGradient<
      Function<EvaluateAndWithGradientTestFunction, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template
          SeparableGradientForm>::value;
  const bool hasEvaluateWithGradient = HasEvaluateWithGradient<
      Function<EvaluateAndWithGradientTestFunction, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template
          SeparableEvaluateWithGradientForm>::value;

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
  const bool hasEvaluate = HasEvaluate<
      Function<EvaluateAndNonConstGradientTestFunction, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template EvaluateConstForm>::value;
  const bool hasGradient = HasGradient<
      Function<EvaluateAndNonConstGradientTestFunction, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template GradientForm>::value;
  const bool hasEvaluateWithGradient = HasEvaluateWithGradient<
      Function<EvaluateAndNonConstGradientTestFunction, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template
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
  const bool hasEvaluate = HasEvaluate<
      Function<EvaluateAndStaticGradientTestFunction, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template EvaluateConstForm>::value;
  const bool hasGradient = HasGradient<
      Function<EvaluateAndStaticGradientTestFunction, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template GradientStaticForm>::value;
  const bool hasEvaluateWithGradient = HasEvaluateWithGradient<
      Function<EvaluateAndStaticGradientTestFunction, arma::mat, arma::mat>,
      TypedForms<arma::mat, arma::mat>::template
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
 * Test the correctness of the static check for SeparableFunctionType API.
 */
TEST_CASE("SeparableFunctionTypeCheckTest", "[FunctionTest]")
{
  static_assert(CheckNumFunctions<A, arma::mat, arma::mat>::value,
      "CheckNumFunctions static check failed.");
  static_assert(CheckNumFunctions<B, arma::mat, arma::mat>::value,
      "CheckNumFunctions static check failed.");
  static_assert(!CheckNumFunctions<C, arma::mat, arma::mat>::value,
      "CheckNumFunctions static check failed.");
  static_assert(!CheckNumFunctions<D, arma::mat, arma::mat>::value,
      "CheckNumFunctions static check failed.");

  static_assert(CheckSeparableEvaluate<A, arma::mat, arma::mat>::value,
      "CheckSeparableEvaluate static check failed.");
  static_assert(CheckSeparableEvaluate<B, arma::mat, arma::mat>::value,
      "CheckSeparableEvaluate static check failed.");
  static_assert(!CheckSeparableEvaluate<C, arma::mat, arma::mat>::value,
      "CheckSeparableEvaluate static check failed.");
  static_assert(!CheckSeparableEvaluate<D, arma::mat, arma::mat>::value,
      "CheckSeparableEvaluate static check failed.");

  static_assert(CheckSeparableGradient<A, arma::mat, arma::mat>::value,
      "CheckSeparableGradient static check failed.");
  static_assert(CheckSeparableGradient<B, arma::mat, arma::mat>::value,
      "CheckSeparableGradient static check failed.");
  static_assert(!CheckSeparableGradient<C, arma::mat, arma::mat>::value,
      "CheckSeparableGradient static check failed.");
  static_assert(!CheckSeparableGradient<D, arma::mat, arma::mat>::value,
      "CheckSeparableGradient static check failed.");
}

/**
 * Test the correctness of the static check for LagrangianFunctionType API.
 */
TEST_CASE("LagrangianFunctionTypeCheckTest", "[FunctionTest]")
{
  static_assert(!CheckEvaluate<A, arma::mat, arma::mat>::value,
      "CheckEvaluate static check failed.");
  static_assert(!CheckEvaluate<B, arma::mat, arma::mat>::value,
      "CheckEvaluate static check failed.");
  static_assert(CheckEvaluate<C, arma::mat, arma::mat>::value,
      "CheckEvaluate static check failed.");
  static_assert(CheckEvaluate<D, arma::mat, arma::mat>::value,
      "CheckEvaluate static check failed.");

  static_assert(!CheckGradient<A, arma::mat, arma::mat>::value,
      "CheckGradient static check failed.");
  static_assert(!CheckGradient<B, arma::mat, arma::mat>::value,
      "CheckGradient static check failed.");
  static_assert(CheckGradient<C, arma::mat, arma::mat>::value,
      "CheckGradient static check failed.");
  static_assert(CheckGradient<D, arma::mat, arma::mat>::value,
      "CheckGradient static check failed.");

  static_assert(!CheckNumConstraints<A, arma::mat, arma::mat>::value,
      "CheckNumConstraints static check failed.");
  static_assert(!CheckNumConstraints<B, arma::mat, arma::mat>::value,
      "CheckNumConstraints static check failed.");
  static_assert(CheckNumConstraints<C, arma::mat, arma::mat>::value,
      "CheckNumConstraints static check failed.");
  static_assert(CheckNumConstraints<D, arma::mat, arma::mat>::value,
      "CheckNumConstraints static check failed.");

  static_assert(!CheckEvaluateConstraint<A, arma::mat, arma::mat>::value,
      "CheckEvaluateConstraint static check failed.");
  static_assert(!CheckEvaluateConstraint<B, arma::mat, arma::mat>::value,
      "CheckEvaluateConstraint static check failed.");
  static_assert(CheckEvaluateConstraint<C, arma::mat, arma::mat>::value,
      "CheckEvaluateConstraint static check failed.");
  static_assert(CheckEvaluateConstraint<D, arma::mat, arma::mat>::value,
      "CheckEvaluateConstraint static check failed.");

  static_assert(!CheckGradientConstraint<A, arma::mat, arma::mat>::value,
      "CheckGradientConstraint static check failed.");
  static_assert(!CheckGradientConstraint<B, arma::mat, arma::mat>::value,
      "CheckGradientConstraint static check failed.");
  static_assert(CheckGradientConstraint<C, arma::mat, arma::mat>::value,
      "CheckGradientConstraint static check failed.");
  static_assert(CheckGradientConstraint<D, arma::mat, arma::mat>::value,
      "CheckGradientConstraint static check failed.");
}

/**
 * Test the correctness of the static check for SparseFunctionType API.
 */
TEST_CASE("SparseFunctionTypeCheckTest", "[FunctionTest]")
{
  static_assert(CheckSparseGradient<A, arma::mat, arma::mat>::value,
      "CheckSparseGradient static check failed.");
  static_assert(CheckSparseGradient<B, arma::mat, arma::mat>::value,
      "CheckSparseGradient static check failed.");
  static_assert(!CheckSparseGradient<C, arma::mat, arma::mat>::value,
      "CheckSparseGradient static check failed.");
  static_assert(!CheckSparseGradient<D, arma::mat, arma::mat>::value,
      "CheckSparseGradient static check failed.");
}

/**
 * Test the correctness of the static check for SparseFunctionType API.
 */
TEST_CASE("ResolvableFunctionTypeCheckTest", "[FunctionTest]")
{
  static_assert(CheckNumFeatures<A, arma::mat, arma::mat>::value,
      "CheckNumFeatures static check failed.");
  static_assert(CheckNumFeatures<B, arma::mat, arma::mat>::value,
      "CheckNumFeatures static check failed.");
  static_assert(!CheckNumFeatures<C, arma::mat, arma::mat>::value,
      "CheckNumFeatures static check failed.");
  static_assert(!CheckNumFeatures<D, arma::mat, arma::mat>::value,
      "CheckNumFeatures static check failed.");

  static_assert(CheckPartialGradient<A, arma::mat, arma::sp_mat>::value,
      "CheckPartialGradient static check failed.");
  static_assert(CheckPartialGradient<B, arma::mat, arma::sp_mat>::value,
      "CheckPartialGradient static check failed.");
  static_assert(!CheckPartialGradient<C, arma::mat, arma::sp_mat>::value,
      "CheckPartialGradient static check failed.");
  static_assert(!CheckPartialGradient<D, arma::mat, arma::sp_mat>::value,
      "CheckPartialGradient static check failed.");
}

/**
 * @file aug_lagrangian_test.cpp
 * @author Ryan Curtin
 * @author Marcus Edel
 * @author Conrad Sanderson
 *
 * Test of the AugmentedLagrangian class using the test functions defined in
 * aug_lagrangian_test_functions.hpp.
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

 /**
  * Tests the Augmented Lagrangian optimizer using the
  * AugmentedLagrangianTestFunction class.
  */
 TEMPLATE_TEST_CASE("AugLagrangian_AugLagrangianTestFunction",
     "[AugLagrangian]", arma::mat)
 {
   typedef typename TestType::elem_type ElemType;

   // The choice of 10 memory slots is arbitrary.
   AugLagrangianTestFunction f;
   AugLagrangian aug;

   arma::Col<ElemType> coords = f.GetInitialPoint();

   if (!aug.Optimize(f, coords))
     FAIL("Optimization reported failure.");

   double finalValue = f.Evaluate(coords);

   REQUIRE(finalValue == Approx(70.0).epsilon(1e-7));
   REQUIRE(coords(0) == Approx(1.0).epsilon(1e-7));
   REQUIRE(coords(1) == Approx(4.0).epsilon(1e-7));
 }

 /**
  * Tests the Augmented Lagrangian optimizer using the Gockenbach function.
  */
 TEMPLATE_TEST_CASE("AugLagrangian_GockenbachFunction", "[AugLagrangian]",
     arma::mat, arma::fmat)
 {
   typedef typename TestType::elem_type ElemType;

   GockenbachFunction f;
   AugLagrangian aug;

   TestType coords = f.GetInitialPoint<TestType>();

   if (!aug.Optimize(f, coords))
     FAIL("Optimization reported failure.");

   ElemType finalValue = f.Evaluate(coords);

   // Higher tolerance for smaller values.
   REQUIRE(finalValue == Approx(29.633926).epsilon(1e-3));
   REQUIRE(coords(0) == Approx(0.12288178).epsilon(0.1));
   REQUIRE(coords(1) == Approx(-1.10778185).epsilon(1e-3));
   REQUIRE(coords(2) == Approx(0.015099932).epsilon(0.1));
 }

 /**
  * Tests the Augmented Lagrangian optimizer using the Gockenbach function.  Uses
  * arma::sp_mat.
  */
 TEMPLATE_TEST_CASE("AugLagrangian_GockenbachFunction", "[AugLagrangian]",
     arma::sp_mat)
 {
   typedef typename TestType::elem_type ElemType;

   GockenbachFunction f;
   AugLagrangian aug;

   TestType coords = f.GetInitialPoint<TestType>();

   if (!aug.Optimize(f, coords))
     FAIL("Optimization reported failure.");

   ElemType finalValue = f.Evaluate(coords);

   // Higher tolerance for smaller values.
   REQUIRE(finalValue == Approx(29.633926).epsilon(1e-7));
   REQUIRE(coords(0) == Approx(0.12288178).epsilon(1e-5));
   REQUIRE(coords(1) == Approx(-1.10778185).epsilon(1e-7));
   REQUIRE(coords(2) == Approx(0.015099932).epsilon(1e-5));
 }

 #ifdef USE_COOT

 TEMPLATE_TEST_CASE("AugLagrangian_GockenbachFunction", "[AugLagrangian]",
   coot::mat, coot::fmat)
 {
   typedef typename ForwardType<TestType>::bvec BaseVecType;
   typedef typename TestType::elem_type ElemType;

   GockenbachFunctionType<TestType> f;
   AugLagrangianType<BaseVecType> aug;

   TestType coords = f.template GetInitialPoint<TestType>();

   if (!aug.Optimize(f, coords))
     FAIL("Optimization reported failure.");

   ElemType finalValue = f.Evaluate(coords);

   // Higher tolerance for smaller values.
   REQUIRE(finalValue == Approx(29.633926).epsilon(1e-3));
   REQUIRE(coords(0) == Approx(0.12288178).epsilon(0.1));
   REQUIRE(coords(1) == Approx(-1.10778185).epsilon(1e-3));
   REQUIRE(coords(2) == Approx(0.015099932).epsilon(0.1));
 }

 #endif

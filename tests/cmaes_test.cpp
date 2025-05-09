/**
 * @file cmaes_test.cpp
 * @author Marcus Edel
 * @author Kartik Nighania
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

 /**
  * Run CMA-ES with the full selection policy on logistic regression and
  * make sure the results are acceptable.
  */
 TEMPLATE_TEST_CASE("CMAES_LogisticRegression", "[CMAES]", arma::mat)
 {
   BoundaryBoxConstraint<TestType> b(-10, 10);
   CMAES<FullSelection, BoundaryBoxConstraint<TestType>> cmaes(
       0, b, 32, 500, 1e-3);
   cmaes.StepSize() = 0.6;
   LogisticRegressionFunctionTest<TestType>(cmaes, 0.003, 0.006, 5);
 }

 /**
  * Run CMA-ES with the random selection policy on logistic regression and
  * make sure the results are acceptable.
  */
 TEMPLATE_TEST_CASE("ApproxCMAES_LogisticRegressionTest", "[CMAES]", arma::mat)
 {
   BoundaryBoxConstraint<> b(-10, 10);
   ApproxCMAES<BoundaryBoxConstraint<TestType>> cmaes(256, b, 16, 500, 1e-3);
   cmaes.StepSize() = 0.6;
   LogisticRegressionFunctionTest<TestType>(cmaes, 0.003, 0.006, 5);
 }

 /**
  * Run CMA-ES with the full selection policy on logistic regression and
  * make sure the results are acceptable.  Use arma::fmat.
  */
 TEMPLATE_TEST_CASE("CMAES_LogisticRegression", "[CMAES]", arma::fmat)
 {
   BoundaryBoxConstraint<TestType> b(-10, 10);
   CMAES<FullSelection, BoundaryBoxConstraint<TestType>> cmaes(
       120, b, 32, 500, 1e-3);
   LogisticRegressionFunctionTest<TestType>(cmaes, 0.01, 0.02, 5);
 }

 /**
  * Run CMA-ES with the random selection policy on logistic regression and
  * make sure the results are acceptable.  Use arma::fmat.
  */
 TEMPLATE_TEST_CASE("ApproxCMAES_LogisticRegression", "[CMAES]", arma::fmat)
 {
   BoundaryBoxConstraint<TestType> b(-10, 10);
   ApproxCMAES<BoundaryBoxConstraint<TestType>> cmaes(0, b, 16, 500, 1e-3);
   LogisticRegressionFunctionTest<TestType>(cmaes, 0.01, 0.02, 5);
 }

 /**
  * Run CMA-ES with the random selection and empty transformation policies
  * on logistic regression and make sure the results are acceptable.
  * Use arma::fmat.
  */
 TEMPLATE_TEST_CASE("ApproxCMAESEmptyTransformationLogisticRegressionFMatTest",
     "[CMAESTest]", arma::fmat)
 {
   ApproxCMAES<EmptyTransformation<TestType>>
       cmaes(0, EmptyTransformation<TestType>(), 16, 500, 1e-3);
   LogisticRegressionFunctionTest<TestType>(cmaes, 0.01, 0.02, 5);
 }

 #ifdef USE_COOT

 TEMPLATE_TEST_CASE("CMAES_LogisticRegression", "[CMAES]", coot::mat)
 {
   BoundaryBoxConstraint<TestType> b(-10, 10);
   CMAES<FullSelection, BoundaryBoxConstraint<TestType>> cmaes(
       0, b, 32, 500, 1e-3);
   cmaes.StepSize() = 0.6;
   LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
       cmaes, 0.003, 0.006, 5);
 }

 TEMPLATE_TEST_CASE("CMAES_LogisticRegression", "[CMAES]", coot::fmat)
 {
   BoundaryBoxConstraint<TestType> b(-10, 10);
   CMAES<FullSelection, BoundaryBoxConstraint<TestType>> cmaes(
       120, b, 32, 500, 1e-3);
   LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
       cmaes, 0.01, 0.02, 5);
 }

 #endif

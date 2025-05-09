/**
 * @file ada_belief_test.cpp
 * @author Marcus Edel
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


 TEMPLATE_TEST_CASE("AdaBelief_LogisticRegressionFunction", "[AdaBelief]",
     arma::mat, arma::fmat)
 {
   AdaBelief adaBelief;
   LogisticRegressionFunctionTest<TestType, arma::Row<size_t>>(
       adaBelief, 0.003, 0.006, 1);
 }

 #ifdef USE_COOT

 TEMPLATE_TEST_CASE("AdaBelief_LogisticRegressionFunction", "[AdaBelief]",
     coot::mat, coot::fmat)
 {
   AdaBelief adaBelief;
   LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
       adaBelief, 0.003, 0.006, 1);
 }

 #endif

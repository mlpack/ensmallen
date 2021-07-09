
 //Created by o


#include <ensmallen.hpp>
#include "catch.hpp"
#include "test_function_tools.hpp"

using namespace ens;
using namespace ens::test;

TEST_CASE("LMCMAInitTest", "[LMCMATest]")
{
  RadermacherLMCMA<> lmcma(0, 1e5, 1e-4);
  LogisticRegressionFunctionTest(lmcma, 0.003, 0.006, 5);
}

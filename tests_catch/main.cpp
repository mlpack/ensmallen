// #include <ensmallen>
// OR
// #include <ensmallen.hpp>

//#define CATCH_CONFIG_MAIN  // catch.hpp will define main()
#define CATCH_CONFIG_RUNNER  // we will define main()
#include "catch.hpp"

int main(int argc, char** argv)
{
  // std::cout << "ensmallen version: " << ens::version::as_string() << '\n';
  
  return Catch::Session().run(argc, argv);
}

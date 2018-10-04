// #include <ensmallen>
// OR
// #include <ensmallen.hpp>

#include "catch.hpp"

// dummy test

TEST_CASE( "dummy test", "[dummy]" ) {
    
    double a = 1.23;
    REQUIRE( a == Approx(1.23) );

}

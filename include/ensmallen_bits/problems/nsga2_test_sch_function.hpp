/**
 * @file nsga2_test_sch_function.hpp
 * @author Sayan Goswami
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_PROBLEMS_NSGA2_TEST_SCH_FUNCTION_HPP
#define ENSMALLEN_PROBLEMS_NSGA2_TEST_SCH_FUNCTION_HPP

namespace ens {
namespace test {

template<typename MatType = arma::mat>
class NSGA2TestFuncSCH {
 public:
  NSGA2TestFuncSCH() : numObjectives(2), numVariables(1)
  {/* Nothing to do here. */}

  arma::vec Evaluate(const std::vector<double> coords) {
    arma::vec objectives(numObjectives);

    objectives[0] = std::pow(coords[0], 2);
    objectives[1] = std::pow(coords[0] - 2, 2);

    return objectives;
  }

  MatType GetInitialPoint() {
    return arma::vec(numVariables, 1, arma::fill::zeros);
  }

  size_t NumObjectives() const { return numObjectives; }

  double GetMaximum(size_t objectiveNumber) {
    return 1000;
  }

  double GetMinimum(size_t objectiveNumber) {
    return -1000;
  }

 private:
  size_t numObjectives;
  size_t numVariables;
};
}
}

#endif
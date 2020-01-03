/**
 * @file schaffer_function_n1.hpp
 * @author Sayan Goswami
 *
 * Implementation of Schaffer function N.1.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_PROBLEMS_SCHAFFER_FUNCTION_N1_HPP
#define ENSMALLEN_PROBLEMS_SCHAFFER_FUNCTION_N1_HPP

namespace ens {
namespace test {

template<typename MatType = arma::mat>
class SchafferFunctionN1
{
 public:
  SchafferFunctionN1() : numObjectives(2), numVariables(1)
  {/* Nothing to do here. */}

  arma::Col<typename MatType::elem_type> Evaluate(const MatType& coords)
  {
    // Convenience typedef.
    typedef typename MatType::elem_type ElemType;

    arma::Col<ElemType> objectives(numObjectives);

    objectives(0) = std::pow(coords[0], 2);
    objectives(1) = std::pow(coords[0] - 2, 2);

    return objectives;
  }

  MatType GetInitialPoint()
  {
    return arma::vec(numVariables, 1, arma::fill::randn);
  }

  size_t NumObjectives() const { return numObjectives; }

  double GetMaximum(size_t objectiveNumber)
  {
    if (objectiveNumber != 0 && objectiveNumber != 1)
    {
      throw std::logic_error("SchafferFunctionN1::GetMaximum() objectiveNumber must"
          " be either 0 or 1");
    }
    return 1000;
  }

  double GetMinimum(size_t objectiveNumber) {
    if (objectiveNumber != 0 && objectiveNumber != 1)
    {
      throw std::logic_error("SchafferFunctionN1::GetMinimum() objectiveNumber must"
          " be either 0 or 1");
    }
    return -1000;
  }

 private:
  size_t numObjectives;
  size_t numVariables;
};
}
}

#endif
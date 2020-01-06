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

/**
 * The Schaffer function N.1 is defined by
 *
 * \f[
 * f_1(x) = x^2
 * f_2(x) = (x-2)^2
 * \f]
 *
 * The optimal solutions to this multi-objective function lie in the
 * range [0, 2].
 *
 * @tparam arma::mat Type of matrix to optimize.
 */
template<typename MatType = arma::mat>
class SchafferFunctionN1
{
 public:
 //! Initialize the SchafferFunctionN1
  SchafferFunctionN1() : numObjectives(2), numVariables(1)
  {/* Nothing to do here. */}

  /**
   * Evaluate the objectives with the given coordinate.
   *
   * @param coords The function coordinates.
   * @return arma::Col<typename MatType::elem_type>
   */
  arma::Col<typename MatType::elem_type> Evaluate(const MatType& coords)
  {
    // Convenience typedef.
    typedef typename MatType::elem_type ElemType;

    arma::Col<ElemType> objectives(numObjectives);

    objectives(0) = std::pow(coords[0], 2);
    objectives(1) = std::pow(coords[0] - 2, 2);

    return objectives;
  }

  //! Get the starting point.
  MatType GetInitialPoint()
  {
    return arma::vec(numVariables, 1, arma::fill::randn);
  }

  //! Number of objectives to optimize
  size_t NumObjectives() const { return numObjectives; }

  /**
   * Get the upper bound for a particular objective
   *
   * @param objectiveNumber The objective for which to get the bound for.
   * @return double
   */
  double GetMaximum(size_t objectiveNumber)
  {
    if (objectiveNumber != 0 && objectiveNumber != 1)
    {
      throw std::logic_error("SchafferFunctionN1::GetMaximum() objectiveNumber"
          " must be either 0 or 1");
    }
    return 1000;
  }

  /**
   * Get the lower bound for a particular objective
   *
   * @param objectiveNumber The objective for which to get the bound for.
   * @return double
   */
  double GetMinimum(size_t objectiveNumber)
  {
    if (objectiveNumber != 0 && objectiveNumber != 1)
    {
      throw std::logic_error("SchafferFunctionN1::GetMinimum() objectiveNumber"
          " must be either 0 or 1");
    }
    return -1000;
  }

 private:
  size_t numObjectives;
  size_t numVariables;
};
} // namespace test
} // namespace ens

#endif
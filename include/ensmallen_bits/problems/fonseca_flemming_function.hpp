/**
 * @file fonseca_flemming_function_n1.hpp
 * @author Sayan Goswami
 *
 * Implementation of Fonseca Flemming function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_PROBLEMS_FONSECA_FLEEMING_FUNCTION_HPP
#define ENSMALLEN_PROBLEMS_FONSECA_FLEEMING_FUNCTION_HPP

namespace ens {
namespace test {

/**
 * The Fonseca Flemming function N.1 is defined by
 *
 * \f[
 * f_1(x) = 1 - \exp(\sum_1^3{(x_i - \frac{1}{\sqrt3})^2})
 * f_2(x) = 1 - \exp(\sum_1^3{(x_i + \frac{1}{\sqrt3})^2})
 * \f]
 *
 * The optimal solutions to this multi-objective function lie in the
 * range [-1/sqrt(3), 1/sqrt(3)].
 *
 * @tparam arma::mat Type of matrix to optimize.
 */
template<typename MatType = arma::mat>
class FonsecaFlemmingFunction
{
 public:
  FonsecaFlemmingFunction() : numObjectives(2), numVariables(3)
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

    objectives(0) = 1.0f - exp(- pow(coords[0] - 1.0f/sqrt(3), 2) -
        - pow(coords[1] - 1.0f/sqrt(3), 2) - pow(coords[0] - 1.0f/sqrt(3), 2));
    objectives(1) = 1.0f - exp(- pow(coords[0] + 1.0f/sqrt(3), 2) -
        - pow(coords[1] + 1.0f/sqrt(3), 2) - pow(coords[0] + 1.0f/sqrt(3), 2));

    return objectives;
  }

  //! Get the starting point.
  MatType GetInitialPoint()
  {
    return arma::vec(numVariables, 1, arma::fill::zeros);
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
    return 4;
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
    return -4;
  }

 private:
  size_t numObjectives;
  size_t numVariables;
};
} // namespace test
} // namespace ens

#endif
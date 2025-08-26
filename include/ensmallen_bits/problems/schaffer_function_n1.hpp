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
 private:
  size_t numObjectives;
  size_t numVariables;

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
    // Convenience typedef.
    typedef typename MatType::elem_type ElemType;

    return arma::Col<ElemType>(numVariables, 1, arma::fill::zeros);
  }

  struct ObjectiveA
  {
    typename MatType::elem_type Evaluate(const MatType& coords)
    {
        return std::pow(coords[0], 2);
    }
  } objectiveA;

  struct ObjectiveB
  {
    typename MatType::elem_type Evaluate(const MatType& coords)
    {
        return std::pow(coords[0] - 2, 2);
    }
  } objectiveB;

  //! Get objective functions.
  std::tuple<ObjectiveA, ObjectiveB> GetObjectives()
  {
    return std::make_tuple(objectiveA, objectiveB);
  }
};
} // namespace test
} // namespace ens

#endif
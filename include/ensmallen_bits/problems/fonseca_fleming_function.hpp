/**
 * @file fonseca_fleming_function_n1.hpp
 * @author Sayan Goswami
 *
 * Implementation of Fonseca Fleming function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_PROBLEMS_FONSECA_FLEEMING_FUNCTION_HPP
#define ENSMALLEN_PROBLEMS_FONSECA_FLEEMING_FUNCTION_HPP

#include <tuple>

namespace ens {
namespace test {

/**
 * The Fonseca Fleming function N.1 is defined by
 *
 * \f[
 * f_{1}\left(\boldsymbol{x}\right) = 1 - \exp \left[-\sum_{i=1}^{3} \left(x_{i} - \frac{1}{\sqrt{n}} \right)^{2} \right] \\
 * f_{2}\left(\boldsymbol{x}\right) = 1 - \exp \left[-\sum_{i=1}^{3} \left(x_{i} + \frac{1}{\sqrt{n}} \right)^{2} \right] \\
 * \f]
 *
 * The optimal solutions to this multi-objective function lie in the
 * range [-1/sqrt(3), 1/sqrt(3)].
 *
 * @tparam arma::mat Type of matrix to optimize.
 */
template<typename MatType = arma::mat>
class FonsecaFlemingFunction
{
 private:
  size_t numObjectives;
  size_t numVariables;

 public:
  FonsecaFlemingFunction() : numObjectives(2), numVariables(3)
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

    objectives(0) = objectiveA.Evaluate(coords);
    objectives(1) = objectiveB.Evaluate(coords);

    return objectives;
  }

  //! Get the starting point.
  MatType GetInitialPoint()
  {
    return arma::vec(numVariables, 1, arma::fill::zeros);
  }

  struct ObjectiveA
  {
    typename MatType::elem_type Evaluate(const MatType& coords)
    {
        return 1.0 - exp(
             -pow(static_cast<double>(coords[0]) - 1.0 / sqrt(3.0), 2.0)
             -pow(static_cast<double>(coords[1]) - 1.0 / sqrt(3.0), 2.0)
             -pow(static_cast<double>(coords[2]) - 1.0 / sqrt(3.0), 2.0)
        );
    }
  } objectiveA;

  struct ObjectiveB
  {
    typename MatType::elem_type Evaluate(const MatType& coords)
    {
        return 1.0 - exp(
            -pow(static_cast<double>(coords[0]) + 1.0 / sqrt(3.0), 2.0)
            -pow(static_cast<double>(coords[1]) + 1.0 / sqrt(3.0), 2.0)
            -pow(static_cast<double>(coords[2]) + 1.0 / sqrt(3.0), 2.0)
        );
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

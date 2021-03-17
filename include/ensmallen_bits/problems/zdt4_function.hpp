/**
 * @file zdt4_function.hpp
 * @author Nanubala Gnana Sai
 *
 * Implementation of the fourth ZDT(Zitzler, Deb and Thiele) test.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_PROBLEMS_ZDT_FOUR_FUNCTION_HPP
#define ENSMALLEN_PROBLEMS_ZDT_FOUR_FUNCTION_HPP

namespace ens {
namespace test {
/**
 * The ZDT4 function, defined by:
 * \f[
 * g(x) = 1 + 10(n-1) + \sum_{i=2}^{n}(x_i^2 - 10cos(4\pi x_i))
 * f_1(x) = x_i
 * h(f_1,g) = 1 - \sqrt{f_i/g}
 * f_2(x) = g(x) * h(f_1, g)
 * \f]
 *
 * This is a 10-variable problem(n = 10) with a convex
 * optimal front. This problem contains several local
 * optimum, making it difficult to reach the global optimum.
 *
 * Bounds of the variable space is:
 *  0 <= x_1 <= 1;
 * -10 <= x_i <= 10 for i = 2,...,n.
 *
 * This should be optimized to g(x) = 1.0, at:
 * x_1* in [0, 1] ; x_i* = 0 for i = 2,...,n
 *
 * For more information, please refer to:
 *
 * @code
 * @article{Zitzler2000,
 *   title   = {Comparison of multiobjective evolutionary algorithms:
 *              Empirical results},
 *   author  = {Zitzler, Eckart and Deb, Kalyanmoy and Thiele, Lothar},
 *   journal = {Evolutionary computation},
 *   year    = {2000},
 *   doi     = {10.1162/106365600568202}
 * }
 * @endcode
 */
  template<typename MatType = arma::mat>
  class ZDT4
  {
    private:
      size_t numObjectives;
      size_t numVariables;

    public:
     //! Initialize the ZDT4
    ZDT4() : numObjectives(2), numVariables(10)
    {
      if(numVariables < 2)
      {
        std::ostringstream oss;
        oss << "ZDT4::ZDT4(): expected variable space of "
        << "dimensions atleast 2, but got " <<  numVariables
        << std::endl;
        throw std::invalid_argument(oss.str());
      }
    }

    /**
     * Evaluate the objectives with the given coordinate.
     *
     * @param coords The function coordinates.
     * @return arma::Col<typename MatType::elem_type>
     */
    arma::Col<typename MatType::elem_type> Evaluate(const MatType& coords)
    {
      double pi = arma::datum::pi;
      typedef typename MatType::elem_type ElemType;

      if(coords.size() != numVariables)
      {
        std::ostringstream oss;
        oss << "ZDT4::Evaluate(): Provided coordinate's dimension is: "
            << coords.size() << "expected: " << numVariables
            << std::endl;
        throw std::invalid_argument(oss.str());
      }

      arma::Col<ElemType> objectives(numObjectives);
      objectives(0) = coords[0];
      arma::vec truncatedCoords = coords(arma::span(1, numVariables - 1));
      double sum = arma::accu(arma::square(truncatedCoords) -
          10. * arma::cos(4 * pi * truncatedCoords));
      double g = 1. + 10 * static_cast<double>(numVariables - 1) + sum;
      double objectiveRatio = objectives(0) / g;
	    objectives(1) = g * (1. - std::sqrt(objectiveRatio));

	    return objectives;
    }

    //! Get the starting point.
    MatType GetInitialPoint()
    {
      return arma::vec(numVariables, 1, arma::fill::zeros);
    }

    struct ObjectiveF1
    {
      typename MatType::elem_type Evaluate(const MatType& coords)
      {
        if(coords.size() != numVariables)
        {
          std::ostringstream oss;
          oss << "ZDT4::Evaluate(): Provided coordinate's dimension is: "
              << coords.size() << "expected: " << numVariables
              << std::endl;
          throw std::invalid_argument(oss.str());
        }

        return coords[0];
      }
    } objectiveF1;

    struct ObjectiveF2
    {
      typename MatType::elem_type Evaluate(const MatType& coords)
      {
        double pi = arma::datum::pi;

        arma::vec truncatedCoords = coords(arma::span(1, numVariables - 1));
        double sum = arma::accu(arma::square(truncatedCoords) -
            10. * arma::cos(4 * pi * truncatedCoords));
        double g = 1. + 10 * static_cast<double>(numVariables - 1) + sum;
        double objectiveRatio = objectiveF1.Evaluate(coords) / g;

        return  g * (1. - std::sqrt(objectiveRatio));
	    }
    } objectiveF2;

    //! Get objective functions.
    std::tuple<ObjectiveF1, ObjectiveF2> GetObjectives()
    {
      return std::make_tuple(objectiveF1, objectiveF2);
    }
  };
  } //namespace test
  } //namespace ens
#endif
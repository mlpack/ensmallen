/**
 * @file zdt5_function.hpp
 * @author Nanubala Gnana Sai
 *
 * Implementation of the fourth ZDT(Zitzler, Deb and Thiele) test.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_PROBLEMS_ZDT_FIVE_FUNCTION_HPP
#define ENSMALLEN_PROBLEMS_ZDT_FIVE_FUNCTION_HPP

namespace ens {
namespace test {
/**
 * The ZDT5 function, defined by:
 * \f[
 * g(x) = \sum_{i=2}^{n}v(u(x_i))
 * v(u(x_i)) = 2 + u(x_i) if u(x_i) < 5; 1 if u(x_i) = 5
 * f_1(x) = 1 + u(x_i)
 * h(f_1,g) = 1 - 1/f_i
 * f_2(x) = g(x) * h(f_1, g)
 * \f]
 *
 * This is a 11-variable problem(n = 11) with a deceptive convex
 * optimal front. This function stands out from the rest
 * since x_i is a bitstring.
 *
 * Bounds of the variable space is:
 *  x_1 is a bitstring of 30 bits.
 *  x_2 to x_11 are 5 bits bitstring.
 *
 * This should be optimized to g(x) = 10.0, at:
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
 *
 * @tparam MatType Type of matrix to optimize.
 */
  template<typename MatType = arma::mat>
  class ZDT5
  {
   public:
     //! Initialize the ZDT5
    ZDT5() :
        numObjectives(2),
        numVariables(11),
        numParetoPoints(100),
        objectiveF1(*this),
        objectiveF2(*this)
    {/* Nothing to do here. */}

    /**
     * Evaluate the objectives with the given coordinate.
     *
     * @param coords The function coordinates.
     * @return arma::Col<typename MatType::elem_type>
     */
    arma::Col<typename MatType::elem_type> Evaluate(const MatType& coords)
    {
      typedef typename MatType::elem_type ElemType;

      arma::Col<ElemType> objectives(numObjectives);
      objectives(0) = coords[0];
      size_t numVectors = (coords[0].size() - 30u/5u) + 1u;


      arma::Col<ElemType> truncatedCoords = coords(arma::span(1, numVariables - 1));
      ElemType sum = arma::accu(arma::square(truncatedCoords) -
          10. * arma::cos(4 * arma::datum::pi * truncatedCoords));
      ElemType g = 1. + 10. * static_cast<ElemType>(numVariables - 1) + sum;
      ElemType objectiveRatio = objectives(0) / g;
	    objectives(1) = g * (1. - std::sqrt(objectiveRatio));

      return objectives;
    }

    //! Get the starting point.
    MatType GetInitialPoint()
    {
      // Convenience typedef.
      typedef typename MatType::elem_type ElemType;

      return arma::Col<ElemType>(numVariables, 1, arma::fill::zeros);
    }

    struct ObjectiveF1
    {
      ObjectiveF1(ZDT5& zdtClass) : zdtClass(zdtClass)
      {/*Nothing to do here */}

      typename MatType::elem_type Evaluate(const MatType& coords)
      {
        return coords[0];
      }

      ZDT5& zdtClass;
    };

    struct ObjectiveF2
    {
      ObjectiveF2(ZDT5& zdtClass) : zdtClass(zdtClass)
      {/*Nothing to do here */}

      typename MatType::elem_type Evaluate(const MatType& coords)
      {
        typedef typename MatType::elem_type ElemType;

        size_t numVariables = zdtClass.numVariables;
        arma::Col<ElemType> truncatedCoords = coords(arma::span(1, numVariables - 1));
        ElemType sum = arma::accu(arma::square(truncatedCoords) -
            10. * arma::cos(4 * arma::datum::pi * truncatedCoords));
        ElemType g = 1. + 10 * static_cast<ElemType>(numVariables - 1) + sum;
        ElemType objectiveRatio = zdtClass.objectiveF1.Evaluate(coords) / g;

        return  g * (1. - std::sqrt(objectiveRatio));
      }

      ZDT5& zdtClass;
    };

    //! Get objective functions.
    std::tuple<ObjectiveF1, ObjectiveF2> GetObjectives()
    {
      return std::make_tuple(objectiveF1, objectiveF2);
    }

    //! Get the true Pareto Front
    arma::cube GetParetoFront()
    {
      arma::cube front(2, 1, numParetoPoints);
      arma::vec x = 1 + 30 * arma::linspace(0, 1, numParetoPoints);
      arma::vec y = 10.0 / x;
      for (size_t idx = 0; idx < numParetoPoints; ++idx)
        front.slice(idx) = arma::vec{ x(idx), y(idx) };

      return front;
    }

    ObjectiveF1 objectiveF1;
    ObjectiveF2 objectiveF2;

   private:
    size_t numObjectives;
    size_t numVariables;
    size_t numParetoPoints;
  };
  } //namespace test
  } //namespace ens
#endif
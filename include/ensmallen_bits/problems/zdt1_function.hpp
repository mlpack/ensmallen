/**
 * @file zdt1_function.hpp
 * @author Nanubala Gnana Sai
 *
 * Implementation of the first ZDT(Zitzler, Deb and Thiele) test.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_PROBLEMS_ZDT_ONE_FUNCTION_HPP
#define ENSMALLEN_PROBLEMS_ZDT_ONE_FUNCTION_HPP

namespace ens {
namespace test {

/**
 * The ZDT1 function, defined by:
 * \f[
 * g(x) = 1 + 9(\sum_{i=2}^{n} x_i )/(n-1)
 * f_1(x) = x_i
 * h(f_1, g) = g(x)[1-\sqrt{f_1/g}\ ]
 * f_2(x) = g(x) * h(f_1, g)
 * \f]
 *
 * This is a 30-variable problem(n = 30) with a convex optimal set.
 *
 * Bounds of the variable space is:
 * 0 <= x_i <= 1 for i = 1,...,n.
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
 *
 * @tparam arma::mat Type of matrix to optimize.
 */
  template<typename MatType = arma::mat>
  class ZDT1
  {
   private:
    size_t numObjectives;
    size_t numVariables;

   public:
     //! Initialize the ZDT1
    ZDT1() :
        numObjectives(2),
        numVariables(30),
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
      // Convenience typedef.
      typedef typename MatType::elem_type ElemType;

	    arma::Col<ElemType> objectives(numObjectives);
      objectives(0) = coords[0];
	    double sum = arma::accu(coords(arma::span(1, numVariables - 1), 0));
	    double g = 1. + 9. * sum / (static_cast<double>(numVariables) - 1.);
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
      ObjectiveF1(ZDT1& zdtClass) : zdtClass(zdtClass)
      {/*Nothing to do here */}

      typename MatType::elem_type Evaluate(const MatType& coords)
      {
        return coords[0];
      }

      ZDT1& zdtClass;
    };

    struct ObjectiveF2
    {
      ObjectiveF2(ZDT1& zdtClass) : zdtClass(zdtClass)
      {/*Nothing to do here */}

      typename MatType::elem_type Evaluate(const MatType& coords)
      {
        double numVariables = zdtClass.numVariables;
        double sum = arma::accu(coords(arma::span(1, numVariables - 1), 0));
        double g = 1. + 9. * sum / (static_cast<double>(numVariables - 1));
        double objectiveRatio = zdtClass.objectiveF1.Evaluate(coords) / g;

        return g * (1. - std::sqrt(objectiveRatio));
		  }

      ZDT1& zdtClass;
    };

    //! Get objective functions.
    std::tuple<ObjectiveF1, ObjectiveF2> GetObjectives()
    {
      return std::make_tuple(objectiveF1, objectiveF2);
    }

    ObjectiveF1 objectiveF1;
    ObjectiveF2 objectiveF2;
  };
  } //namespace test
  } //namespace ens

#endif
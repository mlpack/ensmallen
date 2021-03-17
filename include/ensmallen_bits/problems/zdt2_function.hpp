/**
 * @file zdt2_function.hpp
 * @author Nanubala Gnana Sai
 *
 * Implementation of the second ZDT(Zitzler, Deb and Thiele) test.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_PROBLEMS_ZDT_TWO_FUNCTION_HPP
#define ENSMALLEN_PROBLEMS_ZDT_TWO_FUNCTION_HPP

namespace ens {
namespace test {

/**
 * The ZDT2 function, defined by:
 * \f[
 * g(x) = 1 + 9(\sum_{i=2}^{n} x_i )/(n-1)
 * f_1(x) = x_i
 * h(f_1, g) = 1 - (f_1/g)^2
 * f_2(x) = g(x) * h(f_1, g)
 * \f]
 *
 * This is a 30-variable problem(n = 30) with a
 * non convex optimal set.
 *
 * Bounds of the variable space is:
 * 0 <= x_i <= 1 for i = 1,..,n.
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
  class ZDT2
  {
  private:
      size_t numObjectives;
      size_t numVariables;

   public:
     //! Initialize the ZDT2
    ZDT2() : numObjectives(2), numVariables(30)
    {/* Nothing to do here. */}

    ZDT2(const size_t numVariables) :
        numObjectives(2),
        numVariables(numVariables)
    {
      if(numVariables < 2)
      {
        std::ostringstream oss;
        oss << "ZDT2::ZDT2(): expected variable space of "
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
      // Convenience typedef.
      typedef typename MatType::elem_type ElemType;

      if(coords.size() != numVariables)
      {
        std::ostringstream oss;
        oss << "ZDT2::Evaluate(): Provided coordinate's dimension is: "
            << coords.size() << "expected: " << numVariables
            << std::endl;
        throw std::invalid_argument(oss.str());
      }

      arma::Col<ElemType> objectives(numObjectives);
      objectives(0) = coords[0];
      double sum = arma::accu(coords(arma::span(1, numVariables - 1), 0));
      double g = 1. + 9. * sum / (static_cast<double>(numVariables) - 1.);
      double objectiveRatio = objectives(0) / g;
      objectives(1) = g * (1. - std::pow(objectiveRatio, 2));

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
          oss << "ZDT2::Evaluate(): Provided coordinate's dimension is: "
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
        if(coords.size() != numVariables)
        {
          std::ostringstream oss;
          oss << "ZDT2::Evaluate(): Provided coordinate's dimension is: "
              << coords.size() << "expected: " << numVariables
              << std::endl;
          throw std::invalid_argument(oss.str());
        }

		    double sum = arma::accu(coords(arma::span(1, numVariables - 1), 0));
			  double g = 1. + 9. * sum / (static_cast<double>(numVariables) - 1.);
        double objectiveRatio = objectiveF1.evaluate(coords) / g;

			  return g * (1. - std::pow(objectiveRatio, 2));
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
/**
 * @file dtlz4_function.hpp
 * @author Satyam Shukla
 *
 * Implementation of the fourth DTLZ(Deb, Thiele, Laumanns, and Zitzler) test.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_PROBLEMS_DTLZ_FOUR_FUNCTION_HPP
#define ENSMALLEN_PROBLEMS_DTLZ_FOUR_FUNCTION_HPP

#include "../../moead/weight_init_policies/uniform_init.hpp"

namespace ens {
namespace test {

/**
 * The DTLZ4 function, defined by:
 * \f[
 * x_M = [x_i, n - M + 1 <= i <= n]
 * g(x) = \Sigma{i = n - M + 1}^n (x_i - 0.5)^2
 * 
 * f_1(x) = 0.5 * cos(x_1^alpha * pi * 0.5) * cos(x_2^alpha * pi * 0.5) * ... cos(x_2^alpha * pi * 0.5) * (1 + g(x_M)) 
 * f_2(x) = 0.5 * cos(x_1^alpha * pi * 0.5) * cos(x_2^alpha * pi * 0.5) * ... sin(x_M-1^alpha * pi * 0.5) * (1 + g(x_M))
 * .
 * .
 * f_M(x) = 0.5 * sin(x_1^alpha * pi * 0.5) * (1 + g(x_M))
 * \f]
 *
 * Bounds of the variable space is:
 * 0 <= x_i <= 1 for i = 1,...,n.
 *
 * This should be optimized to x_i = 0.5 (for all x_i in x_M), at:
 * 
 * For more information, please refer to:
 * 
 * @code
 * @incollection{deb2005scalable,
 * title={Scalable test problems for evolutionary multiobjective optimization},
 * author={Deb, Kalyanmoy and Thiele, Lothar and Laumanns, Marco and Zitzler, Eckart},
 * booktitle={Evolutionary multiobjective optimization: theoretical advances and applications},
 * pages={105--145},
 * year={2005},
 * publisher={Springer}
 * }
 * @endcode
 *
 * @tparam MatType Type of matrix to optimize.
 */
  template <typename MatType = arma::mat>
  class DTLZ4
  {
    private:

    // A fixed no. of Objectives and Variables(|x| = 7, M = 3).
    size_t numObjectives {3};
    size_t numVariables {7};
    size_t numParetoPoints;
    size_t alpha;

    public:

      /**
       * Object Constructor.
       * Initializes the individual objective functions.
       *
       * @param alpha The power which each variable is raised to.
       * @param numParetoPoint No. of pareto points in the reference front.
       */
      DTLZ4 (size_t alpha = 100, size_t numParetoPoints = 136) :
        alpha(alpha),
        numParetoPoints(numParetoPoints),
        objectiveF1(0, *this),
        objectiveF2(1, *this),
        objectiveF3(2, *this)
      {/*Nothing to do here.*/}

      //! Get the starting point.
      arma::Col<typename MatType::elem_type> GetInitialPoint()
      {
        // Convenience typedef.
        typedef typename MatType::elem_type ElemType;
        return arma::Col<ElemType>(numVariables, 1, arma::fill::zeros);
      } 
      
      // Get the private variables.

      // Get the number of objectives.
      size_t GetNumObjectives()
      { return this -> numObjectives; }

      // Get the number of variables.
      size_t GetNumVariables()
      { return this -> numVariables; }

      /**
       * Set the no. of pareto points.
       *
       * @param numParetoPoint
       */
      void SetNumParetoPoint(size_t numParetoPoint)
      { this -> numParetoPoints = numParetoPoint; }

      /**
       * Evaluate the G(x) with the given coordinate.
       *
       * @param coords The function coordinates.
       * @return arma::Row<typename MatType::elem_type>
       */
      arma::Row<typename MatType::elem_type> g(const MatType& coords)
      {
        size_t k = numVariables - numObjectives + 1;

        // Convenience typedef.
        typedef typename MatType::elem_type ElemType;
        
        arma::Row<ElemType> innerSum(size(coords)[1], arma::fill::zeros);
        
        for(size_t i = numObjectives - 1; i < numVariables; i++)
        {
          innerSum += arma::pow((coords.row(i) - 0.5), 2); 
        } 
        
        return innerSum;
      }    

      /**
       * Evaluate the objectives with the given coordinate.
       *
       * @param coords The function coordinates.
       * @return arma::Mat<typename MatType::elem_type>
       */
      arma::Mat<typename MatType::elem_type> Evaluate(const MatType& coords)
      {
        // Convenience typedef.
        typedef typename MatType::elem_type ElemType;

        arma::Mat<ElemType> objectives(numObjectives, size(coords)[1]);
        arma::Row<ElemType> G = g(coords);
        arma::Row<ElemType> value = 0.5 * (1.0 + G);
        for(size_t i = 0; i < numObjectives - 1; i++)
        {
          objectives.row(i) =  value %  
              arma::sin(arma::pow(coords.row(i), alpha) * arma::datum::pi * 0.5);
          value = value % arma::cos(arma::pow(coords.row(i), alpha) * arma::datum::pi * 0.5); 
        }
        objectives.row(numObjectives - 1) = value;
        return objectives;    
      }
      
      // Individual Objective function.
      // Changes based on stop variable provided. 
      struct DTLZ4Objective
      {
        DTLZ4Objective(size_t stop, DTLZ4& dtlz): stop(stop), dtlz(dtlz)
        {/* Nothing to do here.*/}  
        
        /**
         * Evaluate one objective with the given coordinate.
         *
         * @param coords The function coordinates.
         * @return arma::Col<typename MatType::elem_type>
         */
        typename MatType::elem_type Evaluate(const MatType& coords)
        {
          // Convenience typedef.
          typedef typename MatType::elem_type ElemType;
          ElemType value = 1.0;
          for(size_t i = 0; i < stop; i++)
          {
            value = value * std::cos(std::pow(coords[i], dtlz.alpha) * arma::datum::pi * 0.5);
          }

          if(stop != dtlz.numObjectives - 1)
          {
            value = value * std::sin(std::pow(coords[stop], dtlz.alpha) * arma::datum::pi * 0.5);
          }
          else
          {
            value = value * std::cos(std::pow(coords[stop], dtlz.alpha) * arma::datum::pi * 0.5);
          }

          value = value * (1 + dtlz.g(coords)[0]);
          return value;  
        }        

        DTLZ4& dtlz;
        size_t stop;
      };

      // Return back a tuple of objective functions.
      std::tuple<DTLZ4Objective, DTLZ4Objective, DTLZ4Objective> GetObjectives()
      {
          return std::make_tuple(objectiveF1, objectiveF2, objectiveF3);
      } 

      //! Get the Reference Front.
      //! Front. The implementation has been taken from pymoo.
      arma::mat GetReferenceFront()
      { 
      	Uniform refGenerator;
        arma::mat refDirs = refGenerator.Generate<arma::mat>(3, this -> numParetoPoints, 0);
        arma::colvec x = arma::normalise(refDirs, 2, 1);
        arma::mat A(size(refDirs), arma::fill::ones);
        A.each_col() = x;
        return refDirs / A;
      }

    DTLZ4Objective objectiveF1;
    DTLZ4Objective objectiveF2;
    DTLZ4Objective objectiveF3;
  };
  } //namespace test
  } //namespace ens

#endif
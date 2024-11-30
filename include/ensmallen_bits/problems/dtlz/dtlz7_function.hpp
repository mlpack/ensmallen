/**
 * @file dtlz7_function.hpp
 * @author Satyam Shukla
 *
 * Implementation of the seventh DTLZ(Deb, Thiele, Laumanns, and Zitzler) test.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_PROBLEMS_DTLZ_SEVEN_FUNCTION_HPP
#define ENSMALLEN_PROBLEMS_DTLZ_SEVEN_FUNCTION_HPP

namespace ens {
namespace test {

/**
 * The DTLZ7 function, defined by:
 * \f[
 * x_M = [x_i, n - M + 1 <= i <= n]
 * g(x) = 1 + (9 / |X_M|) * (\Sigma{i = n - M + 1}^n x_i) 
 * 
 * f_1(x) = x_1
 * f_2(x) = x_2
 * .
 * .
 * f_M(x) = (1 + g(X_M)) * h(f_1, f_2,...,g)
 * \f]
 *
 * Bounds of the variable space is:
 * 0 <= x_i <= 1 for i = 1,...,n.
 *
 * This should be optimized to x_i = 0.5 (for all x_i in x_M), at:
 * the objective function values lie on the linear hyper-plane: 
 * \Sigma { m = 1}^M f_m* =0.5.
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
  class DTLZ7
  {
    private:

    // A fixed no. of Objectives and Variables(|x| = 7, M = 3).
    size_t numObjectives {3};
    size_t numVariables {7};
    size_t numParetoPoints;

    public:

      /**
       * Object Constructor.
       * Initializes the individual objective functions.
       *
       * @param numParetoPoint No. of pareto points in the reference front.
       */
      DTLZ7(size_t numParetoPoint = 136) :
        numParetoPoints(numParetoPoint),
        objectiveF1(0, *this),
        objectiveF2(1, *this),
        objectiveF3(2, *this)
      {/* Nothing to do here */}

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
       * @param numParetoPoint No. of pareto points in the reference front.
       */
      void SetNumParetoPoint(size_t numParetoPoint)
      { this -> numParetoPoints = numParetoPoint; }

      // Get the starting point.
      arma::Col<typename MatType::elem_type> GetInitialPoint()
      {
        // Convenience typedef.
        typedef typename MatType::elem_type ElemType;
        return arma::Col<ElemType>(numVariables, arma::fill::zeros);
      } 

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
        
        innerSum = (9.0 / k) * arma::sum(coords.rows(numObjectives - 1,
            numVariables - 1) , 0) + 1.0; 
        return innerSum;
      }

      /**
       * Evaluate the H(f_i,...) with the given coordinate.
       *
       * @param coords The function coordinates.
       * @return arma::Row<typename MatType::elem_type>
       */
      arma::Row<typename MatType::elem_type> h(
          const MatType& coords, const arma::Row<typename MatType::elem_type>& G)
      {

        // Convenience typedef.
        typedef typename MatType::elem_type ElemType;
        
        arma::Row<ElemType> innerSum(size(coords)[1], arma::fill::ones);
        innerSum = innerSum * numObjectives;
        for(size_t i = 0;i < numObjectives - 1;i++)
        {
            innerSum -= coords.row(i) % (1.0 + 
                    arma::cos(arma::datum::pi * 3 * coords.row(i))) / (1 + G); 
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
        arma::Row<ElemType> H = h(coords, G);
        objectives.rows(0, numObjectives - 2) = coords.rows(0, numObjectives - 2);
        objectives.row(numObjectives - 1) = (1 + G) % H;
        return objectives;    
      }
      
      // Individual Objective function.
      // Changes based on stop variable provided. 
      struct DTLZ7Objective
      {
        DTLZ7Objective(size_t stop, DTLZ7& dtlz): stop(stop), dtlz(dtlz)
        {/* Nothing to do here. */}  
        
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
          ElemType value = 0.5;
          if(stop != dtlz.numObjectives - 1)
          { return coords[stop];}
 
          value = (1.0 + dtlz.g(coords)[0]) * dtlz.h(coords, dtlz.g(coords))[0];
          return value; 
        }        

        DTLZ7& dtlz;
        size_t stop;
      };

      // Return back a tuple of objective functions.
      std::tuple<DTLZ7Objective, DTLZ7Objective, DTLZ7Objective> GetObjectives()
      {
          return std::make_tuple(objectiveF1, objectiveF2, objectiveF3);
      } 

    DTLZ7Objective objectiveF1;
    DTLZ7Objective objectiveF2;
    DTLZ7Objective objectiveF3;
    using MAF7Objective = DTLZ7Objective;
  };

  template<typename MatType>
  using MAF7 = DTLZ7<MatType>;
  } //namespace test
  } //namespace ens

#endif
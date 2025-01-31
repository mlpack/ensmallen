/**
 * @file maf3_function.hpp
 * @author Satyam Shukla
 *
 * Implementation of the third Maf test.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_PROBLEMS_MAF_THREE_FUNCTION_HPP
#define ENSMALLEN_PROBLEMS_MAF_THREE_FUNCTION_HPP

#include "../../moead/weight_init_policies/uniform_init.hpp"

namespace ens {
namespace test {

/**
 * The MAF3 function, defined by:
 * \f[
 * x_M = [x_i, n - M + 1 <= i <= n]
 * g(x) = 100 * [|x_M| + \Sigma{i = n - M + 1}^n (x_i - 0.5)^2 - cos(20 * pi * 
 *   (x_i - 0.5))]
 * 
 * f_1(x) = (cos(x_1 * pi * 0.5) * cos(x_2 * pi * 0.5) * ... cos(x_2 * pi * 0.5) * (1 + g(x_M)))^4
 * f_2(x) = (cos(x_1 * pi * 0.5) * cos(x_2 * pi * 0.5) * ... sin(x_M-1 * pi * 0.5) * (1 + g(x_M)))^4
 * .
 * .
 * f_M(x) = (sin(x_1 * pi * 0.5) * (1 + g(x_M)))^2
 * \f]
 *
 * Bounds of the variable space is:
 * 0 <= x_i <= 1 for i = 1,...,n.
 * 
 * For more information, please refer to:
 * 
 * @code
 * @article{cheng2017benchmark,
 * title={A benchmark test suite for evolutionary many-objective optimization},
 * author={Cheng, Ran and Li, Miqing and Tian, Ye and Zhang, Xingyi and Yang, Shengxiang and Jin, Yaochu and Yao, Xin},
 * journal={Complex \& Intelligent Systems},
 * volume={3},
 * pages={67--81},
 * year={2017},
 * publisher={Springer}
 * }
 * @endcode
 *
 * @tparam MatType Type of matrix to optimize.
 */
  template <typename MatType = arma::mat>
  class MAF3
  {
    private:

    // A fixed no. of Objectives and Variables(|x| = 12, M = 3).
    size_t numObjectives {3};
    size_t numVariables {12};

    public:

      /**
       * Object Constructor.
       * Initializes the individual objective functions.
       *
       * @param numParetoPoint No. of pareto points in the reference front.
       */
      MAF3() :
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
      { return this -> numVariables;}

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
        
        for (size_t i = numObjectives - 1; i < numVariables; i++)
        {
          innerSum += arma::pow((coords.row(i) - 0.5), 2) - 
              arma::cos(20 * arma::datum::pi * (coords.row(i) - 0.5)); 
        } 
        
        return 100 * (k + innerSum);
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

        arma::Mat<ElemType> objectives(numObjectives, size(coords)[1], arma::fill::ones);
        arma::Row<ElemType> G = g(coords);
        arma::Row<ElemType> value = (1.0 + G);
        for (size_t i = 0; i < numObjectives - 1; i++)
        {
          objectives.row(i) =  arma::pow(value, i == 0 ? 2:4) % 
              arma::pow(arma::sin(coords.row(i) * arma::datum::pi * 0.5), i == 0 ? 2:4);
          value = value % arma::cos(coords.row(i) * arma::datum::pi * 0.5);
        }
        objectives.row(numObjectives - 1) = arma::pow(value, 4);
        return objectives;
      }
      
      // Individual Objective function.
      // Changes based on stop variable provided. 
      struct MAF3Objective
      {
        MAF3Objective(size_t stop, MAF3& maf): stop(stop), maf(maf)
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
          ElemType value = 1.0;
          for (size_t i = 0; i < stop; i++)
          {
            value = value * std::cos(coords[i] * arma::datum::pi * 0.5);
          }

          if(stop != maf.GetNumObjectives() - 1)
          {
            value = value * std::sin(coords[stop] * arma::datum::pi * 0.5);
          }

          value = value * (1. + maf.g(coords)[0]);

          if(stop == 0) {
            return std::pow(value, 2); 
          }
          return std::pow(value, 4);  
        }        

        MAF3& maf;
        size_t stop;
      };

      // Return back a tuple of objective functions.
      std::tuple<MAF3Objective, MAF3Objective, MAF3Objective> GetObjectives()
      {
          return std::make_tuple(objectiveF1, objectiveF2, objectiveF3);
      } 

    MAF3Objective objectiveF1;
    MAF3Objective objectiveF2;
    MAF3Objective objectiveF3;
  };
  } //namespace test
  } //namespace ens

#endif
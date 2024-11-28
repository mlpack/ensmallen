/**
 * @file maf1_function.hpp
 * @author Satyam Shukla
 *
 * Implementation of the first Maf test.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_PROBLEMS_MAF_ONE_FUNCTION_HPP
#define ENSMALLEN_PROBLEMS_MAF_ONE_FUNCTION_HPP

#include "../../moead/weight_init_policies/uniform_init.hpp"

namespace ens {
namespace test {

/**
 * The MAF1 function, defined by:
 * \f[
 * x_M = [x_i, n - M + 1 <= i <= n]
 * g(x) = \Sigma{i = n - M + 1}^n (x_i - 0.5)^2
 * 
 * f_1(x) = 1 - x_1 * x_2 * ... x_M-1 * (1 + g(x_M)) 
 * f_2(x) = 1 - x_1 * x_2 * ... (1 - x_M-1) * (1 + g(x_M))
 * .
 * .
 * f_M(x) = (x_1) * (1 + g(x_M))
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
  class MAF1
  {
    private:

    // A fixed no. of Objectives and Variables(|x| = 7, M = 3).
    size_t numObjectives {3};
    size_t numVariables {12};

    public:

      /**
      * Object Constructor.
      * Initializes the individual objective functions.
      *
      * @param numParetoPoint No. of pareto points in the reference front.
      */
      MAF1() :
          objectiveF1(0, *this),
          objectiveF2(1, *this),
          objectiveF3(2, *this)
      {/* Nothing to do here */}

      // Get the private variables.
      size_t GetNumObjectives()
      { return this -> numObjectives; }

      size_t GetNumVariables()
      { return this -> numVariables; }

      // Get the starting point.
      arma::Col<typename MatType::elem_type> GetInitialPoint()
      {
        // Convenience typedef.
        typedef typename MatType::elem_type ElemType;
        return arma::Col<ElemType>(numVariables, arma::fill::ones);
      }

      /**
      * Evaluate the G(x) with the given coordinate.
      *
      * @param coords The function coordinates.
      * @return arma::Row<typename MatType::elem_type>
      */
      arma::Row<typename MatType::elem_type> g(const MatType& coords)
      {
        // Convenience typedef.
        typedef typename MatType::elem_type ElemType;
        
        arma::Row<ElemType> innerSum(size(coords)[1], arma::fill::zeros);
        
        for (size_t i = numObjectives - 1;i < numVariables;i++)
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
        arma::Row<ElemType> value(coords.n_cols, arma::fill::ones);
        for (size_t i = 0;i < numObjectives - 1;i++)
        {
          objectives.row(i) = (1 - value % (1.0 - coords.row(i))) % (1. + G);
          value = value % coords.row(i);
        }
        objectives.row(numObjectives - 1) = (1 - value) % (1. + G);
        return objectives;    
      }
      
      // Individual Objective function.
      // Changes based on stop variable provided. 
      struct MAF1Objective
      {
        MAF1Objective(size_t stop, MAF1& maf): stop(stop), maf(maf)
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
          if(stop == 0)
          {
            return coords[0] * (1. + maf.g(coords)[0]);
          }
          typedef typename MatType::elem_type ElemType;
          ElemType value = 1.0;
          for (size_t i = 0; i < stop; i++)
          {
            value = value * coords[i];
          }

          if(stop != maf.GetNumObjectives() - 1)
          {
            value = value * (1. - coords[stop]);
          }

          value = (1.0 - value) * (1. + maf.g(coords)[0]);
          return value;
        }        

        MAF1& maf;
        size_t stop;
      };

      //! Get objective functions.
      std::tuple<MAF1Objective, MAF1Objective, MAF1Objective> GetObjectives()
      {
        return std::make_tuple(objectiveF1, objectiveF2, objectiveF3);
      }

      MAF1Objective objectiveF1;
      MAF1Objective objectiveF2;
      MAF1Objective objectiveF3;
  };
  } //namespace test
  } //namespace ens

#endif
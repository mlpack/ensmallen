/**
 * @file maf4_function.hpp
 * @author Satyam Shukla
 *
 * Implementation of the fourth Maf test.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_PROBLEMS_MAF_FOUR_FUNCTION_HPP
#define ENSMALLEN_PROBLEMS_MAF_FOUR_FUNCTION_HPP

#include "../../moead/weight_init_policies/uniform_init.hpp"

namespace ens {
namespace test {

/**
 * The MAF4 function, defined by:
 * \f[
 * x_M = [x_i, n - M + 1 <= i <= n]
 * g(x) = 100 * [|x_M| + \Sigma{i = n - M + 1}^n (x_i - 0.5)^2 - cos(20 * pi * 
 *   (x_i - 0.5))]
 * 
 * f_1(x) = a * (1 - cos(x_1 * pi * 0.5) * cos(x_2 * pi * 0.5) * ... cos(x_2 * pi * 0.5))* (1 + g(x_M)) 
 * f_2(x) = a^2 * (1 - cos(x_1 * pi * 0.5) * cos(x_2 * pi * 0.5) * ... sin(x_M-1 * pi * 0.5)) * (1 + g(x_M))
 * .
 * .
 * f_M(x) = a^M * (1 - sin(x_1 * pi * 0.5)) * (1 + g(x_M))
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
  class MAF4
  {
    private:

    // A fixed no. of Objectives and Variables(|x| = 7, M = 3).
    size_t numObjectives {3};
    size_t numVariables {12}; 
    double a;

    public:

      /**
       * Object Constructor.
       * Initializes the individual objective functions.
       *
       * @param numParetoPoint No. of pareto points in the reference front.
       * @param a The scale factor of the objectives.
       */
      MAF4(double a = 2) :
          objectiveF1(0, *this),
          objectiveF2(1, *this),
          objectiveF3(2, *this),
          a(a)
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

      //Get the scaling parameter a.
      size_t GetA()
      { return this -> a; }

      /**
       * Set the scale factor of the objectives.
       * 
       * @param a The scale factor a of the objectives.
       */
      void SetA(double a)
      { this -> a = a; }

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

        arma::Mat<ElemType> objectives(numObjectives, size(coords)[1]);
        arma::Row<ElemType> G = g(coords);
        arma::Row<ElemType> value(coords.n_cols, arma::fill::ones);
        for (size_t i = 0; i < numObjectives - 1; i++)
        {
          objectives.row(i) = (1.0 - value % 
                arma::sin(coords.row(i) * arma::datum::pi * 0.5)) % (1. + G) * 
                std::pow(a, numObjectives - i);
          value = value % arma::cos(coords.row(i) * arma::datum::pi * 0.5); 
        }
        objectives.row(numObjectives - 1) = (1 - value) % (1. + G) * 
                                    std::pow(a, 1);  
        return objectives;    
      }
      
      // Individual Objective function.
      // Changes based on stop variable provided. 
      struct MAF4Objective
      {
        MAF4Objective(size_t stop, MAF4& maf): stop(stop), maf(maf)
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

          value = std::pow(maf.GetA(), maf.GetNumObjectives() - stop) * 
              (1 - value) * (1. + maf.g(coords)[0]);

          return value;
        }        

        MAF4& maf;
        size_t stop;
      };

      // Return back a tuple of objective functions.
      std::tuple<MAF4Objective, MAF4Objective, MAF4Objective> GetObjectives()
      {
          return std::make_tuple(objectiveF1, objectiveF2, objectiveF3);
      } 

    MAF4Objective objectiveF1;
    MAF4Objective objectiveF2;
    MAF4Objective objectiveF3;
  };
  } //namespace test
  } //namespace ens

#endif

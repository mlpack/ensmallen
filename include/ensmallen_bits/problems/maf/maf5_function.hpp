/**
 * @file maf5_function.hpp
 * @author Satyam Shukla
 *
 * Implementation of the fifth MAF test.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_PROBLEMS_MAF_FIVE_FUNCTION_HPP
#define ENSMALLEN_PROBLEMS_MAF_FIVE_FUNCTION_HPP

#include "../../moead/weight_init_policies/uniform_init.hpp"

namespace ens {
namespace test {

/**
 * The MAF5 function, defined by:
 * \f[
 * x_M = [x_i, n - M + 1 <= i <= n]
 * g(x) = \Sigma{i = n - M + 1}^n (x_i - 0.5)^2
 * 
 * f_1(x) = a^M * cos(x_1^alpha * pi * 0.5) * cos(x_2^alpha * pi * 0.5) * ... cos(x_2^alpha * pi * 0.5) * (1 + g(x_M)) 
 * f_2(x) = a^M-1 * cos(x_1^alpha * pi * 0.5) * cos(x_2^alpha * pi * 0.5) * ... sin(x_M-1^alpha * pi * 0.5) * (1 + g(x_M))
 * .
 * .
 * f_M(x) = a * sin(x_1^alpha * pi * 0.5) * (1 + g(x_M))
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
 * @article{cheng2017benchmark,
 * title={A benchmark test suite for evolutionary many-objective optimization},
 * author={Cheng, Ran and Li, Miqing and Tian, Ye and Zhang, Xingyi and Yang, Shengxiang and Jin, Yaochu and Yao, Xin},
 * journal={Complex \& Intelligent Systems},
 * volume={3},
 * pages={67--81},
 * year={2017},
 * publisher={Springer}
 * }
 * @endcodes
 *
 * @tparam MatType Type of matrix to optimize.
 */
  template <typename MatType = arma::mat>
  class MAF5
  {
    private:

    // A fixed no. of Objectives and Variables(|x| = 7, M = 3).
    size_t numObjectives {3};
    size_t numVariables {12};
    size_t alpha;
    size_t a;

    public:

      /**
       * Object Constructor.
       * Initializes the individual objective functions.
       *
       * @param alpha The power which each variable is raised to.
       * @param numParetoPoint No. of pareto points in the reference front.
       * @param a The scale factor of the objectives.
       */
      MAF5(size_t alpha = 100, double a = 2) :
          alpha(alpha),
          a(a),
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

      // Get the scale factor a.
      double GetA()
      { return this -> a; }

      // Get the power alpha of each variable.
      size_t GetAlpha()
      { return this -> alpha; }

      /**
       * Set the scale factor a.
       * 
       * @param a The scale factor of the objectives.
       */
      void SetA(double a)
      { this -> a = a; }

      /**
       * Set the power of each variable alpha.
       * 
       * @param alpha The power of each variable.
       */
      void SetAlpha(size_t alpha)
      { this -> alpha = alpha; }

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
        
        for (size_t i = numObjectives - 1; i < numVariables; i++)
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
        arma::Row<ElemType> value = (1.0 + G);
        for (size_t i = 0; i < numObjectives - 1; i++)
        {
          objectives.row(i) = std::pow(a, i + 1) * arma::pow(value, 4) %  
              arma::pow(arma::sin(arma::pow(coords.row(i), alpha) * 
              arma::datum::pi * 0.5), 4);
          value = value % arma::cos(arma::pow(coords.row(i), alpha) * arma::datum::pi * 0.5); 
        }
        objectives.row(numObjectives - 1) = arma::pow(value, 4) * std::pow(a, numObjectives);
        return objectives;
      }
      
      // Individual Objective function.
      // Changes based on stop variable provided. 
      struct MAF5Objective
      {
        MAF5Objective(size_t stop, MAF5& maf): stop(stop), maf(maf)
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
          for (size_t i = 0; i < stop; i++)
          {
            value = value * std::cos(std::pow(coords[i], maf.GetAlpha()) 
                * arma::datum::pi * 0.5);
          }

          if(stop != maf.GetNumObjectives() - 1)
          {
            value = value * std::sin(std::pow(coords[stop], maf.GetAlpha()) 
                * arma::datum::pi * 0.5);
          }

          value = value * (1 + maf.g(coords)[0]);
          value = std::pow(value, 4);
          value = value * std::pow(maf.GetA(), stop + 1); 
          return value;
        }        

        MAF5& maf;
        size_t stop;
      };

      // Return back a tuple of objective functions.
      std::tuple<MAF5Objective, MAF5Objective, MAF5Objective> GetObjectives()
      {
          return std::make_tuple(objectiveF1, objectiveF2, objectiveF3);
      } 

    MAF5Objective objectiveF1;
    MAF5Objective objectiveF2;
    MAF5Objective objectiveF3;
  };
  } //namespace test
  } //namespace ens

#endif
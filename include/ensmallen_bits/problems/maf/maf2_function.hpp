/**
 * @file maf2_function.hpp
 * @author Satyam Shukla
 *
 * Implementation of the second Maf test.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_PROBLEMS_MAF_TWO_FUNCTION_HPP
#define ENSMALLEN_PROBLEMS_MAF_TWO_FUNCTION_HPP

namespace ens {
namespace test {

/**
 * The MAF2 function, defined by:
 * \f[
 * theta_M = [theta_i, n - M + 1 <= i <= n]
 * g_i(x) = \Sigma{i = M + (i - 1) * (n - M + 1) / N}^
 *                        {M - 1 + (i) * (n - M + 1) / N} (x_i - 0.5)^2 * 0.25
 * 
 * f_1(x) = cos(theta_1) * cos(theta_2) * ... cos(theta_M-1) * (1 + g_1(theta_M)) 
 * f_2(x) = cos(theta_1) * cos(theta_2) * ... sin(theta_M-1) * (1 + g_2(theta_M))
 * .
 * .
 * f_M(x) = sin(theta_1) * (1 + g_M(theta_M))
 * \f]
 *
 * Bounds of the variable space is:
 * 0 <= x_i <= 1 for i = 1,...,n.
 * 
 * Where theta_i = 0.5 * (1 + 2 * g(X_M) * x_i) / (1 + g(X_M))
 * 
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
  class MAF2
  {
    private:

    // A fixed no. of Objectives and Variables(|x| = 7, M = 3).
    size_t numObjectives {3};
    size_t numVariables {12};
    size_t numParetoPoints;

    public:

      /**
       * Object Constructor.
       * Initializes the individual objective functions.
       *
       * @param numParetoPoint No. of pareto points in the reference front.
       */
      MAF2() :
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
       * Evaluate the G(x) with the given coordinate.
       *
       * @param coords The function coordinates.
       * @return arma::Row<typename MatType::elem_type>
       */
      arma::Mat<typename MatType::elem_type> g(const MatType& coords)
      {
        size_t k = numVariables - numObjectives + 1;
        size_t c = std::floor(k / numObjectives);
        // Convenience typedef.
        typedef typename MatType::elem_type ElemType;
        
        arma::Mat<ElemType> innerSum(numObjectives, size(coords)[1], 
            arma::fill::zeros);
        
        for (size_t i = 0; i < numObjectives; i++)
        {
          size_t j = numObjectives - 1 + (i * c);
          for(; j < numVariables - 1 + (i + 1) *c && j < numObjectives; j++)
          {
            innerSum.row(i) += arma::pow((coords.row(i) - 0.5), 2) * 0.25; 
          }
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
        arma::Mat<ElemType> G = g(coords); 
        arma::Row<ElemType> value(size(coords)[1], arma::fill::ones);
        arma::Row<ElemType> theta;
        for (size_t i = 0; i < numObjectives - 1; i++)
        {
          theta = arma::datum::pi * 0.5 * ((coords.row(i) / 2) + 0.25);
          objectives.row(i) =  value %  
              arma::sin(theta) % (1.0 + G.row(numObjectives - 1 - i));
          value = value % arma::cos(theta); 
        }
        objectives.row(numObjectives - 1) = value % 
            (1.0 + G.row(0));
        return objectives;
      }
      
      // Individual Objective function.
      // Changes based on stop variable provided. 
      struct MAF2Objective
      {
        MAF2Objective(size_t stop, MAF2& maf): stop(stop), maf(maf)
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
          ElemType theta;
          arma::Col<ElemType> G = maf.g(coords).col(0);
          for (size_t i = 0; i < stop; i++)
          {
            theta = arma::datum::pi * 0.5 * ((coords[i] / 2) + 0.25); 
            value = value * std::cos(theta);
          }
	        theta = arma::datum::pi * 0.5 * ((coords[stop] / 2) + 0.25);
          if(stop != maf.numObjectives - 1)
          {
            value = value * std::sin(theta);
          }

          value = value * (1.0 + G[maf.GetNumObjectives() - 1 - stop]);
          return value;  
        }

        MAF2& maf;
        size_t stop;
      };

      // Return back a tuple of objective functions.
      std::tuple<MAF2Objective, MAF2Objective, MAF2Objective> GetObjectives()
      {
          return std::make_tuple(objectiveF1, objectiveF2, objectiveF3);
      }

    MAF2Objective objectiveF1;
    MAF2Objective objectiveF2;
    MAF2Objective objectiveF3;
  };
  } //namespace test
  } //namespace ens

#endif
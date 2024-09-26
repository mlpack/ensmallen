/**
 * @file maf6_function.hpp
 * @author Satyam Shukla
 *
 * Implementation of the sixth Maf test.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_PROBLEMS_MAF_SIX_FUNCTION_HPP
#define ENSMALLEN_PROBLEMS_MAF_SIX_FUNCTION_HPP

namespace ens {
namespace test {

/**
 * The MAF6 function, defined by:
 * \f[
 * theta_M = [theta_i, n - M + 1 <= i <= n]
 * g(x) = \Sigma{i = n - M + 1}^n (x_i - 0.5)^2
 * 
 * f_1(x) = 0.5 * cos(theta_1 * pi * 0.5) * cos(theta_2 * pi * 0.5) * ... cos(theta_2 * pi * 0.5) * (1 + g(theta_M)) 
 * f_2(x) = 0.5 * cos(theta_1 * pi * 0.5) * cos(theta_2 * pi * 0.5) * ... sin(theta_M-1 * pi * 0.5) * (1 + g(theta_M))
 * .
 * .
 * f_M(x) = 0.5 * sin(theta_1 * pi * 0.5) * (1 + g(theta_M))
 * \f]
 *
 * Bounds of the variable space is:
 * 0 <= x_i <= 1 for i = 1,...,n.
 * 
 * Where theta_i = 0.5 * (1 + 2 * g(X_M) * x_i) / (1 + g(X_M))
 * 
 * This should be optimized to x_i = 0.5 (for all x_i in X_M), at:
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
  class MAF6
  {
    private:

    // A fixed no. of Objectives and Variables(|x| = 7, M = 3).
    size_t numObjectives {3};
    size_t numVariables {12};
    size_t I;

    public:

      /**
       * Object Constructor.
       * Initializes the individual objective functions.
       *
       * @param numParetoPoint No. of pareto points in the reference front.
       * @param I The manifold dimension (zero indexed).
       */
      MAF6(size_t I = 2) :
          objectiveF1(0, *this),
          objectiveF2(1, *this),
          objectiveF3(2, *this),
          I(I)
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

      // Get the manifold dimension.
      size_t GetI()
      { return this -> I; }

      /**
       * Set the no. of pareto points.
       *
       * @param I The manifold dimension (0 indexed).
       */
      void SetI(size_t I)
      { this -> I = I; }

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
        arma::Row<ElemType> value = (1.0 + 100 * G);
        arma::Row<ElemType> theta;
        for (size_t i = 0; i < numObjectives - 1; i++)
        {
          if(i < I)
          { 
            theta = coords.row(i) * arma::datum::pi * 0.5;
          }
          else
          {
            theta = 0.25 * (1.0  + 2.0 * coords.row(i) % G) / (1.0 + G);
          }
          objectives.row(i) =  value %  
              arma::sin(theta);
          value = value % arma::cos(theta); 
        }
        objectives.row(numObjectives - 1) = value;
        return objectives;
      }
      
      // Individual Objective function.
      // Changes based on stop variable provided. 
      struct MAF6Objective
      {
        MAF6Objective(size_t stop, MAF6& maf): stop(stop), maf(maf)
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
          ElemType G = maf.g(coords)[0];
          for (size_t i = 0; i < stop; i++)
          {
            if(i < maf.GetI())
            {
             theta  = arma::datum::pi * coords[i] * 0.5;
            }
            else
            {
                theta = 0.25 * (1.0  + 2.0 * coords[i] * G) / (1.0 + G);
            }
            value = value * std::cos(theta);
          }

          if(stop < maf.GetI())
          {
            theta  = arma::datum::pi * coords[stop] * 0.5;
          }
          else
          {
            theta = 0.25 * (1.0  + 2.0 * coords[stop] * G) / (1.0 + G);
          }

          if (stop != maf.GetNumObjectives() - 1)
          {
            value = value * std::sin(theta);
          }

          value = value * (1.0 + 100 * G);
          return value;  
        }       

        MAF6& maf;
        size_t stop;
      };

      // Return back a tuple of objective functions.
      std::tuple<MAF6Objective, MAF6Objective, MAF6Objective> GetObjectives()
      {
          return std::make_tuple(objectiveF1, objectiveF2, objectiveF3);
      }

    MAF6Objective objectiveF1;
    MAF6Objective objectiveF2;
    MAF6Objective objectiveF3;
  };
  } //namespace test
  } //namespace ens

#endif
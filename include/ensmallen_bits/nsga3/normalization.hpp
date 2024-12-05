/**
 * @file normalization.hpp
 * @author Satyam Shukla
 *
 * The Optimized normalization technique as described in Investigating the 
 * normalization procedure of nsga-iii.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_NSGA3_NORMALIZATION_HPP
#define ENSMALLEN_NSGA3_NORMALIZATION_HPP

namespace ens {

/**
 * 
 * This normalization technique is an improved version of the algorithm mentioned
 * in the NSGA III paper.
 * 
 * This class solves the problem of negative intercepts and non unique hyperplane
 * by using worst point estimation as a backup value when ther are any abnormal values
 * in the nadir point estimation.
 * 
 * For more information, see the following:
 * 
 * @code
 * @inproceedings{blank2019investigating,
 * title={Investigating the normalization procedure of NSGA-III},
 * author={Blank, Julian and Deb, Kalyanmoy and Roy, Proteek Chandan},
 * booktitle={International Conference on Evolutionary Multi-Criterion Optimization},
 * pages={229--240},
 * year={2019},
 * organization={Springer}
 * }
 * @endcode
 */
template <typename MatType>
class Normalization
{
 public:

  typedef typename MatType::elem_type ElemType;

  /**
   * @param dimension The no. of elements in a single point.
   */
  Normalization(size_t dimensions = 3): 
      dimensions(dimensions),
      idealPoint(arma::Col<ElemType>(dimensions, 
          arma::fill::value(arma::datum::inf))),
      worstPoint(arma::Col<ElemType>(dimensions, 
          arma::fill::value(-1 * arma::datum::inf)))
  {/* Nothing to do here */}
  
  /**
   * @param calculatedObjectives The given population.
   * @param indexes The index of the lements of the front
   */
  void update(const std::vector<arma::Col<ElemType>>& calculatedObjectives,
              const std::vector<size_t> indexes)
  {
    // Calculate the front worst, the population worst and the ideal point.
    arma::Col<ElemType> worstOfFront(dimensions, 
        arma::fill::value(-1 * arma::datum::inf));
    arma::Col<ElemType> worstOfPop(dimensions, 
        arma::fill::value(-1 * arma::datum::inf));

    for (const arma::Col<ElemType> member: calculatedObjectives)
    {
      idealPoint = arma::min(idealPoint, member);
      worstPoint = arma::max(worstPoint, member);
      worstOfPop = arma::max(worstOfPop, member);
    }
 
    for (size_t index : indexes)
    {
      worstOfFront = arma::max(worstOfFront, calculatedObjectives[index]);
    }
    
    arma::Mat<ElemType> f(dimensions, dimensions);
    
    // Find the extremes.
    FindExtremes(calculatedObjectives, indexes, f);
    vectorizedExtremes = f;

    // Update the nadir point.
    GetNadirPoint(calculatedObjectives, indexes, worstOfPop, worstOfFront);

  }
  
  /**
   * @param calculatedObjectives The given population.
   * @param indexes The index of the lements of the front
   * @param f The matrix to store the vector extremes.
   * @param useCurrentExtremes If the previously calculaeted extremes should 
   * be considered.
   */
  void FindExtremes(const std::vector<arma::Col<ElemType>>& calculatedObjectives, 
                    const std::vector<size_t>& indexes,
                    arma::Mat<ElemType>& f,
                    bool useCurrentExtremes = true)
  {
    arma::Mat<ElemType> W(dimensions, dimensions, arma::fill::eye);
    W = W + (W == 0) * 1e6;
    arma::Mat<ElemType> vectorizedObjectives(dimensions, indexes.size());

    for (size_t i = 0; i < indexes.size(); i++)
    {
      vectorizedObjectives.col(i) = calculatedObjectives[indexes[i]];
    }

    if (useCurrentExtremes)
    {
      vectorizedObjectives = arma::join_rows(vectorizedObjectives, 
        vectorizedExtremes);
    }
    vectorizedObjectives.each_col() -= idealPoint;
    vectorizedObjectives = vectorizedObjectives + (vectorizedObjectives < 1e-3) * 0.;
    
    //Calculate ASF score and get the extreme vectors.
    arma::Mat<ElemType> asfScore(vectorizedObjectives.n_cols, dimensions);
    for(size_t i = 0; i < vectorizedObjectives.n_cols; i++)
    {
      for(size_t j = 0; j < dimensions; j++)
      {
        asfScore(i, j) = arma::max(W.col(j) % vectorizedObjectives.col(i));
      }
    }
    arma::urowvec extremes = arma::index_min(asfScore);
    for(size_t i = 0; i < dimensions; i++)
    {
      if(extremes(i) >= indexes.size())
      {
        f.col(i) = vectorizedExtremes.col(extremes(i) - indexes.size());
      }
      else
      {
        f.col(i) = calculatedObjectives[indexes[extremes(i)]];
      }
    }
  }

  /**
   * @param calculatedObjectives The given population.
   * @param indexes The index of the lements of the front
   * @param worstOfFront Worst point of the given front.
   * @param worstOfPop Worst point of the population.
   */
  void GetNadirPoint(const std::vector<arma::Col<ElemType>>& calculatedObjectives, 
                const std::vector<size_t>& indexes,
                const arma::Col<ElemType> worstOfFront,
                const arma::Col<ElemType> worstOfPop)
  {
    try
    {
      arma::Mat<ElemType> M = vectorizedExtremes;
      M.each_col() -= idealPoint;
      arma::Col<ElemType> b(dimensions, arma::fill::ones);
      arma::Col<ElemType> hyperplane = arma::solve(M.t(), b);

      if (hyperplane.has_inf() || hyperplane.has_nan() || (arma::accu(hyperplane < 0.0) > 0))
      {
        throw 1024;
      }

      arma::Col<ElemType> intercepts = 1.0 / hyperplane;
      
      if(arma::accu(arma::abs((M.t() * hyperplane) - b) > 1e-8) || 
          arma::accu(intercepts < 1e-6) > 0 || intercepts.has_inf() || 
          intercepts.has_nan())
      {
        throw 1025;
      }

      nadirPoint = idealPoint + intercepts;
      nadirPoint = nadirPoint % (nadirPoint <= worstPoint);
      nadirPoint += (nadirPoint == 0) % worstPoint;
    }
    catch(...)
    {
      nadirPoint = worstOfFront;
    }
    nadirPoint = ((nadirPoint - idealPoint) >= 1e-6) % nadirPoint + 
        ((nadirPoint - idealPoint) < 1e-6) % worstOfPop;
  }

  //! Retrieve value of dimensions.
  size_t Dimensions() const { return dimensions; }
  //! Modify value of dimensions.
  size_t& Dimensions() {return dimensions;}

  //! Retrieve value of ideal point.
  arma::Col<ElemType> IdealPoint() const {return idealPoint; }
  //! Modify value of ideal point.
  arma::Col<ElemType>& IdealPoint() {return idealPoint; }

  //! Retrieve value of worst point.
  arma::Col<ElemType> WorstPoint() const {return worstPoint; }
  //! Modify value of worst point.
  arma::Col<ElemType>& WorstPoint() {return worstPoint; }

  //! Retrieve value of nadir point.
  arma::Col<ElemType> NadirPoint() const {return nadirPoint; }
  //! Modify value of nadir point.
  arma::Col<ElemType>& NadirPoint() {return nadirPoint; }

  //! Retrieve value of extreme points.
  arma::Mat<ElemType> VectorizedExtremes() const {return vectorizedExtremes; }
  //! Modify value of extreme points.
  arma::Mat<ElemType>& VectorizedExtremes() {return vectorizedExtremes; }

 private:

  size_t dimensions;

  //The ideal point of the current population .
  //! includes previous ideal point in calculations as well.
  arma::Col<ElemType> idealPoint;

  // The worst point in the current population.
  //! includes previous worst point in calculations as well.
  arma::Col<ElemType> worstPoint;
  
  // The nadir point of the previous front.
  arma::Col<ElemType> nadirPoint;
  
  // A Matrix containing the extreme vectors as columns.
  arma::Mat<ElemType> vectorizedExtremes;
}; 
}

#endif

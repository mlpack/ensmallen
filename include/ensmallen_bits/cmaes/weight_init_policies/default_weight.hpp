/**
 * @file default_weight.hpp
 * @author John Hoang 
 * 
 * Default initialization weight
 * 
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 * 
 */
#ifndef ENSMALLEN_CMAES_DEFAULT_WEIGHT_HPP
#define ENSMALLEN_CMAES_DEFAULT_WEIGHT_HPP

namespace ens{

template<typename MatType>
class DefaultWeight{

typedef typename MatType::elem_type ElemType;
typedef typename MatTypeTraits<MatType>::BaseMatType BaseMatType; 

 public:
  /**
   * Constructor
   * 
   */
  DefaultWeight(const size_t len) : len(len)
  {
    mu = std::round(len/2);
    weights = std::log(mu + 0.5) - arma::log(arma::linspace<arma::Row<ElemType>>(0, len - 1, len) + 1.0);
    for(size_t i = mu; i < len; ++i)
    {
      weights(i) = 0;
    }
    assert(weights(mu-1) > 0);
    double sumPos = arma::accu(weights.cols(0, mu-1));
    // positive weights sum to one
    for(size_t i = 0; i < mu; ++i)
    {
      weights(i) /= sumPos;
    }
    mu_eff = 1 / arma::accu(arma::pow(weights.cols(0, mu-1), 2));
  }
  /**
   * Generate default weight for new population
   * 
   *  @tparam ElemType The type of elements in weight vector
   *  @param dim Dimension of iterate variable
   *  @param c1 
   *  @param cmu
   */
  template<typename ElemType> 
  arma::Row<ElemType> Generate(const size_t dim,
                               const double c1,
                               const double cmu)
  {
    return weights;
  }

  // Return variance-effective before the Generate function is called since c1 and cmu is 
  // calculated beforehand 
  size_t Mu_eff() const { return mu_eff; }
  size_t& Mu_eff() { return mu_eff; }

  // These functions might be unnecessary since Generate function is already return the desired weights 
  arma::Row<ElemType> Weights() const { return weights; }
  arma::Row<ElemType>& Weights() { return weights; }

  private:
    size_t len;
    size_t mu;
    size_t mu_eff;
    size_t mu_eff_negative;
    arma::Row<ElemType> weights
};

} // namespace ens

#endif
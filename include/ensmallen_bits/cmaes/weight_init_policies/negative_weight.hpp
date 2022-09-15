/**
 * @file default_weight.hpp
 * @author John Hoang 
 * 
 * Negative initialization weight
 * 
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 * 
 */
#ifndef ENSMALLEN_CMAES_NEGATIVE_WEIGHT_HPP
#define ENSMALLEN_CMAES_NEGATIVE_WEIGHT_HPP

namespace ens{

class NegativeWeight
{
 public:
  /**
   * Constructor
   *
   */
  NegativeWeight(bool test = false) : test(test)
  {
    // Doing nothing.
  }
  
  /**
   * This function will generate raw weights and mueff first.
   *
   * @param lambda The length of raw weights.
   */
  void GenerateRaw(const size_t lambda)
  {
    // Checking the length of the weights vector.
    len = lambda;
    assert(len >= 2 && "Number of weights must be >= 2");
    weights = std::log((len + 1) / 2) - 
        arma::log(arma::linspace<arma::Row<double> >(0, len - 1, len) + 1.0);
    
    assert(weights(0) > 0 && "The first weight must be >0");
    assert(weights(len - 1) <= 0 && "The last weight must be <= 0");
    // mu is expected a half of len(allias of lambda in CMAparameters class).
    mu = 0;
    for (size_t i = 0; i < len; ++i)
    {
      if (weights(i) > 0) mu++;
    }
    double sumPos = arma::accu(weights.cols(0, mu - 1));
    // Positive weights sum to one.
    for (size_t i = 0; i < mu; ++i)
    {
      weights(i) /= sumPos;
    }
    mueff = 1 / arma::accu(arma::pow(weights.cols(0, mu - 1), 2));
  }

  /**
   * Generate negative weight for new population.
   *
   * @tparam ElemType The type of elements in weight vector.
   * @param dim Dimension of iterate variable.
   * @param c1
   * @param cmu
   */
  arma::Row<double> Generate(const size_t dim,
                             const double c1,
                             const double cmu)
  {
    if (c1 > 10 * cmu) 
    {
      std::cout << "Warning: c1/cmu seems to assume a too large value for"
          << "negative weight setting" << std::endl;
    }   
    double sumNeg = std::abs(arma::accu(weights.cols(mu, len - 1)));

    const double alphaMuNegative = 1 + c1 / cmu;
    const double alphaPosdefNegative = (1 - c1 - cmu) / (dim * cmu);
    double factor = std::min(alphaMuNegative, alphaPosdefNegative); 

    for (size_t i = mu; i < len; ++i)
    {
      weights(i) *= factor;
      weights(i) /= sumNeg;
    } 

    double alphamueffNegative = 1 + 2 * NegativeEff(weights) / (mueff + 2);
    if (std::abs(arma::accu(weights.cols(mu, len-1))) >= 
        -std::abs(alphamueffNegative))
    {
      factor = abs(alphamueffNegative) / 
          std::abs(arma::accu(weights.cols(mu, len - 1)));
      if (factor < 1)
      {
        for (size_t i = mu; i < len; ++i)
        {
          weights(i) *= factor;
        }
      }
    }
    mueffNeg = NegativeEff(weights);
    if (test) 
    { 
      Checking();
    }
    return weights;
  }
  /**
   * function will check all the conditions weights vector has to stastify.
   */
  void Checking()
  {
    assert(weights(0) > 0);
    assert(weights(len - 1) < 0);
    for (size_t i = 0; i < len - 1; ++i)
    {
      assert(weights(i) > weights(i+1));
    }
    assert(mu > 0);
    assert(weights(mu - 1) > 0 && 0 >= weights(mu));
    assert(0.999 < arma::accu(weights.cols(0, mu - 1)) && 
        arma::accu(weights.cols(0, mu - 1)) < 1.001);

    double mueffChk = std::pow(arma::accu(weights.cols(0, mu - 1)), 2) / 
        arma::accu(arma::pow(weights.cols(0, mu - 1), 2));
    double muNegEffChk = std::pow(arma::accu(weights.cols(mu, len - 1)), 2) / 
        arma::accu(arma::pow(weights.cols(mu, len - 1), 2));

    assert(mueff / 1.001 < mueffChk && mueffChk < mueff * 1.001);
    assert(mueffNeg / 1.001 < muNegEffChk && muNegEffChk < mueffNeg * 1.001);
  }

  double NegativeEff(const arma::Row<double>& weights)
  {
    double sumNeg = 0.0, sumNegSquare = 0.0;
    for (size_t i = 0; i < weights.n_elem; ++i) 
    {
      if (weights(i) < 0)
      {
        sumNeg += std::abs(weights(i));
        sumNegSquare += std::pow(weights(i), 2);
      } 
    }
    return std::pow(sumNeg, 2) / sumNegSquare;
  }
  // Return variance-effective before the Generate function is called since 
  // c1 and cmu is calculated beforehand.
  double Mueff() const { return mueff; }
  double& Mueff() { return mueff; }

  // These functions might be unnecessary since Generate function is already 
  // return the desired weights.
  arma::Row<double> Weights() const { return weights; }
  arma::Row<double>& Weights() { return weights; }

 private:
  bool test; // Test mode bool variable
  size_t len; // The size of weight vector
  size_t mu; // NUmber of candidate solutions
  double mueff; // Effective of weights vector. 
  double mueffNeg; // Effective of negative weights only
  arma::Row<double> weights; // Vector stored weights in mutation stage
};

} // namespace ens

#endif
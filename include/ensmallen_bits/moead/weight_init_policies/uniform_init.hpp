/**
 * @file das_dennis_init.hpp
 * @author Nanubala Gnana Sai
 *
 * The Uniform(Das Dennis) methodology of Weight Initialization.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_MOEAD_UNIFORM_HPP
#define ENSMALLEN_MOEAD_UNIFORM_HPP

namespace ens {

class Uniform
{
 public:
  Uniform()
  {
    /* Nothing to do. */
  }

  template<typename MatType>
  MatType Generate(size_t numObjectives, size_t numPoints, double epsilon)
  {
    size_t numPartitions  = FindNumParitions(numPoints, numObjectives);
    size_t validNumPoints = FindNumUniformPoints(numPartitions, numObjectives);

    //! The requested number of points is not matching any partition number.
    if (numPoints != validNumPoints)
    {
      size_t nextValidNumPoints = FindNumUniformPoints(numPartitions + 1, numObjectives);
      std::ostringstream oss;
      oss << "DasDennis::Generate(): " << "The requested numPoints " << numPoints
          << " cannot be generated uniformly.\n " << "Either choose numPoints as "
          << validNumPoints << "(numPartition = " << numPartitions << ") or "
          << "numPoints as " << nextValidNumPoints << "(numPartition = "
          << numPartitions + 1 << ").";
      throw std::logic_error(oss.str());
    }

    return DasDennis<MatType>(numPartitions, numObjectives,
        numPoints, epsilon);
  }

 private:
   /**
    * Finds the number of points which can be sampled from the unit
    *  simplex given the number of partitions.
    */
  size_t FindNumUniformPoints(size_t numPartitions, size_t numObjectives)
  {
    auto BinomialCoefficient =
        [](size_t n, size_t k) -> double
        {
          return std::tgamma(n + 1) / (std::tgamma(k + 1) * std::tgamma(n - k + 1));
        };
    return static_cast<size_t>(BinomialCoefficient(
        numObjectives + numPartitions - 1, numPartitions));
  }

  /**
   *  Calculates the appropriate number of partitions such that, the binomial
   *  coefficient value is closest to the number of points requested.
   */
  size_t FindNumParitions(size_t numPoints, size_t numObjectives)
  {
    if (numObjectives == 1) return 0;
    // Iteratively increase numPartitions so that the binomial coefficient
    // comes near to numPoints;
    size_t numPartitions {1};
    size_t sampledNumPoints = FindNumUniformPoints(numPartitions,
        numObjectives);
    while (sampledNumPoints <= numPoints)
    {
       ++numPartitions;
       sampledNumPoints = FindNumUniformPoints(numPartitions,
          numObjectives);
    }

    return numPartitions - 1;
  }

  template<typename ElemType>
  void DasDennisRecursion(std::vector<arma::Col<ElemType>>& weightMatrix,
                          arma::Col<ElemType>& referenceDirection,
                          size_t numPartitions,
                          size_t beta,
                          size_t depth)
  {
    if (depth == referenceDirection.size() - 1)
    {
      referenceDirection(depth) =(ElemType) beta / (ElemType) numPartitions;
      weightMatrix.push_back(referenceDirection);
    }
    for (size_t i = 0; i <= beta; ++i)
    {
      referenceDirection(depth) = (ElemType) i / (ElemType) numPartitions;
      DasDennisRecursion(weightMatrix, referenceDirection, numPartitions,
          beta - i, depth + 1);
    }
  }

  template <typename MatType>
  MatType DasDennis(size_t numPartitions,
                    size_t numObjectives,
                    size_t numPoints,
                    double epsilon)
  {
    typedef typename MatType::elem_type ElemType;

    if (numPartitions == 0)
      return MatType(numObjectives, 1).fill(1 / numObjectives);

    arma::Col<ElemType> weightVector(numObjectives);
    weightVector.fill(arma::datum::nan);
    // A temporary container for weights.
    std::vector<arma::Col<ElemType>> weightMatrixContainer;
    MatType weightMatrix(numObjectives, numPoints);
    DasDennisRecursion(weightMatrixContainer, weightVector, numPartitions,
        numPartitions, 0);

    for (size_t pointIdx = 0; pointIdx < numPoints; ++pointIdx)
      weightMatrix.col(pointIdx) = std::move(weightVector[pointIdx]);

    return weightMatrix + epsilon;
  }

};

} // namespace ens

#endif
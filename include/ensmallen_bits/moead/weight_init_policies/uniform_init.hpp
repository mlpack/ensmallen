/**
 * @file uniform_init.hpp
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
    size_t numPartitions  = FindNumParitions(numObjectives, numPoints);
    size_t validNumPoints = FindNumUniformPoints(numObjectives, numPartitions);

    //! The requested number of points is not matching any partition number.
    if (numPoints != validNumPoints)
    {
      size_t nextValidNumPoints = FindNumUniformPoints(numObjectives, numPartitions + 1);
      std::ostringstream oss;
      oss << "DasDennis::Generate(): " << "The requested numPoints " << numPoints
          << " cannot be generated uniformly.\n " << "Either choose numPoints as "
          << validNumPoints << "(numPartition = " << numPartitions << ") or "
          << "numPoints as " << nextValidNumPoints << "(numPartition = "
          << numPartitions + 1 << ").";
      throw std::logic_error(oss.str());
    }

    return DasDennis<MatType>(numObjectives, numPoints,
        numPartitions, epsilon);
  }

 private:
  /**
   * Finds the number of points which can be sampled from the unit 
   * simplex given the number of partitions.
   */
  size_t FindNumUniformPoints(size_t numObjectives, size_t numPartitions)
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
  size_t FindNumParitions(size_t numObjectives, size_t numPoints)
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
       sampledNumPoints = FindNumUniformPoints(numObjectives,
          numPartitions);
    }

    return numPartitions - 1;
  }

  template<typename AuxInfoStackType,
           typename MatType>
  void DasDennisHelper(AuxInfoStackType& progressStack,
                       MatType& weights,
                       const size_t numObjectives,
                       const size_t numPoints,
                       const size_t numPartitions,
                       const double epsilon)
  {
    typedef typename MatType::elem_type ElemType;
    typedef typename arma::Row<ElemType> RowType;

    size_t counter = 0;
    const ElemType delta = 1.0 / (ElemType)numPartitions;

    while (counter < numPoints and not progressStack.empty())
    {
      MatType point{};
      size_t beta{};
      std::tie(point, beta) = progressStack.back();
      progressStack.pop_back();

      if (point.size() + 1 == numObjectives)
      {
        point.insert_rows(point.n_rows,
            RowType(1).fill(delta * static_cast<ElemType>(beta)));
        weights.col(counter) = point + epsilon;
        ++counter;
      }

      else
      {
        for (size_t i = 0; i <= beta; ++i)
        {
          MatType pointClone(point);
          pointClone.insert_rows(pointClone.n_rows,
              RowType(1).fill(delta * static_cast<ElemType>(i)));
          progressStack.push_back({pointClone, beta - i});
        }
      }
    }
  }

  template <typename MatType>
  MatType DasDennis(const size_t numObjectives,
                    const size_t numPoints,
                    const size_t numPartitions,
                    const double epsilon)
  {
    //! Holds auxillary information required for the recursion step. 
    //! More specifically, holds the current point and beta value.
    using AuxContainer = std::pair<MatType, size_t>;

    std::vector<AuxContainer> progressStack{};
    //! Init the progress stack.
    progressStack.push_back({{}, numPartitions});
    MatType weights(numObjectives, numPoints);
    weights.fill(arma::datum::nan);
    DasDennisHelper<decltype(progressStack), MatType>(
        progressStack,
        weights,
        numObjectives,
        numPoints,
        numPartitions,
        epsilon);

    return weights;
  }

};

} // namespace ens

#endif
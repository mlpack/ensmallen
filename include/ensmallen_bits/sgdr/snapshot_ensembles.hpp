/**
 * @file snapshot_ensembles.hpp
 * @author Marcus Edel
 *
 * Definition of the Snapshot ensembles technique described in:
 * "Snapshot ensembles: Train 1, get m for free" by G. Huang et al.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_SGDR_SNAPSHOT_ENSEMBLES_HPP
#define ENSMALLEN_SGDR_SNAPSHOT_ENSEMBLES_HPP

namespace ens {

/**
 * Simulate a new warm-started run/restart once a number of epochs are
 * performed. Importantly, the restarts are not performed from scratch but
 * emulated by increasing the step size while the old step size value of as an
 * initial parameter.
 *
 * For more information, please refer to:
 *
 * @code
 * @inproceedings{Huang2017,
 *   title     = {Snapshot ensembles: Train 1, get m for free},
 *   author    = {Gao Huang, Yixuan Li, Geoff Pleiss, Zhuang Liu,
 *                John E. Hopcroft, and Kilian Q. Weinberger},
 *   booktitle = {Proceedings of the International Conference on Learning
 *                Representations (ICLR)},
 *   year      = {2017},
 *   url       = {https://arxiv.org/abs/1704.00109}
 * }
 * @endcode
 */
class SnapshotEnsembles
{
 public:
  /**
   * Construct the CyclicalDecay technique a restart method, where the
   * step size decays after each batch and peridically resets to its initial
   * value.
   *
   * @param epochRestart Initial epoch where decay is applied.
   * @param multFactor Factor to increase the number of epochs before a restart.
   * @param stepSize Initial step size for each restart.
   * @param batchSize Size of each mini-batch.
   * @param numFunctions The number of separable functions (the number of
   *        predictor points).
   * @param maxIterations Maximum number of iterations allowed (0 means no
   *        limit).
   * @param snapshots Maximum number of snapshots.
   */
  SnapshotEnsembles(const size_t epochRestart,
                    const double multFactor,
                    const double stepSize,
                    const size_t maxIterations,
                    const size_t snapshots) :
    epochRestart(epochRestart),
    multFactor(multFactor),
    constStepSize(stepSize),
    nextRestart(epochRestart),
    batchRestart(0),
    epochBatches(0),
    epoch(0)
  {
    snapshotEpochs = 0;
    for (size_t i = 0, er = epochRestart, nr = nextRestart;
        i < maxIterations; ++i)
    {
      if (i > nr)
      {
        er *= multFactor;
        nr += er;
        snapshotEpochs++;
      }
    }

    snapshotEpochs = epochRestart * std::pow(multFactor,
        snapshotEpochs - snapshots + 1);
  }

  //! Get the step size.
  double StepSize() const { return constStepSize; }
  //! Modify the step size.
  double& StepSize() { return constStepSize; }

  //! Get the restart fraction.
  double EpochBatches() const { return epochBatches; }
  //! Modify the restart fraction.
  double& EpochBatches() { return epochBatches; }

  //! Get the epoch where decay is applied.
  size_t EpochRestart() const { return epochRestart; }
  //! Modify the epoch where decay is applied.
  size_t& EpochRestart() { return epochRestart; }

  //! Get the factor for modifying epochs in a restart.
  double MultFactor() const { return multFactor; }
  //! Modify the factor for modifying epochs in a restart.
  double& MultFactor() { return multFactor; }

  //! Get the next restart time.
  size_t NextRestart() const { return nextRestart; }
  //! Modify the next restart time.
  size_t& NextRestart() { return nextRestart; }

  //! Get the number of batches since the last restart.
  size_t BatchRestart() const { return batchRestart; }
  //! Modify the number of batches since the last restart.
  size_t& BatchRestart() { return batchRestart; }

  //! Get the current epoch.
  size_t Epoch() const { return epoch; }
  //! Modify the current epoch.
  size_t& Epoch() { return epoch; }

  //! Get the number of epochs needed for a new snapshot.
  size_t SnapshotEpochs() const { return snapshotEpochs; }
  //! Modify the number of epochs needed for a new snapshot.
  size_t& SnapshotEpochs() { return snapshotEpochs; }

  /**
   * The DecayPolicyType policy classes must contain an internal 'Policy'
   * template class with two template arguments: MatType and GradType.  This is
   * initialized at the start of the optimization, and holds parameters specific
   * to an individual optimization.
   */
  template<typename MatType, typename GradType>
  class Policy
  {
   public:
    /**
     * This constructor is called by the SGD Optimize() method before the start
     * of the iteration update process.
     */
    Policy(SnapshotEnsembles& parent) : parent(parent) { }

    /**
     * This function is called in each iteration after the policy update.
     *
     * @param iterate Parameters that minimize the function.
     * @param stepSize Step size to be used for the given iteration.
     * @param gradient The gradient matrix.
     */
    void Update(MatType& iterate,
                double& stepSize,
                const GradType& /* gradient */)
    {
      // Time to adjust the step size.
      if (parent.epoch >= parent.epochRestart)
      {
        // n_t = n_min^i + 0.5(n_max^i - n_min^i)(1 + cos(T_cur/T_i * pi)).
        stepSize = 0.5 * parent.constStepSize *
            (1 + cos((parent.batchRestart / parent.epochBatches)
            * arma::datum::pi));

        // Keep track of the number of batches since the last restart.
        parent.batchRestart++;
      }

      // Time to restart.
      if (parent.epoch > parent.nextRestart)
      {
        parent.batchRestart = 0;

        // Adjust the period of restarts.
        parent.epochRestart *= parent.multFactor;

        // Create a new snapshot.
        if (parent.epochRestart >= parent.snapshotEpochs)
        {
          snapshots.push_back(iterate);
        }

        // Update the time for the next restart.
        parent.nextRestart += parent.epochRestart;
      }

      parent.epoch++;
    }

    //! Get the snapshots.
    std::vector<MatType> Snapshots() const { return snapshots; }
    //! Modify the snapshots.
    std::vector<MatType>& Snapshots() { return snapshots; }

   private:
    // Reference to the instantiated parent object.
    SnapshotEnsembles& parent;
    //! Locally-stored parameter snapshots.
    std::vector<MatType> snapshots;
  };

 private:
  //! Epoch where decay is applied.
  size_t epochRestart;

  //! Parameter to increase the number of epochs before a restart.
  double multFactor;

  //! The step size for each example.
  double constStepSize;

  //! Locally-stored restart time.
  size_t nextRestart;

  //! Locally-stored number of batches since the last restart.
  size_t batchRestart;

  //! Locally-stored restart fraction.
  double epochBatches;

  //! Locally-stored epoch.
  size_t epoch;

  //! Epochs where a new snapshot is created.
  size_t snapshotEpochs;
};

} // namespace ens

#endif // ENSMALLEN_SGDR_CYCLICAL_DECAY_HPP

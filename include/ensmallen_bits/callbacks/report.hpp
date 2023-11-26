/**
 * @file report.hpp
 * @author Marcus Edel
 *
 * Implementation of a simple report callback function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_CALLBACKS_REPORT_HPP
#define ENSMALLEN_CALLBACKS_REPORT_HPP

#include <ensmallen_bits/function.hpp>
#include <iomanip>

namespace ens {

/**
 * A simple optimization report.
 */
class Report
{
 public:
  /**
   * Set up the report callback class with the given output stream.
   *
   * @param iterationsPercentageIn The number of iterations to report in
   *     percent, between [0, 1]).
   * @param outputIn Ostream which receives output from this object.
   * @param outputMatrixSizeIn The number of values to output for the function
   *     coordinates.
   */
  Report(const double iterationsPercentageIn = 0.1,
         std::ostream& outputIn = arma::get_cout_stream(),
         const size_t outputMatrixSizeIn = 4) :
      iterationsPercentage(iterationsPercentageIn),
      output(outputIn),
      outputMatrixSize(outputMatrixSizeIn),
      objective(0),
      gradientNorm(0),
      hasGradient(false),
      hasEndEpoch(false),
      gradientCalls(0),
      evaluateCalls(0),
      epochCalls(0)
  { /* Nothing to do here. */ }

  /**
   * Callback function called at the begin of the optimization process.
   *
   * @param optimizer The optimizer used to update the function.
   * @param function Function to optimize.
   * @param coordinates Starting point.
   */
  template<typename OptimizerType, typename FunctionType, typename MatType>
  void BeginOptimization(OptimizerType& /* optimizer */,
                         FunctionType& /* function */,
                         MatType& coordinates)
  {
    initialCoordinates = coordinates;
    optimizationTimer.tic();
  }

  /**
   * Callback function called at the begin of the optimization process.
   *
   * @param optimizer The optimizer used to update the function.
   * @param function Function to optimize.
   * @param coordinates Starting point.
   */
  template<typename OptimizerType, typename FunctionType, typename MatType>
  void EndOptimization(OptimizerType& optimizer,
                       FunctionType& function,
                       MatType& coordinates)
  {
    output << "Optimization Report" << std::endl;
    output << std::string(80, '-') << std::endl << std::endl;

    std::streamsize streamPrecision = output.precision(4);

    if (coordinates.n_rows > outputMatrixSize ||
        coordinates.n_cols > outputMatrixSize)
    {
      output << "Initial coordinates: " << std::endl;
      TruncatePrint(initialCoordinates, outputMatrixSize);
      output << std::endl << "Final coordinates: " << std::endl;
      TruncatePrint(coordinates, outputMatrixSize);
      output << std::endl;
    }
    else
    {
      output << "Initial Coordinates:" << std::endl << initialCoordinates.t();
      output << std::endl << "Final coordinates:" << std::endl
          << coordinates.t() << std::endl;
    }

    PrettyPrintElement("iter");
    PrettyPrintElement("loss");
    PrettyPrintElement("loss change");

    if (hasGradient)
      PrettyPrintElement("|gradient|");

    if (!stepsizes.empty())
      PrettyPrintElement("step size");

    PrettyPrintElement("total time");
    output << std::endl;

    size_t iterationStep = objectives.size() / (iterationsPercentage * 100);
    if (iterationStep <= 0)
      iterationStep = 1;

    for (size_t i = 0; i < objectives.size(); i += iterationStep)
    {
      PrettyPrintElement(i);
      PrettyPrintElement(objectives[i]);
      PrettyPrintElement(
          i > 0 ? objectives[i - iterationStep] - objectives[i] : 0);

      if (hasGradient)
        PrettyPrintElement(gradientsNorm[i]);

      if (!stepsizes.empty())
        PrettyPrintElement(stepsizes[i]);

      PrettyPrintElement(timings[i]);
      output << std::endl;
    }

    output << std::endl << std::string(80, '-') << std::endl << std::endl;
    output << "Version:" << std::endl;
    PrettyPrintElement("ensmallen:", 30);
    output << ens::version::as_string() << std::endl;
    PrettyPrintElement("armadillo:", 30);
    output << arma::arma_version::as_string() << std::endl << std::endl;

    output << "Function:" << std::endl;
    std::stringstream functionStream;

    PrintNumFunctions(function, functionStream);
    if (functionStream.rdbuf()->in_avail() > 0)
      output << functionStream.str();

    PrettyPrintElement("Coordinates rows:", 30);
    output << coordinates.n_rows << std::endl;
    PrettyPrintElement("Coordinates columns:", 30);
    output << coordinates.n_cols << std::endl;
    output << std::endl;

    // If we did not take any steps, at least fill what the initial objective
    // was.
    const bool tookStep = (objectives.size() > 0);
    if (objectives.size() == 0 && evaluateCalls > 0)
    {
      objectives.push_back(objective);
      timings.push_back(optimizationTimer.toc());
    }
    else if (evaluateCalls == 0)
    {
      // It's not entirely clear how to compute the objective (since the
      // function could implement many different ways of evaluating the
      // objective), so issue an error and return.
      output << "Objective never computed.  Did the optimization fail?"
          << std::endl;
      PrettyPrintElement("Time (in seconds):", 30);
      output << optimizationTimer.toc() << std::endl;
      return;
    }

    output << "Loss:" << std::endl;
    PrettyPrintElement("Initial", 30);
    output << objectives[0] << std::endl;
    PrettyPrintElement("Final", 30);
    output << objectives[objectives.size() - 1] << std::endl;
    PrettyPrintElement("Change", 30);
    output << objectives[0] - objectives[objectives.size() - 1] << std::endl;

    output << std::endl << "Optimizer:" << std::endl;
    std::stringstream optimizerStream;

    PrintMaxIterations(optimizer, optimizerStream);
    PrintBatchSize(optimizer, optimizerStream);
    if (functionStream.rdbuf()->in_avail() > 0)
      output << optimizerStream.str();

    PrettyPrintElement("Iterations:", 30);
    if (tookStep)
      output << objectives.size() << std::endl;
    else
      output << "0 (No steps taken! Did the optimization fail?)" << std::endl;

    if (epochCalls > 0)
    {
      PrettyPrintElement("Number of epochs:", 30);
      output << epochCalls << std::endl;
    }

    if (!stepsizes.empty())
    {
      PrettyPrintElement("Initial step size:", 30);
      output << stepsizes.front() << std::endl;

      PrettyPrintElement("Final step size:", 30);
      output << stepsizes.back() << std::endl;
    }

    if (hasGradient && gradientsNorm.size() > 0)
    {
      PrettyPrintElement("Coordinates max. norm:", 30);
      output << *std::max_element(std::begin(gradientsNorm),
          std::end(gradientsNorm)) << std::endl;
    }

    PrettyPrintElement("Evaluate calls:", 30);
    output << evaluateCalls << std::endl;

    if (hasGradient)
    {
      PrettyPrintElement("Gradient calls:", 30);
      output << gradientCalls << std::endl;
    }

    PrettyPrintElement("Time (in seconds):", 30);
    output << timings[timings.size() - 1] << std::endl;

    // Restore precision.
    output.precision(streamPrecision);
  }

  /**
   * Callback function called at the beginning of a pass over the data.
   *
   * @param optimizer The optimizer used to update the function.
   * @param function Function to optimize.
   * @param coordinates Starting point.
   * @param epoch The index of the current epoch.
   * @param objective Objective value of the current point.
   */
  template<typename OptimizerType, typename FunctionType, typename MatType>
  bool BeginEpoch(OptimizerType& /* optimizer */,
                  FunctionType& /* function */,
                  const MatType& /* coordinates */,
                  const size_t /* epoch */,
                  const double /* objective */)
  {
    epochCalls++;
    return false;
  }

  /**
   * Callback function called at the end of a pass over the data.
   *
   * @param optimizer The optimizer used to update the function.
   * @param function Function to optimize.
   * @param coordinates Starting point.
   * @param epoch The index of the current epoch.
   * @param objective Objective value of the current point.
   */
  template<typename OptimizerType, typename FunctionType, typename MatType>
  bool EndEpoch(OptimizerType& optimizer,
                FunctionType& /* function */,
                const MatType& /* coordinates */,
                const size_t /* epoch */,
                const double objective)
  {
    // In case StepTaken() has been called first we clear the existing data.
    if (!hasEndEpoch)
    {
      hasEndEpoch = true;

      objectives.clear();
      timings.clear();
      gradientsNorm.clear();
      stepsizes.clear();
    }

    objectives.push_back(objective);
    timings.push_back(optimizationTimer.toc());

    if (hasGradient)
      gradientsNorm.push_back(gradientNorm);

    SaveStepSize(optimizer);
    return false;
  }

  /**
   * Callback function called once a step is taken.
   *
   * @param optimizer The optimizer used to update the function.
   * @param function Function to optimize.
   * @param coordinates Starting point.
   * @param objective Objective value of the current point.
   */
  template<typename OptimizerType, typename FunctionType, typename MatType>
  bool StepTaken(OptimizerType& optimizer,
                 FunctionType& /* function */,
                 const MatType& /* coordinates */)
  {
    if (!hasEndEpoch)
    {
      objectives.push_back(objective);
      timings.push_back(optimizationTimer.toc());

      if (hasGradient)
        gradientsNorm.push_back(gradientNorm);

      SaveStepSize(optimizer);
    }
    return false;
  }

  /**
   * Callback function called at any call to Evaluate().
   *
   * @param optimizer The optimizer used to update the function.
   * @param function Function to optimize.
   * @param coordinates Starting point.
   * @param objectiveIn Objective value of the current point.
   */
  template<typename OptimizerType, typename FunctionType, typename MatType>
  bool Evaluate(OptimizerType& /* optimizer */,
                FunctionType& /* function */,
                const MatType& /* coordinates */,
                const double objectiveIn)
  {
    objective = objectiveIn;
    evaluateCalls++;
    return false;
  }

  /**
    * Callback function called at any call to EvaluateConstraint().
    *
    * @param optimizer The optimizer used to update the function.
    * @param function Function to optimize.
    * @param coordinates Starting point.
    * @param constraint The index of the constraint;
    * @param objectiveIn Objective value of the current point.
    */
  template<typename OptimizerType, typename FunctionType, typename MatType>
  bool EvaluateConstraint(OptimizerType& /* optimizer */,
                          FunctionType& /* function */,
                          const MatType& /* coordinates */,
                          const size_t /* constraint */,
                          const double objectiveIn)
  {
    objective += objectiveIn;
    evaluateCalls++;
    return false;
  }

  /**
   * Callback function called at any call to Gradient().
   *
   * @param optimizer The optimizer used to update the function.
   * @param function Function to optimize.
   * @param coordinates Starting point.
   * @param gradientIn Matrix that holds the gradient.
   */
  template<typename OptimizerType, typename FunctionType, typename MatType>
  bool Gradient(OptimizerType& /* optimizer */,
                FunctionType& /* function */,
                const MatType& /* coordinates */,
                const MatType& gradientIn)
  {
    hasGradient = true;
    gradientNorm = arma::norm(gradientIn);
    gradientCalls++;
    return false;
  }

  /**
   * Callback function called at any call to GradientConstraint().
   *
   * @param optimizer The optimizer used to update the function.
   * @param function Function to optimize.
   * @param coordinates Starting point.
   * @param constraint The index of the constraint;
   * @param gradient Matrix that holds the gradient;
   */
  template<typename OptimizerType, typename FunctionType, typename MatType>
  bool GradientConstraint(OptimizerType& optimizer,
                          FunctionType& function,
                          const MatType& coordinates,
                          const size_t /* constraint */,
                          const MatType& gradient)
  {
    Gradient(optimizer, function, coordinates, gradient);
    return false;
  }

 private:
  /**
   * Helper function to print the number of function to the specified output
   * stream.
   *
   * @param function The instantiated function that implements NumFunctions().
   * @param stream The output stream.
   */
  template<typename FunctionType>
  typename std::enable_if<
      traits::HasNumFunctionsSignature<FunctionType>::value, void>::type
  PrintNumFunctions(const FunctionType& function, std::stringstream& stream)
  {
    PrettyPrintElement(stream, "Number of functions:", 30);
    stream << function.NumFunctions() << std::endl;
  }

  template<typename FunctionType>
  typename std::enable_if<
      !traits::HasNumFunctionsSignature<FunctionType>::value, void>::type
  PrintNumFunctions(const FunctionType& /* function */,
                    std::stringstream& /* stream */) { }

  /**
   * Helper function to output the max-iterations to the specified output
   * stream.
   *
   * @param optimizer The instantiated optimizer that implements
   *     MaxIterations().
   * @param stream The output stream.
   */
  template<typename OptimizerType>
  typename std::enable_if<
      traits::HasMaxIterationsSignature<OptimizerType>::value, void>::type
  PrintMaxIterations(const OptimizerType& optimizer, std::stringstream& stream)
  {
    PrettyPrintElement(stream, "Maximum iterations:", 30);
    stream << optimizer.MaxIterations() << std::endl;

    PrettyPrintElement(stream, "Reached maximum iterations:", 30);
    stream << std::string(optimizer.MaxIterations() == objectives.size() ?
        "true" : "false") << std::endl;
  }

  template<typename OptimizerType>
  typename std::enable_if<
      !traits::HasMaxIterationsSignature<OptimizerType>::value, void>::type
  PrintMaxIterations(const OptimizerType& /* optimizer */,
                     std::stringstream& /* stream */) { }

  /**
   * Helper function to output the batch-size to the specified output stream.
   *
   * @param optimizer The instantiated optimizer that implements BatchSize().
   * @param stream The output stream.
   */
  template<typename OptimizerType>
  typename std::enable_if<traits::HasBatchSizeSignature<OptimizerType>::value,
      void>::type
  PrintBatchSize(const OptimizerType& optimizer, std::stringstream& stream)
  {
    PrettyPrintElement(stream, "Batch size:", 30);
    stream << optimizer.BatchSize() << std::endl;
  }

  template<typename OptimizerType>
  typename std::enable_if<!traits::HasBatchSizeSignature<OptimizerType>::value,
      void>::type
  PrintBatchSize(const OptimizerType& /* optimizer */,
                 std::stringstream& /* stream */) { }

  /**
   * Output formatted data.
   *
   * @param out Output stream.
   * @param data The data to print on the given stream.
   * @param width The width of the the formatted output data.
   */
  template<typename T>
  void PrettyPrintElement(std::ostream& out,
                          const T& data,
                          const size_t width = 14)
  {
    out << std::left << std::setw(width) << std::setfill(' ')
        << std::setprecision(3) << data;
  }

  /**
   * Output formatted data.
   *
   * @param data The data to print on the given stream.
   * @param width The width of the the formatted output data.
   */
  template<typename T>
  void PrettyPrintElement(const T& data, const size_t width = 14)
  {
    PrettyPrintElement(output, data, width);
  }

  /**
   * Outputs the given matrix in a truncated format. For example, the matrix:
   *
   * 1 2 3 4 5
   * 6 7 8 9 10
   * 11 12 13 14
   * 15 16 17 18
   *
   * will be truncated to:
   *
   * 1 2 ... 5
   * 6 7 ... 10
   * ...
   * 15 16 ... 18
   *
   * @param data The data to print on the given stream in a truncated format.
   * @param size The number of elements per column/row.
   */
  template<typename T>
  void TruncatePrint(const T& data, const size_t size)
  {
    // We can't directly output the result of submat or use .print, because
    // both introduce a new line at the end, so we iterate over the elements.
    for (size_t c = 0, n = 0; c < data.n_cols; ++c)
    {
      // Skip to the last column.
      if (c >= (size - 1))
      {
        output << "..." << std::endl;
        n = (data.n_cols - 2) * data.n_rows - 1;
      }

      for (size_t r = 0; r < data.n_rows; ++r)
      {
        // Check if need to skip to the last row.
        if (r < (size - 1))
        {
          output << std::fixed;

          // Add space for positive value, to align with negative values.
          if (data(n) >= 0)
            output << " ";

          output << data(n++) << " ";
        }
        else
        {
          n = (c + 1) * data.n_rows - 1;
          output << " ... " << data(n) << std::endl;
          break;
        }
      }

      if (c >= (size - 1))
        break;
    }
  }

  /**
   * Helper function to store the step-size.
   *
   * @param optimizer The instantiated optimzer that implements StepSize().
   */
  template<typename OptimizerType>
  typename std::enable_if<traits::HasStepSizeSignature<OptimizerType>::value,
      void>::type
  SaveStepSize(const OptimizerType& optimizer)
  {
    stepsizes.push_back(optimizer.StepSize());
  }

  template<typename OptimizerType>
  typename std::enable_if<!traits::HasStepSizeSignature<OptimizerType>::value,
      void>::type
  SaveStepSize(const OptimizerType& /* optimizer */) { }

  //! The number of iterations to print in percent.
  double iterationsPercentage;

  //! The output stream that all data is to be sent to; example: std::cout.
  std::ostream& output;

  //! The number of values to print for the function coordinates.
  size_t outputMatrixSize;
  //! The initial coordinates.
  arma::mat initialCoordinates;

  //! Gradient norm storage.
  std::vector<double> gradientsNorm;

  //! Objective storage.
  std::vector<double> objectives;

  //! Timing storage.
  std::vector<double> timings;

  //! Step-size storage.
  std::vector<double> stepsizes;

  //! Objective over the current epoch.
  double objective;

  //! Locally-stored gradient norm for a single step.
  double gradientNorm;

  //! Whether Gradient() was called.
  bool hasGradient;

  //! Whether EndEpoch() was called.
  bool hasEndEpoch;

  //! The number of Gradient() calls.
  size_t gradientCalls;

  //! The number of Evaluate() calls.
  size_t evaluateCalls;

  //! The number of BeginEpoch() calls.
  size_t epochCalls;

  //! Locally-stored optimization step timer object.
  arma::wall_clock optimizationTimer;
};

} // namespace ens

#endif

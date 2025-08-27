Callbacks in ensmallen are methods that are called at various stages of the
optimization process.  These can be used to print information about the optimization, modify behavior of the optimization, or a wide range of other possibilities.  Some examples of what callbacks can be used for include:

* Changing the learning rate.
* Printing the current objective.
* Sending a message when the optimization hits a specific state such us a
  minimal objective.

Callbacks can be passed as an argument to the `Optimize()` function:

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();

MomentumSGD optimizer(0.01, 32, 100000, 1e-5, true, MomentumUpdate(0.5));

// Pass the built-in *PrintLoss* callback as the last argument to the
// *Optimize()* function.
optimizer.Optimize(f, coordinates, PrintLoss());
```

</details>

Passing multiple callbacks is just the same as passing a single callback:

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();

MomentumSGD optimizer(0.01, 32, 100000, 1e-5, true, MomentumUpdate(0.5));

// Pass the built-in *PrintLoss* and *EarlyStopAtMinLoss* callback as the last
// argument to the *Optimize()* function.
optimizer.Optimize(f, coordinates, PrintLoss(), EarlyStopAtMinLoss());
```

</details>

It is also possible to pass a callback instantiation that allows accessing of
internal callback parameters at a later state:

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();

MomentumSGD optimizer(0.01, 32, 100000, 1e-5, true, MomentumUpdate(0.5));

// Create an instantiation of the built-in *StoreBestCoordinates* callback,
// which will store the best objective and the corresponding model parameter
// that can be accessed later.
StoreBestCoordinates<> callback;

// Pass an instantiation of the built-in *StoreBestCoordinates* callback as the
// last argument to the *Optimize()* function.
optimizer.Optimize(f, coordinates, callback);

// Print the minimum objective that is stored inside the *StoreBestCoordinates*
// callback that was passed to the *Optimize()* call.
std::cout << callback.BestObjective() << std::endl;
```

</details>

Numerous implemented and ready-to-use callbacks are included with ensmallen, and
it is also easy to write a custom callback.

 * [`EarlyStopAtMinLoss`](#earlystopatminloss): stop the optimization if no
   improvement has been made
 * [`GradClipByNorm`](#gradclipbynorm): reduce the norm of the gradient to
   prevent the exploding gradient problem
 * [`GradClipByValue`](#gradclipbyvalue): clip the gradient to specified minimum
   and maximum values
 * [`PrintLoss`](#printloss): print the objective at each iteration to a
   specified stream
 * [`ProgressBar`](#progressbar): print a progress bar to the screen at each
   iteration
 * [`Report`](#report): print a report at the end of optimization
 * [`StoreBestCoordinates`](#storebestcoordinates): store the coordinates that
   give the best objective value at the end of an epoch
 * [`TimerStop`](#timerstop): stop the optimization after a given amount of time

A [guide for implementing custom callbacks](#custom-callbacks) is below, and a
few [example custom callbacks](#custom-callback-examples) are given too.

## Built-in Callbacks

### EarlyStopAtMinLoss

Stops the optimization process if the loss stops decreasing or no improvement
has been made.

#### Constructors

 * `EarlyStopAtMinLoss()`
 * `EarlyStopAtMinLoss(`_`patience`_`)`
 * `EarlyStopAtMinLoss(`_`func`_`)`
 * `EarlyStopAtMinLoss(`_`func`_`,`_`patience`_`)`

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `size_t` | **`patience`** | The number of epochs to wait after the minimum loss has been reached. | `10` |
| `std::function<double(const arma::mat&)>` | **`func`** | A callback to return immediate loss evaluated by the function. | |

Note that for the `func` argument above, if a
[different matrix type](#alternate-matrix-types) is desired, instead of using
the class `EarlyStopAtMinLoss`, the class `EarlyStopAtMinLossType<MatType>`
should be used.

#### Examples:

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
AdaDelta optimizer(1.0, 1, 0.99, 1e-8, 1000, 1e-9, true);

RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();
optimizer.Optimize(f, coordinates, EarlyStopAtMinLoss());
```
Another example of using lambda in the constructor.

```c++
// Generate random training data and labels.
arma::mat trainingData(5, 100, arma::fill::randu);
arma::Row<size_t> trainingLabels =
    arma::randi<arma::Row<size_t>>(100, arma::distr_param(0, 1));
// Generate a validation set.
arma::mat validationData(5, 100, arma::fill::randu);
arma::Row<size_t> validationLabels =
    arma::randi<arma::Row<size_t>>(100, arma::distr_param(0, 1));

// Create a LogisticRegressionFunction for both the training and validation data.
LogisticRegressionFunction lrfTrain(trainingData, trainingLabels);
LogisticRegressionFunction lrfValidation(validationData, validationLabels);

// Create a callback that will terminate when the validation loss starts to
// increase.
EarlyStopAtMinLoss cb(
    [&](const arma::mat& coordinates)
    {
      // You could also, e.g., print the validation loss here to watch it converge.
      return lrfValidation.Evaluate(coordinates);
    });

arma::mat coordinates = lrfTrain.GetInitialPoint();
SMORMS3 smorms3;
smorms3.Optimize(lrfTrain, coordinates, cb);
```

</details>

### Gradient Clipping

One challenge in optimization is dealing with "exploding gradients", where large
parameter gradients can cause the optimizer to make excessively large updates,
potentially pushing the model into regions of high loss or causing numerical
instability. This can happen due to:

* A high learning rate, leading to large gradient updates.
* Poorly scaled datasets, resulting in significant variance between data points.
* A loss function that generates disproportionately large error values.

Common solutions for this problem are:

#### GradClipByNorm

In this method, the solution is to change the derivative
of the error before applying the update step.  One option is to clip the norm
`||g||` of the gradient `g` before a parameter update. So given the gradient,
and a maximum norm value, the callback normalizes the gradient so that its
L2-norm is less than or equal to the given maximum norm value.

#### Constructors

 * `GradClipByNorm(`_`maxNorm`_`)`

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `double` | **`maxNorm`** | The maximum clipping value. | |

#### Examples:

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
AdaDelta optimizer(1.0, 1, 0.99, 1e-8, 1000, 1e-9, true);

RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();
optimizer.Optimize(f, coordinates, GradClipByNorm(0.3));
```

#### GradClipByValue

In this method, the solution is to change the derivative
of the error before applying the update step.  One option is to clip the
parameter gradient element-wise before a parameter update.

#### Constructors

 * `GradClipByValue(`_`min, max`_`)`

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `double` | **`min`** | The minimum value to clip to. | |
| `double` | **`max`** | The maximum value to clip to. | |

#### Examples:

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
AdaDelta optimizer(1.0, 1, 0.99, 1e-8, 1000, 1e-9, true);

RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();
optimizer.Optimize(f, coordinates, GradClipByValue(0, 1.3));
```

### PrintLoss

Callback that prints loss to stdout or a specified output stream.

#### Constructors

 * `PrintLoss()`
 * `PrintLoss(`_`output`_`)`

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `std::ostream` | **`output`** | Ostream which receives output from this object. | `stdout` |

#### Examples:

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
AdaDelta optimizer(1.0, 1, 0.99, 1e-8, 1000, 1e-9, true);

RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();
optimizer.Optimize(f, coordinates, PrintLoss());
```

</details>

### ProgressBar

Callback that prints a progress bar to stdout or a specified output stream.

#### Constructors

 * `ProgressBar()`
 * `ProgressBar(`_`width`_`)`
 * `ProgressBar(`_`width, output`_`)`

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `size_t` | **`width`** | Width of the bar. | `70` |
| `std::ostream` | **`output`** | Ostream which receives output from this object. | `stdout` |

#### Examples:

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
AdaDelta optimizer(1.0, 1, 0.99, 1e-8, 1000, 1e-9, true);

RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();
optimizer.Optimize(f, coordinates, ProgressBar());
```

</details>

### Report

Callback that prints a optimizer report to stdout or a specified output stream.

#### Constructors

 * `Report()`
 * `Report(`_`iterationsPercentage`_`)`
 * `Report(`_`iterationsPercentage, output`_`)`
 * `Report(`_`iterationsPercentage, output, outputMatrixSize`_`)`

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `double` | **`iterationsPercentage`** | The number of iterations to report in percent, between [0, 1]. | `0.1` |
| `std::ostream` | **`output`** | Ostream which receives output from this object. | `stdout` |
| `size_t` | **`outputMatrixSize`** | The number of values to output for the function coordinates. | `4` |

#### Examples:

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
AdaDelta optimizer(1.0, 1, 0.99, 1e-8, 1000, 1e-9, true);

RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();
optimizer.Optimize(f, coordinates, Report(0.1));
```

<details open>
<summary>Click to collapse/expand example output.
</summary>

```
Optimization Report
--------------------------------------------------------------------------------

Initial Coordinates:
  -1.2000   1.0000

Final coordinates:
  -1.0490   1.1070

iter          loss          loss change   |gradient|    step size     total time
0             24.2          0             233           1             4.27e-05
100           8.6           15.6          104           1             0.000215
200           5.26          3.35          48.7          1             0.000373
300           4.49          0.767         23.4          1             0.000533
400           4.31          0.181         11.3          1             0.000689
500           4.27          0.0431        5.4           1             0.000846
600           4.26          0.012         2.86          1             0.00101
700           4.25          0.00734       2.09          1             0.00117
800           4.24          0.00971       1.95          1             0.00132
900           4.22          0.0146        1.91          1             0.00148

--------------------------------------------------------------------------------

Version:
ensmallen:                    2.13.0 (Automatically Automated Automation)
armadillo:                    9.900.1 (Nocturnal Misbehaviour)

Function:
Number of functions:          1
Coordinates rows:             2
Coordinates columns:          1

Loss:
Initial                       24.2
Final                         4.2
Change                        20

Optimizer:
Maximum iterations:           1000
Reached maximum iterations:   true
Batchsize:                    1
Iterations:                   1000
Number of epochs:             1001
Initial step size:            1
Final step size:              1
Coordinates max. norm:        233
Evaluate calls:               1000
Gradient calls:               1000
Time (in seconds):            0.00163
```

### StoreBestCoordinates

Callback that stores the model parameter after every epoch if the objective
decreased.

#### Constructors

 * `StoreBestCoordinates<`_`ModelMatType`_`>()`

The _`ModelMatType`_ template parameter refers to the matrix type of the model
parameter.

#### Attributes

The stored model parameter can be accessed via the member method
`BestCoordinates()` and the best objective via `BestObjective()`.

#### Examples:


<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
AdaDelta optimizer(1.0, 1, 0.99, 1e-8, 1000, 1e-9, true);

RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();

StoreBestCoordinates<arma::mat> cb;
optimizer.Optimize(f, coordinates, cb);

std::cout << "The optimized model found by AdaDelta has the "
      << "parameters " << cb.BestCoordinates();
```

</details>

### TimerStop

Callback that stops optimization after a certain amount of time has elapsed.

#### Constructors

 * `TimerStop(`_`seconds`_`)`

#### Examples:

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
AdaDelta optimizer(1.0, 1, 0.99, 1e-8, 1000, 1e-9, true);

RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();

// Limit optimization to 15 seconds.
optimizer.Optimize(f, coordinates, TimerStop(15));
```

</details>

## Custom Callbacks

Custom callbacks can be easily implemented by creating a class and simply
implementing functions for each individual callback that you are interested in
handling, plus any other functionality you might need (e.g. constructors,
accessors).

Thus, when writing a custom callback, start with an empty class like this:

```c++
class CustomCallback
{
 public:
  // Add individual callback handlers that you are interested in handling!
};
```

and add any of the individual callback handler functions described below.

### BeginOptimization

Called at the beginning of the optimization process.  Add this function to your
callback class with your desired implementation:

```c++
  template<typename OptimizerType,
           typename FunctionType,
           typename MatType>
  void BeginOptimization(OptimizerType& optimizer,
                         FunctionType& function,
                         MatType& coordinates);
```

 * `optimizer`: the actual object on which `Optimize()` was called.
 * `function`: the function to be optimized (e.g. the first argument given to
   `optimizer.Optimize()`.
 * `coordinates`: the current coordinates for optimization; since optimization
   is just beginning, this is the exact same matrix given to `Optimize()`.

### EndOptimization

Called at the end of the optimization process.  Add this function to your
callback class with your desired implementation:

```c++
  template<typename OptimizerType,
           typename FunctionType,
           typename MatType>
  void EndOptimization(OptimizerType& optimizer,
                       FunctionType& function,
                       MatType& coordinates);
```

 * `optimizer`: the actual object on which `Optimize()` was called.
 * `function`: the function that has been optimized (e.g. the first argument
   given to `optimizer.Optimize()`.
 * `coordinates`: the final coordinates for optimization; since optimization is
   ending, these are the same values that will be in the resulting matrix after
   `Optimize()` finishes.

### Evaluate

Called after any call to `Evaluate()` or `EvaluateWithGradient()`.  Add this
function to your callback class with your desired implementation:

```c++
  template<typename OptimizerType,
           typename FunctionType,
           typename MatType>
  bool Evaluate(OptimizerType& optimizer,
                FunctionType& function,
                const MatType& coordinates,
                const double objective);
```

 * `optimizer`: the actual object on which `Optimize()` was called.
 * `function`: the function that is being optimized (e.g. the first argument
   given to `optimizer.Optimize()`.
 * `coordinates`: the coordinates with which `function.Evaluate()` was called.
 * `objective`: the result of `function.Evaluate(coordinates)`.

If the callback returns `true`, the optimization will be terminated.

### EvaluateConstraint

Called after any call to `EvaluateConstraint()`, for
[constrained functions](#constrained-functions).  Add this function to your
callback class with your desired implementation:

```c++
  template<typename OptimizerType,
           typename FunctionType,
           typename MatType>
  bool EvaluateConstraint(OptimizerType& optimizer,
                          FunctionType& function,
                          const MatType& coordinates,
                          const size_t constraintIndex,
                          const double constraintValue);
```

 * `optimizer`: the actual object on which `Optimize()` was called.
 * `function`: the function that is being optimized (e.g. the first argument
   given to `optimizer.Optimize()`.
 * `coordinates`: the coordinates with which `function.EvaluateConstraint()` was
   called.
 * `constraintIndex`: the index of the constraint that was evaluated
 * `constraintValue`: the result of
   `function.EvaluateConstraint(coordinates, constraintIndex)`.

If the callback returns `true`, the optimization will be terminated.

### Gradient

Called after any call to `Gradient()` or `EvaluateWithGradient()`.  Add this
function to your callback class with your desired implementation:

```c++
  template<typename OptimizerType,
           typename FunctionType,
           typename MatType,
           typename GradType>
  bool Gradient(OptimizerType& optimizer,
                FunctionType& function,
                const MatType& coordinates,
                GradType& gradient);
```

 * `optimizer`: the actual object on which `Optimize()` was called.
 * `function`: the function that is being optimized (e.g. the first argument
   given to `optimizer.Optimize()`.
 * `coordinates`: the coordinates with which `function.Gradient()` was called.
 * `gradient`: the computed gradient (can be modified!).

If the callback returns `true`, the optimization will be terminated.

### GradientConstraint

Called after any call to `GradientConstraint()` for
[constrained functions](#constrained-functions).  Add this function to your
callback class with your desired implementation:

```c++
  template<typename OptimizerType,
           typename FunctionType,
           typename MatType,
           typename GradType>
  bool GradientConstraint(OptimizerType& optimizer,
                          FunctionType& function,
                          const MatType& coordinates,
                          const size_t constraintIndex,
                          GradType& constraintGradient);
```

 * `optimizer`: the actual object on which `Optimize()` was called.
 * `function`: the function that is being optimized (e.g. the first argument
   given to `optimizer.Optimize()`.
 * `coordinates`: the coordinates with which `function.GradientConstraint()` was
   called.
 * `constraintIndex`: the index of the constraint whose gradient was computed.
 * `constraintGradient`: the computed result of
   `function.GradientConstraint()`.

If the callback returns `true`, the optimization will be terminated.

### BeginEpoch

Called at the beginning of a pass over the data, for
[separable functions](#separable-functions). The objective may be exact or
an estimate depending on the optimizer's `ExactObjective()` value.  Add this
function to your callback class with your desired implementation:

```c++
  template<typename OptimizerType,
           typename FunctionType,
           typename MatType>
  bool BeginEpoch(OptimizerType& optimizer,
                  FunctionType& function,
                  const MatType& coordinates,
                  const size_t epoch,
                  const double objective);
```

 * `optimizer`: the actual object on which `Optimize()` was called.
 * `function`: the function that is being optimized (e.g. the first argument
   given to `optimizer.Optimize()`.
 * `coordinates`: the coordinates at the start of the epoch.
 * `epoch`: the epoch number.
 * `objective`: the exact or approximate objective at the end of the previous
   epoch.

If the callback returns `true`, the optimization will be terminated.

### EndEpoch

Called at the end of a pass over the data, for
[separable functions](#separable-functions). The objective may be exact or an
estimate depending on the optimizer's `ExactObjective()` value.  Add this
function to your callback class with your desired implementation:

```c++
  template<typename OptimizerType,
           typename FunctionType,
           typename MatType>
  bool EndEpoch(OptimizerType& optimizer,
                FunctionType& function,
                const MatType& coordinates,
                const size_t epoch,
                const double objective);
```

 * `optimizer`: the actual object on which `Optimize()` was called.
 * `function`: the function that is being optimized (e.g. the first argument
   given to `optimizer.Optimize()`.
 * `coordinates`: the coordinates at the end of the epoch.
 * `epoch`: the epoch number.
 * `objective`: the exact or approximate objective at the end of the epoch.

If the callback returns `true`, the optimization will be terminated.

### StepTaken

Called after the optimizer has taken any step that modifies the coordinates.
Add this function to your callback class with your desired implementation:

```c++
  template<typename OptimizerType,
           typename FunctionType,
           typename MatType>
  bool StepTaken(OptimizerType& optimizer,
                FunctionType& function,
                MatType& coordinates);
```

 * `optimizer`: the actual object on which `Optimize()` was called.
 * `function`: the function that is being optimized (e.g. the first argument
   given to `optimizer.Optimize()`.
 * `coordinates`: the coordinates after the step (can be modified!).

If the callback returns `true`, the optimization will be terminated.  Note that
changing the `coordinates` matrix may cause strange behavior for certain
optimizers---your mileage may vary!

### GenerationalStepTaken

Called after the evolution of a single generation, for
[multi-objective functions](#multi-objective-functions).  Add this function to
your callback class with your desired implementation:

```c++
  template<typename OptimizerType,
           typename... FunctionTypes,
           typename MatType,
           typename ObjectivesVecType,
           typename IndicesType>
  bool GenerationalStepTaken(OptimizerType& optimizer,
                             std::tuple<FunctionTypes...>& functions,
                             MatType& coordinates,
                             ObjectivesVecType& objectives,
                             IndicesType& frontIndices);
```

 * `optimizer`: the actual object on which `Optimize()` was called.
 * `functions`: the functions that are being optimized (e.g. the first argument
   given to `optimizer.Optimize()`.
 * `coordinates`: the coordinates after taking the step.
 * `objectives`: a vector of column vectors indicating the objective for each
   element in the population on each objective function.
 * `frontIndices`: indices of the population that are on the Pareto front.

If the callback returns `true`, the optimization will be terminated.

## Custom Callback Examples

### Learning rate scheduling

Setting the learning rate is crucially important when training because it
controls both the speed of convergence and the ultimate performance of the
model. One of the simplest learning rate strategies is to have a fixed learning
rate throughout the training process. Choosing a small learning rate allows the
optimizer to find good solutions, but this comes at the expense of limiting the
initial speed of convergence. To overcome this tradeoff, changing the learning
rate as more epochs have passed is commonly done in model training. The
`Evaluate` method in combination with the ``StepSize`` method of the optimizer
can be used to update the variables.

Example code showing how to implement a custom callback to change the learning
rate is given below.

<details>
<summary>Click to collapse/expand example code.
</summary>

```c++
class ExponentialDecay
{
  // Set up the exponential decay learning rate scheduler with the user
  // specified decay value.
  ExponentialDecay(const double decay) : decay(decay), learningRate(0) { }


  // Callback function called at the start of the optimization process.
  // In this example we will use this to save the initial learning rate.
  template<typename OptimizerType, typename FunctionType, typename MatType>
  void BeginOptimization(OptimizerType& /* optimizer */,
                         FunctionType& /* function */,
                         MatType& /* coordinates */)
  {
    // Save the initial learning rate.
    learningRate = optimizer.StepSize();
  }

  // Callback function called at the end of a pass over the data. We are only
  // interested in the current epoch and the optimizer, we ignore the rest.
  template<typename OptimizerType, typename FunctionType, typename MatType>
  bool EndEpoch(OptimizerType& optimizer,
                FunctionType& /* function */,
                const MatType& /* coordinates */,
                const size_t epoch,
                const double objective)
  {
    // Update the learning rate.
    optimizer.StepSize() = learningRate * (1.0 - std::pow(decay,
        (double) epoch));

    // Do not terminate the optimization.
    return false;
  }

  double learningRate;
};

int main()
{
  // First, generate some random data, with 10000 points and 10 dimensions.
  // This data has no pattern and as such will make a model that's not very
  // useful---but the purpose here is just demonstration. :)
  //
  // For a more "real world" situation, load a dataset from file using X.load()
  // and y.load() (but make sure the matrix is column-major, so that each
  // observation/data point corresponds to a *column*, *not* a row.
  arma::mat data(10, 10000, arma::fill::randn);
  arma::rowvec responses(10000, arma::fill::randn);

  // Create a starting point for our optimization randomly.  The model has 10
  // parameters, so the shape is 10x1.
  arma::mat startingPoint(10, 1, arma::fill::randn);

  // Construct the objective function.
  LinearRegressionFunction lrf(data, responses);
  arma::mat lrfParams(startingPoint);

  // Create the StandardSGD optimizer with specified parameters.
  // The ens::StandardSGD type can be replaced with any ensmallen optimizer
  //that can handle differentiable functions.
  StandardSGD optimizer(0.001, 1, 0, 1e-15, true);

  // Use the StandardSGD optimizer with specified parameters to minimize the
  // LinearRegressionFunction and pass the *exponential decay*
  // callback function from above.
  optimizer.Optimize(lrf, lrfParams, ExponentialDecay(0.01));

  // Print the trained model parameter.
  std::cout << lrfParams.t();
}
```

</details>

### Early stopping at minimum loss

Early stopping is a technique for controlling overfitting in machine learning
models, especially neural networks, by stopping the optimization process before
the model has trained for the maximum number of iterations.

Example code showing how to implement a custom callback to stop the optimization
when the minimum of loss has been reached is given below.

<details>
<summary>Click to collapse/expand example code.
</summary>

```c++
#include <ensmallen.hpp>

// This class implements early stopping at minimum loss callback function to
// terminate the optimization process early if the loss stops decreasing.
class EarlyStop
{
 public:
  // Set up the early stop at min loss class, which keeps track of the minimum
  // loss.
  EarlyStop() : bestObjective(std::numeric_limits<double>::max()) { }

  // Callback function called at the end of a pass over the data, which provides
  // the current objective. We are only interested in the objective and ignore
  // the rest.
  template<typename OptimizerType, typename FunctionType, typename MatType>
  bool EndEpoch(OptimizerType& /* optimizer */,
                FunctionType& /* function */,
                const MatType& /* coordinates */,
                const size_t /* epoch */,
                const double objective)
  {
    // Check if the given objective is lower as the previous objective.
    if (objective < bestObjective)
    {
      // Update the local objective.
      bestObjective = objective;
    }
    else
    {
      // Stop the optimization process.
      return true;
    }

    // Do not stop the optimization process.
    return false;
  }

  // Locally-stored best objective.
  double bestObjective;
};

int main()
{
  // First, generate some random data, with 10000 points and 10 dimensions.
  // This data has no pattern and as such will make a model that's not very
  // useful---but the purpose here is just demonstration. :)
  //
  // For a more "real world" situation, load a dataset from file using X.load()
  // and y.load() (but make sure the matrix is column-major, so that each
  // observation/data point corresponds to a *column*, *not* a row.
  arma::mat data(10, 10000, arma::fill::randn);
  arma::rowvec responses(10000, arma::fill::randn);

  // Create a starting point for our optimization randomly.  The model has 10
  // parameters, so the shape is 10x1.
  arma::mat startingPoint(10, 1, arma::fill::randn);

  // Construct the objective function.
  LinearRegressionFunction lrf(data, responses);
  arma::mat lrfParams(startingPoint);

  // Create the L_BFGS optimizer with default parameters.
  // The ens::L_BFGS type can be replaced with any ensmallen optimizer that can
  // handle differentiable functions.
  ens::L_BFGS lbfgs;

  // Use the L_BFGS optimizer with default parameters to minimize the
  // LinearRegressionFunction and pass the *early stopping at minimum loss*
  // callback function from above.
  lbfgs.Optimize(lrf, lrfParams, EarlyStop());

  // Print the trained model parameter.
  std::cout << lrfParams.t();
}
```

</details>

Note that we have simply passed an instantiation of `EarlyStop` the
rest is handled inside the optimizer.

ensmallen provides a more complete and general implementation of a
[early stopping](#EarlyStopAtMinLoss) at minimum loss callback function.

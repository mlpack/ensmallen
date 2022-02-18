Callbacks in ensmallen are methods that are called at various states during the
optimization process, which can be used to implement and control behaviors such
as:

* Changing the learning rate.
* Printing of the current objective.
* Sending a message when the optimization hits a specific state such us a minimal objective.

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

### GradClipByNorm

One difficulty with optimization is that large parameter gradients can lead an
optimizer to update the parameters strongly into a region where the loss
function is much greater, effectively undoing much of the work done to get to
the current solution. Such large updates during the optimization can cause a
numerical overflow or underflow, often referred to as "exploding gradients". The
exploding gradient problem can be caused by: Choosing the wrong learning rate
which leads to huge updates in the gradients.  Failing to scale a data set
leading to very large differences between data points.  Applying a loss function
that computes very large error values.

A common answer to the exploding gradients problem is to change the derivative
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

### GradClipByValue

One difficulty with optimization is that large parameter gradients can lead an
optimizer to update the parameters strongly into a region where the loss
function is much greater, effectively undoing much of the work done to get to
the current solution. Such large updates during the optimization can cause a
numerical overflow or underflow, often referred to as "exploding gradients". The
exploding gradient problem can be caused by: Choosing the wrong learning rate
which leads to huge updates in the gradients.  Failing to scale a data set
leading to very large differences between data points.  Applying a loss function
that computes very large error values.

A common answer to the exploding gradients problem is to change the derivative
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

## Callback States

Callbacks are called at several states during the optimization process:

* At the beginning and end of the optimization process.
* After any call to `Evaluate()` and `EvaluateConstraint`.
* After any call to `Gradient()` and `GradientConstraint`.
* At the start and end of an epoch.

Each callback provides optimization relevant information that can be accessed or
modified.

### BeginOptimization

Called at the beginning of the optimization process.

 * `BeginOptimization(`_`optimizer, function, coordinates`_`)`

#### Attributes

| **type** | **name** | **description** |
|----------|----------|-----------------|
| `OptimizerType` | **`optimizer`** | The optimizer used to update the function. |
| `FunctionType` | **`function`** | The function to be optimized. |
| `MatType` | **`coordinates`** | The current function parameter. |

### EndOptimization

Called at the end of the optimization process.

 * `EndOptimization(`_`optimizer, function, coordinates`_`)`

#### Attributes

| **type** | **name** | **description** |
|----------|----------|-----------------|
| `OptimizerType` | **`optimizer`** | The optimizer used to update the function. |
| `FunctionType` | **`function`** | The function to be optimized. |
| `MatType` | **`coordinates`** | The current function parameter. |

### Evaluate

Called after any call to `Evaluate()`.

 * `Evaluate(`_`optimizer, function, coordinates, objective`_`)`

#### Attributes

| **type** | **name** | **description** |
|----------|----------|-----------------|
| `OptimizerType` | **`optimizer`** | The optimizer used to update the function. |
| `FunctionType` | **`function`** | The function to be optimized. |
| `MatType` | **`coordinates`** | The current function parameter. |
| `double` | **`objective`** | Objective value of the current point. |

### EvaluateConstraint

 Called after any call to `EvaluateConstraint()`.

 * `EvaluateConstraint(`_`optimizer, function, coordinates, constraint, constraintValue`_`)`

#### Attributes

| **type** | **name** | **description** |
|----------|----------|-----------------|
| `OptimizerType` | **`optimizer`** | The optimizer used to update the function. |
| `FunctionType` | **`function`** | The function to be optimized. |
| `MatType` | **`coordinates`** | The current function parameter. |
| `size_t` | **`constraint`** | The index of the constraint. |
| `double` | **`constraintValue`** | Constraint value of the current point. |

### Gradient

 Called after any call to `Gradient()`.

 * `Gradient(`_`optimizer, function, coordinates, gradient`_`)`

#### Attributes

| **type** | **name** | **description** |
|----------|----------|-----------------|
| `OptimizerType` | **`optimizer`** | The optimizer used to update the function. |
| `FunctionType` | **`function`** | The function to be optimized. |
| `MatType` | **`coordinates`** | The current function parameter. |
| `GradType` | **`gradient`** | Matrix that holds the gradient. |

### GradientConstraint

 Called after any call to `GradientConstraint()`.

 * `GradientConstraint(`_`optimizer, function, coordinates, constraint, gradient`_`)`

#### Attributes

| **type** | **name** | **description** |
|----------|----------|-----------------|
| `OptimizerType` | **`optimizer`** | The optimizer used to update the function. |
| `FunctionType` | **`function`** | The function to be optimized. |
| `MatType` | **`coordinates`** | The current function parameter. |
| `size_t` | **`constraint`** | The index of the constraint. |
| `GradType` | **`gradient`** | Matrix that holds the gradient. |

### BeginEpoch

Called at the beginning of a pass over the data. The objective may be exact or
an estimate depending on `exactObjective` value.

 * `BeginEpoch(`_`optimizer, function, coordinates, epoch, objective`_`)`

#### Attributes

| **type** | **name** | **description** |
|----------|----------|-----------------|
| `OptimizerType` | **`optimizer`** | The optimizer used to update the function. |
| `FunctionType` | **`function`** | The function to be optimized. |
| `MatType` | **`coordinates`** | The current function parameter. |
| `size_t` | **`epoch`** | The index of the current epoch. |
| `double` | **`objective`** | Objective value of the current point. |

### EndEpoch

Called at the end of a pass over the data. The objective may be exact or
an estimate depending on `exactObjective` value.

 * `EndEpoch(`_`optimizer, function, coordinates, epoch, objective`_`)`

#### Attributes

| **type** | **name** | **description** |
|----------|----------|-----------------|
| `OptimizerType` | **`optimizer`** | The optimizer used to update the function. |
| `FunctionType` | **`function`** | The function to be optimized. |
| `MatType` | **`coordinates`** | The current function parameter. |
| `size_t` | **`epoch`** | The index of the current epoch. |
| `double` | **`objective`** | Objective value of the current point. |

### GenerationalStepTaken

Called after the evolution of a single generation. Intended specifically for
MultiObjective Optimizers.

 * `GenerationalStepTaken(`_`optimizer, function, coordinates, objectives, frontIndices`_`)`

#### Attributes

| **type** | **name** | **description** |
|----------|----------|-----------------|
| `OptimizerType` | **`optimizer`** | The optimizer used to update the function. |
| `FunctionType` | **`function`** | The function to be optimized. |
| `MatType` | **`coordinates`** | The current function parameter. |
| `ObjectivesVecType` | **`objectives`** | The set of calculated objectives so far. |
| `IndicesType` | **`frontIndices`** | The indices of the members belonging to Pareto Front. |

## Custom Callbacks

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
  void EndEpoch(OptimizerType& optimizer,
                FunctionType& /* function */,
                const MatType& /* coordinates */,
                const size_t epoch,
                const double objective)
  {
    // Update the learning rate.
    optimizer.StepSize() = learningRate * (1.0 - std::pow(decay,
        (double) epoch));
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
  void EndEpoch(OptimizerType& /* optimizer */,
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

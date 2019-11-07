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

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `size_t` | **`patience`** | The number of epochs to wait after the minimum loss has been reached. | `10` |

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

</details>

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
      << "parameters " << cb.BestCoordinatest();
```

</details>

## Callback States

Callbacks are called at different states during the optimization process:

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

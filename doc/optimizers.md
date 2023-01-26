## AdaBelief

*An optimizer for [differentiable separable functions](#differentiable-separable-functions).*

AdaBelief uses a different denominator from Adam, and is orthogonal to other
techniques such as recification, decoupled weight decay. The intuition for
AdaBelief is to adapt the stepsize according to the "belief" in the current
gradient direction.

#### Constructors

 * `AdaBelief()`
 * `AdaBelief(`_`stepSize, batchSize`_`)`
 * `AdaBelief(`_`stepSize, batchSize, beta1, beta2, epsilon, maxIterations, tolerance, shuffle`_`)`
 * `AdaBelief(`_`stepSize, batchSize, beta1, beta2, epsilon, maxIterations, tolerance, shuffle, resetPolicy, exactObjective`_`)`

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `double` | **`stepSize`** | Step size for each iteration. | `0.001` |
| `size_t` | **`batchSize`** | Number of points to process in a single step. | `32` |
| `double` | **`beta1`** | The exponential decay rate for the 1st moment estimates. | `0.9` |
| `double` | **`beta2`** | The exponential decay rate for the 2nd moment estimates. | `0.999` |
| `double` | **`epsilon`** | A small constant for numerical stability. | `1e-8` |
| `size_t` | **`max_iterations`** | Maximum number of iterations allowed (0 means no limit). | `100000` |
| `double` | **`tolerance`** | Maximum absolute tolerance to terminate algorithm. | `1e-5` |
| `bool` | **`shuffle`** | If true, the function order is shuffled; otherwise, each function is visited in linear order. | `true` |
| `bool` | **`resetPolicy`** | If true, parameters are reset before every Optimize call; otherwise, their values are retained. | `true` |
| `bool` | **`exactObjective`** | Calculate the exact objective (Default: estimate the final objective obtained on the last pass over the data). | `false` |

The attributes of the optimizer may also be modified via the member methods
`StepSize()`, `BatchSize()`, `Beta1()`, `Beta2()`, `Epsilon()`, `MaxIterations()`,
`Tolerance()`, `Shuffle()`, `ResetPolicy()`, and `ExactObjective()`.

#### Examples

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();

AdaBelief optimizer(0.001, 32, 0.9, 0.999, 1e-12, 100000, 1e-5, true);
optimizer.Optimize(f, coordinates);
```

</details>

#### See also:

 * [SGD in Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
 * [SGD](#standard-sgd)
 * [AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients](https://arxiv.org/abs/2010.07468)
 * [Differentiable separable functions](#differentiable-separable-functions)

## AdaBound

*An optimizer for [differentiable separable functions](#differentiable-separable-functions).*

AdaBound is a variant of Adam which employs dynamic bounds on learning rates.

#### Constructors

 * `AdaBound()`
 * `AdaBound(`_`stepSize, batchSize`_`)`
 * `AdaBound(`_`stepSize, batchSize, finalLr, gamma, beta1, beta2, epsilon, maxIterations, tolerance, shuffle`_`)`
 * `AdaBound(`_`stepSize, batchSize, finalLr, gamma, beta1, beta2, epsilon, maxIterations, tolerance, shuffle, resetPolicy, exactObjective`_`)`

Note that the `AdaBound` class is based on the `AdaBoundType<`_`UpdateRule`_`>`
class with _`UpdateRule`_` = AdaBoundUpdate`.

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `double` | **`stepSize`** | Step size for each iteration. | `0.001` |
| `size_t` | **`batchSize`** | Number of points to process in a single step. | `32` |
| `double` | **`finalLr`** | The final (SGD) learning rate. | `0.1` |
| `double` | **`gamma`** | The convergence speed of the bound functions. | `0.001` |
| `double` | **`beta1`** | Exponential decay rate for the first moment estimates. | `0.9` |
| `double` | **`beta2`** | Exponential decay rate for the weighted infinity norm estimates. | `0.999` |
| `double` | **`epsilon`** | Value used to initialize the mean squared gradient parameter. | `1e-8` |
| `size_t` | **`maxIterations`** | Maximum number of iterations allowed (0 means no limit). | `100000` |
| `double` | **`tolerance`** | Maximum absolute tolerance to terminate algorithm. | `1e-5` |
| `bool` | **`shuffle`** | If true, the function order is shuffled; otherwise, each function is visited in linear order. | `true` |
| `bool` | **`resetPolicy`** | If true, parameters are reset before every Optimize call; otherwise, their values are retained. | `true` |
| `bool` | **`exactObjective`** | Calculate the exact objective (Default: estimate the final objective obtained on the last pass over the data). | `false` |

The attributes of the optimizer may also be modified via the member methods
`FinalLr()`, `Gamma()`, `StepSize()`, `BatchSize()`, `Beta1()`, `Beta2()`,
`Eps()`, `MaxIterations()`, `Tolerance()`, `Shuffle()`, `ResetPolicy()`, and
`ExactObjective()`.

#### Examples

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
SphereFunction f(2);
arma::mat coordinates = f.GetInitialPoint();

AdaBound optimizer(0.001, 2, 0.1, 1e-3, 0.9, 0.999, 1e-8, 500000, 1e-3);
optimizer.Optimize(f, coordinates);
```

</details>

#### See also:

 * [SGD in Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
 * [SGD](#standard-sgd)
 * [Adaptive Gradient Methods with Dynamic Bound of Learning Rate](https://arxiv.org/abs/1902.09843)
 * [Adam: A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980)
 * [Differentiable separable functions](#differentiable-separable-functions)

## AdaDelta

*An optimizer for [differentiable separable functions](#differentiable-separable-functions).*

AdaDelta is an extension of [AdaGrad](#adagrad) that adapts learning rates
based on a moving window of gradient updates, instead of accumulating all past
gradients. Instead of accumulating all past squared gradients, the sum of
gradients is recursively defined as a decaying average of all past squared
gradients.

#### Constructors

 * `AdaDelta()`
 * `AdaDelta(`_`stepSize`_`)`
 * `AdaDelta(`_`stepSize, batchSize`_`)`
 * `AdaDelta(`_`stepSize, batchSize, rho, epsilon, maxIterations, tolerance, shuffle`_`)`
 * `AdaDelta(`_`stepSize, batchSize, rho, epsilon, maxIterations, tolerance, shuffle, resetPolicy, exactObjective`_`)`

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `double` | **`stepSize`** | Step size for each iteration. | `1.0` |
| `size_t` | **`batchSize`**| Number of points to process in one step. | `32` |
| `double` | **`rho`** | Smoothing constant. Corresponding to fraction of gradient to keep at each time step. | `0.95` |
| `double` | **`epsilon`** | Value used to initialize the mean squared gradient parameter. | `1e-6` |
| `size_t` | **`maxIterations`** | Maximum number of iterations allowed (0 means no limit). | `100000` |
| `double` | **`tolerance`** | Maximum absolute tolerance to terminate algorithm. | `1e-5` |
| `bool` | **`shuffle`** | If true, the function order is shuffled; otherwise, each function is visited in linear order. | `true` |
| `bool` | **`resetPolicy`** | If true, parameters are reset before every Optimize call; otherwise, their values are retained. | `true` |
| `bool` | **`exactObjective`** | Calculate the exact objective (Default: estimate the final objective obtained on the last pass over the data). | `false` |

Attributes of the optimizer may also be changed via the member methods
`StepSize()`, `BatchSize()`, `Rho()`, `Epsilon()`, `MaxIterations()`,
`Shuffle()`, `ResetPolicy()`, and `ExactObjective()`.

#### Examples:

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
AdaDelta optimizer(1.0, 1, 0.99, 1e-8, 1000, 1e-9, true);

RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();
optimizer.Optimize(f, coordinates);
```

</details>

#### See also:

 * [Adadelta - an adaptive learning rate method](https://arxiv.org/abs/1212.5701)
 * [AdaGrad](#adagrad)
 * [Differentiable separable functions](#differentiable-separable-functions)

## Adagrad

*An optimizer for [differentiable separable functions](#differentiable-separable-functions).*

AdaGrad is an optimizer with parameter-specific learning rates, which are
adapted relative to how frequently a parameter gets updated during training.
Larger updates for more sparse parameters and smaller updates for less sparse
parameters.

#### Constructors

 - `AdaGrad()`
 - `AdaGrad(`_`stepSize`_`)`
 - `AdaGrad(`_`stepSize, batchSize`_`)`
 - `AdaGrad(`_`stepSize, batchSize, epsilon, maxIterations, tolerance, shuffle`_`)`
 - `AdaGrad(`_`stepSize, batchSize, epsilon, maxIterations, tolerance, shuffle, resetPolicy, exactObjective`_`)`

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `double` | **`stepSize`** | Step size for each iteration. | `0.01` |
| `size_t` | **`batchSize`** | Number of points to process in one step. | `32` |
| `double` | **`epsilon`** | Value used to initialize the mean squared gradient parameter. | `1e-8` |
| `size_t` | **`maxIterations`** | Maximum number of iterations allowed (0 means no limit). | `100000` |
| `double` | **`tolerance`** | Maximum absolute tolerance to terminate algorithm. | `tolerance` |
| `bool` | **`shuffle`** | If true, the function order is shuffled; otherwise, each function is visited in linear order. | `true` |
| `bool` | **`resetPolicy`** | If true, parameters are reset before every Optimize call; otherwise, their values are retained. | `true` |
| `bool` | **`exactObjective`** | Calculate the exact objective (Default: estimate the final objective obtained on the last pass over the data). | `false` |

Attributes of the optimizer may also be changed via the member methods
`StepSize()`, `BatchSize()`, `Epsilon()`, `MaxIterations()`, `Tolerance()`,
`Shuffle()`, `ResetPolicy()`, and `ExactObjective()`.

#### Examples:

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
AdaGrad optimizer(1.0, 1, 1e-8, 1000, 1e-9, true);

RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();
optimizer.Optimize(f, coordinates);
```

</details>

#### See also:

 * [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
 * [AdaGrad in Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#AdaGrad)
 * [AdaDelta](#adadelta)
 * [Differentiable separable functions](#differentiable-separable-functions)

## AdaSqrt

*An optimizer for [differentiable separable functions](#differentiable-separable-functions).*

AdaSqrt is an optimizer with parameter-specific learning rates, which are
adapted relative to how frequently a parameter gets updated during training.
Larger updates for more sparse parameters and smaller updates for less sparse
parameters. AdaSqrt, removes the square root in the denominator and scales the
learning rate by sqrt(T).

#### Constructors

 - `AdaSqrt()`
 - `AdaSqrt(`_`stepSize`_`)`
 - `AdaSqrt(`_`stepSize, batchSize`_`)`
 - `AdaSqrt(`_`stepSize, batchSize, epsilon, maxIterations, tolerance, shuffle`_`)`
 - `AdaSqrt(`_`stepSize, batchSize, epsilon, maxIterations, tolerance, shuffle, resetPolicy, exactObjective`_`)`

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `double` | **`stepSize`** | Step size for each iteration. | `0.01` |
| `size_t` | **`batchSize`** | Number of points to process in one step. | `32` |
| `double` | **`epsilon`** | Value used to initialize the mean squared gradient parameter. | `1e-8` |
| `size_t` | **`maxIterations`** | Maximum number of iterations allowed (0 means no limit). | `100000` |
| `double` | **`tolerance`** | Maximum absolute tolerance to terminate algorithm. | `tolerance` |
| `bool` | **`shuffle`** | If true, the function order is shuffled; otherwise, each function is visited in linear order. | `true` |
| `bool` | **`resetPolicy`** | If true, parameters are reset before every Optimize call; otherwise, their values are retained. | `true` |
| `bool` | **`exactObjective`** | Calculate the exact objective (Default: estimate the final objective obtained on the last pass over the data). | `false` |

Attributes of the optimizer may also be changed via the member methods
`StepSize()`, `BatchSize()`, `Epsilon()`, `MaxIterations()`, `Tolerance()`,
`Shuffle()`, `ResetPolicy()`, and `ExactObjective()`.

#### Examples:

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
AdaSqrt optimizer(1.0, 1, 1e-8, 1000, 1e-9, true);

RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();
optimizer.Optimize(f, coordinates);
```

</details>

#### See also:

 * [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
 * [AdaGrad in Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#AdaGrad)
 * [AdaDelta](#adadelta)
 * [Differentiable separable functions](#differentiable-separable-functions)

## Adam

*An optimizer for [differentiable separable functions](#differentiable-separable-functions).*

Adam is an algorithm for first-order gradient-based optimization of
stochastic objective functions, based on adaptive estimates of lower-order
moments.

#### Constructors

 * `Adam()`
 * `Adam(`_`stepSize, batchSize`_`)`
 * `Adam(`_`stepSize, batchSize, beta1, beta2, eps, maxIterations, tolerance, shuffle`_`)`
 * `Adam(`_`stepSize, batchSize, beta1, beta2, eps, maxIterations, tolerance, shuffle, resetPolicy, exactObjective`_`)`

Note that the `Adam` class is based on the `AdamType<`_`UpdateRule`_`>` class
with _`UpdateRule`_` = AdamUpdate`.

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `double` | **`stepSize`** | Step size for each iteration. | `0.001` |
| `size_t` | **`batchSize`** | Number of points to process in a single step. | `32` |
| `double` | **`beta1`** | Exponential decay rate for the first moment estimates. | `0.9` |
| `double` | **`beta2`** | Exponential decay rate for the weighted infinity norm estimates. | `0.999` |
| `double` | **`eps`** | Value used to initialize the mean squared gradient parameter. | `1e-8` |
| `size_t` | **`maxIterations`** | Maximum number of iterations allowed (0 means no limit). | `100000` |
| `double` | **`tolerance`** | Maximum absolute tolerance to terminate algorithm. | `1e-5` |
| `bool` | **`shuffle`** | If true, the function order is shuffled; otherwise, each function is visited in linear order. | `true` |
| `bool` | **`resetPolicy`** | If true, parameters are reset before every Optimize call; otherwise, their values are retained. | `true` |
| `bool` | **`exactObjective`** | Calculate the exact objective (Default: estimate the final objective obtained on the last pass over the data). | `false` |

The attributes of the optimizer may also be modified via the member methods
`StepSize()`, `BatchSize()`, `Beta1()`, `Beta2()`, `Epsilon()`, `MaxIterations()`,
`Tolerance()`, `Shuffle()`, `ResetPolicy()`, and `ExactObjective()`.

#### Examples

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();

Adam optimizer(0.001, 32, 0.9, 0.999, 1e-8, 100000, 1e-5, true);
optimizer.Optimize(f, coordinates);
```

</details>

#### See also:

 * [SGD in Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
 * [SGD](#standard-sgd)
 * [Adam: A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980)
 * [Differentiable separable functions](#differentiable-separable-functions)

## AdaMax

*An optimizer for [differentiable separable functions](#differentiable-separable-functions).*

AdaMax is simply a variant of Adam based on the infinity norm.

#### Constructors

 * `AdaMax()`
 * `AdaMax(`_`stepSize, batchSize`_`)`
 * `AdaMax(`_`stepSize, batchSize, beta1, beta2, eps, maxIterations, tolerance, shuffle`_`)`
 * `AdaMax(`_`stepSize, batchSize, beta1, beta2, eps, maxIterations, tolerance, shuffle, exactObjective, resetPolicy`_`)`

Note that the `AdaMax` class is based on the `AdamType<`_`UpdateRule`_`>` class
with _`UpdateRule`_` = AdaMaxUpdate`.

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `double` | **`stepSize`** | Step size for each iteration. | `0.001` |
| `size_t` | **`batchSize`** | Number of points to process in a single step. | `32` |
| `double` | **`beta1`** | Exponential decay rate for the first moment estimates. | `0.9` |
| `double` | **`beta2`** | Exponential decay rate for the weighted infinity norm estimates. | `0.999` |
| `double` | **`eps`** | Value used to initialize the mean squared gradient parameter. | `1e-8` |
| `size_t` | **`maxIterations`** | Maximum number of iterations allowed (0 means no limit). | `100000` |
| `double` | **`tolerance`** | Maximum absolute tolerance to terminate algorithm. | `1e-5` |
| `bool` | **`shuffle`** | If true, the function order is shuffled; otherwise, each function is visited in linear order. | `true` |
| `bool` | **`exactObjective`** | Calculate the exact objective (Default: estimate the final objective obtained on the last pass over the data). | `false` |
| `bool` | **`resetPolicy`** | If true, parameters are reset before every Optimize call; otherwise, their values are retained. | `true` |

The attributes of the optimizer may also be modified via the member methods
`StepSize()`, `BatchSize()`, `Beta1()`, `Beta2()`, `Eps()`, `MaxIterations()`,
`Tolerance()`, `Shuffle()`, `ExactObjective()`, and `ResetPolicy()`.

#### Examples

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();

AdaMax optimizer(0.001, 32, 0.9, 0.999, 1e-8, 100000, 1e-5, true);
optimizer.Optimize(f, coordinates);
```

</details>

#### See also:

 * [SGD in Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
 * [SGD](#standard-sgd)
 * [Adam: A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980) (see section 7)
 * [Differentiable separable functions](#differentiable-separable-functions)

## AMSBound

*An optimizer for [differentiable separable functions](#differentiable-separable-functions).*

AMSBound is a variant of Adam which employs dynamic bounds on learning rates.

#### Constructors

 * `AMSBound()`
 * `AMSBound(`_`stepSize, batchSize`_`)`
 * `AMSBound(`_`stepSize, batchSize, finalLr, gamma, beta1, beta2, epsilon, maxIterations, tolerance, shuffle`_`)`
 * `AMSBound(`_`stepSize, batchSize, finalLr, gamma, beta1, beta2, epsilon, maxIterations, tolerance, shuffle, resetPolicy, exactObjective`_`)`

Note that the `AMSBound` class is based on the `AdaBoundType<`_`UpdateRule`_`>`
class with _`UpdateRule`_` = AdaBoundUpdate`.

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `double` | **`stepSize`** | Step size for each iteration. | `0.001` |
| `size_t` | **`batchSize`** | Number of points to process in a single step. | `32` |
| `double` | **`finalLr`** | The final (SGD) learning rate. | `0.1` |
| `double` | **`gamma`** | The convergence speed of the bound functions. | `0.001` |
| `double` | **`beta1`** | Exponential decay rate for the first moment estimates. | `0.9` |
| `double` | **`beta2`** | Exponential decay rate for the weighted infinity norm estimates. | `0.999` |
| `double` | **`epsilon`** | Value used to initialize the mean squared gradient parameter. | `1e-8` |
| `size_t` | **`maxIterations`** | Maximum number of iterations allowed (0 means no limit). | `100000` |
| `double` | **`tolerance`** | Maximum absolute tolerance to terminate algorithm. | `1e-5` |
| `bool` | **`shuffle`** | If true, the function order is shuffled; otherwise, each function is visited in linear order. | `true` |
| `bool` | **`resetPolicy`** | If true, parameters are reset before every Optimize call; otherwise, their values are retained. | `true` |
| `bool` | **`exactObjective`** | Calculate the exact objective (Default: estimate the final objective obtained on the last pass over the data). | `false` |

The attributes of the optimizer may also be modified via the member methods
`FinalLr()`, `Gamma()`, `StepSize()`, `BatchSize()`, `Beta1()`, `Beta2()`,
`Eps()`, `MaxIterations()`, `Tolerance()`, `Shuffle()`, `ResetPolicy()`, and
`ExactObjective()`.

#### Examples

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
SphereFunction f(2);
arma::mat coordinates = f.GetInitialPoint();

AMSBound optimizer(0.001, 2, 0.1, 1e-3, 0.9, 0.999, 1e-8, 500000, 1e-3);
optimizer.Optimize(f, coordinates);
```

</details>

#### See also:

 * [SGD in Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
 * [SGD](#standard-sgd)
 * [Adaptive Gradient Methods with Dynamic Bound of Learning Rate](https://arxiv.org/abs/1902.09843)
 * [Adam: A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980)
 * [Differentiable separable functions](#differentiable-separable-functions)

## AMSGrad

*An optimizer for [differentiable separable functions](#differentiable-separable-functions).*

AMSGrad is a variant of Adam with guaranteed convergence.

#### Constructors

 * `AMSGrad()`
 * `AMSGrad(`_`stepSize, batchSize`_`)`
 * `AMSGrad(`_`stepSize, batchSize, beta1, beta2, eps, maxIterations, tolerance, shuffle`_`)`
 * `AMSGrad(`_`stepSize, batchSize, beta1, beta2, eps, maxIterations, tolerance, shuffle, exactObjective, resetPolicy`_`)`

Note that the `AMSGrad` class is based on the `AdamType<`_`UpdateRule`_`>` class
with _`UpdateRule`_` = AMSGradUpdate`.

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `double` | **`stepSize`** | Step size for each iteration. | `0.001` |
| `size_t` | **`batchSize`** | Number of points to process in a single step. | `32` |
| `double` | **`beta1`** | Exponential decay rate for the first moment estimates. | `0.9` |
| `double` | **`beta2`** | Exponential decay rate for the weighted infinity norm estimates. | `0.999` |
| `double` | **`eps`** | Value used to initialize the mean squared gradient parameter. | `1e-8` |
| `size_t` | **`maxIterations`** | Maximum number of iterations allowed (0 means no limit). | `100000` |
| `double` | **`tolerance`** | Maximum absolute tolerance to terminate algorithm. | `1e-5` |
| `bool` | **`shuffle`** | If true, the function order is shuffled; otherwise, each function is visited in linear order. | `true` |
| `bool` | **`exactObjective`** | Calculate the exact objective (Default: estimate the final objective obtained on the last pass over the data). | `false` |
| `bool` | **`resetPolicy`** | If true, parameters are reset before every Optimize call; otherwise, their values are retained. | `true` |

The attributes of the optimizer may also be modified via the member methods
`StepSize()`, `BatchSize()`, `Beta1()`, `Beta2()`, `Eps()`, `MaxIterations()`,
`Tolerance()`, `Shuffle()`, `ExactObjective()`, and `ResetPolicy()`.

#### Examples

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();

AMSGrad optimizer(0.001, 32, 0.9, 0.999, 1e-8, 100000, 1e-5, true);
optimizer.Optimize(f, coordinates);
```

</details>

#### See also:

 * [SGD in Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
 * [SGD](#standard-sgd)
 * [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)
 * [Differentiable separable functions](#differentiable-separable-functions)

## Augmented Lagrangian

*An optimizer for [differentiable constrained functions](#constrained-functions).*

The `AugLagrangian` class implements the Augmented Lagrangian method of
optimization.  In this scheme, a penalty term is added to the Lagrangian.
This method is also called the "method of multipliers".  Internally, the
optimizer uses [L-BFGS](#l-bfgs).

#### Constructors

 * `AugLagrangian(`_`maxIterations, penaltyThresholdFactor, sigmaUpdateFactor`_`)`

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `size_t` | **`maxIterations`** | Maximum number of iterations allowed (0 means no limit). | `1000` |
| `double` | **`penaltyThresholdFactor`** | When penalty threshold is updated, set it to this multiplied by the penalty. | `0.25` |
| `double` | **`sigmaUpdateFactor`** | When sigma is updated, multiply it by this. | `10.0` |
| `L_BFGS&` | **`lbfgs`** | Internal l-bfgs optimizer. | `L_BFGS()` |

The attributes of the optimizer may also be modified via the member methods
`MaxIterations()`, `PenaltyThresholdFactor()`, `SigmaUpdateFactor()` and `LBFGS()`.

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
/**
 * Optimize the function.  The value '1' is used for the initial value of each
 * Lagrange multiplier.  To set the Lagrange multipliers yourself, use the
 * other overload of Optimize().
 *
 * @tparam LagrangianFunctionType Function which can be optimized by this
 *     class.
 * @param function The function to optimize.
 * @param coordinates Output matrix to store the optimized coordinates in.
 */
template<typename LagrangianFunctionType>
bool Optimize(LagrangianFunctionType& function,
              arma::mat& coordinates);

/**
 * Optimize the function, giving initial estimates for the Lagrange
 * multipliers.  The vector of Lagrange multipliers will be modified to
 * contain the Lagrange multipliers of the final solution (if one is found).
 *
 * @tparam LagrangianFunctionType Function which can be optimized by this
 *      class.
 * @param function The function to optimize.
 * @param coordinates Output matrix to store the optimized coordinates in.
 * @param initLambda Vector of initial Lagrange multipliers.  Should have
 *     length equal to the number of constraints.
 * @param initSigma Initial penalty parameter.
 */
template<typename LagrangianFunctionType>
bool Optimize(LagrangianFunctionType& function,
              arma::mat& coordinates,
              const arma::vec& initLambda,
              const double initSigma);
```

</details>

#### Examples

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
GockenbachFunction f;
arma::mat coordinates = f.GetInitialPoint();

AugLagrangian optimizer;
optimizer.Optimize(f, coords);
```

</details>

#### See also:

 * [Augmented Lagrangian method on Wikipedia](https://en.wikipedia.org/wiki/Augmented_Lagrangian_method)
 * [L-BFGS](#l-bfgs)
 * [Constrained functions](#constrained-functions)

## Big Batch SGD

*An optimizer for [differentiable separable functions](#differentiable-separable-functions).*

Big-batch stochastic gradient descent adaptively grows the batch size over time
to maintain a nearly constant signal-to-noise ratio in the gradient
approximation, so the Big Batch SGD optimizer is able to adaptively adjust batch
sizes without user oversight.

#### Constructors

 * `BigBatchSGD<`_`UpdatePolicy`_`>()`
 * `BigBatchSGD<`_`UpdatePolicy`_`>(`_`stepSize`_`)`
 * `BigBatchSGD<`_`UpdatePolicy`_`>(`_`stepSize, batchSize`_`)`
 * `BigBatchSGD<`_`UpdatePolicy`_`>(`_`stepSize, batchSize, epsilon, maxIterations, tolerance, shuffle, exactObjective`_`)`

The _`UpdatePolicy`_ template parameter refers to the way that a new step size
is computed.  The `AdaptiveStepsize` and `BacktrackingLineSearch` classes are
available for use; custom behavior can be achieved by implementing a class
with the same method signatures.

For convenience the following typedefs have been defined:

 * `BBS_Armijo = BigBatchSGD<BacktrackingLineSearch>`
 * `BBS_BB = BigBatchSGD<AdaptiveStepsize>`

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `size_t` | **`batchSize`** | Initial batch size. | `1000` |
| `double` | **`stepSize`** | Step size for each iteration. | `0.01` |
| `double` | **`batchDelta`** | Factor for the batch update step. | `0.1` |
| `size_t` | **`maxIterations`** | Maximum number of iterations allowed (0 means no limit). | `100000` |
| `double` | **`tolerance`** | Maximum absolute tolerance to terminate algorithm. | `1e-5` |
| `bool` | **`shuffle`** | If true, the batch order is shuffled; otherwise, each batch is visited in linear order. | `true` |
| `bool` | **`exactObjective`** | Calculate the exact objective (Default: estimate the final objective obtained on the last pass over the data). | `false` |

Attributes of the optimizer may also be changed via the member methods
`BatchSize()`, `StepSize()`, `BatchDelta()`, `MaxIterations()`, `Tolerance()`,
`Shuffle()`, and `ExactObjective()`.

#### Examples:

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();

// Big-Batch SGD with the adaptive stepsize policy.
BBS_BB optimizer(10, 0.01, 0.1, 8000, 1e-4);
optimizer.Optimize(f, coordinates);

// Big-Batch SGD with backtracking line search.
BBS_Armijo optimizer2(10, 0.01, 0.1, 8000, 1e-4);
optimizer2.Optimize(f, coordinates);
```

</details>

#### See also:

 * [Big Batch SGD: Automated Inference using Adaptive Batch Sizes](https://arxiv.org/pdf/1610.05792.pdf)
 * [SGD in Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
 * [SGD](#standard-sgd)

## CMAES

*An optimizer for [separable functions](#separable-functions).*

CMA-ES (Covariance Matrix Adaptation Evolution Strategy) is a stochastic search
algorithm. CMA-ES is a second order approach estimating a positive definite
matrix within an iterative procedure using the covariance matrix.

#### Constructors

 * `CMAES<`_`SelectionPolicyType`_`>()`
 * `CMAES<`_`SelectionPolicyType`_`>(`_`lambda, lowerBound, upperBound`_`)`
 * `CMAES<`_`SelectionPolicyType`_`>(`_`lambda, lowerBound, upperBound, batchSize`_`)`
 * `CMAES<`_`SelectionPolicyType`_`>(`_`lambda, lowerBound, upperBound, batchSize, maxIterations, tolerance, selectionPolicy`_`)`

The _`SelectionPolicyType`_ template parameter refers to the strategy used to
compute the (approximate) objective function.  The `FullSelection` and
`RandomSelection` classes are available for use; custom behavior can be achieved
by implementing a class with the same method signatures.

For convenience the following types can be used:

 * **`CMAES<>`** (equivalent to `CMAES<FullSelection>`): uses all separable functions to compute objective
 * **`ApproxCMAES`** (equivalent to `CMAES<RandomSelection>`): uses a small amount of separable functions to compute approximate objective

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `size_t` | **`lambda`** | The population size (0 uses a default size). | `0` |
| `double` | **`lowerBound`** | Lower bound of decision variables. | `-10.0` |
| `double` | **`upperBound`** | Upper bound of decision variables. | `10.0` |
| `size_t` | **`batchSize`** | Batch size to use for the objective calculation. | `32` |
| `size_t` | **`maxIterations`** | Maximum number of iterations. | `1000` |
| `double` | **`tolerance`** | Maximum absolute tolerance to terminate algorithm. | `1e-5` |
| `SelectionPolicyType` | **`selectionPolicy`** | Instantiated selection policy used to calculate the objective. | `SelectionPolicyType()` |

Attributes of the optimizer may also be changed via the member methods
`Lambda()`, `LowerBound()`, `UpperBound()`, `BatchSize()`, `MaxIterations()`,
`Tolerance()`, and `SelectionPolicy()`.

The `selectionPolicy` attribute allows an instantiated `SelectionPolicyType` to
be given.  The `FullSelection` policy has no need to be instantiated and thus
the option is not relevant when the `CMAES<>` optimizer type is being used; the
`RandomSelection` policy has the constructor `RandomSelection(`_`fraction`_`)`
where _`fraction`_ specifies the percentage of separable functions to use to
estimate the objective function.

#### Examples:

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();

// CMAES with the FullSelection policy.
CMAES<> optimizer(0, -1, 1, 32, 200, 1e-4);
optimizer.Optimize(f, coordinates);

// CMAES with the RandomSelection policy.
ApproxCMAES<> approxOptimizer(0, -1, 1. 32, 200, 1e-4);
approxOptimizer.Optimize(f, coordinates);
```

</details>

#### See also:

 * [Completely Derandomized Self-Adaptation in Evolution Strategies](http://www.cmap.polytechnique.fr/~nikolaus.hansen/cmaartic.pdf)
 * [CMA-ES in Wikipedia](https://en.wikipedia.org/wiki/CMA-ES)
 * [Evolution strategy in Wikipedia](https://en.wikipedia.org/wiki/Evolution_strategy)

## CNE

*An optimizer for [arbitrary functions](#arbitrary-functions).*

Conventional Neural Evolution is an optimizer that works like biological evolution which selects best candidates based on their fitness scores and creates new generation by mutation and crossover of population. The initial population is generated based on a random normal distribution centered at the given starting point.

#### Constructors

 * `CNE()`
 * `CNE(`_`populationSize, maxGenerations`_`)`
 * `CNE(`_`populationSize, maxGenerations, mutationProb, mutationSize`_`)`
 * `CNE(`_`populationSize, maxGenerations, mutationProb, mutationSize, selectPercent, tolerance`_`)`

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `size_t` | **`populationSize`** | The number of candidates in the population. This should be at least 4 in size. | `500` |
| `size_t` | **`maxGenerations`** | The maximum number of generations allowed for CNE. | `5000` |
| `double` | **`mutationProb`** | Probability that a weight will get mutated. | `0.1` |
| `double` | **`mutationSize`** | The range of mutation noise to be added. This range is between 0 and mutationSize. | `0.02` |
| `double` | **`selectPercent`** | The percentage of candidates to select to become the the next generation. | `0.2` |
| `double` | **`tolerance`** | The final value of the objective function for termination. If set to negative value, tolerance is not considered. | `1e-5` |

Attributes of the optimizer may also be changed via the member methods
`PopulationSize()`, `MaxGenerations()`, `MutationProb()`, `SelectPercent()`
and `Tolerance()`.

#### Examples:

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();

CNE optimizer(200, 10000, 0.2, 0.2, 0.3, 1e-5);
optimizer.Optimize(f, coordinates);
```

</details>

#### See also:

 * [Neuroevolution in Wikipedia](https://en.wikipedia.org/wiki/Neuroevolution)
 * [Arbitrary functions](#arbitrary-functions)

## DE

*An optimizer for [arbitrary functions](#arbitrary-functions).*

Differential Evolution is an evolutionary optimization algorithm which selects best candidates based on their fitness scores and creates new generation by mutation and crossover of population.

#### Constructors

* `DE()`
* `DE(`_`populationSize, maxGenerations`_`)`
* `DE(`_`populationSize, maxGenerations, crossoverRate`_`)`
* `DE(`_`populationSize, maxGenerations, crossoverRate, differentialWeight`_`)`
* `DE(`_`populationSize, maxGenerations, crossoverRate, differentialWeight, tolerance`_`)`

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `size_t` | **`populationSize`** | The number of candidates in the population. This should be at least 3 in size. | `100` |
| `size_t` | **`maxGenerations`** | The maximum number of generations allowed for DE. | `2000` |
| `double` | **`crossoverRate`** | Probability that a candidate will undergo crossover. | `0.6` |
| `double` | **`differentialWeight`** | Amplification factor for differentiation. | `0.8` |
| `double` | **`tolerance`** | The final value of the objective function for termination. If set to negative value, tolerance is not considered. | `1e-5` |

Attributes of the optimizer may also be changed via the member methods
`PopulationSize()`, `MaxGenerations()`, `CrossoverRate()`, `DifferentialWeight()`
and `Tolerance()`.

#### Examples:

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();

DE optimizer(200, 1000, 0.6, 0.8, 1e-5);
optimizer.Optimize(f, coordinates);
```

</details>

#### See also:

 * [Differential Evolution - A simple and efficient adaptive scheme for global optimization over continuous spaces](http://www1.icsi.berkeley.edu/~storn/TR-95-012.pdf)
 * [Differential Evolution in Wikipedia](https://en.wikipedia.org/wiki/Differential_Evolution)
 * [Arbitrary functions](#arbitrary-functions)

## DemonAdam

*An optimizer for [differentiable separable functions](#differentiable-separable-functions).*

DemonAdam is an Adam based optimizer. DemonAdam is motivated by decaying the
total contribution of a gradient to all future updates.

#### Constructors

 * `DemonAdam()`
 * `DemonAdam(`_`stepSize, batchSize`_`)`
 * `DemonAdam(`_`stepSize, batchSize, momentum, beta1, beta2, eps, maxIterations, tolerance, shuffle`_`)`
 * `DemonAdam(`_`stepSize, batchSize, momentum, beta1, beta2, eps, maxIterations, tolerance, shuffle, resetPolicy`_`)`

Note that the `DemonAdam` class is based on
the `DemonAdamType<`_`UpdateRule`_`>` class with _`UpdateRule`_` = AdamUpdate`.

For convenience the following typedefs have been defined:

 * `DemonAdaMax` (equivalent to `DemonAdamType<AdaMaxUpdate>`): DemonAdam that
   uses the AdaMax update rule.
 * `DemonAMSGrad` (equivalent to `DemonAdamType<AMSGradUpdate>`): DemonAdam that
   uses the AMSGrad update rule.
 * `DemonNadam` (equivalent to `DemonAdamType<NadamUpdate>`): DemonAdam that
   uses the Nadam update rule.
 * `NadamUpdate` (equivalent to `DemonAdamType<NadaMaxUpdate>`): DemonAdam that
   uses the NadaMax update rule.
 * `DemonOptimisticAdam` (equivalent to `DemonAdamType<OptimisticAdamUpdate>`):
   DemonAdam that uses the OptimisticAdam update rule.

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `double` | **`stepSize`** | Step size for each iteration. | `0.001` |
| `size_t` | **`batchSize`** | Number of points to process in a single step. | `32` |
| `double` | **`momentum`** | The initial momentum coefficient. | `0.9` |
| `double` | **`beta1`** | Exponential decay rate for the first moment estimates. | `0.9` |
| `double` | **`beta2`** | Exponential decay rate for the weighted infinity norm estimates. | `0.999` |
| `double` | **`eps`** | Value used to initialize the mean squared gradient parameter. | `1e-8` |
| `size_t` | **`maxIterations`** | Maximum number of iterations allowed (0 means no limit). | `100000` |
| `double` | **`tolerance`** | Maximum absolute tolerance to terminate algorithm. | `1e-5` |
| `bool` | **`shuffle`** | If true, the function order is shuffled; otherwise, each function is visited in linear order. | `true` |
| `bool` | **`resetPolicy`** | If true, parameters are reset before every Optimize call; otherwise, their values are retained. | `true` |

The attributes of the optimizer may also be modified via the member methods
`StepSize()`, `BatchSize()`, `Momentum()`, `MomentumIterations()`, `Beta1()`,
`Beta2()`, `Eps()`, `MaxIterations()`, `Tolerance()`, `Shuffle()`, and
`ResetPolicy()`.

#### Examples

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
MatyasFunction f;
arma::mat coordinates = f.GetInitialPoint();

DemonAdam optimizer(0.5, 1, 0.9);
optimizer.Optimize(f, coordinates);
```

</details>

#### See also:

 * [SGD in Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
 * [SGD](#standard-sgd)
 * [Decaying momentum helps neural network training](https://arxiv.org/abs/1910.04952)
 * [Differentiable separable functions](#differentiable-separable-functions)

## DemonSGD

*An optimizer for [differentiable separable functions](#differentiable-separable-functions).*

DemonSGD is an SGD based optimizer. DemonSGD is motivated by decaying the total
contribution of a gradient to all future updates.

For convenience ensmallen implements various Adam based versions of the Demon
optimizer:

 * `DemonAdam` (equivalent to `DemonAdamType<AdamUpdate>`): DemonAdam that uses
   the Adam update rule.
 * `DemonAdaMax` (equivalent to `DemonAdamType<AdaMaxUpdate>`): DemonAdam that
   uses the AdaMax update rule.
 * `DemonAMSGrad` (equivalent to `DemonAdamType<AMSGradUpdate>`): DemonAdam that
   uses the AMSGrad update rule.
 * `DemonNadam` (equivalent to `DemonAdamType<NadamUpdate>`): DemonAdam that
   uses the Nadam update rule.
 * `NadamUpdate` (equivalent to `DemonAdamType<NadaMaxUpdate>`): DemonAdam that
   uses the NadaMax update rule.
 * `DemonOptimisticAdam` (equivalent to `DemonAdamType<OptimisticAdamUpdate>`):
   DemonAdam that uses the OptimisticAdam update rule.

#### Constructors

 * `DemonSGD()`
 * `DemonSGD(`_`stepSize, batchSize`_`)`
 * `DemonSGD(`_`stepSize, batchSize, momentum, maxIterations, tolerance, shuffle`_`)`
 * `DemonSGD(`_`stepSize, batchSize, momentum, maxIterations, tolerance, shuffle, resetPolicy`_`)`

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `double` | **`stepSize`** | Step size for each iteration. | `0.001` |
| `size_t` | **`batchSize`** | Number of points to process in a single step. | `32` |
| `double` | **`momentum`** | The initial momentum coefficient. | `0.9` |
| `size_t` | **`maxIterations`** | Maximum number of iterations allowed (0 means no limit). | `100000` |
| `double` | **`tolerance`** | Maximum absolute tolerance to terminate algorithm. | `1e-5` |
| `bool` | **`shuffle`** | If true, the function order is shuffled; otherwise, each function is visited in linear order. | `true` |
| `bool` | **`resetPolicy`** | If true, parameters are reset before every Optimize call; otherwise, their values are retained. | `true` |

The attributes of the optimizer may also be modified via the member methods
`StepSize()`, `BatchSize()`, `Momentum()`, `MomentumIterations()`,
`MaxIterations()`, `Tolerance()`, `Shuffle()`, and `ResetPolicy()`.

#### Examples

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
MatyasFunction f;
arma::mat coordinates = f.GetInitialPoint();

DemonSGD optimizer(0.5, 1, 0.9);
optimizer.Optimize(f, coordinates);
```

</details>

#### See also:

 * [SGD in Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
 * [SGD](#standard-sgd)
 * [Decaying momentum helps neural network training](https://arxiv.org/abs/1910.04952)
 * [Differentiable separable functions](#differentiable-separable-functions)

## Eve

*An optimizer for [differentiable separable functions](#differentiable-separable-functions).*

Eve is a stochastic gradient based optimization method with locally and globally adaptive learning rates.

#### Constructors

 * `Eve()`
 * `Eve(`_`stepSize, batchSize`_`)`
 * `Eve(`_`stepSize, batchSize, beta1, beta2, beta3, epsilon, clip, maxIterations, tolerance, shuffle`_`)`

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `double` | **`stepSize`** | Step size for each iteration. | `0.001` |
| `size_t` | **`batchSize`** | Number of points to process in a single step. | `32` |
| `double` | **`beta1`** | Exponential decay rate for the first moment estimates. | `0.9` |
| `double` | **`beta2`** | Exponential decay rate for the weighted infinity norm estimates. | `0.999` |
| `double` | **`beta3`** | Exponential decay rate for relative change. | `0.999` |
| `double` | **`epsilon`** | Value used to initialize the mean squared gradient parameter. | `1e-8` |
| `double` | **`clip`** | Clipping range to avoid extreme valus. | `10` |
| `size_t` | **`maxIterations`** | Maximum number of iterations allowed (0 means no limit). | `100000` |
| `double` | **`tolerance`** | Maximum absolute tolerance to terminate algorithm. | `1e-5` |
| `bool` | **`shuffle`** | If true, the function order is shuffled; otherwise, each function is visited in linear order. | `true` |
| `bool` | **`exactObjective`** | Calculate the exact objective (Default: estimate the final objective obtained on the last pass over the data). | `false` |

The attributes of the optimizer may also be modified via the member methods
`StepSize()`, `BatchSize()`, `Beta1()`, `Beta2()`, `Beta3()`, `Epsilon()`, `Clip()`, `MaxIterations()`,
`Tolerance()`, `Shuffle()`, and `ExactObjective()`.

#### Examples

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();

Eve optimizer(0.001, 32, 0.9, 0.999, 0.999, 10, 1e-8, 100000, 1e-5, true);
optimizer.Optimize(f, coordinates);
```

</details>

#### See also:

 * [SGD in Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
 * [SGD](#standard-sgd)
 * [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](https://arxiv.org/pdf/1611.01505.pdf)
 * [Differentiable separable functions](#differentiable-separable-functions)

## Frank-Wolfe

*An optimizer for [differentiable functions](#differentiable-functions) that may also be constrained.*

Frank-Wolfe is a technique to minimize a continuously differentiable convex function f over a compact convex subset D of a vector space. It is also known as conditional gradient method.

#### Constructors

 * `FrankWolfe<`_`LinearConstrSolverType, UpdateRuleType`_`>(`_`linearConstrSolver, updateRule`_`)`
 * `FrankWolfe<`_`LinearConstrSolverType, UpdateRuleType`_`>(`_`linearConstrSolver, updateRule, maxIterations, tolerance`_`)`

The _`LinearConstrSolverType`_ template parameter specifies the constraint
domain D for the problem.  The `ConstrLpBallSolver` and
`ConstrStructGroupSolver<GroupLpBall>` classes are available for use; the former
restricts D to the unit ball of the specified l-p norm.  Other constraint types
may be implemented as a class with the same method signatures as either of the
existing classes.

The _`UpdateRuleType`_ template parameter specifies the update rule used by the
optimizer.  The `UpdateClassic` and `UpdateLineSearch` classes are available for
use and represent a simple update step rule and a line search based update rule,
respectively.  The `UpdateSpan` and `UpdateFulLCorrection` classes are also
available and may be used with the `FuncSq` function class (which is a squared
matrix loss).

For convenience the following typedefs have been defined:

 * `OMP` (equivalent to `FrankWolfe<ConstrLpBallSolver, UpdateSpan>`): a solver for the orthogonal matching pursuit problem
 * `StandardFrankWolfe` (equivalent to `FrankWolfe<ConstrLpBallSolver, ClassicUpdate>`): the standard Frank-Wolfe algorithm with the solution restricted to lie within the unit ball

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `LinearConstrSolverType` | **`linearConstrSolver`** | Solver for linear constrained problem. | **n/a** |
| `UpdateRuleType` | **`updateRule`** | Rule for updating solution in each iteration. | **n/a** |
| `size_t` | **`maxIterations`** | Maximum number of iterations allowed (0 means no limit). | `100000` |
| `size_t` | **`tolerance`** | Maximum absolute tolerance to terminate algorithm. | `1e-10` |

Attributes of the optimizer may also be changed via the member methods
`LinearConstrSolver()`, `UpdateRule()`, `MaxIterations()`, and `Tolerance()`.

#### Examples:

TODO

#### See also:

 * [An algorithm for quadratic programming](https://pdfs.semanticscholar.org/3a24/54478a94f1e66a3fc5d209e69217087acbc0.pdf)
 * [Frank-Wolfe in Wikipedia](https://en.wikipedia.org/wiki/Frank%E2%80%93Wolfe_algorithm)
 * [Differentiable functions](#differentiable-functions)

## FTML (Follow the Moving Leader)

*An optimizer for [differentiable separable functions](#differentiable-separable-functions).*

Follow the Moving Leader (FTML) is an optimizer where recent samples are
weighted more heavily in each iteration, so FTML can adapt more quickly to
changes.

#### Constructors

 * `FTML()`
 * `FTML(`_`stepSize, batchSize`_`)`
 * `FTML(`_`stepSize, batchSize, beta1, beta2, epsilon, maxIterations, tolerance, shuffle`_`)`
 * `FTML(`_`stepSize, batchSize, beta1, beta2, epsilon, maxIterations, tolerance, shuffle, resetPolicy, exactObjective`_`)`

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `double` | **`stepSize`** | Step size for each iteration. | `0.001` |
| `size_t` | **`batchSize`** | Number of points to process in a single step. | `32` |
| `double` | **`beta1`** | Exponential decay rate for the first moment estimates. | `0.9` |
| `double` | **`beta2`** | Exponential decay rate for the weighted infinity norm estimates. | `0.999` |
| `double` | **`eps`** | Value used to initialize the mean squared gradient parameter. | `1e-8` |
| `size_t` | **`maxIterations`** | Maximum number of iterations allowed (0 means no limit). | `100000` |
| `double` | **`tolerance`** | Maximum absolute tolerance to terminate algorithm. | `1e-5` |
| `bool` | **`shuffle`** | If true, the function order is shuffled; otherwise, each function is visited in linear order. | `true` |
| `bool` | **`resetPolicy`** | If true, parameters are reset before every Optimize call; otherwise, their values are retained. | `true` |
| `bool` | **`exactObjective`** | Calculate the exact objective (Default: estimate the final objective obtained on the last pass over the data). | `false` |

The attributes of the optimizer may also be modified via the member methods
`StepSize()`, `BatchSize()`, `Beta1()`, `Beta2()`, `Epsilon()`, `MaxIterations()`,
`Tolerance()`, `Shuffle()`, `ResetPolicy()`, and `ExactObjective()`.

#### Examples

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();

FTML optimizer(0.001, 32, 0.9, 0.999, 1e-8, 100000, 1e-5, true);
optimizer.Optimize(f, coordinates);
```

</details>

#### See also:
 * [Follow the Moving Leader in Deep Learning](http://proceedings.mlr.press/v70/zheng17a/zheng17a.pdf)
 * [SGD in Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
 * [SGD](#standard-sgd)
 * [Differentiable separable functions](#differentiable-separable-functions)

## Gradient Descent

*An optimizer for [differentiable functions](#differentiable-functions).*

Gradient Descent is a technique to minimize a function. To find a local minimum
of a function using gradient descent, one takes steps proportional to the
negative of the gradient of the function at the current point.

#### Constructors

 * `GradientDescent()`
 * `GradientDescent(`_`stepSize`_`)`
 * `GradientDescent(`_`stepSize, maxIterations, tolerance`_`)`

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `double` | **`stepSize`** | Step size for each iteration. | `0.01` |
| `size_t` | **`maxIterations`** | Maximum number of iterations allowed (0 means no limit). | `100000` |
| `size_t` | **`tolerance`**  | Maximum absolute tolerance to terminate algorithm. | `1e-5` |

Attributes of the optimizer may also be changed via the member methods
`StepSize()`, `MaxIterations()`, and `Tolerance()`.

#### Examples:

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();

GradientDescent optimizer(0.001, 0, 1e-15);
optimizer.Optimize(f, coordinates);
```

</details>

#### See also:

 * [Gradient descent in Wikipedia](https://en.wikipedia.org/wiki/Gradient_descent)
 * [Differentiable functions](#differentiable-functions)

## Grid Search

*An optimizer for [categorical functions](#categorical-functions).*

An optimizer that finds the minimum of a given function by iterating through
points on a multidimensional grid.

#### Constructors

 * `GridSearch()`

#### Attributes

The `GridSearch` class has no configurable attributes.

**Note**: the `GridSearch` class can only optimize categorical functions where
*every* parameter is categorical.

#### See also:

 * [Categorical functions](#categorical-functions) (includes an example for `GridSearch`)
 * [Grid search on Wikipedia](https://en.wikipedia.org/wiki/Hyperparameter_optimization#Grid_search)

## Hogwild! (Parallel SGD)

*An optimizer for [sparse differentiable separable functions](#differentiable-separable-functions).*

An implementation of parallel stochastic gradient descent using the lock-free
HOGWILD! approach.  This implementation requires OpenMP to be enabled during
compilation (i.e., `-fopenmp` specified as a compiler flag).

Note that the requirements for Hogwild! are slightly different than for most
[differentiable separable functions](#differentiable-separable-functions) but it
is often possible to use Hogwild! by implementing `Gradient()` with a template
parameter.  See the [sparse differentiable separable
functions](#sparse-differentiable-separable-functions) documentation for more
details.

#### Constructors

 * `ParallelSGD<`_`DecayPolicyType`_`>(`_`maxIterations, threadShareSize`_`)`
 * `ParallelSGD<`_`DecayPolicyType`_`>(`_`maxIterations, threadShareSize, tolerance, shuffle, decayPolicy`_`)`

The _`DecayPolicyType`_ template parameter specifies the policy used to update
the step size after each iteration.  The `ConstantStep` class is available for
use.  Custom behavior can be achieved by implementing a class with the same
method signatures.

The default type for _`DecayPolicyType`_ is `ConstantStep`, so the shorter type
`ParallelSGD<>` can be used instead of the equivalent
`ParallelSGD<ConstantStep>`.

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `size_t` | **`maxIterations`** | Maximum number of iterations allowed (0 means no limit). | **n/a** |
| `size_t` | **`threadShareSize`** | Number of datapoints to be processed in one iteration by each thread. | **n/a** |
| `double` | **`tolerance`** | Maximum absolute tolerance to terminate the algorithm. | `1e-5` |
| `bool` | **`shuffle`** | If true, the function order is shuffled; otherwise, each function is visited in linear order. | `true` |
| `DecayPolicyType` | **`decayPolicy`** | An instantiated step size update policy to use. | `DecayPolicyType()` |

Attributes of the optimizer may also be modified via the member methods
`MaxIterations()`, `ThreadShareSize()`, `Tolerance()`, `Shuffle()`, and
`DecayPolicy()`.

Note that the default value for `decayPolicy` is the default constructor for the
`DecayPolicyType`.

#### Examples

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
GeneralizedRosenbrockFunction f(50); // 50-dimensional Rosenbrock function.
arma::mat coordinates = f.GetInitialPoint();

ParallelSGD<> optimizer(100000, f.NumFunctions(), 1e-5, true);
optimizer.Optimize(f, coordinates);
```

</details>

#### See also:

 * [SGD in Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
 * [SGD](#standard-sgd)
 * [HOGWILD!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent](https://arxiv.org/abs/1106.5730)
 * [Sparse differentiable separable functions](#sparse-differentiable-separable-functions)

## IQN

*An optimizer for [differentiable separable functions](#differentiable-separable-functions).*

The Incremental Quasi-Newton belongs to the family of stochastic and incremental
methods that have a cost per iteration independent of n. IQN iterations are a
stochastic version of BFGS iterations that use memory to reduce the variance of
stochastic approximations.

#### Constructors

 * `IQN()`
 * `IQN(`_`stepSize`_`)`
 * `IQN(`_`stepSize, batchSize, maxIterations, tolerance`_`)`

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `double` | **`stepSize`** | Step size for each iteration. | `0.01` |
| `size_t` | **`batchSize`** | Size of each batch. | `10` |
| `size_t` | **`maxIterations`** | Maximum number of iterations allowed (0 means no limit). | `100000` |
| `size_t` | **`tolerance`** | Maximum absolute tolerance to terminate algorithm. | `1e-5` |

Attributes of the optimizer may also be changed via the member methods
`StepSize()`, `BatchSize()`, `MaxIterations()`, and `Tolerance()`.

#### Examples:

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();

IQN optimizer(0.01, 1, 5000, 1e-5);
optimizer.Optimize(f, coordinates);
```

</details>

#### See also:

 * [IQN: An Incremental Quasi-Newton Method with Local Superlinear Convergence Rate](https://arxiv.org/abs/1702.00709)
 * [A Stochastic Quasi-Newton Method for Large-Scale Optimization](https://arxiv.org/abs/1401.7020)
 * [Differentiable functions](#differentiable-functions)

## Katyusha

*An optimizer for [differentiable separable functions](#differentiable-separable-functions).*

Katyusha is a direct, primal-only stochastic gradient method which uses a
"negative momentum" on top of Nesterov's momentum.  Two types are
available---one that uses a proximal update step, and one that uses the standard
update step.

#### Constructors

 * `KatyushaType<`_`proximal`_`>()`
 * `KatyushaType<`_`proximal`_`>(`_`convexity, lipschitz`_`)`
 * `KatyushaType<`_`proximal`_`>(`_`convexity, lipschitz, batchSize`_`)`
 * `KatyushaType<`_`proximal`_`>(`_`convexity, lipschitz, batchSize, maxIterations, innerIterations, tolerance, shuffle, exactObjective`_`)`

The _`proximal`_ template parameter is a boolean value (`true` or `false`) that
specifies whether or not the proximal update should be used.

For convenience the following typedefs have been defined:

 * `Katyusha` (equivalent to `KatyushaType<false>`): Katyusha with the standard update step
 * `KatyushaProximal` (equivalent to `KatyushaType<true>`): Katyusha with the proximal update step

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `double` | **`convexity`** | The regularization parameter. | `1.0` |
| `double` | **`lipschitz`** | The Lipschitz constant. | `10.0` |
| `size_t` | **`batchSize`** | Batch size to use for each step. | `32` |
| `size_t` | **`maxIterations`** | Maximum number of iterations allowed (0 means no limit). | `1000` |
| `size_t` | **`innerIterations`** | The number of inner iterations allowed (0 means n / batchSize). Note that the full gradient is only calculated in the outer iteration. | `0` |
| `double` | **`tolerance`** | Maximum absolute tolerance to terminate algorithm. | `1e-5` |
| `bool` | **`shuffle`** | If true, the function order is shuffled; otherwise, each function is visited in linear order. | `true` |
| `bool` | **`exactObjective`** | Calculate the exact objective (Default: estimate the final objective obtained on the last pass over the data). | `false` |


Attributes of the optimizer may also be changed via the member methods
`Convexity()`, `Lipschitz()`, `BatchSize()`, `MaxIterations()`,
`InnerIterations()`, `Tolerance()`, `Shuffle()`, and `ExactObjective()`.

#### Examples:

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();

// Without proximal update.
Katyusha optimizer(1.0, 10.0, 1, 100, 0, 1e-10, true);
optimizer.Optimize(f, coordinates);

// With proximal update.
KatyushaProximal proximalOptimizer(1.0, 10.0, 1, 100, 0, 1e-10, true);
proximalOptimizer.Optimize(f, coordinates);
```

</details>

#### See also:

 * [Katyusha: The First Direct Acceleration of Stochastic Gradient Methods](https://arxiv.org/abs/1603.05953)
 * [Stochastic gradient descent in Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
 * [Differentiable separable functions](#differentiable-separable-functions)

## L-BFGS

*An optimizer for [differentiable functions](#differentiable-functions)*

L-BFGS is an optimization algorithm in the family of quasi-Newton methods that approximates the Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm using a limited amount of computer memory.

#### Constructors

 * `L_BFGS()`
 * `L_BFGS(`_`numBasis, maxIterations`_`)`
 * `L_BFGS(`_`numBasis, maxIterations, armijoConstant, wolfe, minGradientNorm, factr, maxLineSearchTrials`_`)`
 * `L_BFGS(`_`numBasis, maxIterations, armijoConstant, wolfe, minGradientNorm, factr, maxLineSearchTrials, minStep, maxStep`_`)`

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `size_t` | **`numBasis`** | Number of memory points to be stored (default 10). | `10` |
| `size_t` | **`maxIterations`** | Maximum number of iterations for the optimization (0 means no limit and may run indefinitely). | `10000` |
| `double` | **`armijoConstant`** | Controls the accuracy of the line search routine for determining the Armijo condition. | `1e-4` |
| `double` | **`wolfe`** | Parameter for detecting the Wolfe condition. | `0.9` |
| `double` | **`minGradientNorm`** | Minimum gradient norm required to continue the optimization. | `1e-6` |
| `double` | **`factr`** | Minimum relative function value decrease to continue the optimization. | `1e-15` |
| `size_t` | **`maxLineSearchTrials`** | The maximum number of trials for the line search (before giving up). | `50` |
| `double` | **`minStep`** | The minimum step of the line search. | `1e-20` |
| `double` | **`maxStep`** | The maximum step of the line search. | `1e20` |

Attributes of the optimizer may also be changed via the member methods
`NumBasis()`, `MaxIterations()`, `ArmijoConstant()`, `Wolfe()`,
`MinGradientNorm()`, `Factr()`, `MaxLineSearchTrials()`, `MinStep()`, and
`MaxStep()`.

#### Examples:

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();

L_BFGS optimizer(20);
optimizer.Optimize(f, coordinates);
```

</details>

#### See also:

 * [The solution of non linear finite element equations](https://onlinelibrary.wiley.com/doi/full/10.1002/nme.1620141104)
 * [Updating Quasi-Newton Matrices with Limited Storage](https://www.jstor.org/stable/2006193)
 * [Limited-memory BFGS in Wikipedia](https://en.wikipedia.org/wiki/Limited-memory_BFGS)
 * [Differentiable functions](#differentiable-functions)

## Lookahead

*An optimizer for [differentiable separable functions](#differentiable-separable-functions).*

Lookahead is a stochastic gradient based optimization method which chooses a
search direction by looking ahead at the sequence of "fast weights" generated
by another optimizer.

#### Constructors
 * `Lookahead<>()`
 * `Lookahead<>(`_`stepSize`_`)`
 * `Lookahead<>(`_`stepSize, k`_`)`
 * `Lookahead<>(`_`stepSize, k, maxIterations, tolerance, decayPolicy, exactObjective`_`)`
 * `Lookahead<>(`_`baseOptimizer, stepSize, k, maxIterations, tolerance, decayPolicy, exactObjective`_`)`

Note that `Lookahead<>` is based on the templated type
`LookaheadType<`_`BaseOptimizerType, DecayPolicyType`_`>` with _`BaseOptimizerType`_` = Adam` and _`DecayPolicyType`_` = NoDecay`.

Any optimizer that implements the differentiable separable functions interface
can be paired with the `Lookahead` optimizer.

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `BaseOptimizerType` | **`baseOptimizer`** |  Optimizer for the forward step. | Adam |
| `double` | **`stepSize`** | Step size for each iteration. | `0.5` |
| `size_t` | **`k`** | The synchronization period. | `5` |
| `size_t` | **`maxIterations`** | Maximum number of iterations allowed (0 means no limit). | `100000` |
| `double` | **`tolerance`** | Maximum absolute tolerance to terminate algorithm. | `1e-5` |
| `DecayPolicyType` | **`decayPolicy`** | Instantiated decay policy used to adjust the step size. | `DecayPolicyType()` |
| `bool` | **`exactObjective`** | Calculate the exact objective (Default: estimate the final objective obtained on the last pass over the data). | `false` |

The attributes of the optimizer may also be modified via the member methods
`BaseOptimizer()`, `StepSize()`, `K()`, `MaxIterations()`,
`Tolerance()`, `DecayPolicy()` and `ExactObjective()`.

#### Examples

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();

Lookahead<> optimizer(0.5, 5, 100000, 1e-5);
optimizer.Optimize(f, coordinates);
```

</details>

#### See also:

 * [SGD in Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
 * [SGD](#standard-sgd)
 * [Lookahead Optimizer: k steps forward, 1 step back](https://arxiv.org/abs/1907.08610)
 * [Differentiable separable functions](#differentiable-separable-functions)

## LRSDP (low-rank SDP solver)

*An optimizer for [semidefinite programs](#semidefinite-programs).*

LRSDP is the implementation of Monteiro and Burer's formulation of low-rank
semidefinite programs (LR-SDP).  This solver uses the augmented Lagrangian
optimizer to solve low-rank semidefinite programs.

The assumption here is that the solution matrix for the SDP is low-rank.  If
this assumption is not true, the algorithm should not be expected to converge.

#### Constructors

 * `LRSDP<`_`SDPType`_`>()`

The _`SDPType`_ template parameter specifies the type of SDP to solve.  The
`SDP<arma::mat>` and `SDP<arma::sp_mat>` classes are available for use; these
represent SDPs with dense and sparse `C` matrices, respectively.  The `SDP<>`
class is detailed in the [semidefinite program
documentation](#semidefinite-programs).

Once the `LRSDP<>` object is constructed, the SDP may be specified by calling
the `SDP()` member method, which returns a reference to the _`SDPType`_.

#### Attributes

The attributes of the LRSDP optimizer may only be accessed via member methods.

| **type** | **method name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `size_t` | **`MaxIterations()`** | Maximum number of iterations before termination. | `1000` |
| `AugLagrangian` | **`AugLag()`** | The internally-held Augmented Lagrangian optimizer. | **n/a** |

#### See also:

 * [A Nonlinear Programming Algorithm for Solving Semidefinite Programs via Low-rank Factorization](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.682.1520&rep=rep1&type=pdf)
 * [Semidefinite programming on Wikipedia](https://en.wikipedia.org/wiki/Semidefinite_programming)
 * [Semidefinite programs](#semidefinite-programs) (includes example usage of `PrimalDualSolver`)

## Momentum SGD

*An optimizer for [differentiable separable functions](#differentiable-separable-functions).*

Stochastic Gradient Descent is a technique for minimizing a function which
can be expressed as a sum of other functions.  This is an SGD variant that uses
momentum for its updates.  Using momentum updates for parameter learning can
accelerate the rate of convergence, specifically in the cases where the surface
curves much more steeply (a steep hilly terrain with high curvature).

#### Constructors

 * `MomentumSGD()`
 * `MomentumSGD(`_`stepSize, batchSize`_`)`
 * `MomentumSGD(`_`stepSize, batchSize, maxIterations, tolerance, shuffle`_`)`
 * `MomentumSGD(`_`stepSize, batchSize, maxIterations, tolerance, shuffle, momentumPolicy, decayPolicy, resetPolicy, exactObjective`_`)`

Note that `MomentumSGD` is based on the templated type
`SGD<`_`UpdatePolicyType, DecayPolicyType`_`>` with _`UpdatePolicyType`_` =
MomentumUpdate` and _`DecayPolicyType`_` = NoDecay`.

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `double` | **`stepSize`** | Step size for each iteration. | `0.01` |
| `size_t` | **`batchSize`** | Batch size to use for each step. | `32` |
| `size_t` | **`maxIterations`** | Maximum number of iterations allowed (0 means no limit). | `100000` |
| `double` | **`tolerance`** | Maximum absolute tolerance to terminate algorithm. | `1e-5` |
| `bool` | **`shuffle`** | If true, the function order is shuffled; otherwise, each function is visited in linear order. | `true` |
| `MomentumUpdate` | **`updatePolicy`** | An instantiated `MomentumUpdate`. | `MomentumUpdate()` |
| `DecayPolicyType` | **`decayPolicy`** | Instantiated decay policy used to adjust the step size. | `DecayPolicyType()` |
| `bool` | **`resetPolicy`** | Flag that determines whether update policy parameters are reset before every Optimize call. | `true` |
| `bool` | **`exactObjective`** | Calculate the exact objective (Default: estimate the final objective obtained on the last pass over the data). | `false` |

Attributes of the optimizer may also be modified via the member methods
`StepSize()`, `BatchSize()`, `MaxIterations()`, `Tolerance()`, `Shuffle()`, `UpdatePolicy()`, `DecayPolicy()`, `ResetPolicy()`, and
`ExactObjective()`.

Note that the `MomentumUpdate` class has the constructor
`MomentumUpdate(`_`momentum`_`)` with a default value of `0.5` for the momentum.

#### Examples

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();

MomentumSGD optimizer(0.01, 32, 100000, 1e-5, true, MomentumUpdate(0.5));
optimizer.Optimize(f, coordinates);
```

</details>

#### See also:

 * [Standard SGD](#standard-sgd)
 * [Nesterov Momentum SGD](#nesterov-momentum-sgd)
 * [SGD in Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
 * [Differentiable separable functions](#differentiable-separable-functions)

## Nadam

*An optimizer for [differentiable separable functions](#differentiable-separable-functions).*

Nadam is a variant of Adam based on NAG (Nesterov accelerated gradient).  It
uses Nesterov momentum for faster convergence.

#### Constructors

 * `Nadam()`
 * `Nadam(`_`stepSize, batchSize`_`)`
 * `Nadam(`_`stepSize, batchSize, beta1, beta2, eps, maxIterations, tolerance, shuffle`_`)`
 * `Nadam(`_`stepSize, batchSize, beta1, beta2, eps, maxIterations, tolerance, shuffle, resetPolicy`_`)`

Note that the `Nadam` class is based on the `AdamType<`_`UpdateRule`_`>` class
with _`UpdateRule`_` = NadamUpdate`.

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `double` | **`stepSize`** | Step size for each iteration. | `0.001` |
| `size_t` | **`batchSize`** | Number of points to process in a single step. | `32` |
| `double` | **`beta1`** | Exponential decay rate for the first moment estimates. | `0.9` |
| `double` | **`beta2`** | Exponential decay rate for the weighted infinity norm estimates. | `0.999` |
| `double` | **`eps`** | Value used to initialize the mean squared gradient parameter. | `1e-8` |
| `size_t` | **`maxIterations`** | Maximum number of iterations allowed (0 means no limit). | `100000` |
| `double` | **`tolerance`** | Maximum absolute tolerance to terminate algorithm. | `1e-5` |
| `bool` | **`shuffle`** | If true, the function order is shuffled; otherwise, each function is visited in linear order. | `true` |
| `bool` | **`resetPolicy`** | If true, parameters are reset before every Optimize call; otherwise, their values are retained. | `true` |

The attributes of the optimizer may also be modified via the member methods
`StepSize()`, `BatchSize()`, `Beta1()`, `Beta2()`, `Eps()`, `MaxIterations()`,
`Tolerance()`, `Shuffle()`, and `ResetPolicy()`.

#### Examples

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();

Nadam optimizer(0.001, 32, 0.9, 0.999, 1e-8, 100000, 1e-5, true);
optimizer.Optimize(f, coordinates);
```

</details>

#### See also:

 * [SGD in Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
 * [SGD](#standard-sgd)
 * [Incorporating Nesterov Momentum into Adam](http://cs229.stanford.edu/proj2015/054_report.pdf)
 * [Differentiable separable functions](#differentiable-separable-functions)

## NadaMax

*An optimizer for [differentiable separable functions](#differentiable-separable-functions).*

NadaMax is a variant of AdaMax based on NAG (Nesterov accelerated gradient).  It
uses Nesterov momentum for faster convergence.

#### Constructors

 * `NadaMax()`
 * `NadaMax(`_`stepSize, batchSize`_`)`
 * `NadaMax(`_`stepSize, batchSize, beta1, beta2, eps, maxIterations, tolerance, shuffle`_`)`
 * `NadaMax(`_`stepSize, batchSize, beta1, beta2, eps, maxIterations, tolerance, shuffle, resetPolicy`_`)`

Note that the `NadaMax` class is based on the `AdamType<`_`UpdateRule`_`>` class
with _`UpdateRule`_` = NadaMaxUpdate`.

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `double` | **`stepSize`** | Step size for each iteration. | `0.001` |
| `size_t` | **`batchSize`** | Number of points to process in a single step. | `32` |
| `double` | **`beta1`** | Exponential decay rate for the first moment estimates. | `0.9` |
| `double` | **`beta2`** | Exponential decay rate for the weighted infinity norm estimates. | `0.999` |
| `double` | **`eps`** | Value used to initialize the mean squared gradient parameter. | `1e-8` |
| `size_t` | **`maxIterations`** | Maximum number of iterations allowed (0 means no limit). | `100000` |
| `double` | **`tolerance`** | Maximum absolute tolerance to terminate algorithm. | `1e-5` |
| `bool` | **`shuffle`** | If true, the function order is shuffled; otherwise, each function is visited in linear order. | `true` |
| `bool` | **`resetPolicy`** | If true, parameters are reset before every Optimize call; otherwise, their values are retained. | `true` |

The attributes of the optimizer may also be modified via the member methods
`StepSize()`, `BatchSize()`, `Beta1()`, `Beta2()`, `Eps()`, `MaxIterations()`,
`Tolerance()`, `Shuffle()`, and `ResetPolicy()`.

#### Examples

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();

NadaMax optimizer(0.001, 32, 0.9, 0.999, 1e-8, 100000, 1e-5, true);
optimizer.Optimize(f, coordinates);
```

</details>

#### See also:

 * [SGD in Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
 * [SGD](#standard-sgd)
 * [Incorporating Nesterov Momentum into Adam](http://cs229.stanford.edu/proj2015/054_report.pdf)
 * [Differentiable separable functions](#differentiable-separable-functions)

## Nesterov Momentum SGD

*An optimizer for [differentiable separable functions](#differentiable-separable-functions).*

Stochastic Gradient Descent is a technique for minimizing a function which
can be expressed as a sum of other functions.  This is an SGD variant that uses
Nesterov momentum for its updates.  Nesterov Momentum application can accelerate
the rate of convergence to O(1/k^2).

#### Constructors

 * `NesterovMomentumSGD()`
 * `NesterovMomentumSGD(`_`stepSize, batchSize`_`)`
 * `NesterovMomentumSGD(`_`stepSize, batchSize, maxIterations, tolerance, shuffle`_`)`
 * `NesterovMomentumSGD(`_`stepSize, batchSize, maxIterations, tolerance, shuffle, momentumPolicy, decayPolicy, resetPolicy, exactObjective`_`)`

Note that `MomentumSGD` is based on the templated type
`SGD<`_`UpdatePolicyType, DecayPolicyType`_`>` with _`UpdatePolicyType`_` =
NesterovMomentumUpdate` and _`DecayPolicyType`_` = NoDecay`.

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `double` | **`stepSize`** | Step size for each iteration. | `0.01` |
| `size_t` | **`batchSize`** | Batch size to use for each step. | `32` |
| `size_t` | **`maxIterations`** | Maximum number of iterations allowed (0 means no limit). | `100000` |
| `double` | **`tolerance`** | Maximum absolute tolerance to terminate algorithm. | `1e-5` |
| `bool` | **`shuffle`** | If true, the function order is shuffled; otherwise, each function is visited in linear order. | `true` |
| `NesterovMomentumUpdate` | **`updatePolicy`** | An instantiated `MomentumUpdate`. | `NesterovMomentumUpdate()` |
| `DecayPolicyType` | **`decayPolicy`** | Instantiated decay policy used to adjust the step size. | `DecayPolicyType()` |
| `bool` | **`resetPolicy`** | Flag that determines whether update policy parameters are reset before every Optimize call. | `true` |
| `bool` | **`exactObjective`** | Calculate the exact objective (Default: estimate the final objective obtained on the last pass over the data). | `false` |

Attributes of the optimizer may also be modified via the member methods
`StepSize()`, `BatchSize()`, `MaxIterations()`, `Tolerance()`, `Shuffle()`, `UpdatePolicy()`, `DecayPolicy()`, `ResetPolicy()`, and
`ExactObjective()`.

Note that the `NesterovMomentumUpdate` class has the constructor
`MomentumUpdate(`_`momentum`_`)` with a default value of `0.5` for the momentum.

#### Examples

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();

NesterovMomentumSGD optimizer(0.01, 32, 100000, 1e-5, true,
    MomentumUpdate(0.5));
optimizer.Optimize(f, coordinates);
```

</details>

#### See also:

 * [Standard SGD](#standard-sgd)
 * [Momentum SGD](#momentum-sgd)
 * [SGD in Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
 * [Differentiable separable functions](#differentiable-separable-functions)

## MOEA/D-DE
*An optimizer for arbitrary multi-objective functions.*
MOEA/D-DE (Multi Objective Evolutionary Algorithm based on Decomposition - Differential Evolution) is a multi
objective optimization algorithm. It works by decomposing the problem into a number of scalar optimization
subproblems which are solved simultaneously per generation. MOEA/D in itself is a framework, this particular
algorithm uses Differential Crossover followed by Polynomial Mutation to create offsprings which are then
decomposed to form a Single Objective Problem. A diversity preserving mechanism is also employed which encourages
a varied set of solution.

#### Constructors
* `MOEAD<`_`InitPolicyType, DecompPolicyType`_`>()`
* `MOEAD<`_`InitPolicyType, DecompPolicyType`_`>(`_`populationSize, maxGenerations, crossoverProb,  neighborProb, neighborSize, distributionIndex, differentialWeight, maxReplace, epsilon, lowerBound, upperBound`_`)`

The _`InitPolicyType`_ template parameter refers to the strategy used to
initialize the reference directions.

The following types are available:

 * **`Uniform`**
 * **`BayesianBootstrap`**
 * **`Dirichlet`**

The _`DecompPolicyType`_ template parameter refers to the strategy used to
decompose the weight vectors to form a scalar objective function.

The following types are available:

 * **`Tchebycheff`**
 * **`WeightedAverage`**
 * **`PenaltyBoundaryIntersection`**

For convenience the following types can be used:

 * **`DefaultMOEAD`** (equivalent to `MOEAD<Uniform, Tchebycheff>`): utilizes Uniform method for weight initialization
 and Tchebycheff for weight decomposition.

 * **`BBSMOEAD`** (equivalent to `MOEAD<BayesianBootstrap, Tchebycheff>`): utilizes Bayesian Bootstrap method for weight initialization and Tchebycheff for weight decomposition.

 * **`DirichletMOEAD`** (equivalent to `MOEAD<Dirichlet, Tchebycheff>`): utilizes Dirichlet sampling for weight init
 and Tchebycheff for weight decomposition.

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `size_t` | **`populationSize`** | The number of candidates in the population. | `150` |
| `size_t` | **`maxGenerations`** | The maximum number of generations allowed. | `300` |
| `double` | **`crossoverProb`** | Probability that a crossover will occur. | `1.0` |
| `double` | **`neighborProb`** | The probability of sampling from neighbor. | `0.9` |
| `size_t` | **`neighborSize`** | The number of nearest-neighbours to consider per weight vector.  | `20` |
| `double` | **`distributionIndex`** | The crowding degree of the mutation. | `20` |
| `double` | **`differentialWeight`** | Amplification factor of the differentiation. | `0.5` |
| `size_t` | **`maxReplace`** | The limit of solutions allowed to be replaced by a child. | `2`|
| `double` | **`epsilon`** | Handles numerical stability after weight initialization. | `1E-10`|
| `double`, `arma::vec` | **`lowerBound`** | Lower bound of the coordinates on the coordinates of the whole population during the search process. | `0` |
| `double`, `arma::vec` | **`upperBound`** | Lower bound of the coordinates on the coordinates of the whole population during the search process. | `1` |
| `InitPolicyType` | **`initPolicy`** | Instantiated init policy used to initialize weights. | `InitPolicyType()` |
| `DecompPolicyType` | **`decompPolicy`** | Instantiated decomposition policy used to create scalar objective problem. | `DecompPolicyType()` |

Attributes of the optimizer may also be changed via the member methods
`PopulationSize()`, `MaxGenerations()`, `CrossoverRate()`, `NeighborProb()`, `NeighborSize()`, `DistributionIndex()`,
`DifferentialWeight()`, `MaxReplace()`, `Epsilon()`, `LowerBound()`, `UpperBound()`, `InitPolicy()` and `DecompPolicy()`.

#### Examples:

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
SchafferFunctionN1<arma::mat> SCH;
arma::vec lowerBound("-10");
arma::vec upperBound("10");
DefaultMOEAD opt(300, 300, 1.0, 0.9, 20, 20, 0.5, 2, 1E-10, lowerBound, upperBound);
typedef decltype(SCH.objectiveA) ObjectiveTypeA;
typedef decltype(SCH.objectiveB) ObjectiveTypeB;
arma::mat coords = SCH.GetInitialPoint();
std::tuple<ObjectiveTypeA, ObjectiveTypeB> objectives = SCH.GetObjectives();
// obj will contain the minimum sum of objectiveA and objectiveB found on the best front.
double obj = opt.Optimize(objectives, coords);
// Now obtain the best front.
arma::cube bestFront = opt.ParetoFront();
```
</details>

#### See also
* [MOEA/D-DE Algorithm](https://ieeexplore.ieee.org/document/4633340)
* [Multi-objective Functions in Wikipedia](https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_multi-objective_optimization)
* [Multi-objective functions](#multi-objective-functions)

## NSGA2

*An optimizer for arbitrary multi-objective functions.*

NSGA2 (Non-dominated Sorting Genetic Algorithm - II) is a multi-objective
optimization algorithm. The algorithm works by generating a candidate population
from a fixed starting point. At each stage of optimization, a new population of
children is generated. This new population along with its predecessor is sorted
using non-domination as the metric. Following this, the population is further
segregated into fronts. A new population is generated from these fronts having
size equal to that of the starting population.

#### Constructors

 * `NSGA2()`
 * `NSGA2(`_`populationSize, maxGenerations, crossoverProb, mutationProb, mutationStrength, epsilon, lowerBound, upperBound`_`)`

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `size_t` | **`populationSize`** | The number of candidates in the population. This should be at least 4 in size and a multiple of 4. | `100` |
| `size_t` | **`maxGenerations`** | The maximum number of generations allowed for NSGA2. | `2000` |
| `double` | **`crossoverProb`** | Probability that a crossover will occur. | `0.6` |
| `double` | **`mutationProb`** | Probability that a weight will get mutated. | `0.3` |
| `double` | **`mutationStrength`** | The range of mutation noise to be added. This range is between 0 and mutationStrength. | `0.001` |
| `double` | **`epsilon`** | The value used internally to evaluate approximate equality in crowding distance based sorting. | `1e-6` |
| `double`, `arma::vec` | **`lowerBound`** | Lower bound of the coordinates on the coordinates of the whole population during the search process. | `0` |
| `double`, `arma::vec` | **`upperBound`** | Lower bound of the coordinates on the coordinates of the whole population during the search process. | `1` |

Note that the parameters `lowerBound` and `upperBound` are overloaded. Data types of `double` or `arma::mat` may be used. If they are initialized as single values of `double`, then the same value of the bound applies to all the axes, resulting in an initialization following a uniform distribution in a hypercube. If they are initialized as matrices of `arma::mat`, then the value of `lowerBound[i]` applies to axis `[i]`; similarly, for values in `upperBound`. This results in an initialization following a uniform distribution in a hyperrectangle within the specified bounds.

Attributes of the optimizer may also be changed via the member methods
`PopulationSize()`, `MaxGenerations()`, `CrossoverRate()`, `MutationProbability()`, `MutationStrength()`, `Epsilon()`, `LowerBound()` and `UpperBound()`.

#### Examples:

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
SchafferFunctionN1<arma::mat> SCH;
arma::vec lowerBound("-1000");
arma::vec upperBound("1000");
NSGA2 opt(20, 5000, 0.5, 0.5, 1e-3, 1e-6, lowerBound, upperBound);

typedef decltype(SCH.objectiveA) ObjectiveTypeA;
typedef decltype(SCH.objectiveB) ObjectiveTypeB;

arma::mat coords = SCH.GetInitialPoint();
std::tuple<ObjectiveTypeA, ObjectiveTypeB> objectives = SCH.GetObjectives();

// obj will contain the minimum sum of objectiveA and objectiveB found on the best front.
double obj = opt.Optimize(objectives, coords);
// Now obtain the best front.
arma::cube bestFront = opt.ParetoFront();
```

</details>

#### See also:

 * [NSGA-II Algorithm](https://www.iitk.ac.in/kangal/Deb_NSGA-II.pdf)
 * [Multi-objective Functions in Wikipedia](https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_multi-objective_optimization)
  * [Multi-objective functions](#multi-objective-functions)

## OptimisticAdam

*An optimizer for [differentiable separable functions](#differentiable-separable-functions).*

OptimisticAdam is an optimizer which implements the Optimistic Adam algorithm
which uses Optmistic Mirror Descent with the Adam Optimizer.  It addresses the
problem of limit cycling while training GANs (generative adversarial networks).
It uses OMD to achieve faster regret rates in solving the zero sum game of
training a GAN. It consistently achieves a smaller KL divergence with~ respect to
the true underlying data distribution.  The implementation here can be used with
any differentiable separable function, not just GAN training.

#### Constructors

 * `OptimisticAdam()`
 * `OptimisticAdam(`_`stepSize, batchSize`_`)`
 * `OptimisticAdam(`_`stepSize, batchSize, beta1, beta2, eps, maxIterations, tolerance, shuffle`_`)`
 * `OptimisticAdam(`_`stepSize, batchSize, beta1, beta2, eps, maxIterations, tolerance, shuffle, resetPolicy`_`)`

Note that the `OptimisticAdam` class is based on the
`AdamType<`_`UpdateRule`_`>` class with _`UpdateRule`_` = OptimisticAdamUpdate`.

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `double` | **`stepSize`** | Step size for each iteration. | `0.001` |
| `size_t` | **`batchSize`** | Number of points to process in a single step. | `32` |
| `double` | **`beta1`** | Exponential decay rate for the first moment estimates. | `0.9` |
| `double` | **`beta2`** | Exponential decay rate for the weighted infinity norm estimates. | `0.999` |
| `double` | **`eps`** | Value used to initialize the mean squared gradient parameter. | `1e-8` |
| `size_t` | **`maxIterations`** | Maximum number of iterations allowed (0 means no limit). | `100000` |
| `double` | **`tolerance`** | Maximum absolute tolerance to terminate algorithm. | `1e-5` |
| `bool` | **`shuffle`** | If true, the function order is shuffled; otherwise, each function is visited in linear order. | `true` |
| `bool` | **`resetPolicy`** | If true, parameters are reset before every Optimize call; otherwise, their values are retained. | `true` |

The attributes of the optimizer may also be modified via the member methods
`StepSize()`, `BatchSize()`, `Beta1()`, `Beta2()`, `Eps()`, `MaxIterations()`,
`Tolerance()`, `Shuffle()`, and `ResetPolicy()`.

#### Examples

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();

OptimisticAdam optimizer(0.001, 32, 0.9, 0.999, 1e-8, 100000, 1e-5, true);
optimizer.Optimize(f, coordinates);
```

</details>

#### See also:

 * [SGD in Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
 * [SGD](#standard-sgd)
 * [Training GANs with Optimism](https://arxiv.org/pdf/1711.00141.pdf)
 * [Differentiable separable functions](#differentiable-separable-functions)

## Padam

*An optimizer for [differentiable separable functions](#differentiable-separable-functions).*

Padam is a variant of Adam with a partially adaptive momentum estimation method.

#### Constructors

 * `Padam()`
 * `Padam(`_`stepSize, batchSize`_`)`
 * `Padam(`_`stepSize, batchSize, beta1, beta2, partial, epsilon, maxIterations, tolerance, shuffle, resetPolicy, exactObjective`_`)`

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `double` | **`stepSize`** | Step size for each iteration. | `0.001` |
| `size_t` | **`batchSize`** | Number of points to process in a single step. | `32` |
| `double` | **`beta1`** | Exponential decay rate for the first moment estimates. | `0.9` |
| `double` | **`beta2`** | Exponential decay rate for the weighted infinity norm estimates. | `0.999` |
| `double` | **`partial`** | Partially adaptive parameter. | `0.25` |
| `double` | **`eps`** | Value used to initialize the mean squared gradient parameter. | `1e-8` |
| `size_t` | **`maxIterations`** | Maximum number of iterations allowed (0 means no limit). | `100000` |
| `double` | **`tolerance`** | Maximum absolute tolerance to terminate algorithm. | `1e-5` |
| `bool` | **`shuffle`** | If true, the function order is shuffled; otherwise, each function is visited in linear order. | `true` |
| `bool` | **`resetPolicy`** | If true, parameters are reset before every Optimize call; otherwise, their values are retained. | `true` |
| `bool` | **`exactObjective`** | Calculate the exact objective (Default: estimate the final objective obtained on the last pass over the data). | `false` |

The attributes of the optimizer may also be modified via the member methods
`StepSize()`, `BatchSize()`, `Beta1()`, `Beta2()`, `Partial()`, `Epsilon()`,
`MaxIterations()`, `Tolerance()`, `Shuffle()`, `ResetPolicy()`, and `ExactObjective()`.

#### Examples

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();

Padam optimizer(0.001, 32, 0.9, 0.999, 0.25, 1e-8, 100000, 1e-5, true);
optimizer.Optimize(f, coordinates);
```

</details>

#### See also:
 * [Closing the Generalization Gap of Adaptive Gradient Methods in Training Deep Neural Networks](https://arxiv.org/abs/1806.06763)
 * [SGD in Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
 * [SGD](#standard-sgd)
 * [Adam: A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980)
 * [Differentiable separable functions](#differentiable-separable-functions)

## PSO

*An optimizer for [arbitrary functions](#arbitrary-functions).*

PSO is an evolutionary approach to optimization that is inspired by flocks or birds or fishes. The fundamental analogy is that every creature (particle in a swarm) is at a measurable position of goodness or fitness, and this information can be shared amongst the creatures in the flock, so that iteratively, the entire flock can get close to the global optimum.

#### Constructors

 * `PSOType<`_`VelocityUpdatePolicy, InitPolicy`_`>()`
 * `PSOType<`_`VelocityUpdatePolicy, InitPolicy`_`>(`_`numParticles`_`)`
 * `PSOType<`_`VelocityUpdatePolicy, InitPolicy`_`>(`_`numParticles, lowerBound, upperBound`_`)`
 * `PSOType<`_`VelocityUpdatePolicy, InitPolicy`_`>(`_`numParticles, lowerBound, upperBound, maxIterations`_`)`
 * `PSOType<`_`VelocityUpdatePolicy, InitPolicy`_`>(`_`numParticles, lowerBound, upperBound, maxIterations, horizonSize`_`)`
 * `PSOType<`_`VelocityUpdatePolicy, InitPolicy`_`>(`_`numParticles, lowerBound, upperBound, maxIterations, horizonSize, impTolerance`_`)`
 * `PSOType<`_`VelocityUpdatePolicy, InitPolicy`_`>(`_`numParticles, lowerBound, upperBound, maxIterations, horizonSize, impTolerance, exploitationFactor, explorationFactor`_`)`

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `size_t` | **`numParticles`** |  numParticles Number of particles in the swarm. | `64` |
| `double`, `arma::mat` | **`lowerBound`** | Lower bound of the coordinates of the initial population. | `1` |
| `double`, `arma::mat` | **`upperBound`** |  Upper bound of the coordinates of the initial population. | `1` |
| `size_t` | **`maxIterations`** | Maximum number of iterations allowed. | `3000` |
| `size_t` | **`horizonSize`** | Size of the lookback-horizon for computing improvement. | `350` |
| `double` | **`impTolerance`** | The final value of the objective function for termination. If set to negative value, tolerance is not considered. | `1e-5` |
| `double` | **`exploitationFactor`** | Influence of the personal best of the particle. | `2.05` |
| `double` | **`explorationFactor`** | Influence of the neighbours of the particle. | `2.05` |

Note that the parameters `lowerBound` and `upperBound` are overloaded. Data types of `double` or `arma::mat` may be used. If they are initialized as single values of `double`, then the same value of the bound applies to all the axes, resulting in an initialization following a uniform distribution in a hypercube. If they are initialized as matrices of `arma::mat`, then the value of `lowerBound[i]` applies to axis `[i]`; similarly, for values in `upperBound`. This results in an initialization following a uniform distribution in a hyperrectangle within the specified bounds.

Attributes of the optimizer may also be changed via the member methods
`NumParticles()`, `LowerBound()`, `UpperBound()`, `MaxIterations()`,
`HorizonSize()`, `ImpTolerance()`,`ExploitationFactor()`, and
`ExplorationFactor()`.

At present, only the local-best variant of PSO is present in ensmallen. The optimizer may be initialized using the class type `LBestPSO`, which is an alias for `PSOType<LBestUpdate, DefaultInit>`.

#### Examples:

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
SphereFunction f(4);
arma::vec coordinates = f.GetInitialPoint();

LBestPSO s;
const double result = s.Optimize(f, coordinates)
```

</details>

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
RosenbrockFunction f;
arma::vec coordinates = f.GetInitialPoint();

// Setting bounds for the initial swarm population of size 2.
arma::vec lowerBound("50 50");
arma::vec upperBound("60 60");

LBestPSO s(200, lowerBound, upperBound, 3000, 600, 1e-30, 2.05, 2.05);
const double result = s.Optimize(f, coordinates)
```

</details>

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
RosenbrockFunction f;
arma::vec coordinates = f.GetInitialPoint();

// Setting bounds for the initial swarm population as type double.
double lowerBound = 50;
double upperBound = 60;

LBestPSO s(64, lowerBound, upperBound, 3000, 400, 1e-30, 2.05, 2.05);
const double result = s.Optimize(f, coordinates)
```

</details>

#### See also:

 * [Particle Swarm Optimization](http://www.swarmintelligence.org/)
 * [Arbitrary functions](#arbitrary-functions)


## Primal-dual SDP Solver

*An optimizer for [semidefinite programs](#semidefinite-programs).*

A primal-dual interior point method solver.  This can solve semidefinite
programs.

#### Constructors

 * `PrimalDualSolver<>(`_`maxIterations`_`)`
 * `PrimalDualSolver<>(`_`maxIterations, tau, normXzTol, primalInfeasTol, dualInfeasTol`_`)`

#### Attributes

The `PrimalDualSolver<>` class has several attributes that are only modifiable
as member methods.

| **type** | **method name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `double` | **`Tau()`** | Value of tau used to compute alpha\_hat. | `0.99` |
| `double` | **`NormXZTol()`** | Tolerance for the norm of X\*Z. | `1e-7` |
| `double` | **`PrimalInfeasTol()`** | Tolerance for primal infeasibility. | `1e-7` |
| `double` | **`DualInfeasTol()`** | Tolerance for dual infeasibility. | `1e-7` |
| `size_t` | **`MaxIterations()`** | Maximum number of iterations before convergence. | `1000` |

#### Optimization

The `PrimalDualSolver<>` class offers two overloads of `Optimize()` that
optionally return the converged values for the dual variables.

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
/**
 * Invoke the optimization procedure, returning the converged values for the
 * primal and dual variables.
 */
template<typename SDPType>
double Optimize(SDPType& s,
                arma::mat& X,
                arma::vec& ySparse,
                arma::vec& yDense,
                arma::mat& Z);

/**
 * Invoke the optimization procedure, and only return the primal variable.
 */
template<typename SDPType>
double Optimize(SDPType& s, arma::mat& X);
```

</details>

The _`SDPType`_ template parameter specifies the type of SDP to solve.  The
`SDP<arma::mat>` and `SDP<arma::sp_mat>` classes are available for use; these
represent SDPs with dense and sparse `C` matrices, respectively.  The `SDP<>`
class is detailed in the [semidefinite program
documentation](#semidefinite-programs).  _`SDPType`_ is automatically inferred
when `Optimize()` is called with an SDP.

#### See also:

 * [Primal-dual interior-point methods for semidefinite programming](http://www.dtic.mil/dtic/tr/fulltext/u2/1020236.pdf)
 * [Semidefinite programming on Wikipedia](https://en.wikipedia.org/wiki/Semidefinite_programming)
 * [Semidefinite programs](#semidefinite-programs) (includes example usage of `PrimalDualSolver`)

## Quasi-Hyperbolic Momentum Update SGD (QHSGD)

*An optimizer for [differentiable separable
functions](#differentiable-separable-functions).*

Quasi-hyperbolic momentum update SGD (QHSGD) is an SGD-like optimizer with
momentum where quasi-hyperbolic terms are added to the parametrization.  The
update rule for this optimizer is a weighted average of momentum SGD and vanilla
SGD.

#### Constructors

  * `QHSGD()`
  * `QHSGD(`_`stepSize, batchSize`_`)`
  * `QHSGD(`_`stepSize, batchSize, maxIterations, tolerance, shuffle, exactObjective`_`)`

 Note that `QHSGD` is based on the templated type
 `SGD<`_`UpdatePolicyType, DecayPolicyType`_`>` with _`UpdatePolicyType`_` =
 QHUpdate` and _`DecayPolicyType`_` = NoDecay`.

#### Attributes

 | **type** | **name** | **description** | **default** |
 |----------|----------|-----------------|-------------|
 | `double` | **`stepSize`** | Step size for each iteration. | `0.01` |
 | `size_t` | **`batchSize`** | Batch size to use for each step. | `32` |
 | `size_t` | **`maxIterations`** | Maximum number of iterations allowed (0 means no limit). | `100000` |
 | `double` | **`tolerance`** | Maximum absolute tolerance to terminate algorithm. | `1e-5` |
 | `bool` | **`shuffle`** | If true, the function order is shuffled; otherwise, each function is visited in linear order. | `true` |
 | `bool` | **`exactObjective`** | Calculate the exact objective (Default: estimate the final objective obtained on the last pass over the data). | `false` |

 Attributes of the optimizer may also be modified via the member methods
 `StepSize()`, `BatchSize()`, `MaxIterations()`, `Tolerance()`, `Shuffle()`,  and `ExactObjective()`.

 Note that the `QHUpdate` class has the constructor  `QHUpdate(`_`v,
momentum`_`)` with a default value of `0.7` for the quasi-hyperbolic term `v`
and `0.999` for the momentum term.

#### Examples

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();

QHSGD optimizer(0.01, 32, 100000, 1e-5, true);
optimizer.Optimize(f, coordinates);
```

</details>

#### See also:

  * [Quasi-Hyperbolic Momentum and Adam For Deep Learning](https://arxiv.org/pdf/1810.06801.pdf)
  * [SGD](#sgd)
  * [Momentum SGD](#momentum-sgd)
  * [Nesterov Momentum SGD](#nesterov-momentum-sgd)
  * [SGD in Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
  * [Differentiable separable functions](#differentiable-separable-functions)

## QHAdam

*An optimizer for [differentiable separable functions](#differentiable-separable-functions).*

QHAdam is an optimizer that uses quasi-hyperbolic descent with the Adam
optimizer.  This replaces the moment estimators of Adam with quasi-hyperbolic
terms, and various values of the `v1` and `v2` parameters are equivalent to
the following other optimizers:

 * When `v1 = v2 = 1`, `QHAdam` is equivalent to `Adam`.

 * When `v1 = 0` and `v2 = 1`, `QHAdam` is equivalent to `RMSProp`.

 * When `v1 = beta1` and `v2 = 1`, `QHAdam` is equivalent to `Nadam`.

#### Constructors

  * `QHAdam()`
  * `QHAdam(`_`stepSize, batchSize`_`)`
  * `QHAdam(`_`stepSize, batchSize, v1, v2, beta1, beta2, eps, maxIterations`_`)`
  * `QHAdam(`_`stepSize, batchSize, v1, v2, beta1, beta2, eps, maxIterations, tolerance, shuffle, resetPolicy, exactObjective`_`)`

#### Attributes

 | **type** | **name** | **description** | **default** |
 |----------|----------|-----------------|-------------|
 | `double` | **`stepSize`** | Step size for each iteration. | `0.001` |
 | `size_t` | **`batchSize`** | Number of points to process in a single step. | `32` |
 | `double` | **`v1`** | The First Quasi Hyperbolic Term. | `0.7` |
 | `double` | **`v2`** | The Second Quasi Hyperbolic Term. | `1.00` |
 | `double` | **`beta1`** | Exponential decay rate for the first moment estimates. | `0.9` |
 | `double` | **`beta2`** | Exponential decay rate for the weighted infinity norm estimates. | `0.999` |
 | `double` | **`eps`** | Value used to initialize the mean squared gradient parameter. | `1e-8` |
 | `size_t` | **`maxIterations`** | Maximum number of iterations allowed (0 means no limit). | `100000` |
 | `double` | **`tolerance`** | Maximum absolute tolerance to terminate algorithm. | `1e-5` |
 | `bool` | **`shuffle`** | If true, the function order is shuffled; otherwise, each function is visited in linear order. | `true` |
 | `bool` | **`resetPolicy`** | If true, parameters are reset before every Optimize call; otherwise, their values are retained. | `true` |
 | `bool` | **`exactObjective`** | Calculate the exact objective (Default: estimate the final objective obtained on the last pass over the data). | `false` |

 The attributes of the optimizer may also be modified via the member methods
 `StepSize()`, `BatchSize()`, `Beta1()`, `Beta2()`, `Eps()`, `MaxIterations()`,
 `Tolerance()`, `Shuffle()`, `V1()`, `V2()`, `ResetPolicy()`, and `ExactObjective()`.

#### Examples

 ```c++
 RosenbrockFunction f;
 arma::mat coordinates = f.GetInitialPoint();

 QHAdam optimizer(0.001, 32, 0.7, 0.9, 0.9, 0.999, 1e-8, 100000, 1e-5, true);
 optimizer.Optimize(f, coordinates);
 ```

#### See also:

  * [Quasi-Hyperbolic Momentum and Adam For Deep Learning](https://arxiv.org/pdf/1810.06801.pdf)
  * [SGD in Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
  * [SGD](#standard-sgd)
  * [Adam](#adam)
  * [RMSprop](#rmsprop)
  * [Nadam](#nadam)
  * [Incorporating Nesterov Momentum into Adam](http://cs229.stanford.edu/proj2015/054_report.pdf)
  * [Differentiable separable functions](#differentiable-separable-functions)

## RMSProp

*An optimizer for [differentiable separable functions](#differentiable-separable-functions).*

RMSProp utilizes the magnitude of recent gradients to normalize the gradients.

#### Constructors

 * `RMSProp()`
 * `RMSProp(`_`stepSize, batchSize`_`)`
 * `RMSProp(`_`stepSize, batchSize, alpha, epsilon, maxIterations, tolerance, shuffle`_`)`
 * `RMSProp(`_`stepSize, batchSize, alpha, epsilon, maxIterations, tolerance, shuffle, resetPolicy, exactObjective`_`)`

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `double` | **`stepSize`** | Step size for each iteration. | `0.01` |
| `size_t` | **`batchSize`** | Number of points to process in each step. | `32` |
| `double` | **`alpha`** | Smoothing constant, similar to that used in AdaDelta and momentum methods. | `0.99` |
| `double` | **`epsilon`** | Value used to initialize the mean squared gradient parameter. | `1e-8` |
| `size_t` | **`maxIterations`** | Maximum number of iterations allowed (0 means no limit). | `100000` |
| `double` | **`tolerance`** | Maximum absolute tolerance to terminate algorithm. |
| `bool` | **`shuffle`** | If true, the function order is shuffled; otherwise, each function is visited in linear order. | `true` |
| `bool` | **`resetPolicy`** | If true, parameters are reset before every Optimize call; otherwise, their values are retained. | `true` |
| `bool` | **`exactObjective`** | Calculate the exact objective (Default: estimate the final objective obtained on the last pass over the data). | `false` |

Attributes of the optimizer can also be modified via the member methods
`StepSize()`, `BatchSize()`, `Alpha()`, `Epsilon()`, `MaxIterations()`,
`Tolerance()`, `Shuffle()`, `ResetPolicy()`, and `ExactObjective()`.

#### Examples:

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();

RMSProp optimizer(1e-3, 1, 0.99, 1e-8, 5000000, 1e-9, true);
optimizer.Optimize(f, coordinates);
```

</details>

#### See also:

 * [Divide the gradient by a running average of its recent magnitude](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
 * [Stochastic gradient descent in Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#RMSProp)
 * [Differentiable separable functions](#differentiable-separable-functions)

## Simulated Annealing (SA)

*An optimizer for [arbitrary functions](#arbitrary-functions).*

Simulated Annealing is an stochastic optimization algorithm which is able to
deliver near-optimal results quickly without knowing the gradient of the
function being optimized.  It has a unique hill climbing capability that makes
it less vulnerable to local minima. This implementation uses exponential cooling
schedule and feedback move control by default, but the cooling schedule can be
changed via a template parameter.

#### Constructors

 * `SA<`_`CoolingScheduleType`_`>(`_`coolingSchedule`_`)`
 * `SA<`_`CoolingScheduleType`_`>(`_`coolingSchedule, maxIterations`_`)`
 * `SA<`_`CoolingScheduleType`_`>(`_`coolingSchedule, maxIterations, initT, initMoves, moveCtrlSweep, tolerance, maxToleranceSweep, maxMoveCoef, initMoveCoef, gain`_`)`

The _`CoolingScheduleType`_ template parameter implements a policy to update the
temperature.  The `ExponentialSchedule` class is available for use; it has a
constructor `ExponentialSchedule(`_`lambda`_`)` where _`lambda`_ is the cooling
speed (default `0.001`).  Custom schedules may be created by implementing a
class with at least the single member method below:

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
// Return the next temperature given the current system status.
double NextTemperature(const double currentTemperature,
                       const double currentEnergy);
```

</details>

For convenience, the default cooling schedule is `ExponentialSchedule`, so the
shorter type `SA<>` may be used instead of the equivalent
`SA<ExponentialSchedule>`.

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `CoolingScheduleType` | **`coolingSchedule`** | Instantiated cooling schedule (default ExponentialSchedule). | **CoolingScheduleType()** |
| `size_t` | **`maxIterations`** | Maximum number of iterations allowed (0 indicates no limit). | `1000000` |
| `double` | **`initT`** | Initial temperature. | `10000.0` |
| `size_t` | **`initMoves`** | Number of initial iterations without changing temperature. | `1000` |
| `size_t` | **`moveCtrlSweep`** | Sweeps per feedback move control. | `100` |
| `double` | **`tolerance`** | Tolerance to consider system frozen. | `1e-5` |
| `size_t` | **`maxToleranceSweep`** | Maximum sweeps below tolerance to consider system frozen. | `3` |
| `double` | **`maxMoveCoef`** | Maximum move size. | `20` |
| `double` | **`initMoveCoef`** | Initial move size. | `0.3` |
| `double` | **`gain`** | Proportional control in feedback move control. | `0.3` |

Attributes of the optimizer may also be changed via the member methods
`CoolingSchedule()`, `MaxIterations()`, `InitT()`, `InitMoves()`,
`MoveCtrlSweep()`, `Tolerance()`, `MaxToleranceSweep()`, `MaxMoveCoef()`,
`InitMoveCoef()`, and `Gain()`.

#### Examples:

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();

SA<> optimizer(ExponentialSchedule(), 1000000, 1000., 1000, 100, 1e-10, 3, 1.5,
    0.5, 0.3);
optimizer.Optimize(f, coordinates);
```

</details>

#### See also:

 * [Simulated annealing on Wikipedia](https://en.wikipedia.org/wiki/Simulated_annealing)
 * [Arbitrary functions](#arbitrary-functions)

## Simultaneous Perturbation Stochastic Approximation (SPSA)

*An optimizer for [arbitrary functions](#arbitrary-functions).*

The SPSA algorithm approximates the gradient of the function by finite
differences along stochastic directions.

#### Constructors

 * `SPSA(`_`alpha, gamma, stepSize, evaluationStepSize, maxIterations, tolerance`_`)`

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `double` | **`alpha`** | Scaling exponent for the step size. | `0.602` |
| `double` | **`gamma`** | Scaling exponent for evaluation step size. | `0.101` |
| `double` | **`stepSize`** | Scaling parameter for step size (named as 'a' in the paper). | `0.16` |
| `double` | **`evaluationStepSize`** | Scaling parameter for evaluation step size (named as 'c' in the paper). | `0.3` |
| `size_t` | **`maxIterations`** | Maximum number of iterations allowed (0 means no limit). | `100000` |
| `double` | **`tolerance`** | Maximum absolute tolerance to terminate algorithm. | `1e-5` |

Attributes of the optimizer may also be changed via the member methods
`Alpha()`, `Gamma()`, `StepSize()`, `EvaluationStepSize()`, and `MaxIterations()`.

#### Examples:

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
SphereFunction f(2);
arma::mat coordinates = f.GetInitialPoint();

SPSA optimizer(0.1, 0.102, 0.16, 0.3, 100000, 1e-5);
optimizer.Optimize(f, coordinates);
```

</details>

#### See also:

 * [An Overview of the Simultaneous Perturbation Method for Efficient Optimization](https://pdfs.semanticscholar.org/bf67/0fb6b1bd319938c6a879570fa744cf36b240.pdf)
 * [SPSA on Wikipedia](https://en.wikipedia.org/wiki/Simultaneous_perturbation_stochastic_approximation)
 * [Stochastic gradient descent in Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
 * [Differentiable separable functions](#differentiable-separable-functions)

## Stochastic Recursive Gradient Algorithm (SARAH/SARAH+)

*An optimizer for [differentiable separable functions](#differentiable-separable-functions).*

StochAstic Recusive gRadient algoritHm (SARAH), is a variance reducing
stochastic recursive gradient algorithm which employs the stochastic recursive
gradient, for solving empirical loss minimization for the case of nonconvex
losses.

#### Constructors

 * `SARAHType<`_`UpdatePolicyType`_`>()`
 * `SARAHType<`_`UpdatePolicyType`_`>(`_`stepSize, batchSize`_`)`
 * `SARAHType<`_`UpdatePolicyType`_`>(`_`stepSize, batchSize, maxIterations, innerIterations, tolerance, shuffle, updatePolicy, exactObjective`_`)`

The _`UpdatePolicyType`_ template parameter specifies the update step used for
the optimizer.  The `SARAHUpdate` and `SARAHPlusUpdate` classes are available
for use, and implement the standard SARAH update and SARAH+ update,
respectively.  A custom update rule can be used by implementing a class with the
same method signatures.

For convenience the following typedefs have been defined:

 * `SARAH` (equivalent to `SARAHType<SARAHUpdate>`): the standard SARAH optimizer
 * `SARAH_Plus` (equivalent to `SARAHType<SARAHPlusUpdate>`): the SARAH+ optimizer

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `double` | **`stepSize`** | Step size for each iteration. | `0.01` |
| `size_t` | **`batchSize`** | Batch size to use for each step. | `32` |
| `size_t` | **`maxIterations`** | Maximum number of iterations allowed (0 means no limit). | `1000` |
| `size_t` | **`innerIterations`** | The number of inner iterations allowed (0 means n / batchSize). Note that the full gradient is only calculated in the outer iteration. | `0` |
| `double` | **`tolerance`** | Maximum absolute tolerance to terminate algorithm. | `1e-5` |
| `bool` | **`shuffle`** | If true, the function order is shuffled; otherwise, each function is visited in linear order. | `true` |
| `UpdatePolicyType` | **`updatePolicy`** | Instantiated update policy used to adjust the given parameters. | `UpdatePolicyType()` |
| `bool` | **`exactObjective`** | Calculate the exact objective (Default: estimate the final objective obtained on the last pass over the data). | `false` |

Attributes of the optimizer may also be changed via the member methods
`StepSize()`, `BatchSize()`, `MaxIterations()`, `InnerIterations()`,
`Tolerance()`, `Shuffle()`, `UpdatePolicy()`, and `ExactObjective()`.

Note that the default value for `updatePolicy` is the default constructor for
the `UpdatePolicyType`.

#### Examples:

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();

// Standard stochastic variance reduced gradient.
SARAH optimizer(0.01, 1, 5000, 0, 1e-5, true);
optimizer.Optimize(f, coordinates);

// Stochastic variance reduced gradient with Barzilai-Borwein.
SARAH_Plus optimizerPlus(0.01, 1, 5000, 0, 1e-5, true);
optimizerPlus.Optimize(f, coordinates);
```

</details>

#### See also:

 * [Stochastic Recursive Gradient Algorithm for Nonconvex Optimization](https://arxiv.org/abs/1705.07261)
 * [Stochastic gradient descent in Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
 * [Differentiable separable functions](#differentiable-separable-functions)

## Standard SGD

*An optimizer for [differentiable separable functions](#differentiable-separable-functions).*

Stochastic Gradient Descent is a technique for minimizing a function which
can be expressed as a sum of other functions.  It's likely better to use any of
the other variants of SGD than this class; however, this standard SGD
implementation may still be useful in some situations.

#### Constructors

 * `StandardSGD()`
 * `StandardSGD(`_`stepSize, batchSize`_`)`
 * `StandardSGD(`_`stepSize, batchSize, maxIterations, tolerance, shuffle, updatePolicy, decayPolicy, resetPolicy, exactObjective`_`)`

Note that `StandardSGD` is based on the templated type
`SGD<`_`UpdatePolicyType, DecayPolicyType`_`>` with _`UpdatePolicyType`_` =
VanillaUpdate` and _`DecayPolicyType`_` = NoDecay`.

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `double` | **`stepSize`** | Step size for each iteration. | `0.01` |
| `size_t` | **`batchSize`** | Batch size to use for each step. | `32` |
| `size_t` | **`maxIterations`** | Maximum number of iterations allowed (0 means no limit). | `100000` |
| `double` | **`tolerance`** | Maximum absolute tolerance to terminate algorithm. | `1e-5` |
| `bool` | **`shuffle`** | If true, the function order is shuffled; otherwise, each function is visited in linear order. | `true` |
| `UpdatePolicyType` | **`updatePolicy`** | Instantiated update policy used to adjust the given parameters. | `UpdatePolicyType()` |
| `DecayPolicyType` | **`decayPolicy`** | Instantiated decay policy used to adjust the step size. | `DecayPolicyType()` |
| `bool` | **`resetPolicy`** | Flag that determines whether update policy parameters are reset before every Optimize call. | `true` |
| `bool` | **`exactObjective`** | Calculate the exact objective (Default: estimate the final objective obtained on the last pass over the data). | `false` |

Attributes of the optimizer may also be modified via the member methods
`StepSize()`, `BatchSize()`, `MaxIterations()`, `Tolerance()`, `Shuffle()`, `UpdatePolicy()`, `DecayPolicy()`, `ResetPolicy()`, and
`ExactObjective()`.

#### Examples

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();

StandardSGD optimizer(0.01, 32, 100000, 1e-5, true);
optimizer.Optimize(f, coordinates);
```

</details>

#### See also:

 * [Momentum SGD](#momentum-sgd)
 * [Nesterov Momentum SGD](#nesterov-momentum-sgd)
 * [SGD in Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
 * [Differentiable separable functions](#differentiable-separable-functions)

## Stochastic Coordinate Descent (SCD)

*An optimizer for [partially differentiable functions](#partially-differentiable-functions).*

Stochastic Coordinate descent is a technique for minimizing a function by
doing a line search along a single direction at the current point in the
iteration. The direction (or "coordinate") can be chosen cyclically, randomly
or in a greedy fashion.

#### Constructors

 * `SCD<`_`DescentPolicyType`_`>()`
 * `SCD<`_`DescentPolicyType`_`>(`_`stepSize, maxIterations`_`)`
 * `SCD<`_`DescentPolicyType`_`>(`_`stepSize, maxIterations, tolerance, updateInterval`_`)`
 * `SCD<`_`DescentPolicyType`_`>(`_`stepSize, maxIterations, tolerance, updateInterval, descentPolicy`_`)`

The _`DescentPolicyType`_ template parameter specifies the behavior of SCD when
selecting the next coordinate to descend with.  The `RandomDescent`,
`GreedyDescent`, and `CyclicDescent` classes are available for use.  Custom
behavior can be achieved by implementing a class with the same method
signatures.

For convenience, the following typedefs have been defined:

 * `RandomSCD` (equivalent to `SCD<RandomDescent>`): selects coordinates randomly
 * `GreedySCD` (equivalent to `SCD<GreedyDescent>`): selects the coordinate with the maximum guaranteed descent according to the Gauss-Southwell rule
 * `CyclicSCD` (equivalent to `SCD<CyclicDescent>`): selects coordinates sequentially

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `double` | **`stepSize`** | Step size for each iteration. | `0.01` |
| `size_t` | **`maxIterations`** | Maximum number of iterations allowed (0 means no limit). | `100000` |
| `double` | **`tolerance`** | Maximum absolute tolerance to terminate the algorithm. | `1e-5` |
| `size_t` | **`updateInterval`** | The interval at which the objective is to be reported and checked for convergence. | `1e3` |
| `DescentPolicyType` | **`descentPolicy`** | The policy to use for selecting the coordinate to descend on. | `DescentPolicyType()` |

Attributes of the optimizer may also be modified via the member methods
`StepSize()`, `MaxIterations()`, `Tolerance()`, `UpdateInterval()`, and
`DescentPolicy()`.

Note that the default value for `descentPolicy` is the default constructor for
_`DescentPolicyType`_.

#### Examples

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
SparseTestFunction f;
arma::mat coordinates = f.GetInitialPoint();

RandomSCD randomscd(0.01, 100000, 1e-5, 1e3);
randomscd.Optimize(f, coordinates);

GreedySCD greedyscd(0.01, 100000, 1e-5, 1e3);
greedyscd.Optimize(f, coordinates);

CyclicSCD cyclicscd(0.01, 100000, 1e-5, 1e3);
cyclicscd.Optimize(f, coordinates);
```

</details>

#### See also:

 * [Coordinate descent on Wikipedia](https://en.wikipedia.org/wiki/Coordinate_descent)
 * [Stochastic Methods for L1-Regularized Loss Minimization](https://www.jmlr.org/papers/volume12/shalev-shwartz11a/shalev-shwartz11a.pdf)
 * [Partially differentiable functions](#partially-differentiable-functions)

## Stochastic Gradient Descent with Restarts (SGDR)

*An optimizer for [differentiable separable
functions](#differentiable-separable-functions).*

SGDR is based on Mini-batch Stochastic Gradient Descent class and simulates a
new warm-started run/restart once a number of epochs are performed.

#### Constructors

 * `SGDR<`_`UpdatePolicyType`_`>()`
 * `SGDR<`_`UpdatePolicyType`_`>(`_`epochRestart, multFactor, batchSize, stepSize`_`)`
 * `SGDR<`_`UpdatePolicyType`_`>(`_`epochRestart, multFactor, batchSize, stepSize, maxIterations, tolerance, shuffle, updatePolicy`_`)`
 * `SGDR<`_`UpdatePolicyType`_`>(`_`epochRestart, multFactor, batchSize, stepSize, maxIterations, tolerance, shuffle, updatePolicy, resetPolicy, exactObjective`_`)`

The _`UpdatePolicyType`_ template parameter controls the update policy used
during the iterative update process.  The `MomentumUpdate` class is available
for use, and custom behavior can be achieved by implementing a class with the
same method signatures as `MomentumUpdate`.

For convenience, the default type of _`UpdatePolicyType`_ is `MomentumUpdate`,
so the shorter type `SGDR<>` can be used instead of the equivalent
`SGDR<MomentumUpdate>`.

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `size_t` | **`epochRestart`** | Initial epoch where decay is applied. | `50` |
| `double` | **`multFactor`** | Batch size multiplication factor. | `2.0` |
| `size_t` | **`batchSize`** | Size of each mini-batch. | `1000` |
| `double` | **`stepSize`** | Step size for each iteration. | `0.01` |
| `size_t` | **`maxIterations`** | Maximum number of iterations allowed (0 means no limit). | `100000` |
| `double` | **`tolerance`** | Maximum absolute tolerance to terminate algorithm. | `1e-5` |
| `bool` | **`shuffle`** | If true, the mini-batch order is shuffled; otherwise, each mini-batch is visited in linear order. | `true` |
| `UpdatePolicyType` | **`updatePolicy`** | Instantiated update policy used to adjust the given parameters. | `UpdatePolicyType()` |
| `bool` | **`resetPolicy`** | If true, parameters are reset before every Optimize call; otherwise, their values are retained. | `true` |
| `bool` | **`exactObjective`** | Calculate the exact objective (Default: estimate the final objective obtained on the last pass over the data). | `false` |

Attributes of the optimizer can also be modified via the member methods
`EpochRestart()`, `MultFactor()`, `BatchSize()`, `StepSize()`,
`MaxIterations()`, `Tolerance()`, `Shuffle()`, `UpdatePolicy()`, `ResetPolicy()`, and
`ExactObjective()`.

Note that the default value for `updatePolicy` is the default constructor for
the `UpdatePolicyType`.

#### Examples:

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();

SGDR<> optimizer(50, 2.0, 1, 0.01, 10000, 1e-3);
optimizer.Optimize(f, coordinates);
```

</details>

#### See also:

 * [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983)
 * [Stochastic gradient descent in Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
 * [Differentiable separable functions](#differentiable-separable-functions)

## Snapshot Stochastic Gradient Descent with Restarts (SnapshotSGDR)

*An optimizer for [differentiable separable functions](#differentiable-separable-functions).*

SnapshotSGDR simulates a new warm-started run/restart once a number of epochs
are performed using the Snapshot Ensembles technique.

#### Constructors

 * `SnapshotSGDR<`_`UpdatePolicyType`_`>()`
 * `SnapshotSGDR<`_`UpdatePolicyType`_`>(`_`epochRestart, multFactor, batchSize, stepSize`_`)`
 * `SnapshotSGDR<`_`UpdatePolicyType`_`>(`_`epochRestart, multFactor, batchSize, stepSize, maxIterations, tolerance, shuffle, snapshots, accumulate, updatePolicy`_`)`
 * `SnapshotSGDR<`_`UpdatePolicyType`_`>(`_`epochRestart, multFactor, batchSize, stepSize, maxIterations, tolerance, shuffle, snapshots, accumulate, updatePolicy, resetPolicy, exactObjective`_`)`

The _`UpdatePolicyType`_ template parameter controls the update policy used
during the iterative update process.  The `MomentumUpdate` class is available
for use, and custom behavior can be achieved by implementing a class with the
same method signatures as `MomentumUpdate`.

For convenience, the default type of _`UpdatePolicyType`_ is `MomentumUpdate`,
so the shorter type `SnapshotSGDR<>` can be used instead of the equivalent
`SnapshotSGDR<MomentumUpdate>`.

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `size_t` | **`epochRestart`** | Initial epoch where decay is applied. | `50` |
| `double` | **`multFactor`** | Batch size multiplication factor. | `2.0` |
| `size_t` | **`batchSize`** | Size of each mini-batch. | `1000` |
| `double` | **`stepSize`** | Step size for each iteration. | `0.01` |
| `size_t` | **`maxIterations`** | Maximum number of iterations allowed (0 means no limit). | `100000` |
| `double` | **`tolerance`** | Maximum absolute tolerance to terminate algorithm. | `1e-5` |
| `bool` | **`shuffle`** | If true, the mini-batch order is shuffled; otherwise, each mini-batch is visited in linear order. | `true` |
| `size_t` | **`snapshots`** | Maximum number of snapshots. | `5` |
| `bool` | **`accumulate`** | Accumulate the snapshot parameter. | `true` |
| `UpdatePolicyType` | **`updatePolicy`** | Instantiated update policy used to adjust the given parameters. | `UpdatePolicyType()` |
| `bool` | **`resetPolicy`** | If true, parameters are reset before every Optimize call; otherwise, their values are retained. | `true` |
| `bool` | **`exactObjective`** | Calculate the exact objective (Default: estimate the final objective obtained on the last pass over the data). | `false` |

Attributes of the optimizer can also be modified via the member methods
`EpochRestart()`, `MultFactor()`, `BatchSize()`, `StepSize()`,
`MaxIterations()`, `Tolerance()`, `Shuffle()`, `Snapshots()`, `Accumulate()`,
`UpdatePolicy()`, `ResetPolicy()`, and `ExactObjective()`.

The `Snapshots()` function returns a `std::vector<arma::mat>&` (a vector of
snapshots of the parameters), not a `size_t` representing the maximum number of
snapshots.

Note that the default value for `updatePolicy` is the default constructor for
the `UpdatePolicyType`.

#### Examples:

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();

SnapshotSGDR<> optimizer(50, 2.0, 1, 0.01, 10000, 1e-3);
optimizer.Optimize(f, coordinates);
```

</details>

#### See also:

 * [Snapshot ensembles: Train 1, get m for free](https://arxiv.org/abs/1704.00109)
 * [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983)
 * [Stochastic gradient descent in Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
 * [Differentiable separable functions](#differentiable-separable-functions)

## SMORMS3

*An optimizer for [differentiable separable functions](#differentiable-separable-functions).*

SMORMS3 is a hybrid of RMSprop, which is trying to estimate a safe and optimal
distance based on curvature or perhaps just normalizing the step-size in the
parameter space.

#### Constructors

 * `SMORMS3()`
 * `SMORMS3(`_`stepSize, batchSize`_`)`
 * `SMORMS3(`_`stepSize, batchSize, epsilon, maxIterations, tolerance`_`)`
 * `SMORMS3(`_`stepSize, batchSize, epsilon, maxIterations, tolerance, shuffle, resetPolicy, exactObjective`_`)`

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `double` | **`stepSize`** | Step size for each iteration. | `0.001` |
| `size_t` | **`batchSize`** | Number of points to process at each step. | `32` |
| `double` | **`epsilon`** | Value used to initialize the mean squared gradient parameter. | `1e-16` |
| `size_t` | **`maxIterations`** | Maximum number of iterations allowed (0 means no limit). | `100000` |
| `double` | **`tolerance`** | Maximum absolute tolerance to terminate algorithm. | `1e-5` |
| `bool` | **`shuffle`** | If true, the mini-batch order is shuffled; otherwise, each mini-batch is visited in linear order. | `true` |
| `bool` | **`resetPolicy`** | If true, parameters are reset before every Optimize call; otherwise, their values are retained. | `true` |
| `bool` | **`exactObjective`** | Calculate the exact objective (Default: estimate the final objective obtained on the last pass over the data). | `false` |

Attributes of the optimizer can also be modified via the member methods
`StepSize()`, `BatchSize()`, `Epsilon()`, `MaxIterations()`, `Tolerance()`,
`Shuffle()`, `ResetPolicy()`, and `ExactObjective()`.

#### Examples:

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();

SMORMS3 optimizer(0.001, 1, 1e-16, 5000000, 1e-9, true);
optimizer.Optimize(f, coordinates);
```

</details>

#### See also:

 * [RMSprop loses to SMORMS3 - Beware the Epsilon!](https://sifter.org/simon/journal/20150420.html)
 * [RMSProp](#rmsprop)
 * [Stochastic gradient descent in Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
 * [Differentiable separable functions](#differentiable-separable-functions)

## Standard stochastic variance reduced gradient (SVRG)

*An optimizer for [differentiable separable functions](#differentiable-separable-functions).*

Stochastic Variance Reduced Gradient is a technique for minimizing smooth and
strongly convex problems.

#### Constructors

 * `SVRGType<`_`UpdatePolicyType, DecayPolicyType`_`>()`
 * `SVRGType<`_`UpdatePolicyType, DecayPolicyType`_`>(`_`stepSize`_`)`
 * `SVRGType<`_`UpdatePolicyType, DecayPolicyType`_`>(`_`stepSize, batchSize, maxIterations, innerIterations`_`)`
 * `SVRGType<`_`UpdatePolicyType, DecayPolicyType`_`>(`_`stepSize, batchSize, maxIterations, innerIterations, tolerance, shuffle, updatePolicy, decayPolicy, resetPolicy, exactObjective`_`)`

The _`UpdatePolicyType`_ template parameter controls the update step used by
SVRG during the optimization.  The `SVRGUpdate` class is available for use and
custom update behavior can be achieved by implementing a class with the same
method signatures as `SVRGUpdate`.

The _`DecayPolicyType`_ template parameter controls the decay policy used to
adjust the step size during the optimization.  The `BarzilaiBorweinDecay` and
`NoDecay` classes are available for use.  Custom decay functionality can be
achieved by implementing a class with the same method signatures.

For convenience the following typedefs have been defined:

 * `SVRG` (equivalent to `SVRGType<SVRGUpdate, NoDecay>`): the standard SVRG technique
 * `SVRG_BB` (equivalent to `SVRGType<SVRGUpdate, BarzilaiBorweinDecay>`): SVRG with the Barzilai-Borwein decay policy

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `double` | **`stepSize`** | Step size for each iteration. | `0.01` |
| `size_t` | **`batchSize`** | Initial batch size. | `32` |
| `size_t` | **`maxIterations`** | Maximum number of iterations allowed (0 means no limit). | `1000` |
| `size_t` | **`innerIterations`** | The number of inner iterations allowed (0 means n / batchSize). Note that the full gradient is only calculated in the outer iteration. | `0` |
| `double` | **`tolerance`** | Maximum absolute tolerance to terminate algorithm. | `1e-5` |
| `bool` | **`shuffle`** | If true, the batch order is shuffled; otherwise, each batch is visited in linear order. | `true` |
| `UpdatePolicyType` | **`updatePolicy`** | Instantiated update policy used to adjust the given parameters. | `UpdatePolicyType()` |
| `DecayPolicyType` | **`decayPolicy`** | Instantiated decay policy used to adjust the step size. | `DecayPolicyType()` |
| `bool` | **`resetPolicy`** | Flag that determines whether update policy parameters are reset before every Optimize call. | `true` |
| `bool` | **`exactObjective`** | Calculate the exact objective (Default: estimate the final objective obtained on the last pass over the data). | `false` |

Attributes of the optimizer may also be modified via the member methods
`StepSize()`, `BatchSize()`, `MaxIterations()`, `InnerIterations()`,
`Tolerance()`, `Shuffle()`, `UpdatePolicy()`, `DecayPolicy()`, `ResetPolicy()`, and
`ExactObjective()`.

Note that the default values for the `updatePolicy` and `decayPolicy` parameters
are simply the default constructors of the _`UpdatePolicyType`_ and
_`DecayPolicyType`_ classes.

#### Examples:

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();

// Standard stochastic variance reduced gradient.
SVRG optimizer(0.005, 1, 300, 0, 1e-10, true);
optimizer.Optimize(f, coordinates);

// Stochastic variance reduced gradient with Barzilai-Borwein.
SVRG_BB bbOptimizer(0.005, batchSize, 300, 0, 1e-10, true, SVRGUpdate(),
    BarzilaiBorweinDecay(0.1));
bbOptimizer.Optimize(f, coordinates);
```

</details>

#### See also:

 * [Accelerating Stochastic Gradient Descent using Predictive Variance Reduction](https://papers.nips.cc/paper/4937-accelerating-stochastic-gradient-descent-using-predictive-variance-reduction.pdf)
 * [SGD](#standard-sgd)
 * [SGD in Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
 * [Differentiable separable functions](#differentiable-separable-functions)

## SPALeRA Stochastic Gradient Descent (SPALeRASGD)

*An optimizer for [differentiable separable functions](#differentiable-separable-functions).*

SPALeRA involves two components: a learning rate adaptation scheme, which
ensures that the learning system goes as fast as it can; and a catastrophic
event manager, which is in charge of detecting undesirable behaviors and getting
the system back on track.

#### Constructors

 * `SPALeRASGD<`_`DecayPolicyType`_`>()`
 * `SPALeRASGD<`_`DecayPolicyType`_`>(`_`stepSize, batchSize`_`)`
 * `SPALeRASGD<`_`DecayPolicyType`_`>(`_`stepSize, batchSize, maxIterations, tolerance`_`)`
 * `SPALeRASGD<`_`DecayPolicyType`_`>(`_`stepSize, batchSize, maxIterations, tolerance, lambda, alpha, epsilon, adaptRate, shuffle, decayPolicy, resetPolicy, exactObjective`_`)`

The _`DecayPolicyType`_ template parameter controls the decay in the step size
during the course of the optimization.  The `NoDecay` class is available for
use; custom behavior can be achieved by implementing a class with the same
method signatures.

By default, _`DecayPolicyType`_ is set to `NoDecay`, so the shorter type
`SPALeRASGD<>` can be used instead of the equivalent `SPALeRASGD<NoDecay>`.

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `double` | **`stepSize`** | Step size for each iteration. | `0.01` |
| `size_t` | **`batchSize`** | Initial batch size. | `32` |
| `size_t` | **`maxIterations`** | Maximum number of iterations allowed (0 means no limit). | `100000` |
| `double` | **`tolerance`** | Maximum absolute tolerance to terminate algorithm. | `1e-5` |
| `double` | **`lambda`** | Page-Hinkley update parameter. | `0.01` |
| `double` | **`alpha`** | Memory parameter of the Agnostic Learning Rate adaptation. | `0.001` |
| `double` | **`epsilon`** | Numerical stability parameter. | `1e-6` |
| `double` | **`adaptRate`** | Agnostic learning rate update rate. | `3.10e-8` |
| `bool` | **`shuffle`** | If true, the batch order is shuffled; otherwise, each batch is visited in linear order. | `true` |
| `DecayPolicyType` | **`decayPolicy`** | Instantiated decay policy used to adjust the step size. | `DecayPolicyType()` |
| `bool` | **`resetPolicy`** | Flag that determines whether update policy parameters are reset before every Optimize call. | `true` |
| `bool` | **`exactObjective`** | Calculate the exact objective (Default: estimate the final objective obtained on the last pass over the data). | `false` |

Attributes of the optimizer may also be modified via the member methods
`StepSize()`, `BatchSize()`, `MaxIterations()`, `Tolerance()`, `Lambda()`,
`Alpha()`, `Epsilon()`, `AdaptRate()`, `Shuffle()`, `DecayPolicy()`, `ResetPolicy()`, and `ExactObjective()`.

#### Examples

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();

SPALeRASGD<> optimizer(0.05, 1, 10000, 1e-4);
optimizer.Optimize(f, coordinates);
```

</details>

#### See also:

 * [Stochastic Gradient Descent: Going As Fast As Possible But Not Faster](https://arxiv.org/abs/1709.01427)
 * [SGD](#standard-sgd)
 * [SGD in Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
 * [Differentiable separable functions](#differentiable-separable-functions)

## SWATS

*An optimizer for [differentiable separable functions](#differentiable-separable-functions).*

SWATS is an optimizer that uses a simple strategy to switch from Adam to
standard SGD when a triggering condition is satisfied.  The condition relates to
the projection of Adam steps on the gradient subspace.

#### Constructors

 * `SWATS()`
 * `SWATS(`_`stepSize, batchSize`_`)`
 * `SWATS(`_`stepSize, batchSize, beta1, beta2, epsilon, maxIterations, tolerance`_`)`
 * `SWATS(`_`stepSize, batchSize, beta1, beta2, epsilon, maxIterations, tolerance, shuffle, resetPolicy, exactObjective`_`)`

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `double` | **`stepSize`** | Step size for each iteration. | `0.001` |
| `size_t` | **`batchSize`** | Number of points to process at each step. | `32` |
| `double` | **`beta1`** | Exponential decay rate for the first moment estimates. | `0.9` |
| `double` | **`beta2`** | Exponential decay rate for the weighted infinity norm estimates. | `0.999` |
| `double` | **`epsilon`** | Value used to initialize the mean squared gradient parameter. | `1e-16` |
| `size_t` | **`maxIterations`** | Maximum number of iterations allowed (0 means no limit). | `100000` |
| `double` | **`tolerance`** | Maximum absolute tolerance to terminate algorithm. | `1e-5` |
| `bool` | **`shuffle`** | If true, the mini-batch order is shuffled; otherwise, each mini-batch is visited in linear order. | `true` |
| `bool` | **`resetPolicy`** | If true, parameters are reset before every Optimize call; otherwise, their values are retained. | `true` |
| `bool` | **`exactObjective`** | Calculate the exact objective (Default: estimate the final objective obtained on the last pass over the data). | `false` |

Attributes of the optimizer can also be modified via the member methods
`StepSize()`, `BatchSize()`, `Beta1()`, `Beta2()`, `Epsilon()`,
`MaxIterations()`, `Tolerance()`, `Shuffle()`, `ResetPolicy()`, and  `ExactObjective()`.

#### Examples:

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();

SWATS optimizer(0.001, 1, 0.9, 0.999, 1e-16, 5000000, 1e-9, true);
optimizer.Optimize(f, coordinates);
```

</details>

#### See also:

 * [Improving generalization performance by switching from Adam to SGD](https://arxiv.org/abs/1712.07628)
 * [Adam](#adam)
 * [Standard SGD](#standard-sgd)
 * [Stochastic gradient descent in Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
 * [Differentiable separable functions](#differentiable-separable-functions)

## WNGrad

*An optimizer for [differentiable separable functions](#differentiable-separable-functions).*

WNGrad is a general nonlinear update rule for the learning rate. WNGrad has
near-optimal convergence rates in both the batch and stochastic settings.

#### Constructors

 * `WNGrad()`
 * `WNGrad(`_`stepSize, batchSize`_`)`
 * `WNGrad(`_`stepSize, batchSize, maxIterations, tolerance, shuffle`_`)`
 * `WNGrad(`_`stepSize, batchSize, maxIterations, tolerance, shuffle, resetPolicy, exactObjective`_`)`

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `double` | **`stepSize`** | Step size for each iteration. | `0.562` |
| `size_t` | **`batchSize`** | Initial batch size. | `32` |
| `size_t` | **`maxIterations`** | Maximum number of iterations allowed (0 means no limit). | `100000` |
| `double` | **`tolerance`** | Maximum absolute tolerance to terminate algorithm. | `1e-5` |
| `bool` | **`shuffle`** | If true, the batch order is shuffled; otherwise, each batch is visited in linear order. | `true` |
| `bool` | **`resetPolicy`** | If true, parameters are reset before every Optimize call; otherwise, their values are retained. | `true` |
| `bool` | **`exactObjective`** | Calculate the exact objective (Default: estimate the final objective obtained on the last pass over the data). | `false` |

Attributes of the optimizer may also be modified via the member methods
`StepSize()`, `BatchSize()`, `MaxIterations()`, `Tolerance()`, `Shuffle()`, `ResetPolicy()`, and
`ExactObjective()`.

#### Examples

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();

WNGrad<> optimizer(0.562, 1, 10000, 1e-4);
optimizer.Optimize(f, coordinates);
```

</details>

#### See also:

 * [WNGrad: Learn the Learning Rate in Gradient Descent](https://arxiv.org/abs/1803.02865)
 * [SGD](#standard-sgd)
 * [SGD in Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
 * [Differentiable separable functions](#differentiable-separable-functions)

## Yogi

*An optimizer for [differentiable separable functions](#differentiable-separable-functions).*

Yogi is an optimization algorithm based on Adam with more fine-grained effective
learning rate control, which uses additive updates instead of multiplicative
updates for the moving average of the squared gradient. In addition, Yogi has
similar theoretical guarantees on convergence as Adam.

#### Constructors

 * `Yogi()`
 * `Yogi(`_`stepSize, batchSize`_`)`
 * `Yogi(`_`stepSize, batchSize, beta1, beta2, eps, maxIterations`_`)`
 * `Yogi(`_`stepSize, batchSize, beta1, beta2, eps, maxIterations, tolerance, shuffle, resetPolicy, exactObjective`_`)`

#### Attributes

| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `double` | **`stepSize`** | Step size for each iteration. | `0.001` |
| `size_t` | **`batchSize`** | Number of points to process in a single step. | `32` |
| `double` | **`beta1`** | Exponential decay rate for the first moment estimates. | `0.9` |
| `double` | **`beta2`** | Exponential decay rate for the weighted infinity norm estimates. | `0.999` |
| `double` | **`eps`** | Value used to initialize the mean squared gradient parameter. | `1e-8` |
| `size_t` | **`max_iterations`** | Maximum number of iterations allowed (0 means no limit). | `100000` |
| `double` | **`tolerance`** | Maximum absolute tolerance to terminate algorithm. | `1e-5` |
| `bool` | **`shuffle`** | If true, the function order is shuffled; otherwise, each function is visited in linear order. | `true` |
| `bool` | **`resetPolicy`** | If true, parameters are reset before every Optimize call; otherwise, their values are retained. | `true` |
| `bool` | **`exactObjective`** | Calculate the exact objective (Default: estimate the final objective obtained on the last pass over the data). | `false` |

The attributes of the optimizer may also be modified via the member methods
`StepSize()`, `BatchSize()`, `Beta1()`, `Beta2()`, `Eps()`, `MaxIterations()`,
`Tolerance()`, `Shuffle()`, `ResetPolicy()`, and `ExactObjective()`.

#### Examples

```c++
RosenbrockFunction f;
arma::mat coordinates = f.GetInitialPoint();

Yogi optimizer(0.001, 32, 0.9, 0.999, 1e-8, 100000, 1e-5, true);
optimizer.Optimize(f, coordinates);
```

#### See also:

 * [Adaptive Methods for Nonconvex Optimization](https://papers.nips.cc/paper/8186-adaptive-methods-for-nonconvex-optimization)
 * [SGD in Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
 * [SGD](#standard-sgd)
 * [Adam](#adam)
 * [Differentiable separable functions](#differentiable-separable-functions)

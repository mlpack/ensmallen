## Arbitrary functions

The least restrictive type of function that can be implemented in ensmallen is
a function for which only the objective can be evaluated.  For this, a class
with the following API must be implemented:

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
class ArbitraryFunctionType
{
 public:
  // This should return f(x).
  double Evaluate(const arma::mat& x);
};
```

</details>

For this type of function, we assume that the gradient `f'(x)` is not
computable.  If it is, see [differentiable functions](#differentiable-functions).

The `Evaluate()` method is allowed to have additional cv-modifiers (`static`,
`const`, etc.).

The following optimizers can be used to optimize an arbitrary function:

 - [Simulated Annealing](#simulated-annealing-sa)
 - [CNE](#cne)
 - [DE](#de)
 - [PSO](#pso)
 - [SPSA](#simultaneous-perturbation-stochastic-approximation-spsa)

Each of these optimizers has an `Optimize()` function that is called as
`Optimize(f, x)` where `f` is the function to be optimized (which implements
`Evaluate()`) and `x` holds the initial point of the optimization.  After
`Optimize()` is called, `x` will hold the final result of the optimization
(that is, the best `x` found that minimizes `f(x)`).

#### Example: squared function optimization

An example program that implements the objective function f(x) = 2 |x|^2 is
shown below, using the simulated annealing optimizer.

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
#include <ensmallen.hpp>

class SquaredFunction
{
 public:
  // This returns f(x) = 2 |x|^2.
  double Evaluate(const arma::mat& x)
  {
    return 2 * std::pow(arma::norm(x), 2.0);
  }
};

int main()
{
  // The minimum is at x = [0 0 0].  Our initial point is chosen to be
  // [1.0, -1.0, 1.0].
  arma::mat x("1.0 -1.0 1.0");

  // Create simulated annealing optimizer with default options.
  // The ens::SA<> type can be replaced with any suitable ensmallen optimizer
  // that is able to handle arbitrary functions.
  ens::SA<> optimizer;
  SquaredFunction f; // Create function to be optimized.
  optimizer.Optimize(f, x);

  std::cout << "Minimum of squared function found with simulated annealing is "
      << x;
}
```

</details>

## Differentiable functions

Probably the most common type of function that can be optimized with ensmallen
is a differentiable function, where both f(x) and f'(x) can be calculated.  To
optimize a differentiable function with ensmallen, a class must be implemented
that follows the API below:

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
class DifferentiableFunctionType
{
 public:
  // Given parameters x, return the value of f(x).
  double Evaluate(const arma::mat& x);

  // Given parameters x and a matrix g, store f'(x) in the provided matrix g.
  // g should have the same size (rows, columns) as x.
  void Gradient(const arma::mat& x, arma::mat& gradient);

  // OPTIONAL: this may be implemented in addition to---or instead
  // of---Evaluate() and Gradient().  If this is the only function implemented,
  // implementations of Evaluate() and Gradient() will be automatically
  // generated using template metaprogramming.  Often, implementing
  // EvaluateWithGradient() can result in more efficient optimizations.
  //
  // Given parameters x and a matrix g, return the value of f(x) and store
  // f'(x) in the provided matrix g.  g should have the same size (rows,
  // columns) as x.
  double EvaluateWithGradient(const arma::mat& x, arma::mat& g);
};
```

</details>

Note that you may implement *either* `Evaluate()` and `Gradient()` *or*
`EvaluateWithGradient()`, but it is not mandatory to implement both.  (Of
course, supplying both is okay too.)  It often results in faster code when
`EvaluateWithGradient()` is implemented, especially for functions where f(x)
and f'(x) compute some of the same intermediate quantities.

Each of the implemented methods is allowed to have additional cv-modifiers
(`static`, `const`, etc.).

The following optimizers can be used with differentiable functions:

 * [L-BFGS](#l-bfgs) (`ens::L_BFGS`)
 * [FrankWolfe](#frank-wolfe) (`ens::FrankWolfe`)
 * [GradientDescent](#gradient-descent) (`ens::GradientDescent`)
 - Any optimizer for [arbitrary functions](#arbitrary-functions)

Each of these optimizers has an `Optimize()` function that is called as
`Optimize(f, x)` where `f` is the function to be optimized and `x` holds the
initial point of the optimization.  After `Optimize()` is called, `x` will hold
the final result of the optimization (that is, the best `x` found that
minimizes `f(x)`).

An example program is shown below.  In this program, we optimize a linear
regression model.  In this setting, we have some matrix `data` of data points
that we've observed, and some vector `responses` of the observed responses to
this data.  In our model, we assume that each response is the result of a
weighted linear combination of the data:

$$ \operatorname{responses}_i = x * \operatorname{data}_i $$

where $x$ is a vector of parameters.  This gives the objective function $f(x) =
(\operatorname{responses} - x'\operatorname{data})^2$.

In the example program, we optimize this objective function and compare the
runtime of an implementation that uses `Evaluate()` and `Gradient()`, and the
runtime of an implementation that uses `EvaluateWithGradient()`.

<details>
<summary>Click to collapse/expand example code.
</summary>

```c++
#include <ensmallen.hpp>

// Define a differentiable objective function by implementing both Evaluate()
// and Gradient() separately.
class LinearRegressionFunction
{
 public:
  // Construct the object with the given data matrix and responses.
  LinearRegressionFunction(const arma::mat& dataIn,
                           const arma::rowvec& responsesIn) :
      data(dataIn), responses(responsesIn) { }

  // Return the objective function for model parameters x.
  double Evaluate(const arma::mat& x)
  {
    return std::pow(arma::norm(responses - x.t() * data), 2.0);
  }

  // Compute the gradient for model parameters x.
  void Gradient(const arma::mat& x, arma::mat& g)
  {
    g = -2 * data * (responses - x.t() * data);
  }

 private:
  // The data.
  const arma::mat& data;
  // The responses to each data point.
  const arma::rowvec& responses;
};

// Define the same function, but only implement EvaluateWithGradient().
class LinearRegressionEWGFunction
{
 public:
  // Construct the object with the given data matrix and responses.
  LinearRegressionEWGFunction(const arma::mat& dataIn,
                              const arma::rowvec& responsesIn) :
      data(dataIn), responses(responsesIn) { }

  // Simultaneously compute both the objective function and gradient for model
  // parameters x.  Note that this is faster than implementing Evaluate() and
  // Gradient() individually because it caches the computation of
  // (responses - x.t() * data)!
  double EvaluateWithGradient(const arma::mat& x, arma::mat& g)
  {
    const arma::rowvec v = (responses - x.t() * data);
    g = -2 * data * v;
    return arma::accu(v % v); // equivalent to \| v \|^2
  }
};

int main()
{
  // We'll run a simple speed comparison between both objective functions.

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

  // We'll use Armadillo's wall_clock class to do a timing comparison.
  arma::wall_clock clock;

  // Construct the first objective function.
  LinearRegressionFunction lrf1(data, responses);
  arma::mat lrf1Params(startingPoint);

  // Create the L_BFGS optimizer with default parameters.
  // The ens::L_BFGS type can be replaced with any ensmallen optimizer that can
  // handle differentiable functions.
  ens::L_BFGS lbfgs;

  // Time how long L-BFGS takes for the first Evaluate() and Gradient()
  // objective function.
  clock.tic();
  lbfgs.Optimize(lrf1, lrf1Params);
  const double time1 = clock.toc();

  std::cout << "LinearRegressionFunction with Evaluate() and Gradient() took "
    << time1 << " seconds to converge to the model: " << std::endl;
  std::cout << lrf1Params.t();

  // Create the second objective function, which uses EvaluateWithGradient().
  LinearRegressionEWGFunction lrf2(data, responses);
  arma::mat lrf2Params(startingPoint);

  // Time how long L-BFGS takes for the EvaluateWithGradient() objective
  // function.
  clock.tic();
  lbfgs.Optimize(lrf2, lrf2Params);
  const double time2 = clock.toc();

  std::cout << "LinearRegressionEWGFunction with EvaluateWithGradient() took "
    << time2 << " seconds to converge to the model: " << std::endl;
  std::cout << lrf2Params.t();

  // When this runs, the output parameters will be exactly on the same, but the
  // LinearRegressionEWGFunction will run more quickly!
}
```

</details>

### Partially differentiable functions

Some differentiable functions have the additional property that the gradient
`f'(x)` can be decomposed along a different axis `j` such that the gradient is
sparse.  This makes the most sense in machine learning applications where the
function `f(x)` is being optimized for some dataset `data`, which has `d`
dimensions.  A partially differentiable separable function is partially
differentiable with respect to each dimension `j` of `data`.  This property is
useful for coordinate descent type algorithms.

To use ensmallen optimizers to minimize these types of functions, only two
functions needs to be added to the differentiable function type:

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
// Compute the partial gradient f'_j(x) with respect to data coordinate j and
// store it in the sparse matrix g.
void Gradient(const arma::mat& x, const size_t j, arma::sp_mat& g);

// Get the number of features that f(x) can be partially differentiated with.
size_t NumFeatures();
```

</details>

**Note**: many partially differentiable function optimizers do not require a
regular implementation of the `Gradient()`, so that function may be omitted.

If these functions are implemented, the following partially differentiable
function optimizers can be used:

 - [Stochastic Coordinate Descent](#stochastic-coordinate-descent-scd)

## Arbitrary separable functions

Often, an objective function `f(x)` may be represented as the sum of many
functions:

```
f(x) = f_0(x) + f_1(x) + ... + f_N(x).
```

In this function type, we assume the gradient `f'(x)` is not computable.  If it
is, see [differentiable separable functions](#differentiable-separable-functions).

For machine learning tasks, the objective function may be, e.g., the sum of a
function taken across many data points.  Implementing an arbitrary separable
function type in ensmallen is similar to implementing an arbitrary objective
function, but with a few extra utility methods:

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
class ArbitrarySeparableFunctionType
{
 public:
  // Given parameters x, return the sum of the individual functions
  // f_i(x) + ... + f_{i + batchSize - 1}(x).  i will always be greater than 0,
  // and i + batchSize will be less than or equal to the value of NumFunctions().
  double Evaluate(const arma::mat& x, const size_t i, const size_t batchSize);

  // Shuffle the ordering of the functions f_i(x).
  // (For machine learning problems, this would be equivalent to shuffling the
  // data points, e.g., before an epoch of learning.)
  void Shuffle();

  // Get the number of functions f_i(x).
  // (For machine learning problems, this is often just the number of points in
  // the dataset.)
  size_t NumFunctions();
};
```

</details>

Each of the implemented methods is allowed to have additional cv-modifiers
(`static`, `const`, etc.).

The following optimizers can be used with arbitrary separable functions:

 - [CMAES](#cmaes)

Each of these optimizers has an `Optimize()` function that is called as
`Optimize(f, x)` where `f` is the function to be optimized and `x` holds the
initial point of the optimization.  After `Optimize()` is called, `x` will hold
the final result of the optimization (that is, the best `x` found that
minimizes `f(x)`).

**Note**: using an arbitrary non-separable function optimizer will call
`Evaluate(x, 0, NumFunctions() - 1)`; if this is a very computationally
intensive operation for your objective function, it may be best to avoid using
a non-separable arbitrary function optimizer.

**Note**: if possible, it's often better to try and use a gradient-based
approach.  See [differentiable separable functions](#differentiable-separable-functions)
for separable f(x) where the gradient f'(x) can be computed.

The example program below demonstrates the implementation and use of an
arbitrary separable function.  The function used is the linear regression
objective function, described in [differentiable functions](#example-linear-regression).
Given some dataset `data` and responses `responses`, the linear regression
objective function is separable as

$$ f_i(x) = (\operatorname{responses}(i) - x' * \operatorname{data}(i))^2 $$

where $\operatorname{data}(i)$ represents the data point indexed by $i$ and
$\operatorname{responses}(i)$ represents the observed response indexed by $i$.

<details>
<summary>Click to collapse/expand example code.
</summary>

```c++
#include <ensmallen.hpp>

// This class implements the linear regression objective function as an
// arbitrary separable function type.
class LinearRegressionFunction
{
 public:
  // Create the linear regression function with the given data and the given
  // responses.
  LinearRegressionFunction(const arma::mat& dataIn,
                           const arma::rowvec& responsesIn) :
      data(data), responses(responses) { }

  // Given parameters x, compute the sum of the separable objective
  // functions starting with f_i(x) and ending with
  // f_{i + batchSize - 1}(x).
  double Evaluate(const arma::mat& x, const size_t i, const size_t batchSize)
  {
    // A more complex implementation could avoid the for loop and use
    // submatrices, but it is easier to understand when implemented this way.
    double objective = 0.0;
    for (size_t j = i; j < i + batchSize; ++j)
    {
      objective += std::pow(responses[j] - x.t() * data.col(j), 2.0);
    }
  }

  // Shuffle the ordering of the functions f_i(x).  We do this by simply
  // shuffling the data and responses.
  void Shuffle()
  {
    // Generate a random ordering of data points.
    arma::uvec ordering = arma::shuffle(
        arma::linspace<arma::uvec>(0, data.n_cols - 1, data.n_cols));

    // This reorders the data and responses with our randomly-generated
    // ordering above.
    data = data.cols(ordering);
    responses = responses.cols(ordering);
  }

  // Return the number of functions f_i(x).  In our case this is simply the
  // number of data points.
  size_t NumFunctions() { return data.n_cols; }
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
  arma::mat params(10, 1, arma::fill::randn);

  // Use the CMAES optimizer with default parameters to minimize the
  // LinearRegressionFunction.
  // The ens::CMAES type can be replaced with any suitable ensmallen optimizer
  // that can handle arbitrary separable functions.
  ens::CMAES cmaes;
  LinearRegressionFunction lrf(data, responses);
  cmaes.Optimize(lrf, params);

  std::cout << "The optimized linear regression model found by CMAES has the "
      << "parameters " << params.t();
}
```

</details>

## Differentiable separable functions

Likely the most important type of function to be optimized in machine learning
and some other fields is the differentiable separable function.  A
differentiable separable function f(x) can be represented as the sum of many
functions:

```
f(x) = f_0(x) + f_1(x) + ... + f_N(x).
```

And it also has a computable separable gradient `f'(x)`:

```
f'(x) = f'_0(x) + f'_1(x) + ... + f'_N(x).
```

For machine learning tasks, the objective function may be, e.g., the sum of a
function taken across many data points.  Implementing a differentiable
separable function type in ensmallen is similar to implementing an ordinary
differentiable function, but with a few extra utility methods:

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
class ArbitrarySeparableFunctionType
{
 public:
  // Given parameters x, return the sum of the individual functions
  // f_i(x) + ... + f_{i + batchSize - 1}(x).  i will always be greater than 0,
  // and i + batchSize will be less than or equal to the value of NumFunctions().
  double Evaluate(const arma::mat& x, const size_t i, const size_t batchSize);

  // Given parameters x and a matrix g, store the sum of the gradient of
  // individual functions f'_i(x) + ... + f'_{i + batchSize - 1}(x) into g.  i
  // will always be greater than 0, and i + batchSize will be less than or
  // equal to the value of NumFunctions().
  void Gradient(const arma::mat& x,
                const size_t i,
                arma::mat& g,
                const size_t batchSize);

  // Shuffle the ordering of the functions f_i(x).
  // (For machine learning problems, this would be equivalent to shuffling the
  // data points, e.g., before an epoch of learning.)
  void Shuffle();

  // Get the number of functions f_i(x).
  // (For machine learning problems, this is often just the number of points in
  // the dataset.)
  size_t NumFunctions();

  // OPTIONAL: this may be implemented in addition to---or instead
  // of---Evaluate() and Gradient().  If this is the only function implemented,
  // implementations of Evaluate() and Gradient() will be automatically
  // generated using template metaprogramming.  Often, implementing
  // EvaluateWithGradient() can result in more efficient optimizations.
  //
  // Given parameters x and a matrix g, return the sum of the individual
  // functions f_i(x) + ... + f_{i + batchSize - 1}(x), and store the sum of
  // the gradient of individual functions f'_i(x) + ... +
  // f'_{i + batchSize - 1}(x) into the provided matrix g.  g should have the
  // same size (rows, columns) as x.  i will always be greater than 0, and i +
  // batchSize will be less than or equal to the value of NumFunctions().
  double EvaluateWithGradient(const arma::mat& x,
                              const size_t i,
                              arma::mat& g,
                              const size_t batchSize);
};
```

</details>

Note that you may implement *either* `Evaluate()` and `Gradient()` *or*
`EvaluateWithGradient()`, but it is not mandatory to implement both.  (Of
course, supplying both is okay too.)  It often results in faster code when
`EvaluateWithGradient()` is implemented, especially for functions where f(x)
and f'(x) compute some of the same intermediate quantities.

Each of the implemented methods is allowed to have additional cv-modifiers
(`static`, `const`, etc.).

The following optimizers can be used with differentiable separable functions:

 - [AdaBelief](#adabelief)
 - [AdaBound](#adabound)
 - [AdaDelta](#adadelta)
 - [AdaGrad](#adagrad)
 - [AdaSqrt](#adasqrt)
 - [Adam](#adam)
 - [AdaMax](#adamax)
 - [AMSBound](#amsbound)
 - [AMSGrad](#amsgrad)
 - [Big Batch SGD](#big-batch-sgd)
 - [Eve](#eve)
 - [FTML](#ftml-follow-the-moving-leader)
 - [IQN](#iqn)
 - [Katyusha](#katyusha)
 - [Lookahead](#lookahead)
 - [Momentum SGD](#momentum-sgd)
 - [Nadam](#nadam)
 - [NadaMax](#nadamax)
 - [NesterovMomentumSGD](#nesterov-momentum-sgd)
 - [OptimisticAdam](#optimisticadam)
 - [QHAdam](#qhadam)
 - [QHSGD](#qhsgd)
 - [RMSProp](#rmsprop)
 - [SARAH/SARAH+](#stochastic-recursive-gradient-algorithm-sarahsarah)
 - [SGD](#standard-sgd)
 - [Stochastic Gradient Descent with Restarts (SGDR)](#stochastic-gradient-descent-with-restarts-sgdr)
 - [Snapshot SGDR](#snapshot-stochastic-gradient-descent-with-restarts)
 - [SMORMS3](#smorms3)
 - [SPALeRA](#spalera-stochastic-gradient-descent-spalerasgd)
 - [SWATS](#swats)
 - [SVRG](#standard-stochastic-variance-reduced-gradient-svrg)
 - [WNGrad](#wngrad)

The example program below demonstrates the implementation and use of an
arbitrary separable function.  The function used is the linear regression
objective function, described in [differentiable functions](#example-linear-regression).
Given some dataset `data` and responses `responses`, the linear regression
objective function is separable as

```
f_i(x) = (responses(i) - x' * data(i))^2
```

where `data(i)` represents the data point indexed by `i` and `responses(i)`
represents the observed response indexed by `i`.  This example implementation
only implements `EvaluateWithGradient()` in order to avoid redundant
calculations.

<details>
<summary>Click to collapse/expand example code.
</summary>

```c++
#include <ensmallen.hpp>

// This class implements the linear regression objective function as an
// arbitrary separable function type.
class LinearRegressionFunction
{
 public:
  // Create the linear regression function with the given data and the given
  // responses.
  LinearRegressionFunction(const arma::mat& dataIn,
                           const arma::rowvec& responsesIn) :
      data(data), responses(responses) { }

  // Given parameters x, compute the sum of the separable objective
  // functions starting with f_i(x) and ending with
  // f_{i + batchSize - 1}(x), and also compute the gradient of those functions
  // and store them in g.
  double EvaluateWithGradient(const arma::mat& x,
                              const size_t i,
                              arma::mat& g,
                              const size_t batchSize)
  {
    // This slightly complex implementation uses Armadillo submatrices to
    // compute the objective functions and gradients simultaneously for
    // multiple points.
    //
    // The shared computation between the objective and gradient is the term
    // (response - x * data) so we compute that first, only for points in the
    // batch.
    const arma::rowvec v = (responses.cols(i, i + batchSize - 1) - x.t() *
        data.cols(i, i + batchSize - 1));
    g = -2 * data.cols(i, i + batchSize - 1) * v;
    return arma::accu(v % v); // equivalent to |v|^2
  }

  // Shuffle the ordering of the functions f_i(x).  We do this by simply
  // shuffling the data and responses.
  void Shuffle()
  {
    // Generate a random ordering of data points.
    arma::uvec ordering = arma::shuffle(
        arma::linspace<arma::uvec>(0, data.n_cols - 1, data.n_cols));

    // This reorders the data and responses with our randomly-generated
    // ordering above.
    data = data.cols(ordering);
    responses = responses.cols(ordering);
  }

  // Return the number of functions f_i(x).  In our case this is simply the
  // number of data points.
  size_t NumFunctions() { return data.n_cols; }
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
  arma::mat params(10, 1, arma::fill::randn);

  // Use RMSprop to find the best parameters for the linear regression model.
  // The type 'ens::RMSprop' can be changed for any ensmallen optimizer able to
  // handle differentiable separable functions.
  ens::RMSProp rmsprop;
  LinearRegressionFunction lrf(data, responses);
  rmsprop.Optimize(lrf, params);

  std::cout << "The optimized linear regression model found by RMSprop has the"
      << " parameters " << params.t();
}
```

</details>

### Sparse differentiable separable functions

Some differentiable separable functions have the additional property that
the gradient `f'_i(x)` is sparse.  When this is true, one additional method can
be implemented as part of the class to be optimized:

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
// Add this definition to use sparse differentiable separable function
// optimizers.  Given x, store the sum of the sparse gradient f'_i(x) + ... +
// f'_{i + batchSize - 1}(x) into the provided matrix g.
void Gradient(const arma::mat& x,
              const size_t i,
              arma::sp_mat& g,
              const size_t batchSize);
```

</details>

It's also possible to instead use templates to provide only one `Gradient()`
function for both sparse and non-sparse optimizers:

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
// This provides Gradient() for both sparse and non-sparse optimizers.
template<typename GradType>
void Gradient(const arma::mat& x,
              const size_t i,
              GradType& g,
              const size_t batchSize);
```

</details>

If either of these methods are available, then any ensmallen optimizer that
optimizes sparse separable differentiable functions may be used.  This
includes:

 - [Hogwild!](#hogwild-parallel-sgd) (Parallel SGD)

## Categorical functions

A categorical function is a function f(x) where some of the values of x are
categorical variables (i.e. they take integer values [0, c - 1] and each value
0, 1, ..., c - 1 is unrelated).  In this situation, the function is not
differentiable, and so only the objective f(x) can be implemented.  Therefore
the class requirements for a categorical function are exactly the same as for
an `ArbitraryFunctionType`---but for any categorical dimension `x_i` in `x`, the
value will be in the range [0, c_i - 1] where `c_i` is the number of categories
in dimension `x_i`.

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
class CategoricalFunction
{
 public:
  // Return the objective function for the given parameters x.
  double Evaluate(const arma::mat& x);
};
```

</details>

However, when an optimizer's Optimize() method is called, two additional
parameters must be specified, in addition to the function to optimize and the
matrix holding the parameters:

 - `const std::vector<bool>& categoricalDimensions`: a vector of length equal
   to the number of elements in `x` (the number of dimensions).  If an element
   is true, then the dimension is categorical.

 - `const arma::Row<size_t>& numCategories`: a vector of length equal to the
   number of elements in `x` (the number of dimensions).  If a dimension is
   categorical, then the corresponding value in `numCategories` should hold the
   number of categories in that dimension.

The following optimizers can be used in this way to optimize a categorical function:

 - [Grid Search](#grid-search) (all parameters must be categorical)

An example program showing usage of categorical optimization is shown below.

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
#include <ensmallen.hpp>

// An implementation of a simple categorical function.  The parameters can be
// understood as x = [c1 c2 c3].  When c1 = 0, c2 = 2, and c3 = 1, the value of
// f(x) is 0.  In any other case, the value of f(x) is 10.  Therefore, the
// optimum is found at [0, 2, 1].
class SimpleCategoricalFunction
{
 public:
  // Return the objective function f(x) as described above.
  double Evaluate(const arma::mat& x)
  {
    if (size_t(x[0]) == 0 &&
        size_t(x[1]) == 2 &&
        size_t(x[2]) == 1)
      return 0.0;
    else
      return 10.0;
  }
};

int main()
{
  // Create and optimize the categorical function with the GridSearch
  // optimizer.  We must also create a std::vector<bool> that holds the types
  // of each dimension, and an arma::Row<size_t> that holds the number of
  // categories in each dimension.
  SimpleCategoricalFunction c;

  // We have three categorical dimensions only.
  std::vector<bool> categoricalDimensions;
  categoricalDimensions.push_back(true);
  categoricalDimensions.push_back(true);
  categoricalDimensions.push_back(true);

  // The first category can take 5 values; the second can take 3; the third can
  // take 12.
  arma::Row<size_t> numCategories("5 3 12");

  // The initial point for our optimization will be to set all categories to 0.
  arma::mat params("0 0 0");

  // Now create the GridSearch optimizer with default parameters, and run the
  // optimization.
  // The ens::GridSearch type can be replaced with any ensmallen optimizer that
  // is able to handle categorical functions.
  ens::GridSearch gs;
  gs.Optimize(c, params, categoricalDimensions, numCategories);

  std::cout << "The ens::GridSearch optimizer found the optimal parameters to "
      << "be " << params;
}
```

</details>


## Multi-objective functions

A multi-objective optimizer does not return just one set of coordinates at the
minimum of all objective functions, but instead finds a *front* or *frontier* of
possible coordinates that are Pareto-optimal (that is, no individual objective
function's value can be reduced without increasing at least one other
objective function).

In order to optimize a multi-objective function with ensmallen, a `std::tuple<>`
containing multiple `ArbitraryFunctionType`s ([see here](#arbitrary-functions))
should be passed to a multi-objective optimizer's `Optimize()` function.

An example below simultaneously optimizes the generalized Rosenbrock function
in 6 dimensions and the Wood function using [NSGA2](#nsga2).

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
GeneralizedRosenbrockFunction rf(6);
WoodFunction wf;
std::tuple<GeneralizedRosenbrockFunction, WoodFunction> objectives(rf, wf);

// Create an initial point (a random point in 6 dimensions).
arma::mat coordinates(6, 1, arma::fill::randu);

// `coordinates` will be set to the coordinates on the best front that minimize the
// sum of objective functions, and `bestFrontSum` will be the sum of all objectives
// at that coordinate set.
NSGA2 nsga;
double bestFrontSum = nsga.Optimize(objectives, coordinates);

// Set `bestFront` to contain all of the coordinates on the best front.
arma::cube bestFront = optimizer.ParetoFront();
}
```

</details>

*Note*: all multi-objective function optimizers have both the function `Optimize()` to find the
best front, and also the function `ParetoFront()` to return all sets of solutions that are on the
front.

The following optimizers can be used with multi-objective functions:
- [NSGA2](#nsga2)
- [MOEA/D-DE](#moead)

## Constrained functions

A constrained function is an objective function `f(x)` that is also subject to
some constraints on `x`.  (For instance, perhaps a constraint could be that `x`
is a positive semidefinite matrix.)  ensmallen is able to handle differentiable
objective functions of this type---so, `f'(x)` must also be computable.  Given
some set of constraints `c_0(x)`, ..., `c_M(x)`, we can re-express our constrained
objective function as

```
f_C(x) = f(x) + c_0(x) + ... + c_M(x)
```

where the (soft) constraint `c_i(x)` is a positive value if it is not satisfied, and
`0` if it is satisfied.  The soft constraint `c_i(x)` should take some value
representing how far from a feasible solution `x` is.  It should be
differentiable, since ensmallen's constrained optimizers will use the gradient
of the constraint to find a feasible solution.

In order to optimize a constrained function with ensmallen, a class
implementing the API below is required.

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
class ConstrainedFunctionType
{
 public:
  // Return the objective function f(x) for the given x.
  double Evaluate(const arma::mat& x);

  // Compute the gradient of f(x) for the given x and store the result in g.
  void Gradient(const arma::mat& x, arma::mat& g);

  // Get the number of constraints on the objective function.
  size_t NumConstraints();

  // Evaluate constraint i at the parameters x.  If the constraint is
  // unsatisfied, a value greater than 0 should be returned.  If the constraint
  // is satisfied, 0 should be returned.  The optimizer will add this value to
  // its overall objective that it is trying to minimize.
  double EvaluateConstraint(const size_t i, const arma::mat& x);

  // Evaluate the gradient of constraint i at the parameters x, storing the
  // result in the given matrix g.  If the constraint is not satisfied, the
  // gradient should be set in such a way that the gradient points in the
  // direction where the constraint would be satisfied.
  void GradientConstraint(const size_t i, const arma::mat& x, arma::mat& g);
};
```

</details>

A constrained function can be optimized with the following optimizers:

 - [Augmented Lagrangian](#augmented-lagrangian)

### Semidefinite programs

A special class of constrained function is a semidefinite program.  ensmallen
has support for creating and optimizing semidefinite programs via the
`ens::SDP<>` class.  For this, the SDP must be expressed in the primal form:

```
min_x dot(C, x) such that

dot(A_i, x) = b_i, for i = 1, ..., M;
x >= 0
```

In this case, `A_i` and `C` are square matrices (sparse or dense), and `b_i` is
a scalar.

Once the problem is expressed in this form, a class of type `ens::SDP<>` can be
created.  `SDP<arma::mat>` indicates an SDP with a dense C, and
`SDP<arma::sp_mat>` indicates an SDP with a sparse C.  The class has the
following useful methods:

 - `SDP(cMatrixSize, numDenseConstraints, numSparseConstraints)`: create a new `SDP`
 - `size_t NumSparseConstraints()`: get number of sparse constraint matrices A_i
 - `size_t NumDenseConstraints()`: get number of dense constraint matrices A_i
 - `std::vector<arma::mat>& DenseA()`: get vector of dense A_i matrices
 - `std::vector<arma::sp_mat>& SparseA()`: get vector of sparse A_i matrices
 - `arma::vec& DenseB()`: get vector of b_i values for dense A_i constraints
 - `arma::vec& SparseB()`: get vector of b_i values for sparse A_i constraints

Once these methods are used to set each A_i matrix and corresponding b_i value,
and C objective matrix, the SDP object can be used with any ensmallen SDP
solver.  The list of SDP solvers is below:

 - [Primal-dual SDP solver](#primal-dual-sdp-solver)
 - [Low-rank accelerated SDP solver (LRSDP)](#lrsdp-low-rank-sdp-solver)

Example code showing how to solve an SDP is given below.

<details>
<summary>Click to collapse/expand example code.
</summary>

```c++
int main()
{
  // We will build a toy semidefinite program and then use the PrimalDualSolver to find a solution

  // The semi-definite constraint looks like:
  //
  // [ 1  x_12  x_13  0  0  0  0 ]
  // [     1    x_23  0  0  0  0 ]
  // [            1   0  0  0  0 ]
  // [               s1  0  0  0 ]  >= 0
  // [                  s2  0  0 ]
  // [                     s3  0 ]
  // [                        s4 ]

  // x_11 == 0
  arma::sp_mat A0(7, 7); A0.zeros();
  A0(0, 0) = 1.;

  // x_22 == 0
  arma::sp_mat A1(7, 7); A1.zeros();
  A1(1, 1) = 1.;

  // x_33 == 0
  arma::sp_mat A2(7, 7); A2.zeros();
  A2(2, 2) = 1.;

  // x_12 <= -0.1  <==>  x_12 + s1 == -0.1, s1 >= 0
  arma::sp_mat A3(7, 7); A3.zeros();
  A3(1, 0) = A3(0, 1) = 1.; A3(3, 3) = 2.;

  // -0.2 <= x_12  <==>  x_12 - s2 == -0.2, s2 >= 0
  arma::sp_mat A4(7, 7); A4.zeros();
  A4(1, 0) = A4(0, 1) = 1.; A4(4, 4) = -2.;

  // x_23 <= 0.5  <==>  x_23 + s3 == 0.5, s3 >= 0
  arma::sp_mat A5(7, 7); A5.zeros();
  A5(2, 1) = A5(1, 2) = 1.; A5(5, 5) = 2.;

  // 0.4 <= x_23  <==>  x_23 - s4 == 0.4, s4 >= 0
  arma::sp_mat A6(7, 7); A6.zeros();
  A6(2, 1) = A6(1, 2) = 1.; A6(6, 6) = -2.;

  std::vector<arma::sp_mat> ais({A0, A1, A2, A3, A4, A5, A6});

  SDP<arma::sp_mat> sdp(7, 7 + 4 + 4 + 4 + 3 + 2 + 1, 0);

  for (size_t j = 0; j < 3; j++)
  {
    // x_j4 == x_j5 == x_j6 == x_j7 == 0
    for (size_t i = 0; i < 4; i++)
    {
      arma::sp_mat A(7, 7); A.zeros();
      A(i + 3, j) = A(j, i + 3) = 1;
      ais.emplace_back(A);
    }
  }

  // x_45 == x_46 == x_47 == 0
  for (size_t i = 0; i < 3; i++)
  {
    arma::sp_mat A(7, 7); A.zeros();
    A(i + 4, 3) = A(3, i + 4) = 1;
    ais.emplace_back(A);
  }

  // x_56 == x_57 == 0
  for (size_t i = 0; i < 2; i++)
  {
    arma::sp_mat A(7, 7); A.zeros();
    A(i + 5, 4) = A(4, i + 5) = 1;
    ais.emplace_back(A);
  }

  // x_67 == 0
  arma::sp_mat A(7, 7); A.zeros();
  A(6, 5) = A(5, 6) = 1;
  ais.emplace_back(A);

  std::swap(sdp.SparseA(), ais);

  sdp.SparseB().zeros();
  sdp.SparseB()[0] = sdp.SparseB()[1] = sdp.SparseB()[2] = 1.;
  sdp.SparseB()[3] = -0.2; sdp.SparseB()[4] = -0.4;
  sdp.SparseB()[5] = 1.; sdp.SparseB()[6] = 0.8;

  sdp.C().zeros();
  sdp.C()(0, 2) = sdp.C()(2, 0) = 1.;

  // That took a long time but we finally set up the problem right!  Now we can
  // use the PrimalDualSolver to solve it.
  // ens::PrimalDualSolver could be replaced with ens::LRSDP or other ensmallen
  // SDP solvers.
  PrimalDualSolver solver;
  arma::mat X, Z;
  arma::vec ysparse, ydense;
  // ysparse, ydense, and Z hold the primal and dual variables found during the
  // optimization.
  const double obj = solver.Optimize(sdp, X, ysparse, ydense, Z);

  std::cout << "SDP optimized with objective " << obj << "." << std::endl;
}
```

</details>

## Alternate matrix types

All of the examples above (and throughout the rest of the documentation)
generally assume that the matrix being optimized has type `arma::mat`.  But
ensmallen's optimizers are capable of optimizing more types than just dense
Armadillo matrices.  In fact, the full signature of each optimizer's
`Optimize()` method is this:

```
template<typename FunctionType, typename MatType>
typename MatType::elem_type Optimize(FunctionType& function,
                                     MatType& coordinates);
```

The return type, `typename MatType::elem_type`, is just the numeric type held by
the given matrix type.  So, for `arma::mat`, the return type is just `double`.
In addition, optimizers for differentiable functions have a third template
parameter, `GradType`, which specifies the type of the gradient.  `GradType` can
be manually specified in the situation where, e.g., a sparse gradient is
desired.

It is easy to write a function to optimize, e.g., an `arma::fmat`.  Here is an
example, adapted from the `SquaredFunction` example from the
[arbitrary function documentation](#example__squared_function_optimization).

<details open>
<summary>Click to collapse/expand example code.
</summary>

```c++
#include <ensmallen.hpp>

class SquaredFunction
{
 public:
  // This returns f(x) = 2 |x|^2.
  float Evaluate(const arma::fmat& x)
  {
    return 2 * std::pow(arma::norm(x), 2.0);
  }

  void Gradient(const arma::fmat& x, arma::fmat& gradient)
  {
    gradient = 4 * x;
  }
};

int main()
{
  // The minimum is at x = [0 0 0].  Our initial point is chosen to be
  // [1.0, -1.0, 1.0].
  arma::fmat x("1.0 -1.0 1.0");

  // Create simulated annealing optimizer with default options.
  // The ens::SA<> type can be replaced with any suitable ensmallen optimizer
  // that is able to handle arbitrary functions.
  ens::L_BFGS optimizer;
  SquaredFunction f; // Create function to be optimized.
  optimizer.Optimize(f, x); // The optimizer will infer arma::fmat!

  std::cout << "Minimum of squared function found with simulated annealing is "
      << x;
}
```

</details>

Note that we have simply changed the `SquaredFunction` to accept `arma::fmat`
instead of `arma::mat` as parameters to `Evaluate()`, and the return type has
accordingly been changed to `float` from `double`.  It would even be possible to
optimize functions with sparse coordinates by having `Evaluate()` take a sparse
matrix (i.e. `arma::sp_mat`).

If it were desired to represent the gradient as a sparse type, the `Gradient()`
function would need to be modified to take a sparse matrix (i.e. `arma::sp_mat`
or similar), and then you could call `optimizer.Optimize<SquaredFunction,
arma::mat, arma::sp_mat>(f, x);` to perform the optimization while using sparse
matrix types to represent the gradient.  Using sparse `MatType` or `GradType`
should *only* be done when it is known that the objective matrix and/or
gradients will be sparse; otherwise the code may run very slow!

ensmallen will automatically infer `MatType` from the call to `Optimize()`, and
check that the given `FunctionType` has all of the necessary functions for the
given `MatType`, throwing a `static_assert` error if not.  If you would like to
disable these checks, define the macro `ENS_DISABLE_TYPE_CHECKS` before
including ensmallen:

```
#define ENS_DISABLE_TYPE_CHECKS
#include <ensmallen.hpp>
```

This can be useful for situations where you know that the checks should be
ignored.  However, be aware that the code may fail to compile and give more
confusing and difficult error messages!

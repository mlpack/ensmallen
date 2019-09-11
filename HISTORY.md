### ensmallen 2.10.1: "Fried Chicken"
###### 2019-09-10
  * Documentation fix for callbacks
    ([#129](https://github.com/mlpack/ensmallen/pull/129).

  * Compatibility fixes for ensmallen 1.x
    ([#131](https://github.com/mlpack/ensmallen/pull/131)).

### ensmallen 2.10.0: "Fried Chicken"
###### 2019-09-07
  * All `Optimize()` functions now take any matrix type; so, e.g., `arma::fmat`
    or `arma::sp_mat` can be used for optimization.  See the documentation for
    more details ([#113](https://github.com/mlpack/ensmallen/pull/113),
    [#119](https://github.com/mlpack/ensmallen/pull/119)).

  * Introduce callback support.  Callbacks can be appended as the last arguments
    of an `Optimize()` call, and can perform custom behavior at different points
    during the optimization.  See the documentation for more details
    ([#119](https://github.com/mlpack/ensmallen/pull/119)).

  * Slight speedups for `FrankWolfe` optimizer
    ([#127](https://github.com/mlpack/ensmallen/pull/127)).

### ensmallen 1.16.2: "Loud Alarm Clock"
###### 2019-08-12
  * Fix PSO return type bug
    ([#126](https://github.com/mlpack/ensmallen/pull/126)).

### ensmallen 1.16.1: "Loud Alarm Clock"
###### 2019-08-11
  * Update HISTORY.md to use Markdown links to the PR and add release names.

  * Fix PSO return type bug
    ([#124](https://github.com/mlpack/ensmallen/pull/124)).

### ensmallen 1.16.0: "Loud Alarm Clock"
###### 2019-08-09
  * Add option to avoid computing exact objective at the end of the optimization
    ([#109](https://github.com/mlpack/ensmallen/pull/109)).

  * Fix handling of curvature for BigBatchSGD
    ([#118](https://github.com/mlpack/ensmallen/pull/118)).

  * Reduce runtime of tests
    ([#118](https://github.com/mlpack/ensmallen/pull/118)).

  * Introduce local-best particle swarm optimization, `LBestPSO`, for
    unconstrained optimization problems
    ([#86](https://github.com/mlpack/ensmallen/pull/86)).

### ensmallen 1.15.1: "Wrong Side Of The Road"
###### 2019-05-22
  * Fix `-Wreorder` in `qhadam` warning
    ([#115](https://github.com/mlpack/ensmallen/pull/115)).

  * Fix `-Wunused-private-field` warning in `spsa`
    ([#115](https://github.com/mlpack/ensmallen/pull/115)).

  * Add more warning output for gcc/clang
    ([#116](https://github.com/mlpack/ensmallen/pull/116)).

### ensmallen 1.15.0: "Wrong Side Of The Road"
###### 2019-05-14
  * Added QHAdam and QHSGD optimizers
    ([#81](https://github.com/mlpack/ensmallen/pull/81)).

### ensmallen 1.14.4: "Difficult Crimp"
###### 2019-05-12
  * Fixes for BigBatchSGD ([#91](https://github.com/mlpack/ensmallen/pull/91)).

### ensmallen 1.14.3: "Difficult Crimp"
###### 2019-05-06
  * Handle `eig_sym()` failures correctly
    ([#100](https://github.com/mlpack/ensmallen/pull/100)).

### ensmallen 1.14.2: "Difficult Crimp"
###### 2019-03-14
  * SPSA test tolerance fix
    ([#97](https://github.com/mlpack/ensmallen/pull/97)).

  * Minor documentation fixes (#95, #98).

  * Fix newlines at end of file
    ([#92](https://github.com/mlpack/ensmallen/pull/92)).

### ensmallen 1.14.1: "Difficult Crimp"
###### 2019-03-09
  * Fixes for SPSA ([#87](https://github.com/mlpack/ensmallen/pull/87)).

  * Optimized CNE and DE ([#90](https://github.com/mlpack/ensmallen/pull/90)).
    Changed initial population generation in CNE to be a normal distribution
    about the given starting point, which should accelerate convergence.

### ensmallen 1.14.0: "Difficult Crimp"
###### 2019-02-20
  * Add DE optimizer ([#77](https://github.com/mlpack/ensmallen/pull/77)).

  * Fix for Cholesky decomposition in CMAES
    ([#83](https://github.com/mlpack/ensmallen/pull/83)).

### ensmallen 1.13.2: "Coronavirus Invasion"
###### 2019-02-18
 * Minor documentation fixes ([#82](https://github.com/mlpack/ensmallen/pull/82)).

### ensmallen 1.13.1: "Coronavirus Invasion"
###### 2019-01-24
 * Fix -Wreorder warning ([#75](https://github.com/mlpack/ensmallen/pull/75)).

### ensmallen 1.13.0: "Coronavirus Invasion"
###### 2019-01-14
 * Enhance options for AugLagrangian optimizer
   ([#66](https://github.com/mlpack/ensmallen/pull/66)).

 * Add SPSA optimizer ([#69](https://github.com/mlpack/ensmallen/pull/69)).

### ensmallen 1.12.2: "New Year's Party"
###### 2019-01-05
 * Fix list of contributors.

### ensmallen 1.12.1: "New Year's Party"
###### 2019-01-03
 * Make sure all files end with newlines.

### ensmallen 1.12.0: "New Year's Party"
###### 2018-12-30
 * Add link to ensmallen PDF to README.md.

 * Minor documentation fixes.  Remove too-verbose documentation from source for
   each optimizer ([#61](https://github.com/mlpack/ensmallen/pull/61)).

 * Add FTML optimizer ([#48](https://github.com/mlpack/ensmallen/pull/48)).

 * Add SWATS optimizer ([#42](https://github.com/mlpack/ensmallen/pull/42)).

 * Add Padam optimizer ([#46](https://github.com/mlpack/ensmallen/pull/46)).

 * Add Eve optimizer ([#45](https://github.com/mlpack/ensmallen/pull/45)).

 * Add ResetPolicy() to SGD-like optimizers
   ([#60](https://github.com/mlpack/ensmallen/pull/60)).

### ensmallen 1.11.1: "Jet Lag"
###### 2018-11-29
 * Minor documentation fixes.

### ensmallen 1.11.0: "Jet Lag"
###### 2018-11-28
 * Add WNGrad optimizer.

 * Fix header name in documentation samples.

### ensmallen 1.10.1: "Corporate Catabolism"
###### 2018-11-16
 * Fixes for GridSearch optimizer.

 * Include documentation with release.

### ensmallen 1.10.0: "Corporate Catabolism"
###### 2018-10-20
 * Initial release.

 * Includes the ported optimization framework from mlpack
   (http://www.mlpack.org/).

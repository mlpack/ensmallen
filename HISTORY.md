### ensmallen 3.10.0: "Unexpected Rain"
###### 2025-09-25
 * SGD-like optimizers now all divide the step size by the batch size so that
   step sizes don't need to be tuned in addition to batch sizes.  If you require
   behavior from ensmallen 2, define the `ENS_OLD_SEPARABLE_STEP_BEHAVIOR` macro
   before including `ensmallen.hpp`
   ([#431](https://github.com/mlpack/ensmallen/pull/431)).

 * Remove deprecated `ParetoFront()` and `ParetoSet()` from multi-objective
   optimizers ([#435](https://github.com/mlpack/ensmallen/pull/435)).  Instead,
   pass objects to the `Optimize()` function; see the documentation for each
   multi-objective optimizer for more details.  A typical transition will change
   code like:

    ```c++
    optimizer.Optimize(objectives, coordinates);
    arma::cube paretoFront = optimizer.ParetoFront();
    arma::cube paretoSet = optimizer.ParetoSet();
    ```

   to instead gather the Pareto front and set in the call:

    ```c++
    arma::cube paretoFront, paretoSet;
    optimizer.Optimize(objectives, coordinates, paretoFront, paretoSet);
    ```

 * Remove deprecated constructor for Active CMA-ES that takes `lowerBound` and
   `upperBound` ([#435](https://github.com/mlpack/ensmallen/pull/435)).
   Instead, pass an instantiated `BoundaryBoxConstraint` to the constructor.  A
   typical transition will change code like:

    ```c++
    ActiveCMAES<FullSelection, BoundaryBoxConstraint> opt(lambda,
        lowerBound, upperBound, ...);
    ```

   into

    ```c++
    ActiveCMAES<FullSelection, BoundaryBoxConstraint> opt(lambda,
        BoundaryBoxConstraint(lowerBound, upperBound), ...);
    ```
 * Add proximal gradient optimizers for L1-constrained and other related
   problems: `FBS`, `FISTA`, and `FASTA`
   ([#427](https://github.com/mlpack/ensmallen/pull/427)).  See the
   documentation for more details.

 * The `Lambda()` and `Sigma()` functions of the `AugLagrangian` optimizer,
   which could be used to retrieve the Lagrange multipliers and penalty
   parameter after optimization, are now deprecated
   ([#439](https://github.com/mlpack/ensmallen/pull/439)).  Instead, pass a
   vector and a double to the `Optimize()` function directly:

    ```c++
    augLag.Optimize(function, coordinates, lambda, sigma)
    ```

   and these will be filled with the final Lagrange multiplier estimates and
   penalty parameters.

### ensmallen 2.22.2: "E-Bike Excitement"
###### 2025-04-30
 * Fix include statement in `tests/de_test.cpp`
   ([#419](https://github.com/mlpack/ensmallen/pull/419)).

 * Fix `exactObjective` output for SGD-like optimizers when the number of
   iterations is an even number of epochs
   ([#417](https://github.com/mlpack/ensmallen/pull/417)).

 * Increase tolerance in `demon_sgd_test.cpp`
   ([#420](https://github.com/mlpack/ensmallen/pull/420)).

 * Set cmake version range to 3.5...4.0
   ([#422](https://github.com/mlpack/ensmallen/pull/422)).

### ensmallen 2.22.1: "E-Bike Excitement"
###### 2024-12-02
 * Remove unused variables to fix compiler warnings
   ([#413](https://github.com/mlpack/ensmallen/pull/413)).

### ensmallen 2.22.0: "E-Bike Excitement"
###### 2024-11-29
 * Update to C++14 standard
   ([#400](https://github.com/mlpack/ensmallen/pull/400)).

 * Bump minimum Armadillo version to 10.8
   ([#404](https://github.com/mlpack/ensmallen/pull/404)).

 * For Armadillo 14.2.0 switch to `.index_min()` and `.index_max()`
   ([#409](https://github.com/mlpack/ensmallen/pull/409)).

 * Added IPOP and BIPOP restart mechanisms for CMA-ES.
   ([#403](https://github.com/mlpack/ensmallen/pull/403)).


### ensmallen 2.21.1: "Bent Antenna"
###### 2024-02-15
 * Fix numerical precision issues for small-gradient L-BFGS scaling factor
   computations ([#392](https://github.com/mlpack/ensmallen/pull/392)).

 * Ensure the tests are built with optimisation enabled
   ([#394](https://github.com/mlpack/ensmallen/pull/394)).

### ensmallen 2.21.0: "Bent Antenna"
###### 2023-11-27
 * Clarify return values for different callback types
   ([#383](https://github.com/mlpack/ensmallen/pull/383)).

 * Fix return types of callbacks
   ([#382](https://github.com/mlpack/ensmallen/pull/382)).

 * Minor cleanup for printing optimization reports via `Report()`
   ([#385](https://github.com/mlpack/ensmallen/pull/385)).

### ensmallen 2.20.0: "Stripped Bolt Head"
###### 2023-10-02
 * Implementation of Active CMAES
   ([#367](https://github.com/mlpack/ensmallen/pull/367)).

 * LBFGS: avoid generation of NaNs, and add checks for finite values
   ([#368](https://github.com/mlpack/ensmallen/pull/368)).

 * Fix CNE test tolerances
   ([#360](https://github.com/mlpack/ensmallen/pull/360)).

 * Rename `SCD` optimizer, to `CD`
   ([#379](https://github.com/mlpack/ensmallen/pull/379)).

### ensmallen 2.19.1: "Eight Ball Deluxe"
###### 2023-01-30
 * Avoid deprecation warnings in Armadillo 11.2+
   ([#347](https://github.com/mlpack/ensmallen/pull/347)).

### ensmallen 2.19.0: "Eight Ball Deluxe"
###### 2022-04-06
 * Added DemonSGD and DemonAdam optimizers
   ([#211](https://github.com/mlpack/ensmallen/pull/211)).

 * Fix bug with Adam-like optimizers not resetting when `resetPolicy` is `true`.
   ([#340](https://github.com/mlpack/ensmallen/pull/340)).

 * Add Yogi optimizer
   ([#232](https://github.com/mlpack/ensmallen/pull/232)).

 * Add AdaBelief optimizer
   ([#233](https://github.com/mlpack/ensmallen/pull/233)).

 * Add AdaSqrt optimizer
   ([#234](https://github.com/mlpack/ensmallen/pull/234)).

 * Bump check for minimum supported version of Armadillo
   ([#342](https://github.com/mlpack/ensmallen/pull/342)).

### ensmallen 2.18.2: "Fairmount Bagel"
###### 2022-02-13
 * Update Catch2 to 2.13.8
  ([#336](https://github.com/mlpack/ensmallen/pull/336)).

 * Fix epoch timing output
  ([#337](https://github.com/mlpack/ensmallen/pull/337)).

### ensmallen 2.18.1: "Fairmount Bagel"
###### 2021-11-19
 * Accelerate SGD test time
   ([#330](https://github.com/mlpack/ensmallen/pull/300)).

 * Fix potential infinite loop in CMAES
   ([#331](https://github.com/mlpack/ensmallen/pull/331)).

 * Fix SCD partial gradient test
   ([#332](https://github.com/mlpack/ensmallen/pull/332)).

### ensmallen 2.18.0: "Fairmount Bagel"
###### 2021-10-20
 * Add gradient value clipping and gradient norm scaling callback
   ([#315](https://github.com/mlpack/ensmallen/pull/315)).

 * Remove superfluous CMake option to build the tests
   ([#313](https://github.com/mlpack/ensmallen/pull/313)).

 * Bump minimum Armadillo version to 9.800
   ([#318](https://github.com/mlpack/ensmallen/pull/318)).

 * Update Catch2 to 2.13.7
   ([#322](https://github.com/mlpack/ensmallen/pull/322)).

 * Remove redundant template argument for C++20 compatibility
   ([#324](https://github.com/mlpack/ensmallen/pull/324)).

 * Fix MOEAD test stability
   ([#327](https://github.com/mlpack/ensmallen/pull/327)).

### ensmallen 2.17.0: "Pachis Din Me Pesa Double"
###### 2021-07-06
 * CheckArbitraryFunctionTypeAPI extended for MOO support
   ([#283](https://github.com/mlpack/ensmallen/pull/283)).

 * Refactor NSGA2
   ([#263](https://github.com/mlpack/ensmallen/pull/263),
   [#304](https://github.com/mlpack/ensmallen/pull/304)).

 * Add Indicators for Multiobjective optimizers
   ([#285](https://github.com/mlpack/ensmallen/pull/285)).

 * Make Callback flexible for MultiObjective Optimizers
   ([#289](https://github.com/mlpack/ensmallen/pull/289)).

 * Add ZDT Test Suite
   ([#273](https://github.com/mlpack/ensmallen/pull/273)).

 * Add MOEA-D/DE Optimizer
   ([#269](https://github.com/mlpack/ensmallen/pull/269)).

 * Introduce Policy Methods for MOEA/D-DE
   ([#293](https://github.com/mlpack/ensmallen/pull/293)).

 * Add Das-Dennis weight initialization method
   ([#295](https://github.com/mlpack/ensmallen/pull/295)).

 * Add Dirichlet Weight Initialization
   ([#296](https://github.com/mlpack/ensmallen/pull/296)).

 * Improved installation and compilation instructions
   ([#300](https://github.com/mlpack/ensmallen/pull/300)).

 * Disable building the tests by default for faster installation
   ([#303](https://github.com/mlpack/ensmallen/pull/303)).

 * Modify matrix initialisation to take into account
   default element zeroing in Armadillo 10.5
   ([#305](https://github.com/mlpack/ensmallen/pull/305)).

### ensmallen 2.16.2: "Severely Dented Can Of Polyurethane"
###### 2021-03-24
 * Fix CNE test trials
   ([#267](https://github.com/mlpack/ensmallen/pull/267)).

 * Update Catch2 to 2.13.4
   ([#268](https://github.com/mlpack/ensmallen/pull/268)).

 * Fix typos in documentation
   ([#270](https://github.com/mlpack/ensmallen/pull/270),
    [#271](https://github.com/mlpack/ensmallen/pull/271)).

 * Add clarifying comments in problems/ implementations
   ([#276](https://github.com/mlpack/ensmallen/pull/276)).

### ensmallen 2.16.1: "Severely Dented Can Of Polyurethane"
###### 2021-03-02
 * Fix test compilation issue when `ENS_USE_OPENMP` is set
   ([#255](https://github.com/mlpack/ensmallen/pull/255)).

 * Fix CNE initial population generation to use normal distribution
   ([#258](https://github.com/mlpack/ensmallen/pull/258)).

 * Fix compilation warnings
   ([#259](https://github.com/mlpack/ensmallen/pull/259)).

 * Remove `AdamSchafferFunctionN2Test` test from Adam test suite to prevent
   spurious issue on some aarch64 ([#265](https://github.com/mlpack/ensmallen/pull/259)).

### ensmallen 2.16.0: "Severely Dented Can Of Polyurethane"
###### 2021-02-11
 * Expand README with example installation and add simple example program
   showing usage of the L-BFGS optimizer
   ([#248](https://github.com/mlpack/ensmallen/pull/248)).

 * Refactor tests to increase stability and reduce random errors
   ([#249](https://github.com/mlpack/ensmallen/pull/249)).

### ensmallen 2.15.1: "Why Can't I Manage To Grow Any Plants?"
###### 2020-11-05
 * Fix include order to ensure traits is loaded before reports
   ([#239](https://github.com/mlpack/ensmallen/pull/239)).

### ensmallen 2.15.0: "Why Can't I Manage To Grow Any Plants?"
###### 2020-11-01
 * Make a few tests more robust
   ([#228](https://github.com/mlpack/ensmallen/pull/228)).

 * Add release date to version information. ([#226](https://github.com/mlpack/ensmallen/pull/226))

 * Fix typo in release script
   ([#236](https://github.com/mlpack/ensmallen/pull/236)).

### ensmallen 2.14.2: "No Direction Home"
###### 2020-08-31
 * Fix implementation of fonesca fleming problem function f1 and f2
   type usage and negative signs. ([#223](https://github.com/mlpack/ensmallen/pull/223))

### ensmallen 2.14.1: "No Direction Home"
###### 2020-08-19
 * Fix release script (remove hardcoded information, trim leading whitespaces
   introduced by `wc -l` in MacOS)
    ([#216](https://github.com/mlpack/ensmallen/pull/216),
     [#220](https://github.com/mlpack/ensmallen/pull/220)).

 * Adjust tolerance for AugLagrangian convergence based on element type
   ([#217](https://github.com/mlpack/ensmallen/pull/217)).

### ensmallen 2.14.0: "No Direction Home"
###### 2020-08-10
 * Add NSGA2 optimizer for multi-objective functions
    ([#149](https://github.com/mlpack/ensmallen/pull/149)).

 * Update automatic website update release script
   ([#207](https://github.com/mlpack/ensmallen/pull/207)).

 * Clarify and fix documentation for constrained optimizers
   ([#201](https://github.com/mlpack/ensmallen/pull/201)).

 * Fix L-BFGS convergence when starting from a minimum
   ([#201](https://github.com/mlpack/ensmallen/pull/201)).

* Add optimizer summary report callback
   ([#213](https://github.com/mlpack/ensmallen/pull/213)).

### ensmallen 2.13.0: "Automatically Automated Automation"
###### 2020-07-15
 * Fix CMake package export
    ([#198](https://github.com/mlpack/ensmallen/pull/198)).

 * Allow early stop callback to accept a lambda function
   ([#165](https://github.com/mlpack/ensmallen/pull/165)).

### ensmallen 2.12.1: "Stir Crazy"
###### 2020-04-20
 * Fix total number of epochs and time estimation for ProgressBar callback
    ([#181](https://github.com/mlpack/ensmallen/pull/181)).

 * Handle SpSubview_col and SpSubview_row in Armadillo 9.870
    ([#194](https://github.com/mlpack/ensmallen/pull/194)).

 * Minor documentation fixes
    ([#197](https://github.com/mlpack/ensmallen/pull/197)).

### ensmallen 2.12.0: "Stir Crazy"
###### 2020-03-28
 * Correction in the formulation of sigma in CMA-ES
    ([#183](https://github.com/mlpack/ensmallen/pull/183)).

 * Remove deprecated methods from PrimalDualSolver implementation
    ([#185](https://github.com/mlpack/ensmallen/pull/185).

 * Update logo ([#186](https://github.com/mlpack/ensmallen/pull/186)).

### ensmallen 2.11.5: "The Poster Session Is Full"
###### 2020-03-11
  * Change "mathematical optimization" term to "numerical optimization" in the
    documentation ([#177](https://github.com/mlpack/ensmallen/pull/177)).

### ensmallen 2.11.4: "The Poster Session Is Full"
###### 2020-03-03
  * Require new HISTORY.md entry for each PR.
    ([#171](https://github.com/mlpack/ensmallen/pull/171),
     [#172](https://github.com/mlpack/ensmallen/pull/172),
     [#175](https://github.com/mlpack/ensmallen/pull/175)).

  * Update/fix example documentation
    ([#174](https://github.com/mlpack/ensmallen/pull/174)).

### ensmallen 2.11.3: "The Poster Session Is Full"
###### 2020-02-19
  * Prevent spurious compiler warnings
    ([#161](https://github.com/mlpack/ensmallen/pull/161)).

  * Fix minor memory leaks
    ([#167](https://github.com/mlpack/ensmallen/pull/167)).

  * Revamp CMake configuration
    ([#152](https://github.com/mlpack/ensmallen/pull/152)).

### ensmallen 2.11.2: "The Poster Session Is Full"
###### 2020-01-16
  * Allow callback instantiation for SGD based optimizer
    ([#138](https://github.com/mlpack/ensmallen/pull/155)).

  * Minor test stability fixes on i386
    ([#156](https://github.com/mlpack/ensmallen/pull/156)).

  * Fix Lookahead MaxIterations() check.
    ([#159](https://github.com/mlpack/ensmallen/pull/159)).

### ensmallen 2.11.1: "The Poster Session Is Full"
###### 2019-12-28
  * Fix Lookahead Synchronization period type
    ([#153](https://github.com/mlpack/ensmallen/pull/153)).

### ensmallen 2.11.0: "The Poster Session Is Full"
###### 2019-12-24
  * Add Lookahead
    ([#138](https://github.com/mlpack/ensmallen/pull/138)).

  * Add AdaBound and AMSBound
    ([#137](https://github.com/mlpack/ensmallen/pull/137)).

### ensmallen 2.10.5: "Fried Chicken"
###### 2019-12-13
  * SGD callback test 32-bit safety (big number)
    ([#143](https://github.com/mlpack/ensmallen/pull/143)).

  * Use "arbitrary" and "separable" terms in static function type checks
    ([#145](https://github.com/mlpack/ensmallen/pull/145)).

  * Remove 'using namespace std' from `problems/` files
    ([#147](https://github.com/mlpack/ensmallen/pull/147)).

### ensmallen 2.10.4: "Fried Chicken"
###### 2019-11-18
  * Add optional tests building.
    ([#141](https://github.com/mlpack/ensmallen/pull/141)).

  * Make code samples collapsible in the documentation.
    ([#140](https://github.com/mlpack/ensmallen/pull/140)).

### ensmallen 2.10.3: "Fried Chicken"
###### 2019-09-26
  * Fix ParallelSGD runtime bug.
    ([#135](https://github.com/mlpack/ensmallen/pull/135)).

  * Add additional L-BFGS convergence check
    ([#136](https://github.com/mlpack/ensmallen/pull/136)).

### ensmallen 2.10.2: "Fried Chicken"
###### 2019-09-11
  * Add release script to rel/ for maintainers
    ([#128](https://github.com/mlpack/ensmallen/pull/128)).

  * Fix Armadillo version check
    ([#133](https://github.com/mlpack/ensmallen/pull/133)).

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

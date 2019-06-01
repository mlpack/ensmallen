### ensmallen 1.15.1
###### 2019-05-22
  * Fix -Wreorder in `qhadam` warning (#115).

  * Fix -Wunused-private-field warning in `spsa` (#115).

  * Add more warning output for gcc/clang (#116).

### ensmallen 1.15.0
###### 2019-05-14
  * Added QHAdam and QHSGD optimizers (#81).

### ensmallen 1.14.4
###### 2019-05-12
  * Fixes for BigBatchSGD (#91).

### ensmallen 1.14.3
###### 2019-05-06
  * Handle eig_sym() failures correctly (#100).

### ensmallen 1.14.2
###### 2019-03-14
  * SPSA test tolerance fix (#97).

  * Minor documentation fixes (#95, #98).

  * Fix newlines at end of file (#92).

### ensmallen 1.14.1
###### 2019-03-09
  * Fixes for SPSA (#87).

  * Optimized CNE and DE (#90). Changed initial population generation in CNE
    to be a normal distribution about the given starting point, which should
    accelerate convergence.

### ensmallen 1.14.0
###### 2019-02-20
  * Add DE optimizer (#77).

  * Fix for Cholesky decomposition in CMAES (#83).

### ensmallen 1.13.2
###### 2019-02-18
 * Minor documentation fixes (#82).

### ensmallen 1.13.1
###### 2019-01-24
 * Fix -Wreorder warning (#75).

### ensmallen 1.13.0
###### 2019-01-14
 * Enhance options for AugLagrangian optimizer (#66).

 * Add SPSA optimizer (#69).

### ensmallen 1.12.2
###### 2019-01-05
 * Fix list of contributors.

### ensmallen 1.12.1
###### 2019-01-03
 * Make sure all files end with newlines.

### ensmallen 1.12.0
###### 2018-12-30
 * Add link to ensmallen PDF to README.md.

 * Minor documentation fixes.  Remove too-verbose documentation from source for
   each optimizer (#61).

 * Add FTML optimizer (#48).

 * Add SWATS optimizer (#42).

 * Add Padam optimizer (#46).

 * Add Eve optimizer (#45).

 * Add ResetPolicy() to SGD-like optimizers (#60).

### ensmallen 1.11.1
###### 2018-11-29
 * Minor documentation fixes.

### ensmallen 1.11.0
###### 2018-11-28
 * Add WNGrad optimizer.

 * Fix header name in documentation samples.

### ensmallen 1.10.1
###### 2018-11-16
 * Fixes for GridSearch optimizer.

 * Include documentation with release.

### ensmallen 1.10.0
###### 2018-10-20
 * Initial release.

 * Includes the ported optimization framework from mlpack
   (http://www.mlpack.org/).

**ensmallen** is a C++ header-only library for mathematical optimization.

Documentation and downloads: http://ensmallen.org

ensmallen provides a simple set of abstractions for writing an objective
function to optimize. It also provides a large set of standard and cutting-edge
optimizers that can be used for virtually any mathematical optimization task.
These include full-batch gradient descent techniques, small-batch techniques,
gradient-free optimizers, and constrained optimization.


### Requirements

* C++ compiler with C++11 support
* Armadillo: http://arma.sourceforge.net
* OpenBLAS or Intel MKL or LAPACK (see Armadillo site for details)

### Install Dependencies

<details open>
<summary>Linux</summary>

Use your distributions' package manager to install the required dependencies. The commands shown below are for Ubuntu.

```bash
$ sudo apt-get update
$ sudo apt-get install -y g++ cmake
$ sudo apt-get install -y libopenblas-dev liblapack-dev xz-utils
$ sudo apt-get install -y libarmadillo-dev
```
</details>

<details open>
<summary>MacOS</summary>

Use a package manager like [Homebrew](https://brew.sh) to get the necessary dependencies.

````bash
$ brew install cmake openblas lapack armadillo
````
</details>

<details open>
<summary>Windows</summary>

You can install **ensmallen** directly by using [vcpkg](https://github.com/microsoft/vcpkg).

```
vcpkg install ensmallen:x64-windows
```
</details>

### Build and Test

This section describes how to build and test the **ensmallen** library from source. **ensmallen** uses CMake as its build system.

First, clone the source code from Github and change into the cloned directory. Or alternatively, you can download the latest relese from the [website](http://ensmallen.org) and extract it.

```bash
$ git clone https://github.com/mlpack/ensmallen
$ cd ensmallen

# - or -

$ wget http://ensmallen.org/files/ensmallen-latest.tar.gz
$ tar -xvzpf ensmallen-latest.tar.gz
$ cd ensmallen-latest
```

Next, make a build directory and change into that directory.

```bash
$ mkdir build
$ cd build
```

Then, run the cmake command followed by the make command in the build directory. If the cmake command fails, you probably have missing dependencies.

```bash
$ cmake ..
# or with -w flag to inhibit all warning messages
# $ cmake -DCMAKE_CXX_FLAGS="-w" -DCMAKE_C_FLAGS="-w" .. #
$ make -j4
```

Now, you can either run all of the test or an individual test case with:

```bash
$ ./ensmallen_tests
$ ./ensmallen_tests <test name>
```

Ensmallen uses [Catch2](https://github.com/catchorg/Catch2) as the unit test framework.
You can list all tests with:

```bash
$ ./ensmallen-tests -l

ensmallen version: 2.10.5 (Fried Chicken)
armadillo version: 9.800.3 (Horizon Scraper)
All available test cases:
  SimpleAdaDeltaTestFunction
      [AdaDeltaTest]
...
262 test cases
```


### License

Unless stated otherwise, the source code for **ensmallen**
is licensed under the 3-clause BSD license (the "License").
A copy of the License is included in the "LICENSE.txt" file.
You may also obtain a copy of the License at
http://opensource.org/licenses/BSD-3-Clause


### Citation

Please cite the following paper if you use ensmallen in your research and/or
software. Citations are useful for the continued development and maintenance of
the library.

* S. Bhardwaj, R. Curtin, M. Edel, Y. Mentekidis, C. Sanderson.
  [ensmallen: a flexible C++ library for efficient function optimization](http://www.ensmallen.org/files/ensmallen_2018.pdf).
  Workshop on Systems for ML and Open Source Software at NIPS 2018.


### Developers and Contributors

* Ryan Curtin
* Dongryeol Lee
* Marcus Edel
* Sumedh Ghaisas
* Siddharth Agrawal
* Stephen Tu
* Shikhar Bhardwaj
* Vivek Pal
* Sourabh Varshney
* Chenzhe Diao
* Abhinav Moudgil
* Konstantin Sidorov
* Kirill Mishchenko
* Kartik Nighania
* Haritha Nair
* Moksh Jain
* Abhishek Laddha
* Arun Reddy
* Nishant Mehta
* Trironk Kiatkungwanglai
* Vasanth Kalingeri
* Zhihao Lou
* Conrad Sanderson
* Dan Timson
* N Rajiv Vaidyanathan
* Roberto Hueso

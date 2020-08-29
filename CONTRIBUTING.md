# Contributing to ensmallen

ensmallen is an open-source project, so contributions are welcome and
encouraged!  Anyone can join the ensmallen developer community simply by opening
a pull request with an improvement, bugfix, or feature addition, and an
ensmallen maintainer will review it and help through the merge process.  So if
you have an improvement you would like to see, we would love to include it!

The ensmallen maintainer community overlaps heavily with the
[mlpack](https://github.com/mlpack/mlpack) community, so development discussions
can happen either here on Github, on the [mlpack mailing
list](http://lists.mlpack.org/mailman/listinfo/mlpack), or in the #mlpack
IRC channel on irc.freenode.net.

Once a pull request is submitted, it must be reviewed and approved before a
merge, to ensure that:

 * the design matches the rest of the ensmallen design
 * the style matches the [mlpack style guide](
    https://github.com/mlpack/mlpack/wiki/DesignGuidelines#StyleGuidelines)
 * any new functionality is tested and working

Please do make sure that if you contribute a new optimizer or other new
functionality, that you've added some tests in the `tests/` directory.  And if
you are fixing a bug, it's always nice to include a test case if possible to so
that the bug won't happen again.

## Build/test process

All of the code for ensmallen is located in `include/ensmallen_bits/` and all of
the tests are in `tests/`.  ensmallen is header-only, so anything in
`include/ensmallen_bits/` must be either template methods or marked `inline`.

Adding a new test can be done either by adding a new `TEST_CASE()` block to an
existing file in `tests/` or by creating a new file and adding it to the list of
test sources in `tests/CMakeLists.txt`.  The tests are written with the
[Catch2](https://github.com/catchorg/Catch2) unit test framework.

Sometimes, tests have random behavior and may not pass every time.  (For
instance, consider a test where the initial point is randomly generated.)  If
you have written a test like this, make sure it does not fail often by
uncommenting the code that sets a random seed in `tests/main.cpp` and running
your test many times.

### Install Dependencies

<details open>
<summary>Linux</summary>

Use your distributions' package manager to install the required dependencies. For Ubuntu/Debian, the commands are as shown below.

```bash
$ sudo apt-get update
$ sudo apt-get install libarmadillo-dev cmake
```

To install the dependencies on Fedora/RHEL/CentOS, you can use:

```bash
$ yum install armadillo-devel cmake
```

You can install **ensmallen** on Arch from AUR using:

```bash
$ git clone https://aur.archlinux.org/ensmallen.git
$ cd ensmallen
$ makepkg -si
```

</details>

<details open>
<summary>macOS</summary>

You can install **ensmallen** directly with [Homebrew](https://brew.sh).

```bash
$ brew install ensmallen
```

If you want to build **ensmallen** tests from source, use a package manager like [Homebrew](https://brew.sh) to get the necessary dependencies.

```bash
$ brew install armadillo cmake
```
</details>

<details open>
<summary>Windows</summary>

You can install **ensmallen** directly by using [vcpkg](https://github.com/microsoft/vcpkg).

```
vcpkg install ensmallen:x64-windows
```
</details>

### Build and Test

This section describes how to build the **ensmallen** tests from source. **ensmallen** uses CMake as its build system and [Catch2](https://github.com/catchorg/Catch2) as the unit test framework.

First, clone the source code from Github and change into the cloned directory. Or alternatively, you can download the latest relese from the [website](http://ensmallen.org) and extract it.

```bash
$ git clone https://github.com/mlpack/ensmallen
$ cd ensmallen

# - or -

$ wget http://ensmallen.org/files/ensmallen-2.14.1.tar.gz
$ tar -xvzpf ensmallen-2.14.1.tar.gz
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
$ make
# or alternately make with the -jN flag to run parallel jobs
# $ make -j4
```

Now, you can either run all of the tests or an individual test case with:

```bash
$ ./ensmallen_tests
$ ./ensmallen_tests <test name>
```

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


## Release process

New releases of ensmallen can be performed almost-automatically with the
`rel/ensmallen-release.sh` script.  Releases can only be performed by
contributors with push permissions to the repository.  Before making a release,
make sure that all the tests are passing and the release number satisfies the
versioning guidelines in `UPDATING.md` and make sure that `HISTORY.md` is
up-to-date with the new release's release notes (and date).

The script should be run, e.g.,

```
$ rel/ensmallen-release.sh 2 10 0 "Fried Chicken"
```

If the version is a new minor version (or major version), choose a name for the
release.  Previous release names have generally been entirely arbitrary.

Then, after running the script, a release needs to be done on the Github
website:

https://github.com/mlpack/ensmallen/releases/new

The format for the release notes is just the release date at the top (e.g.,
`Released Sept. 7th, 2019`), followed by the Markdown-formatted HISTORY.md
updates for that release.

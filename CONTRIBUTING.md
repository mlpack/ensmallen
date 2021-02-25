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

 * The design matches the rest of the ensmallen design
 * the style matches the [mlpack style guide](
    https://github.com/mlpack/mlpack/wiki/DesignGuidelines#StyleGuidelines)
 * any new functionality is tested and working

Please do make sure that if you contribute a new optimizer or other new
functionality, that you've added some tests in the `tests/` directory.  And if
you are fixing a bug, it's always nice to include a test case if possible so
that the bug won't happen again.

## Build/test process

All of the code for ensmallen is located in `include/ensmallen_bits/` and all of
the tests are in `tests/`.  ensmallen is header-only, so anything in
`include/ensmallen_bits/` must be either template methods or marked `inline`.
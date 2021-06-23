<h2 align="center">
  <a href="http://ensmallen.org/"><img src="http://ensmallen.org/img/ensmallen_text.svg" style="background-color:rgba(0,0,0,0);" height=230 alt="ensmallen: a C++ header-only library for numerical optimization"></a>
</h2>

**ensmallen** is a C++ header-only library for numerical optimization.

Documentation and downloads: http://ensmallen.org

ensmallen provides a simple set of abstractions for writing an objective
function to optimize. It also provides a large set of standard and cutting-edge
optimizers that can be used for virtually any numerical optimization task.
These include full-batch gradient descent techniques, small-batch techniques,
gradient-free optimizers, and constrained optimization.


### Requirements

* C++ compiler with C++11 support
* Armadillo: http://arma.sourceforge.net
* OpenBLAS or Intel MKL or LAPACK (see Armadillo site for details)


### Installation

ensmallen can be installed in several ways.

A straightforward approach is to simply copy the `include/ensmallen.hpp` header ***and*** the associated `include/ensmallen_bits` directory to a location such as `/usr/include/` which is searched by your C++ compiler.
If you can't use `sudo` or don't have write access to `/usr/include/`, use a directory within your own home directory (eg. `/home/blah/include/`).

Installation can also be done with CMake 3.3+, which will also build the tests.
If CMake is not already available on your system, it can be obtained from [cmake.org](https://cmake.org).
If you are using an older system such as RHEL 7 or CentOS 7,
an updated version of CMake is also available via the EPEL repository via the `cmake3` package.

Example cmake based installation with root access:

```
mkdir build
cd build
cmake ..
sudo make install
```

Example cmake based installation without root access:

```
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX:PATH=/home/blah/include/
make install
```

Change `/home/blah/include/` as required.  


### Example Compilation

If you have installed ensmallen in a standard location such as `/usr/include/`:

    g++ prog.cpp -o prog -O2 -larmadillo
    
If you have installed ensmallen in a non-standard location, such as `/home/blah/include/`, you will need to make sure that your C++ compiler searches `/home/blah/include/` by explicitly specifying the directory as an argument/option. For example, using the `-I` switch in gcc and clang:

    g++ prog.cpp -o prog -O2 -I /home/blah/include/ -larmadillo


### Example Optimization

See [`example.cpp`](example.cpp) for example usage of the L-BFGS optimizer in a linear regression setting.


### License

Unless stated otherwise, the source code for **ensmallen** is licensed under the
3-clause BSD license (the "License").  A copy of the License is included in the
"LICENSE.txt" file.  You may also obtain a copy of the License at
http://opensource.org/licenses/BSD-3-Clause


### Citation

Please cite the following paper if you use ensmallen in your research and/or
software. Citations are useful for the continued development and maintenance of
the library.

* S. Bhardwaj, R. Curtin, M. Edel, Y. Mentekidis, C. Sanderson.  
  [ensmallen: a flexible C++ library for efficient function optimization](http://www.ensmallen.org/files/ensmallen_2018.pdf).  
  Workshop on Systems for ML and Open Source Software at NIPS 2018.

```
@article{DBLP:journals/corr/abs-1810-09361,
  author    = {Shikhar Bhardwaj and
               Ryan R. Curtin and
               Marcus Edel and
               Yannis Mentekidis and
               Conrad Sanderson},
  title     = {ensmallen: a flexible {C++} library for efficient function optimization},
  journal   = {CoRR},
  volume    = {abs/1810.09361},
  doi       = {10.5281/zenodo.2008650},
  year      = {2018},
  url       = {http://arxiv.org/abs/1810.09361},
  archivePrefix = {arXiv},
  eprint    = {1810.09361},
  timestamp = {Wed, 31 Oct 2018 14:24:29 +0100},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1810-09361},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

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
* Sayan Goswami

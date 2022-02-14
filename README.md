<h2 align="center">
  <a href="http://ensmallen.org/"><img src="http://ensmallen.org/img/ensmallen_text.svg" style="background-color:rgba(0,0,0,0);" height=230 alt="ensmallen: a C++ header-only library for numerical optimization"></a>
</h2>

**ensmallen** is a high-quality C++ library for non-linear numerical optimization.

ensmallen provides many types of optimizers that can be used
for virtually any numerical optimization task.
This includes gradient descent techniques, gradient-free optimizers,
and constrained optimization.
ensmallen also allows optional callbacks to customize the optimization process.

Documentation and downloads: http://ensmallen.org

### Requirements

* C++ compiler with C++11 support
* Armadillo: http://arma.sourceforge.net
* OpenBLAS or Intel MKL or LAPACK (see Armadillo site for details)


### Installation

ensmallen can be installed in several ways: either manually or via cmake, 
with or without root access.

The cmake based installation will check the requirements 
and optionally build the tests. If cmake 3.3 (or a later version) 
is not already available on your system, it can be obtained 
from [cmake.org](https://cmake.org). If you are using an older 
system such as RHEL 7 or CentOS 7, an updated version of cmake 
is also available via the EPEL repository (see the `cmake3` package).

Example cmake based installation with root access:

```
mkdir build
cd build
cmake ..
sudo make install
```

Example cmake based installation without root access, 
installing into `/home/blah/` (adapt as required):

```
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX:PATH=/home/blah/
make install
```

The above will create a directory named `/home/blah/include/` 
and place all ensmallen headers there.

To optionally build and run the tests
(after running cmake as above),
use the following additional commands:

```
make ensmallen_tests
./ensmallen_tests --durations yes
```

Manual installation involves simply copying the `include/ensmallen.hpp` header 
***and*** the associated `include/ensmallen_bits` directory to a location 
such as `/usr/include/` which is searched by your C++ compiler.
If you can't use `sudo` or don't have write access to `/usr/include/`, 
use a directory within your own home directory (eg. `/home/blah/include/`).


### Example Compilation

If you have installed ensmallen in a standard location such as `/usr/include/`:

    g++ prog.cpp -o prog -O2 -larmadillo
    
If you have installed ensmallen in a non-standard location, 
such as `/home/blah/include/`, you will need to make sure 
that your C++ compiler searches `/home/blah/include/` 
by explicitly specifying the directory as an argument/option. 
For example, using the `-I` switch in gcc and clang:

    g++ prog.cpp -o prog -O2 -I /home/blah/include/ -larmadillo


### Example Optimization

See [`example.cpp`](example.cpp) for example usage of the L-BFGS optimizer 
in a linear regression setting.


### License

Unless stated otherwise, the source code for **ensmallen** is licensed under the
3-clause BSD license (the "License").  A copy of the License is included in the
"LICENSE.txt" file.  You may also obtain a copy of the License at
http://opensource.org/licenses/BSD-3-Clause


### Citation

Please cite the following paper if you use ensmallen in your research and/or
software. Citations are useful for the continued development and maintenance of
the library.

* Ryan R. Curtin, Marcus Edel, Rahul Ganesh Prabhu, Suryoday Basak, Zhihao Lou, Conrad Sanderson.  
  [The ensmallen library for flexible numerical optimization](https://jmlr.org/papers/volume22/20-416/20-416.pdf).  
  Journal of Machine Learning Research, Vol. 22, No. 166, 2021.

```
@article{ensmallen_JMLR_2021,
  author  = {Ryan R. Curtin and Marcus Edel and Rahul Ganesh Prabhu and Suryoday Basak and Zhihao Lou and Conrad Sanderson},
  title   = {The ensmallen library for flexible numerical optimization},
  journal = {Journal of Machine Learning Research},
  year    = {2021},
  volume  = {22},
  number  = {166},
  pages   = {1--6},
  url     = {http://jmlr.org/papers/v22/20-416.html}
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

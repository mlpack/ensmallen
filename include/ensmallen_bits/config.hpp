// Copyright (c) 2018 ensmallen developers.
// 
// Licensed under the 3-clause BSD license (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.opensource.org/licenses/BSD-3-Clause


#if !defined(ENS_PRINT_INFO)
  // #define ENS_PRINT_INFO
#endif

#if !defined(ENS_PRINT_WARN)
  // #define ENS_PRINT_WARN
#endif

#if defined(ARMA_USE_OPENMP)
  #undef  ENS_USE_OPENMP
  #define ENS_USE_OPENMP
#endif


//


#if defined(ENS_DONT_PRINT_INFO)
  #undef ENS_PRINT_INFO
#endif

#if defined(ENS_DONT_PRINT_WARN)
  #undef ENS_PRINT_WARN
#endif

#if defined(ENS_DONT_USE_OPENMP)
  #undef ENS_USE_OPENMP
#endif


//


#if defined(ENS_USE_OPENMP)
  #define ENS_PRAGMA_OMP_PARALLEL _Pragma("omp parallel")
  #define ENS_PRAGMA_OMP_ATOMIC   _Pragma("omp atomic")
#else
  #define ENS_PRAGMA_OMP_PARALLEL
  #define ENS_PRAGMA_OMP_ATOMIC
#endif


//

// for legacy code
#define ENS_M_PI       (double(3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679))
#define ENS_M_SQRT2    (double(1.4142135623730950488016887242096980785696718753769480731766797379907324784621070388503875343276415727))
#define ENS_M_SQRT1_2  (double(0.7071067811865475244008443621048490392848359376884740365883398689953662392310535194251937671638207864))

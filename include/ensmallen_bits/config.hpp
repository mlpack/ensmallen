// Copyright (c) 2018 ensmallen developers.
// 
// Licensed under the 3-clause BSD license (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.opensource.org/licenses/BSD-3-Clause


#if !defined(ENS_PRINT_INFO)
  #define ENS_PRINT_INFO
#endif

#if !defined(ENS_PRINT_WARN)
  #define ENS_PRINT_WARN
#endif


#if defined(ENS_DONT_PRINT_INFO)
  #undef ENS_PRINT_INFO
#endif

#if defined(ENS_DONT_PRINT_WARN)
  #undef ENS_PRINT_WARN
#endif

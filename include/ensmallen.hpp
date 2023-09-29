/**
 * @file ensmallen.hpp
 *
 * This is the main header to include if you want to use the ensmallen library.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

// NOTE: When using the ensmallen library in your code, only include the ensmallen.hpp header.
// NOTE: Do not include any of the files in the ensmallen_bits folder.

#ifndef ENSMALLEN_HPP
#define ENSMALLEN_HPP

// certain compilers are way behind the curve
#if (defined(_MSVC_LANG) && (_MSVC_LANG >= 201402L))
  #undef  ARMA_USE_CXX11
  #define ARMA_USE_CXX11
#endif

#include <armadillo>

#if !defined(ARMA_USE_CXX11)
  // armadillo automatically enables ARMA_USE_CXX11
  // when a C++11/C++14/C++17/etc compiler is detected
  #error "please enable C++11/C++14 mode in your compiler"
#endif

#if ((ARMA_VERSION_MAJOR < 9) || ((ARMA_VERSION_MAJOR == 9) && (ARMA_VERSION_MINOR < 800)))
  #error "need Armadillo version 9.800 or later"
#endif

#include <cctype>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <set>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// On Visual Studio, disable C4519 (default arguments for function templates)
// since it's by default an error, which doesn't even make any sense because
// it's part of the C++11 standard.
#ifdef _MSC_VER
  #pragma warning(disable : 4519)
#endif

#include "ensmallen_bits/config.hpp"
#include "ensmallen_bits/ens_version.hpp"
#include "ensmallen_bits/log.hpp" // TODO: should move to another place

#include "ensmallen_bits/utility/any.hpp"
#include "ensmallen_bits/utility/arma_traits.hpp"
#include "ensmallen_bits/utility/indicators/epsilon.hpp"
#include "ensmallen_bits/utility/indicators/igd_plus.hpp"

// Contains traits, must be placed before report callback.
#include "ensmallen_bits/function.hpp" // TODO: should move to function/

// Callbacks.
#include "ensmallen_bits/callbacks/callbacks.hpp"
#include "ensmallen_bits/callbacks/early_stop_at_min_loss.hpp"
#include "ensmallen_bits/callbacks/grad_clip_by_norm.hpp"
#include "ensmallen_bits/callbacks/grad_clip_by_value.hpp"
#include "ensmallen_bits/callbacks/print_loss.hpp"
#include "ensmallen_bits/callbacks/progress_bar.hpp"
#include "ensmallen_bits/callbacks/query_front.hpp"
#include "ensmallen_bits/callbacks/report.hpp"
#include "ensmallen_bits/callbacks/store_best_coordinates.hpp"
#include "ensmallen_bits/callbacks/timer_stop.hpp"

#include "ensmallen_bits/problems/problems.hpp" // TODO: should move to another place

#include "ensmallen_bits/ada_belief/ada_belief.hpp"
#include "ensmallen_bits/ada_bound/ada_bound.hpp"
#include "ensmallen_bits/ada_delta/ada_delta.hpp"
#include "ensmallen_bits/ada_grad/ada_grad.hpp"
#include "ensmallen_bits/ada_sqrt/ada_sqrt.hpp"
#include "ensmallen_bits/adam/adam.hpp"
#include "ensmallen_bits/demon_adam/demon_adam.hpp"
#include "ensmallen_bits/demon_sgd/demon_sgd.hpp"
#include "ensmallen_bits/qhadam/qhadam.hpp"
#include "ensmallen_bits/aug_lagrangian/aug_lagrangian.hpp"
#include "ensmallen_bits/bigbatch_sgd/bigbatch_sgd.hpp"
#include "ensmallen_bits/cmaes/cmaes.hpp"
#include "ensmallen_bits/cmaes/active_cmaes.hpp"
#include "ensmallen_bits/cd/cd.hpp"
#include "ensmallen_bits/cne/cne.hpp"
#include "ensmallen_bits/de/de.hpp"
#include "ensmallen_bits/eve/eve.hpp"
#include "ensmallen_bits/ftml/ftml.hpp"

#include "ensmallen_bits/fw/frank_wolfe.hpp"
#include "ensmallen_bits/gradient_descent/gradient_descent.hpp"
#include "ensmallen_bits/grid_search/grid_search.hpp"
#include "ensmallen_bits/iqn/iqn.hpp"
#include "ensmallen_bits/katyusha/katyusha.hpp"
#include "ensmallen_bits/lbfgs/lbfgs.hpp"
#include "ensmallen_bits/lookahead/lookahead.hpp"
#include "ensmallen_bits/moead/moead.hpp"
#include "ensmallen_bits/nsga2/nsga2.hpp"
#include "ensmallen_bits/padam/padam.hpp"
#include "ensmallen_bits/parallel_sgd/parallel_sgd.hpp"
#include "ensmallen_bits/pso/pso.hpp"
#include "ensmallen_bits/rmsprop/rmsprop.hpp"

#include "ensmallen_bits/sa/sa.hpp"
#include "ensmallen_bits/sarah/sarah.hpp"
#include "ensmallen_bits/sdp/sdp.hpp"
#include "ensmallen_bits/sdp/lrsdp.hpp"
#include "ensmallen_bits/sdp/primal_dual.hpp"

#include "ensmallen_bits/sgd/sgd.hpp"
// TODO: this should probably be included in sgd.hpp
#include "ensmallen_bits/sgd/update_policies/gradient_clipping.hpp"
#include "ensmallen_bits/sgdr/sgdr.hpp"
#include "ensmallen_bits/sgdr/snapshot_ensembles.hpp"
#include "ensmallen_bits/sgdr/snapshot_sgdr.hpp"
#include "ensmallen_bits/smorms3/smorms3.hpp"
#include "ensmallen_bits/spalera_sgd/spalera_sgd.hpp"
#include "ensmallen_bits/spsa/spsa.hpp"
#include "ensmallen_bits/svrg/svrg.hpp"
#include "ensmallen_bits/swats/swats.hpp"
#include "ensmallen_bits/wn_grad/wn_grad.hpp"
#include "ensmallen_bits/yogi/yogi.hpp"

#endif

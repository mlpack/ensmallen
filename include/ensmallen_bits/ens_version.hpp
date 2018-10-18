// Copyright (c) 2018 ensmallen developers.
// 
// Licensed under the 3-clause BSD license (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.opensource.org/licenses/BSD-3-Clause


// This follows the Semantic Versioning pattern defined in https://semver.org/.

#define ENS_VERSION_MAJOR 0
// The minor version is two digits so regular numerical comparisons of versions
// work right.
#define ENS_VERSION_MINOR 00
// Release candidates may have patch versions "RC<X>", i.e., RC0, RC1, and so
// forth.
#define ENS_VERSION_PATCH 0
#define ENS_VERSION_NAME  "development"

namespace ens {

struct version
{
  static const unsigned int major = ENS_VERSION_MAJOR;
  static const unsigned int minor = ENS_VERSION_MINOR;
  static const unsigned int patch = ENS_VERSION_PATCH;

  static inline std::string as_string()
  {
    const char* nickname = ENS_VERSION_NAME;

    std::stringstream ss;
    ss << version::major << '.' << version::minor << '.' << version::patch
       << " (" << nickname << ')';

    return ss.str();
  }
};

} // namespace ens

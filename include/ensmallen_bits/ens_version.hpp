// Copyright (c) 2018 ensmallen developers.
// 
// Licensed under the 3-clause BSD license (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.opensource.org/licenses/BSD-3-Clause


// This follows the Semantic Versioning pattern defined in https://semver.org/.

#define ENS_VERSION_MAJOR 0
// The minor version is two digits so regular numerical comparisons of versions
// work right.  The first minor version of a release is always 10.
#define ENS_VERSION_MINOR 10
#define ENS_VERSION_PATCH 0
// If this is a release candidate, it will be reflected in the version name
// (i.e. the version name will be "RC1", "RC2", etc.).  Otherwise the version
// name will typically be a seemingly arbitrary set of words that does not
// contain the capitalized string "RC".
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

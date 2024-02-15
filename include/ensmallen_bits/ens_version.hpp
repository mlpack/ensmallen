/**
 * @file ens_version.hpp
 * @author Conrad Sanderson
 * @author Ryan Curtin
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */


// This follows the Semantic Versioning pattern defined in https://semver.org/.

#define ENS_VERSION_MAJOR 2
// The minor version is two digits so regular numerical comparisons of versions
// work right.  The first minor version of a release is always 10.
#define ENS_VERSION_MINOR 21
#define ENS_VERSION_PATCH 1
// If this is a release candidate, it will be reflected in the version name
// (i.e. the version name will be "RC1", "RC2", etc.).  Otherwise the version
// name will typically be a seemingly arbitrary set of words that does not
// contain the capitalized string "RC".
#define ENS_VERSION_NAME "Bent Antenna"
// Incorporate the date the version was released.
#define ENS_VERSION_YEAR "2024"
#define ENS_VERSION_MONTH "02"
#define ENS_VERSION_DAY "15"

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
    
  static inline std::string date()
  {
    std::stringstream ss;
    ss << ENS_VERSION_YEAR << '-' << ENS_VERSION_MONTH << '-' << ENS_VERSION_DAY;

    return ss.str();
  }
};

} // namespace ens

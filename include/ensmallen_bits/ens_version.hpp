// Copyright (c) 2018 ensmallen developers.
// 
// Licensed under the 3-clause BSD license (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.opensource.org/licenses/BSD-3-Clause


// following the Semantic Versioning pattern defined in https://semver.org/

#define ENS_VERSION_MAJOR 0
#define ENS_VERSION_MINOR 100
#define ENS_VERSION_PATCH 0
#define ENS_VERSION_NAME  "development"
//#define ENS_VERSION_NAME "2018-10-31"
//#define ENS_VERSION_NAME "stable"
//#define ENS_VERSION_NAME "1.199-RC3"

// for normal releases, ENS_VERSION_NAME can be a date, nickname, etc.
// a release candidate (RC) can follow the versioning pattern of normal releases,
// but use the patch level to indicate the RC level,
// and the RC status explicitly denoted via the ENS_VERSION_NAME;
// for example, version 1.199.3 can be denoted as "1.199-RC3".

// CS note: i'm not a fan of misuse of the decimal point in version names.
// CS note: for example, using 1.10 to indicate a newer version than 1.9
// CS note: breaks the usual mathematical meaning of 1.9 > 1.10
// CS note: this is why armadillo uses "large" minor version numbers
// CS note: fixed to 3 digits and no leading zeros (eg. 100).
// CS note: this approach reserves a lot of room for denoting minor
// CS note: version upgrades (eg. 110, 120, 199, 200, etc)
// CS note: without resulting in mathematical ugliness.
// CS note: i suggest using a minor version number that starts at 100
// CS note: and is not allowed go past 999.

struct ens_version
  {
  static const unsigned int major = ENS_VERSION_MAJOR;
  static const unsigned int minor = ENS_VERSION_MINOR;
  static const unsigned int patch = ENS_VERSION_PATCH;
  
  static
  inline
  std::string
  as_string()
    {
    const char* nickname = ENS_VERSION_NAME;
    
    std::stringstream ss;
    ss << ens_version::major
       << '.'
       << ens_version::minor
       << '.'
       << ens_version::patch
       << " ("
       << nickname
       << ')';
    
    return ss.str();
    }
  };

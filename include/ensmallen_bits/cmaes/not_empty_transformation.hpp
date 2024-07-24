/**
 * @file not_empty_transformation.hpp
 * @author Suvarsha Chennareddy
 *
 * Check whether TransformationPolicyType is EmptyTransformation.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_CMAES_NOT_EMPTY_TRANSFORMATION
#define ENSMALLEN_CMAES_NOT_EMPTY_TRANSFORMATION

/**
 * This partial specialization is used to throw an exception when the
 * TransformationPolicyType is EmptyTransformation and call a constructor with
 * parameters 'lowerBound' and 'upperBound' otherwise.  This shall be removed
 * when the deprecated constructor is removed in the next major version of
 * ensmallen.
 */
template<typename T1, typename T2>
struct NotEmptyTransformation : std::true_type
{
  void Assign(T1& obj, double lowerBound, double upperBound)
  {
    obj = T1(lowerBound, upperBound);
  }
};

template<template<typename...> class T, typename... A, typename... B>
struct NotEmptyTransformation<T<A...>, T<B...>> : std::false_type
{
  void Assign(T<A...>& /* obj */,
              double /* lowerBound */,
              double /* upperBound */)
  {
    throw std::logic_error("TransformationPolicyType is EmptyTransformation");
  }
};

#endif

/**
 * @file any.hpp
 * @author Ryan Curtin
 *
 * A simple and dangerous implementation of 'Any' that can be used when the
 * class needs to hold some specific information for which the type is not
 * known.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_UTILITY_ANY_HPP
#define ENSMALLEN_UTILITY_ANY_HPP

#include <typeinfo>
#include <typeindex>

namespace ens {

/**
 * A utility class that can hold any C++ class as a void*.  It does some very
 * basic type checking to ensure that you cast it correctly.  If you call
 * Clean(), it will properly call the destructor on the given type.
 *
 * The Any is holding nothing if Has<void>() returns true.
 */
class Any
{
 public:
  /**
   * Create an Any object that holds nothing.
   */
  Any() :
      held(NULL),
      type(typeid(void)),
      destructor([](const void*) {}) // Fake destructor.
  {
    // Nothing to do.
  }

  /**
   * Get the Any, cast as the thing we want.
   */
  template<typename T>
  const T& As() const
  {
    if (std::type_index(typeid(T)) != type)
    {
      std::string error = "Invalid cast to type '";
      error += typeid(T).name();
      error += "' when Any is holding '";
      error += type.name();
      error += "'!";
      throw std::invalid_argument(error);
    }

    return *reinterpret_cast<const T*>(held);
  }

  /**
   * Get the Any, cast as the thing we want.
   */
  template<typename T>
  T& As()
  {
    if (std::type_index(typeid(T)) != type)
    {
      std::string error = "Invalid cast to type '";
      error += typeid(T).name();
      error += "' when Any is holding '";
      error += type.name();
      error += "'!";
      throw std::invalid_argument(error);
    }

    return *reinterpret_cast<T*>(held);
  }

  /**
   * Set the Any as the given type.
   */
  template<typename T>
  void Set(T* t)
  {
    type = std::type_index(typeid(T));
    held = (void*) t;
    destructor = [](const void* x) { delete static_cast<const T*>(x); };
  }

  /**
   * Determine if the Any is currently holding the given type.
   */
  template<typename T>
  bool Has()
  {
    return (std::type_index(typeid(T)) == type);
  }

  /**
   * Call delete on the thing we are holding.  Be careful with this one.  It
   * automatically does nothing if 'held' is NULL, but that's the only guarantee
   * you get.  Also, I hope you used 'new' to make the thing you're holding.
   */
  void Clean()
  {
    if (held)
    {
      destructor(held);
      held = NULL;
      type = std::type_index(typeid(void));
      destructor = [](const void*) { }; // Fake destructor.
    }
  }

 private:
  // The thing we are holding.
  void* held;
  // The type of the thing we are holding.
  std::type_index type;
  // A pointer to the destructor.
  void (*destructor)(const void*);
};

} // namespace ens

#endif

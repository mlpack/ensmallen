/**
 * @file sfinae_utility.hpp
 * @author Trironk Kiatkungwanglai, Kirill Mishchenko
 *
 * This file contains macro utilities for the SFINAE Paradigm. These utilities
 * determine if classes passed in as template parameters contain members at
 * compile time, which is useful for changing functionality depending on what
 * operations an object is capable of performing.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_SFINAE_UTILITY
#define MLPACK_CORE_SFINAE_UTILITY

#include <type_traits>
#include <cstring>

namespace ens {
namespace sfinae {

// TODO: I think a lot of this can be stripped for ensmallen.

/*
 * MethodFormDetector is a tool that helps to find out whether a given class has
 * a method of the requested form. For that purpose MethodFormDetector defines
 * an operator() that accepts a class member pointer for the given class. If the
 * operator()(&Class::Method) call can be compiled, then the given class has a
 * method of the requested form. For any provided AdditionalArgsCount, the check
 * succeeds only if the given class has exactly one method of the requested form
 * with AdditionalArgsCount additional arguments.
 *
 * The tool is dedicated to be used in type functions (structs) generated by the
 * macro HAS_METHOD_FORM.
 *
 * @tparam MethodForm A template class member pointer type to a method of the
 *   form to look for.
 * @tparam Class A class in which a method of the requested form should be
 *   looked for.
 * @tparam AdditionalArgsCount A number of additional arguments.
 */
template<typename Class,
         template<typename...> class MethodForm,
         size_t AdditionalArgsCount>
struct MethodFormDetector;

template<typename Class, template<typename...> class MethodForm>
struct MethodFormDetector<Class, MethodForm, 0>
{
  void operator()(MethodForm<Class>);
};

template<typename Class, template<typename...> class MethodForm>
struct MethodFormDetector<Class, MethodForm, 1>
{
  template<class T1>
  void operator()(MethodForm<Class, T1>);
};

template<typename Class, template<typename...> class MethodForm>
struct MethodFormDetector<Class, MethodForm, 2>
{
  template<class T1, class T2>
  void operator()(MethodForm<Class, T1, T2>);
};

template<typename Class, template<typename...> class MethodForm>
struct MethodFormDetector<Class, MethodForm, 3>
{
  template<class T1, class T2, class T3>
  void operator()(MethodForm<Class, T1, T2, T3>);
};

template<typename Class, template<typename...> class MethodForm>
struct MethodFormDetector<Class, MethodForm, 4>
{
  template<class T1, class T2, class T3, class T4>
  void operator()(MethodForm<Class, T1, T2, T3, T4>);
};

template<typename Class, template<typename...> class MethodForm>
struct MethodFormDetector<Class, MethodForm, 5>
{
  template<class T1, class T2, class T3, class T4, class T5>
  void operator()(MethodForm<Class, T1, T2, T3, T4, T5>);
};

template<typename Class, template<typename...> class MethodForm>
struct MethodFormDetector<Class, MethodForm, 6>
{
  template<class T1, class T2, class T3, class T4, class T5, class T6>
  void operator()(MethodForm<Class, T1, T2, T3, T4, T5, T6>);
};

template<typename Class, template<typename...> class MethodForm>
struct MethodFormDetector<Class, MethodForm, 7>
{
  template<class T1, class T2, class T3, class T4, class T5, class T6, class T7>
  void operator()(MethodForm<Class, T1, T2, T3, T4, T5, T6, T7>);
};

} // namespace sfinae
} // namespace ens

//! Utility struct for checking signatures.
template<typename U, U> struct SigCheck : std::true_type {};

/*
 * Constructs a template supporting the SFINAE pattern.
 *
 * This macro generates a template struct that is useful for enabling/disabling
 * a method if the template class passed in contains a member function matching
 * a given signature with a specified name.
 *
 * The generated struct should be used in conjunction with std::enable_if_t.
 *
 * For general references, see:
 * http://stackoverflow.com/a/264088/391618
 *
 * For an mlpack specific use case, see /mlpack/core/util/prefixedoutstream.hpp
 * and /mlpack/core/util/prefixedoutstream_impl.hpp
 *
 * @param NAME the name of the struct to construct. For example: HasToString
 * @param FUNC the name of the function to check for. For example: ToString
 */
#define HAS_MEM_FUNC(FUNC, NAME)                                               \
template<typename T, typename sig, typename = std::true_type>                  \
struct NAME : std::false_type {};                                              \
                                                                               \
template<typename T, typename sig>                                             \
struct NAME                                                                    \
<                                                                              \
  T,                                                                           \
  sig,                                                                         \
  std::integral_constant<bool, SigCheck<sig, &T::FUNC>::value>                 \
> : std::true_type {};

/**
 * Base macro for HAS_METHOD_FORM() and HAS_EXACT_METHOD_FORM() macros.
 */
#define HAS_METHOD_FORM_BASE(METHOD, NAME, MAXN)                               \
template<typename Class,                                                       \
         template<typename...> class MF /* MethodForm */,                      \
         size_t MinN = 0 /* MinNumberOfAdditionalArgs */>                      \
struct NAME                                                                    \
{                                                                              \
  /* Making a short alias for MethodFormDetector */                            \
  template<typename C, template<typename...> class MethodForm, int N>          \
  using MFD = ens::sfinae::MethodFormDetector<C, MethodForm, N>;            \
                                                                               \
  template<size_t N>                                                           \
  struct WithNAdditionalArgs                                                   \
  {                                                                            \
    using yes = char[1];                                                       \
    using no = char[2];                                                        \
                                                                               \
    template<typename T, typename ResultType>                                  \
    using EnableIfVoid =                                                       \
        typename std::enable_if<std::is_void<T>::value, ResultType>::type;     \
                                                                               \
    template<typename C>                                                       \
    static EnableIfVoid<decltype(MFD<C, MF, N>()(&C::METHOD)), yes&> chk(int); \
    template<typename>                                                         \
    static no& chk(...);                                                       \
                                                                               \
    static const bool value = sizeof(chk<Class>(0)) == sizeof(yes);            \
  };                                                                           \
                                                                               \
  template<size_t N>                                                           \
  struct WithGreaterOrEqualNumberOfAdditionalArgs                              \
  {                                                                            \
    using type = typename std::conditional<                                    \
        WithNAdditionalArgs<N>::value,                                         \
        std::true_type,                                                        \
        typename std::conditional<                                             \
            N < MAXN,                                                          \
            WithGreaterOrEqualNumberOfAdditionalArgs<N + 1>,                   \
            std::false_type>::type>::type;                                     \
    static const bool value = type::value;                                     \
  };                                                                           \
                                                                               \
  static const bool value =                                                    \
      WithGreaterOrEqualNumberOfAdditionalArgs<MinN>::value;                   \
};

/**
 * Constructs a template structure, which will define a boolean static
 * variable, to true, if the passed template parameter, has a member function
 * with the specified name. The check does not care about the signature or the
 * function parameters.
 *
 * @param FUNC the name of the function, whose existence is to be detected
 * @param NAME the name of the structure that will be generated
 *
 * Use this like: NAME<ClassName>::value to check for the existence of the
 * function in the given class name.
 * This can also be used in conjunction with std::enable_if.
 */
#define HAS_ANY_METHOD_FORM(FUNC, NAME)                                      \
template <typename T>                                                        \
struct NAME                                                                  \
{                                                                            \
  template <typename Q = T>                                                  \
  static typename                                                            \
  std::enable_if<std::is_member_function_pointer<decltype(&Q::FUNC)>::value, \
                 int>::type                                                  \
  f(int) { return 1;}                                                      \
                                                                             \
  template <typename Q = T>                                                  \
  static char f(char) { return 0; }                                        \
                                                                             \
  static const bool value = sizeof(f<T>(0)) != sizeof(char);                 \
};
/*
 * A macro that can be used for passing arguments containing commas to other
 * macros.
 */
#define SINGLE_ARG(...) __VA_ARGS__

/**
 * HAS_METHOD_FORM generates a template that allows to check at compile time
 * whether a given class has a method of the requested form. For example, for
 * the following class
 *
 * class A
 * {
 *  public:
 *   ...
 *   Train(const arma::mat&, const arma::Row<size_t>&, double);
 *   ...
 * };
 *
 * and the following form of Train methods
 *
 * template<typename Class, typename...Ts>
 * using TrainForm =
 *     void(Class::*)(const arma::mat&, const arma::Row<size_t>&, Ts...);
 *
 * we can check whether the class A has a Train method of the specified form:
 *
 * HAS_METHOD_FORM(Train, HasTrain);
 * static_assert(HasTrain<A, TrainFrom>::value, "value should be true");
 *
 * The class generated by this will also return true values if the given class
 * has a method that also has extra parameters.
 *
 * @param METHOD The name of the method to check for.
 * @param NAME The name of the struct to construct.
 * @param MAXN The maximum number of additional arguments.
 */
#define HAS_METHOD_FORM(METHOD, NAME) \
    HAS_METHOD_FORM_BASE(SINGLE_ARG(METHOD), SINGLE_ARG(NAME), 7)

/**
 * HAS_EXACT_METHOD_FORM generates a template that allows to check at compile
 * time whether a given class has a method of the requested form. For example,
 * for the following class
 *
 * class A
 * {
 *  public:
 *   ...
 *   Train(const arma::mat&, const arma::Row<size_t>&);
 *   ...
 * };
 *
 * and the following form of Train methods
 *
 * template<typename Class>
 * using TrainForm =
 *     void(Class::*)(const arma::mat&, const arma::Row<size_t>&);
 *
 * we can check whether the class A has a Train method of the specified form:
 *
 * HAS_METHOD_FORM(Train, HasTrain);
 * static_assert(HasTrain<A, TrainFrom>::value, "value should be true");
 *
 * The class generated by this will only return true values if the signature
 * matches exactly.
 *
 * @param METHOD The name of the method to check for.
 * @param NAME The name of the struct to construct.
 * @param MAXN The maximum number of additional arguments.
 */
#define HAS_EXACT_METHOD_FORM(METHOD, NAME) \
    HAS_METHOD_FORM_BASE(SINGLE_ARG(METHOD), SINGLE_ARG(NAME), 0)

/**
 * A version of HAS_METHOD_FORM() where the maximum number of extra arguments is
 * set to the default of 7.
 *
 * HAS_METHOD_FORM generates a template that allows to check at compile time
 * whether a given class has a method of the requested form. For example, for
 * the following class
 *
 * class A
 * {
 *  public:
 *   ...
 *   Train(const arma::mat&, const arma::Row<size_t>&, double);
 *   ...
 * };
 *
 * and the following form of Train methods
 *
 * template<typename Class, typename...Ts>
 * using TrainForm =
 *     void(Class::*)(const arma::mat&, const arma::Row<size_t>&, Ts...);
 *
 * we can check whether the class A has a Train method of the specified form:
 *
 * HAS_METHOD_FORM(Train, HasTrain);
 * static_assert(HasTrain<A, TrainFrom>::value, "value should be true");
 *
 * The implementation is analogous to implementation of the macro HAS_MEM_FUNC.
 *
 * @param METHOD The name of the method to check for.
 * @param NAME The name of the struct to construct.
 */

#endif

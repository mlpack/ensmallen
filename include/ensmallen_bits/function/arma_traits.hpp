/**
 * @file arma_traits.hpp
 * @author Ryan Curtin
 *
 * Given an Armadillo type, determine its "true" base type.
 */
#ifndef ENSMALLEN_FUNCTION_ARMA_TRAITS_HPP
#define ENSMALLEN_FUNCTION_ARMA_TRAITS_HPP

namespace ens {

/**
 * Extract the base type of a matrix (i.e. if it is a column, return the matrix
 * type).  If the type is unknown (or not a derived type) we just return the
 * type itself as the typedef BaseMatType.
 */

template<typename MatType>
struct MatTypeTraits
{
  typedef MatType BaseMatType;
};

template<typename eT>
struct MatTypeTraits<arma::Col<eT>>
{
  typedef arma::Mat<eT> BaseMatType;
};

template<typename eT>
struct MatTypeTraits<arma::Row<eT>>
{
  typedef arma::Mat<eT> BaseMatType;
};

template<typename eT>
struct MatTypeTraits<arma::SpCol<eT>>
{
  typedef arma::SpMat<eT> BaseMatType;
};

template<typename eT>
struct MatTypeTraits<arma::SpRow<eT>>
{
  typedef arma::SpMat<eT> BaseMatType;
};

/**
 * Disable usage of arma::subviews and related types for optimizers.  It might
 * be nice to also explicitly disable Armadillo expressions, but we'll hope for
 * now nobody even tries that, since those aren't even lvalues and thus can't
 * really work.
 */

template<typename eT>
struct MatTypeTraits<arma::subview<eT>>
{
  static_assert(sizeof(arma::subview<eT>) == 0,
      "Armadillo subviews cannot be passed to Optimize()!  Create a matrix "
      "or a matrix alias instead!");
};

template<typename eT>
struct MatTypeTraits<arma::subview_col<eT>>
{
  static_assert(sizeof(arma::subview_col<eT>) == 0,
      "Armadillo subviews cannot be passed to Optimize()!  Create a matrix "
      "or a matrix alias instead!");
};

template<typename eT>
struct MatTypeTraits<arma::SpSubview<eT>>
{
  static_assert(sizeof(arma::SpSubview<eT>) == 0,
      "Armadillo subviews cannot be passed to Optimize()!  Create a matrix "
      "or a matrix alias instead!");
};


#if ((ARMA_VERSION_MAJOR >= 10) || \
    ((ARMA_VERSION_MAJOR == 9) && (ARMA_VERSION_MINOR >= 869)))

// Armadillo 9.869+ has SpSubview_col and SpSubview_row

template<typename eT>
struct MatTypeTraits<arma::SpSubview_col<eT>>
{
  static_assert(sizeof(arma::SpSubview_col<eT>) == 0,
      "Armadillo subviews cannot be passed to Optimize()!  Create a matrix "
      "or a matrix alias instead!");
};

template<typename eT>
struct MatTypeTraits<arma::SpSubview_row<eT>>
{
  static_assert(sizeof(arma::SpSubview_row<eT>) == 0,
      "Armadillo subviews cannot be passed to Optimize()!  Create a matrix "
      "or a matrix alias instead!");
};

#endif


template<typename eT>
struct MatTypeTraits<arma::Cube<eT>>
{
  static_assert(sizeof(arma::Cube<eT>) == 0,
      "Armadillo cubes cannot be passed to Optimize()!  Create a matrix "
      "or a matrix alias instead!");
};

/**
 * Issue a fatal error if the type is not an Armadillo double or floating point
 * sparse or dense matrix.
 */

template<typename MatType>
void RequireDenseFloatingPointType()
{
#ifndef ENS_DISABLE_TYPE_CHECKS
  static_assert(sizeof(MatType) == 0,
      "The given MatType must be arma::mat or arma::fmat or it is not known "
      "to work!  If you would like to try anyway, set the preprocessor macro "
      "ENS_DISABLE_TYPE_CHECKS before including ensmallen.hpp.  However, you "
      "get to pick up all the pieces if there is a failure!");
#endif
}

template<>
inline void RequireDenseFloatingPointType<arma::mat>() { }
template<>
inline void RequireDenseFloatingPointType<arma::fmat>() { }

template<typename MatType>
void RequireFloatingPointType()
{
#ifndef ENS_DISABLE_TYPE_CHECKS
  static_assert(sizeof(MatType) == 0,
      "The given MatType must be arma::mat, arma::fmat, arma::sp_mat, or "
      "arma::sp_fmat, or it is not known to work!  If you would like to try "
      "anyway, set the preprocessor macro ENS_DISABLE_TYPE_CHECKS before "
      "including ensmallen.hpp.  However, you get to pick up all the pieces if "
      "there is a failure!");
#endif
}

template<>
inline void RequireFloatingPointType<arma::mat>() { }
template<>
inline void RequireFloatingPointType<arma::fmat>() { }
template<>
inline void RequireFloatingPointType<arma::sp_mat>() { }
template<>
inline void RequireFloatingPointType<arma::sp_fmat>() { }

/**
 * Require that the internal element type of the matrix type and gradient type
 * are the same.  A static_assert() will fail if not.
 */
template<typename MatType, typename GradType>
void RequireSameInternalTypes()
{
#ifndef ENS_DISABLE_TYPE_CHECKS
  static_assert(std::is_same<typename MatType::elem_type,
                             typename GradType::elem_type>::value,
      "The internal element types of the given MatType and GradType must be "
      "identical, or it is not known to work!  If you would like to try "
      "anyway, set the preprocessor macro ENS_DISABLE_TYPE_CHECKS before "
      "including ensmallen.hpp.  However, you get to pick up all the pieces if "
      "there is a failure!");
#endif
}

} // namespace ens

#endif

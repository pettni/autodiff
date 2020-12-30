#ifndef CERES_TESTER_HPP_
#define CERES_TESTER_HPP_

#include <ceres/internal/autodiff.h>

#include <utility>

#include "common.hpp"


class CeresTester
{
public:
  static constexpr char name[] = "Ceres";

  template<typename Func, typename Derived>
  void setup(Func &&, const Eigen::PlainObjectBase<Derived> &)
  {}

  template<typename Func, typename Derived>
  void run(
    Func && f,
    const Eigen::PlainObjectBase<Derived> & x,
    typename EigenFunctor<Func, Derived>::JacobianType & J)
  {
    using Scalar = typename Derived::Scalar;

    EigenFunctor<Func, Derived> func(std::forward<Func>(f));
    typename EigenFunctor<Func, Derived>::JacobianTypeRow Jrow;

    const Scalar * prms[1] = {x.data()};
    double ** jac_cols = new double *[func.values()];
    for (size_t i = 0; i < func.values(); i++) {
      jac_cols[i] = Jrow.data() + func.values() * i;
    }

    Scalar F{};
    ceres::internal::AutoDifferentiate<
      EigenFunctor<Func, Derived>::ValuesAtCompileTime,
      ceres::internal::StaticParameterDims<Derived::RowsAtCompileTime>
    >(
      func, prms, func.values(), &F, jac_cols
    );

    delete[] jac_cols;

    J = Jrow;
  }
};

#endif  // CERES_TESTER_HPP_

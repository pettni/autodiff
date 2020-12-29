#ifndef AUTODIFF_TESTERS_HPP_
#define AUTODIFF_TESTERS_HPP_

#include <autodiff/common/meta.hpp>
#include <autodiff/forward.hpp>
#include <autodiff/forward/eigen.hpp>
#include <autodiff/reverse.hpp>
#include <autodiff/reverse/eigen.hpp>

#include <utility>

#include "common.hpp"


class AutodiffFwdTester
{
public:
  static constexpr char name[] = "AutodiffFwd";

  template<typename Func, typename Derived>
  void setup(Func &&, const Eigen::PlainObjectBase<Derived> &)
  {}

  template<typename Func, typename Derived>
  void run(
    Func && f,
    const Eigen::PlainObjectBase<Derived> & x,
    typename EigenFunctor<Func, Derived>::JacobianType & J)
  {
    auto x_ad = x.template cast<autodiff::dual>().eval();
    J = autodiff::forward::jacobian(
      std::forward<Func>(f),
      autodiff::forward::wrt(x_ad),
      autodiff::forward::at(x_ad)
    );
  }
};

class AutodiffRevTester
{
public:
  static constexpr char name[] = "AutodiffRev";

  template<typename Func, typename Derived>
  void setup(Func &&, const Eigen::PlainObjectBase<Derived> &)
  {}

  template<typename Func, typename Derived>
  void run(
    Func && f, const Eigen::PlainObjectBase<Derived> & x,
    typename EigenFunctor<Func, Derived>::JacobianType & J)
  {
    auto x_ad = x.template cast<autodiff::var>().eval();
    auto F = f(x_ad);

    if constexpr (EigenFunctor<Func, Derived>::JacobianType::RowsAtCompileTime != -1) {
      autodiff::detail::For<EigenFunctor<Func, Derived>::JacobianType::RowsAtCompileTime>(
        [&J, &F, &x_ad](auto i) {
          J.row(i) = autodiff::reverse::gradient(F(i), x_ad);
        });
    } else {
      for (auto i = 0u; i != F.rows(); ++i) {
        J.row(i) = autodiff::reverse::gradient(F(i), x_ad);
      }
    }
  }
};

#endif  // AUTODIFF_TESTERS_HPP_

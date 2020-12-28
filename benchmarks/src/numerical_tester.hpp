#ifndef SRC__NUMERICAL_TESTER_HPP_
#define SRC__NUMERICAL_TESTER_HPP_

#include <unsupported/Eigen/NumericalDiff>

#include <utility>

#include "common.hpp"


class NumericalTester
{
public:
  static constexpr char name[] = "Numerical";

  template<typename Func, typename Derived>
  void setup(Func &&, const Eigen::PlainObjectBase<Derived> &)
  {}

  template<typename Func, typename Derived>
  void run(
    Func && f,
    const Eigen::PlainObjectBase<Derived> & x,
    typename EigenFunctor<Func, Derived>::JacobianType & J)
  {
    Eigen::NumericalDiff func(EigenFunctor<Func, Derived>(std::forward<Func>(f)));
    func.df(x, J);
  }
};

#endif  // SRC__NUMERICAL_TESTER_HPP_

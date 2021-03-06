#ifndef SACADO_TESTER_HPP_
#define SACADO_TESTER_HPP_

#include <Sacado.hpp>

#include <utility>

#include "common.hpp"


class SacadoTester
{
public:
  static constexpr char name[] = "Sacado";

  template<typename Func, typename Derived>
  void setup(Func &&, const Eigen::PlainObjectBase<Derived> &)
  {}

  template<typename Func, typename Derived>
  void run(
    Func && f,
    const Eigen::PlainObjectBase<Derived> & x,
    typename EigenFunctor<Func, Derived>::JacobianType & J)
  {
    Eigen::Matrix<Sacado::Fad::DFad<double>, Derived::RowsAtCompileTime, 1> x_ad(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
      x_ad(i) = Sacado::Fad::DFad<double>(x.size(), i, x(i));
    }

    auto y_ad = f(x_ad);

    for (std::size_t j = 0; j < y_ad.size(); ++j) {
      for (std::size_t i = 0; i < x.size(); ++i) {
        J(j, i) = y_ad(j).dx(i);
      }
    }
  }
};

#endif  // SACADO_TESTER_HPP_

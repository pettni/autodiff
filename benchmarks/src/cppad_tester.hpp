#ifndef CPPAD_TESTER_HPP_
#define CPPAD_TESTER_HPP_

#include <cppad/cppad.hpp>

#include "common.hpp"


class CppADTester
{
public:
  static constexpr char name[] = "CppAD";

  template<typename Func, typename Derived>
  void setup(Func && f, const Eigen::PlainObjectBase<Derived> & x)
  {
    Eigen::Matrix<CppAD::AD<double>, Eigen::Dynamic, 1> ax = x.template cast<CppAD::AD<double>>();

    CppAD::Independent(ax);
    Eigen::Matrix<CppAD::AD<double>, Eigen::Dynamic, 1> ay = f(ax);
    f_ad = CppAD::ADFun<double>(ax, ay);

    f_ad.optimize();
  }

  template<typename Func, typename Derived>
  void run(
    Func &&,
    const Eigen::PlainObjectBase<Derived> & x,
    typename EigenFunctor<Func, Derived>::JacobianType & J)
  {
    Eigen::VectorXd x_dyn = x;
    auto j_ad = f_ad.Jacobian(x_dyn);
    J = Eigen::Map<typename EigenFunctor<Func, Derived>::JacobianTypeRowMajor>(
      j_ad.data(), f_ad.Range(), f_ad.Domain()
    );
  }

private:
  CppAD::ADFun<double> f_ad;
};

#endif  // CPPAD_TESTER_HPP_
